"""Pinned d3-mini text adapter runtime: native d3 prompt, FP32 code readout, language LoRA."""

import argparse
import functools
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import sys
from pathlib import Path

from . import ROOT, models, options, settings
from .jevk5 import load_adapter_weights

MAX_TOKENS = 4096
TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)
PINS = models.pins(models.D3)


def base_path(download=False):
    """The weights every d3 load uses: the pinned release, or, for the validator's adapter regression
    suite only, a verified base candidate named by ZILS_D3_CANDIDATE_BASE (weights intentionally unpinned)."""
    candidate = settings.get("ZILS_D3_CANDIDATE_BASE")
    if candidate:
        path = Path(candidate).resolve(strict=True)
        validate_candidate(path)
        return path
    return pinned_base_path(download)


def pinned_base_path(download=False):
    from huggingface_hub import snapshot_download

    override = settings.get("ZILS_D3_BASE_DIR")
    profile = models.spec(models.D3)
    root = (
        Path(override).resolve(strict=True)
        if override
        else Path(
            snapshot_download(
                repo_id=profile["base"],
                revision=profile["base_revision"],
                cache_dir=str(
                    Path(os.environ.get("HF_HOME", str(ROOT / ".cache/huggingface"))) / "hub"
                ),
                allow_patterns=list(PINS["files"]),
                local_files_only=not download,
            )
        )
    )
    for name, expected in PINS["files"].items():
        with (root / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise OSError(f"d3 base file {name} differs from its pin")
    return root


@functools.cache
def vendor():
    """d3_runtime from the pinned release only; a base candidate supplies weights, never code."""
    root = pinned_base_path()
    sys.path.insert(0, str(root))
    import d3_runtime

    if Path(d3_runtime.__file__).parent != root:
        raise OSError("d3_runtime was imported from outside the pinned release")
    return d3_runtime


class Encoder:
    """The vendor's prompt and tokenizer without its weights, so CPU hosts can validate inputs."""

    def __init__(self, root):
        from transformers import AutoTokenizer

        self.native = d3_runtime = vendor()
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        # D3.text renders with these codes; the vendor refuses to load when they differ from the config.
        self.codes, _ = d3_runtime.answer_codes(self.tokenizer)
        config = json.loads((Path(root) / "decision_config.json").read_text())
        if config["prompt"] != "d3" or config["codes"] != self.codes:
            raise OSError("d3 answer codes differ from its tokenizer")

    def encode(self, state, question, images=(), videos=()):
        if images or videos:
            raise ValueError("d3 image inputs arrive in a later release")
        declared = options(question)
        self.native._json_value(state)
        normalized = self.native.normalize_question(question)
        if normalized.keys != declared:
            raise ValueError("d3 outcomes differ from the declared outcomes")
        text = self.native.render(self.tokenizer, "d3", state, normalized.rendered, self.codes)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids or len(ids) > MAX_TOKENS:
            raise ValueError("Decision exceeds the 4096-token d3 input limit; shorten the example")
        return ids, normalized.keys


def validate_inputs(splits):
    from .miner_grading import workload_profile

    encoder, lengths = Encoder(base_path()), []
    for split, rows in splits.items():
        for row in rows:
            ids, _ = encoder.encode(row["state"], row["question"])
            if split == "train":
                lengths.append(len(ids))
    return workload_profile(lengths, model=models.D3)


def verify_runtime():
    if "%d.%d" % sys.version_info[:2] != PINS["python"]:
        raise OSError("d3 requires the pinned isolated Python 3.12 runtime")
    for name, version in PINS["packages"].items():
        if importlib.metadata.version(name).split("+")[0] != version.split("+")[0]:
            raise OSError(f"d3 runtime requires {name}=={version}")
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as operations

    if torch.version.cuda != PINS["cuda"]:
        raise OSError("d3 requires the pinned CUDA 13.0 PyTorch build")
    for name, package in (
        ("causal_conv1d_fn", "causal_conv1d"),
        ("causal_conv1d_update", "causal_conv1d"),
        ("torch_chunk_gated_delta_rule", "fla."),
        ("torch_recurrent_gated_delta_rule", "fla."),
    ):
        scope = inspect.getclosurevars(getattr(operations, name)).nonlocals
        if not scope.get("is_new_implementation") or not getattr(
            scope.get("implementation"), "__module__", ""
        ).startswith(package):
            raise OSError(f"d3 fast kernel is unavailable: {name}")


def validate_adapter_config(checkpoint, targets):
    path = Path(checkpoint) / "adapter_config.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16384:
        raise ValueError("d3 adapter configuration must be a bounded regular file")
    config = json.loads(path.read_text())
    profile = models.spec(models.D3)
    required = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "base_model_name_or_path": profile["base"],
        "revision": profile["base_revision"],
        "modules_to_save": None,
    }
    if not isinstance(config, dict):
        raise ValueError("Adapter architecture differs from the pinned d3 recipe")
    wrong = [k for k, v in required.items() if config.get(k) != v]
    if set(config.get("target_modules", [])) != set(targets):
        wrong.append("target_modules")
    if wrong:
        raise ValueError(f"Adapter {', '.join(wrong)} differs from the pinned d3 recipe")


def validate_candidate(checkpoint):
    """A base candidate changes weights only; every contract file, config.json included, is the pin."""
    checkpoint = Path(checkpoint)
    meta = models.metadata(checkpoint)
    if meta is None or meta["model"] != models.D3 or meta["kind"] != "candidate":
        raise ValueError("base candidate requires d3 candidate metadata")
    for entry in checkpoint.iterdir():
        if entry.name not in models.D3_CANDIDATE_FILES:
            raise ValueError(f"base candidate contains an unexpected entry: {entry.name}")
    for name in models.D3_CANDIDATE_FILES:
        path = checkpoint / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"base candidate requires a regular {name}")
        if name in models.D3_CONTRACT_FILES:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != PINS["files"][name]:
                    raise ValueError(f"base candidate {name} differs from the pinned parent")
    # Weights must load exactly: the parent's tensor names, held in the two pinned shard files.
    from safetensors import SafetensorError, safe_open

    index = "model.safetensors.index.json"
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    parent = json.loads((pinned_base_path() / index).read_text())["weight_map"]
    mapped = json.loads((checkpoint / index).read_text())
    mapped = mapped.get("weight_map") if isinstance(mapped, dict) else None
    if not isinstance(mapped, dict):
        raise ValueError("base candidate index has no weight_map")
    if extra := sorted(set(mapped) ^ set(parent)):
        raise ValueError(f"base candidate index keys differ from the pinned parent: {extra[0]}")
    for key, shard in mapped.items():
        if shard not in shards:
            raise ValueError(f"base candidate index maps {key} outside the pinned shards")
    for shard in shards:
        try:
            with safe_open(checkpoint / shard, framework="pt") as stream:
                held = set(stream.keys())
        except SafetensorError as error:
            raise ValueError(f"base candidate {shard} is not a safetensors file: {error}") from None
        wrong = sorted(held ^ {key for key, value in mapped.items() if value == shard})
        if wrong:
            raise ValueError(f"base candidate {shard} tensors differ from its index: {wrong[0]}")


class DecisionModel:
    def __init__(self, checkpoint, device, train=False):
        import torch
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        from safetensors.torch import load_file

        # Setup invariants raise OSError (node unavailable); only checkpoint faults blame the miner.
        self.meta = models.metadata(checkpoint)
        if self.meta is None or self.meta["model"] != models.D3:
            raise ValueError("d3 requires a d3 checkpoint")
        candidate = self.meta["kind"] == "candidate"
        if candidate and train:
            raise ValueError("adapters train on the pinned base, not on a base candidate")
        if device != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise OSError("d3 requires a BF16-capable CUDA GPU")
        verify_runtime()
        self.device = device
        torch.set_num_threads(6)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        root = base_path()
        if candidate:
            validate_candidate(checkpoint)
            root = Path(checkpoint)
        self.encoder = Encoder(root)
        self.tokenizer = self.encoder.tokenizer
        # verify="none": every pinned file was hashed above, and the vendor manifest also lists
        # README.md and assets/* that the pinned snapshot does not download.
        self.runtime = self.encoder.native.D3.from_pretrained(
            root,
            device="cuda:0",
            batch_size=1,
            max_length=MAX_TOKENS,
            verify="none",
            local_files_only=True,
            permutation_average=False,
        )
        runtime = self.runtime
        if (
            runtime.fast is not None
            or runtime.attention_mode != PINS["attention"]
            or runtime.temperature != PINS["temperature"]
            or runtime.readout_dtype != PINS["readout_dtype"]
            or runtime.prompt != "d3"
            or runtime.backbone_type != "qwen3_5"
            or runtime.max_length != MAX_TOKENS
        ):
            raise OSError("d3 runtime differs from the pinned configuration")
        self.base = runtime.backbone
        self.readout = runtime.readout
        if self.readout.dtype != torch.float32 or self.readout.shape[0] != 255:
            raise OSError("d3 readout must be FP32 with 255 codes")
        if any(p.dtype != torch.bfloat16 or not p.is_cuda for p in self.base.parameters()):
            raise OSError("d3 backbone must be BF16 on the GPU")
        self.base.config.use_cache = False
        self.base.config.text_config.use_cache = False
        self.peft = None
        if train or self.meta["kind"] == "adapter":
            targets = [
                name
                for name, _ in self.base.named_modules()
                if name.startswith("language_model.layers.") and name.rsplit(".", 1)[-1] in TARGETS
            ]
            if len(targets) != 152:
                raise OSError("d3 language adapter target count changed")
            if self.meta["kind"] == "adapter":
                validate_adapter_config(checkpoint, targets)
            self.peft = get_peft_model(
                self.base,
                LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    target_modules=targets,
                    bias="none",
                ),
            )
            # PEFT shortens target names during injection; persist the exact language-only scope.
            self.peft.peft_config["default"].target_modules = set(targets)
            trainable = [(n, p) for n, p in self.peft.named_parameters() if p.requires_grad]
            if (
                len(trainable) != 304
                or sum(p.numel() for _, p in trainable) != 14_376_960
                or not all(
                    "lora_" in n
                    and "language_model.layers." in n
                    and p.dtype == torch.float32
                    and p.is_cuda
                    for n, p in trainable
                )
                or any(torch.count_nonzero(p) for n, p in trainable if "lora_B" in n)
            ):
                raise OSError("d3 LoRA initialisation differs from the pinned recipe")
            if self.meta["kind"] == "adapter":
                stored = load_file(str(Path(checkpoint) / "adapter_model.safetensors"))
                if any(v.dtype != torch.bfloat16 for v in stored.values()):
                    raise ValueError("d3 adapter tensors must be bfloat16")
                load_adapter_weights(self.peft, checkpoint)
                actual = get_peft_model_state_dict(self.peft)
                if any(
                    not torch.equal(actual[k].detach().cpu(), v.to(actual[k].dtype))
                    for k, v in stored.items()
                ):
                    raise OSError("d3 adapter tensors did not load exactly")
                if not train:
                    self.peft.requires_grad_(False)
            self.peft.eval()

    def encode(self, state, question):
        return self.encoder.encode(state, question)

    def logits(self, ids, count):
        import torch

        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        hidden = self.base(
            input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False
        ).last_hidden_state[:, -1]
        if hidden.dtype != torch.bfloat16:
            raise OSError("d3 backbone must return BF16 hidden states")
        # The published d3 readout is FP32 even during BF16 LoRA training.
        with torch.autocast(self.device, enabled=False):
            values = (hidden.float() @ self.readout[:count].T)[0]
        if values.dtype != torch.float32:
            raise OSError("d3 readout must produce FP32 logits")
        if not torch.isfinite(values).all():
            raise RuntimeError("d3 produced nonfinite logits")
        return values

    def probabilities(self, ids, keys, question, temperature):
        import torch

        if self.base.training or torch.is_autocast_enabled("cuda"):
            raise OSError("d3 inference requires eval mode without autocast")
        with torch.inference_mode():
            values = (self.logits(ids, len(keys)).double() / temperature).softmax(-1).cpu().tolist()
        if not all(math.isfinite(v) for v in values) or abs(sum(values) - 1) >= 1e-6:
            raise RuntimeError("d3 produced an invalid distribution")
        return dict(zip(keys, values, strict=True))

    def predict(self, state, question):
        ids, keys = self.encode(state, question)
        return self.probabilities(ids, keys, question, self.meta["temperature"])


def parity(model, cases):
    """Our readout against the vendor's own D3.probabilities on the same token ids."""
    import torch

    worst = 0.0
    for case in cases:
        ids, keys = model.encode(case["state"], case["question"])
        ours = list(model.probabilities(ids, keys, case["question"], 1.0).values())
        with torch.inference_mode():
            reference = model.runtime.probabilities([ids], [len(keys)])[0]
        difference = max(abs(a - b) for a, b in zip(ours, reference, strict=True))
        if difference >= 1e-6 or ours.index(max(ours)) != reference.index(max(reference)):
            raise RuntimeError(f"d3 parity failed on case {case['id']}: {difference}")
        worst = max(worst, difference)
    return {"cases": len(cases), "max_difference": worst}


def main():
    from .jevk5 import train

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reference = sub.add_parser("reference")
    reference.add_argument("--out", required=True)
    reference.add_argument("--no-download", action="store_true")
    trainer = sub.add_parser("train")
    for name in ("data", "reference", "out"):
        trainer.add_argument("--" + name, required=True)
    trainer.add_argument("--seed", required=True, type=int)
    trainer.add_argument("--device", choices=("cuda",), default="cuda")
    checker = sub.add_parser("parity")
    checker.add_argument("--reference", required=True)
    checker.add_argument("--cases", required=True)
    args = parser.parse_args()
    if args.command == "reference":
        base_path(download=not args.no_download)
        path = Path(args.out)
        path.mkdir(parents=True, exist_ok=False)
        models.write_metadata(path, kind="base", model=models.D3)
        (path / "model.json").chmod(0o444)
    elif args.command == "train":
        train(args, model_id=models.D3)
    else:
        cases = [
            json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()
        ]
        print(json.dumps(parity(DecisionModel(args.reference, "cuda"), cases)))


if __name__ == "__main__":
    main()
