"""Pinned d3-family adapter runtimes: native prompts, FP32 readout, language LoRA."""

import argparse
import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import sys
from pathlib import Path

from . import ROOT, models, options, settings
from .jevk5 import asset_bytes, load_adapter_weights

MAX_TOKENS = 4096
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
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


def profile(model):
    """A d3-family profile id: the vendor base or a Zils base promoted from it (same graph, new weights)."""
    if model not in models.d3_family():
        raise ValueError("not a d3 profile")
    return model


def pins(model=models.D3):
    """Pins of a d3-family profile; the vendor base's are the module constant."""
    return PINS if profile(model) == models.D3 else models.pins(model)


def base_path(download=False, model=models.D3):
    """The weights every load of `model` uses: its pinned release, or, for the validator's adapter regression
    suite only, a verified base candidate named by its candidate-base setting (weights intentionally unpinned)."""
    candidate = settings.get(models.candidate_base_env(profile(model)))
    if candidate:
        path = Path(candidate).resolve(strict=True)
        validate_candidate(path, model)
        return path
    return pinned_base_path(download, model)


def pinned_base_path(download=False, model=models.D3):
    from huggingface_hub import snapshot_download

    override = settings.get("ZILS_D3_BASE_DIR") if model == models.D3 else None
    files = pins(model)["files"]
    profile_spec = models.spec(model)
    root = (
        Path(override).resolve(strict=True)
        if override
        else Path(
            snapshot_download(
                repo_id=profile_spec["base"],
                revision=profile_spec["base_revision"],
                cache_dir=str(
                    Path(os.environ.get("HF_HOME", str(ROOT / ".cache/huggingface"))) / "hub"
                ),
                allow_patterns=list(files),
                local_files_only=not download,
            )
        )
    )
    for name, expected in files.items():
        with (root / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise OSError(f"d3 base file {name} differs from its pin")
    return root


@functools.cache
def vendor(model=models.D3):
    """d3_runtime from the vendor's pinned release only; promoted bases and candidates supply weights, never code."""
    # Promoted checkpoints provide weights, while code always comes from their root release.
    source = models.D3_POOL_MODELS[models.pool_id(model)]
    root = pinned_base_path(model=source)
    sys.path.insert(0, str(root))
    import d3_runtime

    if Path(d3_runtime.__file__).parent != root:
        raise OSError("d3_runtime was imported from outside the pinned release")
    return d3_runtime


def check_media(images=(), videos=(), model=models.D3):
    """Refuse media outside the d3 limits before any processor or GPU work; return RGB images."""
    limits = models.spec(model)["inputs"]
    if len(images) > limits["images"]:
        raise ValueError(f"d3 accepts at most {limits['images']} images")
    if videos and not limits["video"]:
        raise ValueError("d3 does not accept video")
    if len(videos) > 1:
        raise ValueError("d3 accepts at most 1 video")
    if any(not isinstance(image, bytes) or image[:8] != PNG_SIGNATURE for image in images):
        raise ValueError("d3 images must be canonical PNG bytes")
    if any(Path(video).is_symlink() or not Path(video).is_file() for video in videos):
        raise ValueError("d3 videos must be regular MP4 file paths")
    pictures = []
    if images:
        from PIL import Image

        for image in images:
            try:
                with Image.open(io.BytesIO(image)) as picture:
                    pictures.append(picture.convert("RGB"))
            except (OSError, SyntaxError, Image.DecompressionBombError):
                raise ValueError("d3 images must be canonical PNG bytes") from None
    return pictures


class Encoder:
    """The vendor's prompt and tokenizer without its weights, so CPU hosts can validate inputs."""

    def __init__(self, root, model=models.D3):
        from transformers import AutoTokenizer

        self.model = profile(model)
        self.native = d3_runtime = vendor(model)
        self.root = root
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        # D3.text renders with these codes; the vendor refuses to load when they differ from the config.
        self.codes, _ = d3_runtime.answer_codes(self.tokenizer)
        config = json.loads((Path(root) / "decision_config.json").read_text())
        if config["prompt"] != "d3" or config["codes"] != self.codes:
            raise OSError("d3 answer codes differ from its tokenizer")

    @functools.cached_property
    def processor(self):
        """The pinned AutoProcessor (no weights), loaded only for image or video prompts."""
        processor = self.native.load_processor(self.root)
        self.native.configure_video_processor(processor)
        return processor

    def inputs(self, state, question, images=(), videos=()):
        """(batch, names): token ids for text, or the processor batch for canonical PNGs and MP4 paths."""
        pictures = check_media(images, videos, model=getattr(self, "model", models.D3))
        declared = options(question)
        self.native._json_value(state)
        normalized = self.native.normalize_question(question)
        if normalized.keys != declared:
            raise ValueError("d3 outcomes differ from the declared outcomes")
        if not images and not videos:
            text = self.native.render(self.tokenizer, "d3", state, normalized.rendered, self.codes)
            batch = ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        else:
            # The vendor's D3.image_text / D3.video_text prompt, expanded by the pinned processor.
            native, rendered = self.native, normalized.rendered
            messages = (
                native.video_messages("d3", state, rendered, self.codes, len(images), len(videos))
                if videos
                else native.image_messages("d3", state, rendered, self.codes, len(images))
            )
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            if text.count(native.IMAGE_TOKEN) != len(images) or text.count(
                native.VIDEO_TOKEN
            ) != len(videos):
                raise ValueError("d3 state or question contains an image or video placeholder")
            media = {"images": pictures} if pictures else {}
            if videos:
                decoded = [native.load_video(str(video)) for video in videos]
                media.update(
                    videos=[video.frames for video in decoded],
                    video_metadata=[video.metadata() for video in decoded],
                )
            batch = self.processor(text=[text], return_tensors="pt", **media)
            ids = batch["input_ids"][0].tolist()
        maximum = models.spec(getattr(self, "model", models.D3))["max_input_tokens"]
        if not ids or len(ids) > maximum:
            raise ValueError("Decision exceeds the 4096-token d3 input limit; shorten the example")
        return batch, normalized.keys

    def encode(self, state, question, images=(), videos=()):
        batch, names = self.inputs(state, question, images, videos)
        return (batch if isinstance(batch, list) else batch["input_ids"][0].tolist()), names


def validate_inputs(splits, root=None, model=models.D3):
    from .miner_grading import workload_profile

    encoder, lengths = Encoder(base_path(model=model), model=model), []
    for split, rows in splits.items():
        for row in rows:
            images = ()
            if "images" in row:
                if root is None:
                    raise ValueError("image rows need a dataset root")
                images = [asset_bytes(Path(root), name) for name in row["images"]]
            ids, _ = encoder.encode(row["state"], row["question"], images=images)
            if split == "train":
                lengths.append(len(ids))
    return workload_profile(lengths, model=model)


def verify_runtime(model=models.D3):
    required = pins(model)
    if "%d.%d" % sys.version_info[:2] != required["python"]:
        raise OSError("d3 requires the pinned isolated Python 3.12 runtime")
    for name, version in required["packages"].items():
        if importlib.metadata.version(name).split("+")[0] != version.split("+")[0]:
            raise OSError(f"d3 runtime requires {name}=={version}")
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as operations

    if torch.version.cuda != required["cuda"]:
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
    meta = models.metadata(checkpoint)
    base = models.spec(profile(meta["model"] if meta else models.D3))
    required = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "base_model_name_or_path": base["base"],
        "revision": base["base_revision"],
        "modules_to_save": None,
    }
    if not isinstance(config, dict):
        raise ValueError("Adapter architecture differs from the pinned d3 recipe")
    wrong = [k for k, v in required.items() if config.get(k) != v]
    if set(config.get("target_modules", [])) != set(targets):
        wrong.append("target_modules")
    if wrong:
        raise ValueError(f"Adapter {', '.join(wrong)} differs from the pinned d3 recipe")


def validate_candidate(checkpoint, model=None):
    """A base candidate changes weights only; every contract file, config.json included, is its parent's pin.
    `model`, when given, is the parent profile the caller expects."""
    checkpoint = Path(checkpoint)
    meta = models.metadata(checkpoint)
    if (
        meta is None
        or meta["model"] not in models.d3_family()
        or meta["kind"] != "candidate"
        or (model is not None and meta["model"] != model)
    ):
        raise ValueError("base candidate requires d3 candidate metadata")
    files = pins(meta["model"])["files"]
    candidate_files = models.candidate_files(meta["model"], "candidate")
    for entry in checkpoint.iterdir():
        if entry.name not in candidate_files:
            raise ValueError(f"base candidate contains an unexpected entry: {entry.name}")
    for name in candidate_files:
        path = checkpoint / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"base candidate requires a regular {name}")
        if name in models.D3_CONTRACT_FILES:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != files[name]:
                    raise ValueError(f"base candidate {name} differs from the pinned parent")
    # Read tensor headers only: names, shapes and dtypes must match the pinned parent.
    from safetensors import SafetensorError, safe_open

    shards = [
        n for n in candidate_files if n.endswith(".safetensors") and n != "readout.safetensors"
    ]
    parent_root = pinned_base_path(model=meta["model"])
    parent = weight_map(parent_root)
    mapped = weight_map(checkpoint)
    if extra := sorted(set(mapped) ^ set(parent)):
        raise ValueError(f"base candidate index keys differ from the pinned parent: {extra[0]}")
    for key, shard in mapped.items():
        if shard not in shards:
            raise ValueError(f"base candidate index maps {key} outside the pinned shards")
    with contextlib.ExitStack() as stack:
        references = {
            name: stack.enter_context(safe_open(parent_root / name, framework="pt"))
            for name in set(parent.values())
        }
        for shard in shards:
            try:
                with safe_open(checkpoint / shard, framework="pt") as stream:
                    held = set(stream.keys())
                    wrong = sorted(held ^ {key for key, value in mapped.items() if value == shard})
                    if wrong:
                        raise ValueError(
                            f"base candidate {shard} tensors differ from its index: {wrong[0]}"
                        )
                    for key in held:
                        actual = stream.get_slice(key)
                        expected = references[parent[key]].get_slice(key)
                        if (
                            actual.get_shape() != expected.get_shape()
                            or actual.get_dtype() != expected.get_dtype()
                        ):
                            raise ValueError(
                                f"base candidate tensor schema differs from parent: {key}"
                            )
            except SafetensorError as error:
                raise ValueError(
                    f"base candidate {shard} is not a safetensors file: {error}"
                ) from None


def weight_map(root):
    """Read the pinned layout without loading model tensors into memory."""
    from safetensors import SafetensorError, safe_open

    root = Path(root)
    index = root / "model.safetensors.index.json"
    if index.exists():
        value = json.loads(index.read_text())
        mapped = value.get("weight_map") if isinstance(value, dict) else None
        if (
            not isinstance(mapped, dict)
            or not mapped
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapped.items())
        ):
            raise ValueError("base candidate index has no valid weight_map")
        return mapped
    try:
        with safe_open(root / "model.safetensors", framework="pt") as stream:
            return {key: "model.safetensors" for key in stream.keys()}
    except SafetensorError as error:
        raise ValueError(f"base candidate is not a safetensors file: {error}") from None


TARGET_MODULES = 152


def write_candidate(adapter, out):
    """Merge a pinned-recipe adapter into the parent backbone and write a d3 base candidate.

    The candidate is the parent's weight files with ``W + (lora_alpha / r) * B @ A`` on every LoRA target
    (computed in FP32, stored BF16), the parent's index and contract files byte for byte, and candidate
    metadata; nothing else, so `validate_candidate` accepts it. Returns a provenance record for the caller
    to keep beside the artifact (the artifact itself may hold no extra file).
    """
    import shutil

    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    adapter, out = Path(adapter), Path(out)
    meta = models.metadata(adapter)
    if meta is None or meta["model"] not in models.d3_family() or meta["kind"] != "adapter":
        raise ValueError("merge requires a d3 adapter checkpoint")
    parent = meta["model"]
    root = pinned_base_path(model=parent)
    index = "model.safetensors.index.json"
    layout = weight_map(root)
    targets = {
        key[: -len(".weight")]
        for key in layout
        if key.startswith("language_model.layers.")
        and key.endswith(".weight")
        and key.rsplit(".", 2)[-2] in TARGETS
    }
    if len(targets) != pins(parent).get("target_modules", TARGET_MODULES):
        raise OSError("d3 language adapter target count changed")
    validate_adapter_config(adapter, targets)
    pairs = {}
    for key, value in load_file(str(adapter / "adapter_model.safetensors")).items():
        module, found, rest = key.removeprefix("base_model.model.").rpartition(".lora_")
        which = rest.split(".")[0]
        if (
            not found
            or module not in targets
            or which not in ("A", "B")
            or value.dtype != torch.bfloat16
        ):
            raise ValueError(f"adapter tensor outside the pinned recipe: {key}")
        pairs.setdefault(module, {})[which] = value
    if set(pairs) != targets or any(set(p) != {"A", "B"} for p in pairs.values()):
        raise ValueError("adapter must hold lora_A and lora_B for every target")
    scaling = 32 / 16  # lora_alpha / r of the pinned recipe
    out.mkdir(parents=True, exist_ok=False)
    merged = 0
    for shard in sorted(set(layout.values())):
        tensors = {}
        with safe_open(str(root / shard), framework="pt") as stream:
            for key in stream.keys():
                tensor = stream.get_tensor(key)
                module = key[: -len(".weight")] if key.endswith(".weight") else None
                if module in pairs:
                    delta = pairs[module]["B"].float() @ pairs[module]["A"].float()
                    if tensor.dtype != torch.bfloat16 or delta.shape != tensor.shape:
                        raise ValueError(f"adapter shape differs from the backbone at {module}")
                    tensor = (tensor.float() + scaling * delta).to(torch.bfloat16)
                    merged += 1
                tensors[key] = tensor
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
    if merged != len(pairs):
        raise RuntimeError("adapter targets missing from the backbone shards")
    for name in (*((index,) if (root / index).exists() else ()), *models.D3_CONTRACT_FILES):
        shutil.copyfile(root / name, out / name)
    models.write_metadata(out, kind="candidate", model=parent)
    validate_candidate(out, parent)

    def sha(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    return {
        "version": "zils-base-candidate/v1",
        "parent": models.spec(parent),
        "parent_files_sha256": dict(pins(parent)["files"]),
        "adapter_sha256": sha(adapter / "adapter_model.safetensors"),
        "recipe": {"merge": "lora_alpha/r scaled B@A in fp32, stored bf16", "scaling": scaling},
        "merged_tensors": merged,
        "files_sha256": {p.name: sha(p) for p in sorted(out.iterdir())},
    }


class DecisionModel:
    def __init__(self, checkpoint, device, train=False):
        import torch
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        from safetensors.torch import load_file

        # Setup invariants raise OSError (node unavailable); only checkpoint faults blame the miner.
        self.meta = models.metadata(checkpoint)
        if self.meta is None or self.meta["model"] not in models.d3_family():
            raise ValueError("d3 requires a d3 checkpoint")
        candidate = self.meta["kind"] == "candidate"
        if candidate and train:
            raise ValueError("adapters train on the pinned base, not on a base candidate")
        if device != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise OSError("d3 requires a BF16-capable CUDA GPU")
        verify_runtime(self.meta["model"])
        required = pins(self.meta["model"])
        maximum = models.spec(self.meta["model"])["max_input_tokens"]
        self.device = device
        torch.set_num_threads(6)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        root = base_path(model=self.meta["model"])
        if candidate:
            validate_candidate(checkpoint, self.meta["model"])
            root = Path(checkpoint)
        self.encoder = Encoder(root, model=self.meta["model"])
        self.tokenizer = self.encoder.tokenizer
        # verify="none": every pinned file was hashed above, and the vendor manifest also lists
        # README.md and assets/* that the pinned snapshot does not download.
        self.runtime = self.encoder.native.D3.from_pretrained(
            root,
            device="cuda:0",
            batch_size=1,
            max_length=maximum,
            verify="none",
            local_files_only=True,
            permutation_average=False,
        )
        runtime = self.runtime
        if (
            runtime.fast is not None
            or runtime.attention_mode != required["attention"]
            or runtime.temperature != required["temperature"]
            or runtime.readout_dtype != required["readout_dtype"]
            or runtime.prompt != "d3"
            or runtime.backbone_type != "qwen3_5"
            or runtime.max_length != maximum
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
            if len(targets) != required.get("target_modules", TARGET_MODULES):
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

    def logits(self, inputs, count):
        """`inputs` is a token-id list (text) or the encoder's processor batch (images, video)."""
        import torch

        if isinstance(inputs, list):
            tokens = torch.tensor([inputs], dtype=torch.long, device=self.device)
            inputs = {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}
        else:
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
        hidden = self.base(**inputs, use_cache=False).last_hidden_state[:, -1]
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

    def predict(self, state, question, images=(), videos=()):
        inputs, keys = self.encoder.inputs(state, question, images, videos)
        return self.probabilities(inputs, keys, question, self.meta["temperature"])


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
    reference.add_argument("--model", choices=models.d3_family(), default=models.D3)
    trainer = sub.add_parser("train")
    for name in ("data", "reference", "out"):
        trainer.add_argument("--" + name, required=True)
    trainer.add_argument("--seed", required=True, type=int)
    trainer.add_argument("--device", choices=("cuda",), default="cuda")
    merger = sub.add_parser("candidate", help="merge an adapter into a full-weight base candidate")
    merger.add_argument("--adapter", required=True)
    merger.add_argument("--out", required=True)
    checker = sub.add_parser("parity")
    checker.add_argument("--reference", required=True)
    checker.add_argument("--cases", required=True)
    args = parser.parse_args()
    if args.command == "reference":
        base_path(download=not args.no_download, model=args.model)
        path = Path(args.out)
        path.mkdir(parents=True, exist_ok=False)
        models.write_metadata(path, kind="base", model=args.model)
        (path / "model.json").chmod(0o444)
    elif args.command == "train":
        # The reference checkpoint names the profile, so promoted bases train without new flags.
        train(args, model_id=profile((models.metadata(args.reference) or {}).get("model")))
    elif args.command == "candidate":
        print(json.dumps(write_candidate(args.adapter, args.out), indent=2))
    else:
        cases = [
            json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()
        ]
        print(json.dumps(parity(DecisionModel(args.reference, "cuda"), cases)))


if __name__ == "__main__":
    main()
