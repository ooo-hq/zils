"""Pinned H2O Lightning text adapter runtime and native decision contract."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

from . import ROOT, models, options, settings
from .jevk5 import load_adapter_weights

MAX_TOKENS = 2048
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
PINS = json.loads(Path(__file__).with_name("h2o-pins.json").read_text())


def base_path(download=False):
    from huggingface_hub import snapshot_download

    override = settings.get("ZILS_H2O_BASE_DIR")
    profile = models.spec(models.H2O)
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
                raise OSError(f"Pinned H2O checksum mismatch: {name}")
    return root


class Encoder:
    def __init__(self, root):
        from transformers import AutoTokenizer

        # Only the operator-owned, checksum-verified release supplies executable code.
        source = importlib.util.spec_from_file_location(
            "zils_h2o_native", root / "h2o_lightning_shim.py"
        )
        self.native = importlib.util.module_from_spec(source)
        source.loader.exec_module(self.native)
        self.contract = self.native.Contract(
            json.loads((root / "serve_config.json").read_text()), env={}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            root, local_files_only=True, trust_remote_code=False
        )
        probe = self.contract.question_prompt(
            "{}",
            {
                "type": "choice",
                "criteria": {str(i): "option" for i in range(255)},
            },
        )[0]
        ids = self.tokenizer.encode(probe, add_special_tokens=False)
        self.label_ids = []
        for label in self.contract.labels:
            full = self.tokenizer.encode(probe + " " + label, add_special_tokens=False)
            if full[:-1] != ids or len(full) != len(ids) + 1:
                raise OSError("H2O answer labels must be single tokens")
            self.label_ids.append(full[-1])
        if len(set(self.label_ids)) != 255:
            raise OSError("H2O answer tokens must be distinct")

    def encode(self, state, question):
        question = dict(question)
        if question.get("instructions") is None:
            question.pop("instructions", None)
        elif not isinstance(question["instructions"], str):
            raise ValueError("Decision instructions must be text")
        declared = options(question)
        state, images = self.native.extract_images({"state": state})
        if images:
            raise ValueError("H2O text adapters do not accept images")
        text = self.native.render_state(state, self.contract.state_compact)
        if len(text.encode("utf-8")) > self.contract.max_state_tokens:
            raise ValueError("H2O state exceeds its byte limit; shorten the example")
        prompt, names, _, _ = self.contract.question_prompt(text, question, "decision")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or len(ids) > MAX_TOKENS:
            raise ValueError("Decision exceeds the 2048-token H2O input limit; shorten the example")
        if set(names) != set(declared):
            raise ValueError("H2O outcomes differ from the declared outcomes")
        return ids, names


def validate_inputs(splits):
    from .miner_grading import workload_profile

    encoder, lengths = Encoder(base_path()), []
    for split, rows in splits.items():
        for row in rows:
            ids, _ = encoder.encode(row["state"], row["question"])
            if split == "train":
                lengths.append(len(ids))
    return workload_profile(lengths, model=models.H2O)


def verify_runtime():
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as operations

    if f"{sys.version_info.major}.{sys.version_info.minor}" != PINS["python"]:
        raise OSError("H2O requires the pinned isolated Python 3.12 runtime")
    for name, version in PINS["packages"].items():
        if importlib.metadata.version(name).split("+")[0] != version:
            raise OSError(f"H2O runtime requires {name}=={version}")
    if torch.version.cuda != PINS["cuda"]:
        raise OSError("H2O requires the pinned CUDA 13.0 PyTorch build")
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
            raise OSError(f"H2O fast kernel is unavailable: {name}")


def validate_adapter_config(checkpoint, targets):
    path = Path(checkpoint) / "adapter_config.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16384:
        raise ValueError("H2O adapter configuration must be a bounded regular file")
    config = json.loads(path.read_text())
    profile = models.spec(models.H2O)
    required = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "peft_type": "LORA",
        "base_model_name_or_path": profile["base"],
        "revision": profile["base_revision"],
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
    }
    if (
        not isinstance(config, dict)
        or any(config.get(k) != v for k, v in required.items())
        or set(config.get("target_modules", [])) != set(targets)
    ):
        raise ValueError("Adapter architecture differs from the pinned H2O recipe")


class DecisionModel:
    def __init__(self, checkpoint, device, train=False):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3_5ForConditionalGeneration

        self.meta = models.metadata(checkpoint)
        if self.meta is None or self.meta["model"] != models.H2O:
            raise ValueError("H2O requires an H2O checkpoint")
        if device != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise OSError("H2O requires a BF16-capable CUDA GPU")
        verify_runtime()
        self.device = device
        torch.set_num_threads(6)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        root = base_path()
        self.encoder = Encoder(root)
        self.tokenizer = self.encoder.tokenizer
        self.base, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": device},
            output_loading_info=True,
        )
        if any(
            info.get(k)
            for k in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs")
        ):
            raise OSError("H2O base weights differ from the pinned architecture")
        self.base.requires_grad_(False).eval()
        self.base.config.use_cache = False
        self.readout = self.base.lm_head.weight[self.encoder.label_ids].detach().float().clone()
        self.peft = None
        if train or self.meta["kind"] == "adapter":
            targets = [
                name
                for name, _ in self.base.named_modules()
                if name.startswith("model.language_model.layers.")
                and name.rsplit(".", 1)[-1] in TARGETS
            ]
            if len(targets) != 152:
                raise OSError("H2O language adapter target count changed")
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
            if self.meta["kind"] == "adapter":
                load_adapter_weights(self.peft, checkpoint)
            self.peft.eval()

    def encode(self, state, question):
        return self.encoder.encode(state, question)

    def logits(self, ids, count):
        import torch

        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        hidden = self.base.model(
            input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False
        ).last_hidden_state[:, -1]
        # The published H2O head requires FP32 projection even during BF16 LoRA training.
        with torch.autocast("cuda", enabled=False):
            return torch.nn.functional.linear(hidden.float(), self.readout[:count])[0]

    def probabilities(self, ids, keys, question, temperature):
        import torch

        kind = question["type"]
        with torch.inference_mode():
            logits = self.logits(ids, len(keys)).double().cpu().tolist()
            contract, native = self.encoder.contract, self.encoder.native
            values = native.probabilities(logits, contract.temperature_by_type[kind])
            if kind == "noul":
                yes = native.commit_noul(values[0], contract.noul_floor)
                values = [yes, 1 - yes]
            # Fit a scalar on the native distribution; this matches the shared held-out fitter.
            values = torch.softmax(
                torch.tensor(values, dtype=torch.float64).clamp_min(1e-9).log() / temperature, dim=0
            ).tolist()
        torch.cuda.synchronize()
        return dict(zip(keys, values, strict=True))

    def predict(self, state, question):
        ids, keys = self.encode(state, question)
        return self.probabilities(ids, keys, question, self.meta["temperature"])


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
    args = parser.parse_args()
    if args.command == "reference":
        base_path(download=not args.no_download)
        path = Path(args.out)
        path.mkdir(parents=True, exist_ok=False)
        models.write_metadata(path, kind="base", model=models.H2O)
        (path / "model.json").chmod(0o444)
    else:
        train(args, model_id=models.H2O)


if __name__ == "__main__":
    main()
