"""Pinned JevK5 decision readout and attention-only LoRA training for queued jobs."""

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

from . import ROOT, models, options, settings

MAX_TOKENS = 2048
TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkvz",
    "in_proj_ba",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)
BASE_HASHES = {
    "config.json": "63f47812d0f11118e4d252d2b3ad488707eb9287a11589f4fd382a1d31182724",
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "model.safetensors": "13824e47f2e40fe052f06943976cf742cb366ba305741a111e75a8ebae907a9c",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
    "jevk5_config.json": "0d689fd13d15dc962265e2ae10b56359706ab5d05ad24e00e6334e4c19cf83d2",
}


def base_path(download=False):
    from huggingface_hub import snapshot_download

    override = settings.get("ZILS_JEVK5_BASE_DIR")
    if override:
        root = Path(override).resolve(strict=True)
    else:
        spec = models.spec(models.JEVK5)
        root = Path(
            snapshot_download(
                repo_id=spec["base"],
                revision=spec["base_revision"],
                cache_dir=str(
                    Path(os.environ.get("HF_HOME", str(ROOT / ".cache/huggingface"))) / "hub"
                ),
                allow_patterns=list(BASE_HASHES),
                local_files_only=not download,
            )
        )
    for name, expected in BASE_HASHES.items():
        with (root / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise OSError("pinned JevK5 model or tokenizer checksum mismatch")
    return root


def decision(state, question):
    from jevk5.prompt import decision_options

    keys = options(question)
    if len(keys) > 16:
        raise ValueError("JevK5 queued training supports at most 16 outcomes")
    criterion = question.get(
        "instructions", state.get("decision", "") if isinstance(state, dict) else ""
    )
    if not isinstance(criterion, str):
        raise ValueError("decision instructions must be text")
    pairs = decision_options(question)
    if {k for k, _ in pairs} != set(keys):
        raise ValueError("decision options differ from the declared outcomes")
    return criterion, pairs


def encode(tokenizer, state, question):
    from jevk5.prompt import messages

    criterion, pairs = decision(state, question)
    prompt = tokenizer.apply_chat_template(
        messages(state, criterion, [text for _, text in pairs]),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > MAX_TOKENS:
        raise ValueError("decision exceeds the 2048-token JevK5 input limit; shorten the example")
    return ids, [key for key, _ in pairs]


def validate_inputs(splits):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_path(), local_files_only=True)
    for rows in splits.values():
        for row in rows:
            encode(tokenizer, row["state"], row["question"])


def save_adapter_weights(model, output):
    import torch

    # Keep FP32 optimizer parameters; serialize only LoRA tensors at inference precision.
    state = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items() if "lora_" in k}
    model.save_pretrained(output, state_dict=state, safe_serialization=True)


def load_adapter_weights(model, checkpoint):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    tensors = load_file(str(Path(checkpoint) / "adapter_model.safetensors"))
    expected = get_peft_model_state_dict(model)
    if set(tensors) != set(expected) or any(
        tensors[k].shape != expected[k].shape
        or tensors[k].dtype not in (torch.float32, torch.bfloat16)
        or not torch.isfinite(tensors[k]).all()
        for k in expected
    ):
        raise ValueError("adapter tensors differ from the pinned JevK5 recipe")
    set_peft_model_state_dict(model, {k: v.to(expected[k].dtype) for k, v in tensors.items()})


class DecisionModel:
    def __init__(self, checkpoint, device, train=False):
        import torch
        from jevk5 import JevK5
        from peft import LoraConfig, get_peft_model

        self.meta = models.metadata(checkpoint)
        if self.meta is None:
            raise ValueError("JevK5 requires a versioned checkpoint")
        if device != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise OSError("JevK5 training and evaluation require a BF16-capable CUDA GPU")
        torch.set_num_threads(6)
        self.runtime = JevK5(
            str(base_path()), device=device, graphs=False, temperature=self.meta["temperature"]
        )
        self.base, self.tokenizer = self.runtime.model, self.runtime.tok
        self.peft = None
        if train or self.meta["kind"] == "adapter":
            names = {name.rsplit(".", 1)[-1] for name, _ in self.base.named_modules()}
            targets = [name for name in TARGETS if name in names]
            self.peft = get_peft_model(
                self.base,
                LoraConfig(
                    r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets, bias="none"
                ),
            )
            if self.meta["kind"] == "adapter":
                config_path = Path(checkpoint) / "adapter_config.json"
                if config_path.stat().st_size > 16384:
                    raise ValueError("adapter configuration exceeds its size limit")
                config = json.loads(config_path.read_text())
                required = {
                    "r": 16,
                    "lora_alpha": 32,
                    "bias": "none",
                    "peft_type": "LORA",
                    "base_model_name_or_path": models.spec(models.JEVK5)["base"],
                    "revision": models.spec(models.JEVK5)["base_revision"],
                }
                if any(config.get(k) != v for k, v in required.items()) or set(
                    config.get("target_modules", [])
                ) != set(targets):
                    raise ValueError("adapter architecture differs from the pinned JevK5 recipe")
                # Never instantiate an architecture from a miner-supplied PEFT config.
                # Only the exact tensor names/shapes of our own fixed LoRA are accepted.
                load_adapter_weights(self.peft, checkpoint)
        self.base.eval()
        if self.peft is not None:
            self.peft.eval()

    def logits(self, ids, count):
        import torch

        tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
        last = torch.tensor([len(ids) - 1], device="cuda")
        return self.runtime._slot_logits(tokens, last)[0, :count].float()

    def predict(self, state, question):
        import torch

        ids, keys = encode(self.tokenizer, state, question)
        with torch.inference_mode():
            logits = self.logits(ids, len(keys))
            probabilities = (
                torch.softmax(logits.double() / self.meta["temperature"], dim=0).cpu().tolist()
            )
        torch.cuda.synchronize()
        return dict(zip(keys, probabilities))


def train(args):
    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError("training output exists; preserve the previous attempt")
    model = DecisionModel(args.reference, args.device, train=True)
    rows = []
    for line in Path(args.data).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != {"state", "questions"} or set(row["questions"]) != {"decision"}:
            raise ValueError("training export must contain exactly one decision per example")
        question = dict(row["questions"]["decision"])
        label = question.pop("label")
        label = str(label).lower() if type(label) is bool else str(label)
        ids, keys = encode(model.tokenizer, row["state"], question)
        if label not in keys:
            raise ValueError("training label is not an allowed outcome")
        rows.append((ids, keys.index(label), len(keys)))
    if not rows:
        raise ValueError("training export is empty")
    params = [p for p in model.peft.parameters() if p.requires_grad]
    if not params or any(
        "lora_" not in n for n, p in model.peft.named_parameters() if p.requires_grad
    ):
        raise RuntimeError("training must update only LoRA parameters")
    model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.peft.train()
    optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=0)
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    started = time.perf_counter()
    for step, index in enumerate(order):
        ids, label, count = rows[index]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.logits(ids, count)
            loss = torch.nn.functional.cross_entropy(
                logits[None], torch.tensor([label], device="cuda")
            )
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        window_size = min(4, len(order) - (step // 4) * 4)
        (loss / window_size).backward()
        if step == 0 and not any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0
            for p in params
        ):
            raise RuntimeError("no finite nonzero LoRA training gradients")
        losses.append(float(loss.detach()))
        if (step + 1) % 4 == 0 or step + 1 == len(order):
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            print(
                json.dumps({"examples": step + 1, "total": len(order), "loss": losses[-1]}),
                flush=True,
            )
    model.base.gradient_checkpointing_disable()
    model.peft.eval()
    save_adapter_weights(model.peft, output)
    config_path = output / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config.update(
        base_model_name_or_path=models.spec(models.JEVK5)["base"],
        revision=models.spec(models.JEVK5)["base_revision"],
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    models.write_metadata(output, temperature=1.0)
    (output / "training_metrics.json").write_text(
        json.dumps(
            {
                "model": models.spec(models.JEVK5),
                "examples": len(rows),
                "optimizer_steps": math.ceil(len(rows) / 4),
                "epochs": 1,
                "learning_rate": 2e-5,
                "seed": args.seed,
                "max_tokens": MAX_TOKENS,
                "mean_loss": sum(losses) / len(losses),
                "training_seconds": time.perf_counter() - started,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    initialize = sub.add_parser(
        "reference", help="download the pinned base and create an immutable reference"
    )
    initialize.add_argument("--out", required=True)
    initialize.add_argument("--no-download", action="store_true")
    trainer = sub.add_parser("train")
    trainer.add_argument("--data", required=True)
    trainer.add_argument("--reference", required=True)
    trainer.add_argument("--device", default="cuda")
    trainer.add_argument("--seed", required=True, type=int)
    trainer.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "reference":
        base_path(download=not args.no_download)
        path = Path(args.out)
        path.mkdir(parents=True, exist_ok=False)
        models.write_metadata(path, kind="base", temperature=1.22)
        (path / "model.json").chmod(0o444)
    else:
        train(args)


if __name__ == "__main__":
    main()
