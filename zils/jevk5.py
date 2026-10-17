"""Shared attention-only LoRA trainer for the pinned text profiles (H2O and d3).

The module keeps its historical name because job and trainer hashes reference it.
"""

import json
import math
import random
import time
from pathlib import Path

from . import models


def asset_bytes(root, name):
    """Bytes of a file named relative to `root` that resolves inside it (no absolute paths or `..`)."""
    path = (root / name).resolve() if isinstance(name, str) else None
    if path is None or Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("image paths must stay inside the dataset directory")
    if not path.is_file():
        raise ValueError(f"no image file inside the dataset directory at {name}")
    return path.read_bytes()


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
        raise ValueError("adapter tensors differ from the pinned LoRA recipe")
    set_peft_model_state_dict(model, {k: v.to(expected[k].dtype) for k, v in tensors.items()})


def train(args, *, model_id):
    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError("training output exists; preserve the previous attempt")
    engine = models.text_runtime(model_id)
    data = Path(args.data)
    examples = []
    for line in data.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) - {"images"} != {"state", "questions"} or set(row["questions"]) != {"decision"}:
            raise ValueError("training export must contain exactly one decision per example")
        images = row.get("images", [])
        if images and "inputs" not in models.spec(model_id):
            raise ValueError(f"{model_id} does not accept images")
        if not isinstance(images, list):
            raise ValueError("training images must be a list of paths or frozen references")
        from .image_jobs import export_path

        # A queue export names frozen assets; the miner caches them under images/<sha256>.png.
        images = [
            asset_bytes(data.parent, export_path(name) if isinstance(name, dict) else name)
            for name in images
        ]
        if images:
            engine.check_media(images)  # media limits are refused before the weights load
        examples.append((row, images))
    if not examples:
        raise ValueError("training export is empty")
    model = engine.DecisionModel(args.reference, args.device, train=True)
    rows = []
    for row, images in examples:
        question = dict(row["questions"]["decision"])
        label = question.pop("label")
        label = str(label).lower() if type(label) is bool else str(label)
        # Image rows keep the processor batch (pixels and expanded ids); text rows keep token ids.
        inputs, keys = (
            model.encoder.inputs(row["state"], question, images=images)
            if images
            else model.encode(row["state"], question)
        )
        if label not in keys:
            raise ValueError("training label is not an allowed outcome")
        temperature = (
            model.encoder.contract.temperature_by_type[question["type"]]
            if model_id == models.H2O
            else 1.0
        )
        rows.append((inputs, keys.index(label), len(keys), temperature))
    params = [p for p in model.peft.parameters() if p.requires_grad]
    if not params or any(
        "lora_" not in n for n, p in model.peft.named_parameters() if p.requires_grad
    ):
        raise RuntimeError("training must update only LoRA parameters")
    if any(p.requires_grad for n, p in model.peft.named_parameters() if "visual" in n):
        raise RuntimeError("training must keep the vision tower frozen")
    model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.peft.train()
    optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=0)
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    getattr(torch, args.device).synchronize()
    started = time.perf_counter()
    for step, index in enumerate(order):
        inputs, label, count, temperature = rows[index]
        with torch.autocast(args.device, dtype=torch.bfloat16):
            logits = model.logits(inputs, count) / temperature
            loss = torch.nn.functional.cross_entropy(
                logits[None], torch.tensor([label], device=args.device)
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
        base_model_name_or_path=models.spec(model_id)["base"],
        revision=models.spec(model_id)["base_revision"],
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    models.write_metadata(output, temperature=1.0, model=model_id)
    getattr(torch, args.device).synchronize()
    (output / "training_metrics.json").write_text(
        json.dumps(
            {
                "model": models.spec(model_id),
                "examples": len(rows),
                "optimizer_steps": math.ceil(len(rows) / 4),
                "epochs": 1,
                "learning_rate": 2e-5,
                "seed": args.seed,
                "max_tokens": engine.MAX_TOKENS,
                "mean_loss": sum(losses) / len(losses),
                "training_seconds": time.perf_counter() - started,
                "device": args.device,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            },
            indent=2,
        )
        + "\n"
    )
