"""Pinned Imajev reference, processor and full-native decision engine."""

import hashlib
import importlib
import json
import math
import random
import re
import shutil
import sys
import time
from pathlib import Path

from . import models
from .decisions import DecisionError
from .image_contract import UNKNOWN

RELEASE_ID = "zils-imajev-4b-v1-r1"
PINS = json.loads(Path(__file__).with_name("imajev-pins.json").read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in value.items() if k != "fingerprint"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build_manifest(reference):
    root = Path(reference)
    files = {}
    for folder, items in PINS.items():
        for name, expected in items.items():
            path = root / folder / name
            if path.is_symlink() or not path.is_file() or sha(path) != expected:
                raise ValueError("Pinned image reference files differ")
            files[folder + "/" + name] = expected
    manifest = {
        "release_id": RELEASE_ID,
        "profile": models.spec(models.IMAJEV),
        "files": files,
        "temperature": 1.0,
        "dtype": "bfloat16",
    }
    manifest["fingerprint"] = fingerprint(manifest)
    return manifest


def verify_reference(reference):
    manifest = json.loads((Path(reference) / "release.json").read_text())
    if manifest != build_manifest(reference):
        raise ValueError("Image release identity or files changed")
    return manifest


def resolve_reference(reference):
    verify_reference(reference)
    root = Path(reference)
    return root / "base", root / "adapter"


class ImageEngine:
    def __init__(self, reference, device, *, trainable=False):
        import torch
        from peft import PeftModel
        from transformers.image_utils import SizeDict

        if device != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("The image profile requires a BF16-capable CUDA GPU")
        self.reference = Path(reference)
        self.manifest = verify_reference(reference)
        base, stock = self.reference / "base", self.reference / "adapter"
        self.stock = stock
        runtime = (self.reference / "runtime").resolve()
        sys.path[:0] = [str(runtime / "src"), str(runtime / "scripts")]
        upstream = importlib.import_module("torch_decision")
        vision = importlib.import_module("vision_decision")
        for module in (upstream, vision):
            if not Path(module.__file__).resolve().is_relative_to(runtime):
                raise ValueError("A different image runtime is already imported")
        self.engine = upstream.TorchDecision(
            str(base), device, dtype=torch.bfloat16, max_length=4096
        )
        self.engine.model = PeftModel.from_pretrained(
            self.engine.model, str(stock), is_trainable=trainable
        )
        if not self.engine.enable_readout(stock, trainable=trainable, codes=256):
            raise ValueError("The pinned image decision head is missing")
        self.engine.processor._vision().image_processor.size = SizeDict(
            shortest_edge=65536, longest_edge=400000
        )
        self.engine.model.eval()
        self.device = device

    def prepare(self, image, state, question):
        from PIL import Image
        from vision_decision.jev_api import to_request_with_plan
        from vision_decision.scoring import compile_question

        q = {
            **question,
            "instructions": question.get("instructions")
            or "Choose the best answer from the image and permitted state.",
        }
        request, _ = to_request_with_plan(
            {"state": state, "questions": {"inspection": q}}, max_options=self.engine.max_options
        )
        header, choices, texts = compile_question(
            request.fields[0], request.state, self.engine.prompt_layout
        )
        keys = [key for key, _ in choices]
        if keys != [*question["criteria"], UNKNOWN]:
            raise ValueError("Image candidate ordering changed")
        labels = self.engine.labels(len(choices), 1)
        prompt = header + "\n".join(
            f"{label}: {text}" for label, text in zip(labels, texts, strict=True)
        )
        with Image.open(image) as photo:
            _, inputs, ids = self.engine.prepare([photo.convert("RGB")], prompt, labels)
        count = int(inputs["input_ids"].shape[-1])
        if count > 4096:
            raise DecisionError(
                413, "context_limit", "The image and question exceed this model context limit."
            )
        if (
            "pixel_values" not in inputs
            or "image_grid_thw" not in inputs
            or not inputs["pixel_values"].numel()
        ):
            raise ValueError("Image processor omitted visual input")
        return {"inputs": inputs, "token_ids": ids, "keys": keys, "input_tokens": count}

    def logits(self, prepared):
        return self.engine.candidate_logits(prepared["inputs"], prepared["token_ids"]).float()

    def predict(self, prepared, temperature=1.0):
        import torch

        if (
            type(temperature) not in (float, int)
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("Invalid image calibration temperature")
        with torch.inference_mode():
            logits = self.logits(prepared)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Nonfinite image logits")
            values = torch.softmax(logits.double() / temperature, dim=0).cpu().tolist()
        return {
            "probabilities": dict(zip(prepared["keys"], values, strict=True)),
            "input_tokens": prepared["input_tokens"],
        }


def verify_starting_checkpoint(reference):
    """A training reference is an exact copy of the published LoRA and readout."""
    root = Path(reference)
    meta = models.metadata(root)
    if (
        not meta
        or meta["model"] != models.IMAJEV
        or meta["temperature"] != 1.0
        or meta.get("calibration")
    ):
        raise ValueError("Image training must start from the published checkpoint")
    for name, expected in PINS["adapter"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError("Starting image adapter differs from the published checkpoint")


def create_reference(release, destination):
    verify_reference(release)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    for name in PINS["adapter"]:
        shutil.copyfile(Path(release) / "adapter" / name, root / name)
    models.write_metadata(root, model=models.IMAJEV)
    verify_starting_checkpoint(root)
    return root


def accumulation_batches(rows, size):
    if type(size) is not int or size < 1:
        raise ValueError("Accumulation size must be positive")
    return [rows[start : start + size] for start in range(0, len(rows), size)]


def accumulate_gradients(rows, loss_for_example):
    import torch

    if not rows:
        raise ValueError("An accumulation group cannot be empty")
    losses = []
    for row in rows:
        loss = loss_for_example(row)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite image training loss")
        (loss / len(rows)).backward()
        losses.append(float(loss.detach()))
    return losses


def trainable_parameters(engine, *, expected_count=122552320):
    import torch

    parameters = []
    for name, parameter in engine.model.named_parameters():
        if parameter.requires_grad:
            if "language_model" not in name or not any(
                part in name for part in (".lora_A.", ".lora_B.")
            ):
                raise ValueError("Only language LoRA weights may be trained")
            parameters.append(parameter)
    parameters.append(engine.readout.weight)
    if sum(p.numel() for p in parameters) != expected_count or any(
        not p.requires_grad or p.dtype != torch.float32 for p in parameters
    ):
        raise ValueError("Image trainable parameter count or precision changed")
    return parameters


def validate_tensors(actual, expected):
    import torch

    if set(actual) != set(expected) or not actual:
        raise ValueError("Image checkpoint tensor names differ")
    for key, tensor in actual.items():
        if (
            tensor.shape != expected[key].shape
            or tensor.dtype != torch.float32
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError("Image checkpoint has invalid shapes, precision or nonfinite weights")


def load_checkpoint(engine, checkpoint):
    """Copy inert weights into the verified architecture; never execute candidate config."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    meta = models.metadata(checkpoint)
    if not meta or meta["model"] != models.IMAJEV:
        raise ValueError("An image adapter is required")
    for name in models.IMAJEV_FILES:
        path = checkpoint / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Image artifacts must be regular files")
    config = json.loads((checkpoint / "adapter_config.json").read_text())
    stock = json.loads((engine.stock / "adapter_config.json").read_text())
    # These fields are serialization metadata. The architecture always comes from stock.
    ignored = {"base_model_name_or_path", "inference_mode", "peft_version"}
    if {k: v for k, v in config.items() if k not in ignored} != {
        k: v for k, v in stock.items() if k not in ignored
    }:
        raise ValueError("Image adapter configuration changed")
    if json.loads((checkpoint / "decision_readout.json").read_text()) != json.loads(
        (engine.stock / "decision_readout.json").read_text()
    ):
        raise ValueError("Image readout configuration changed")
    weights = load_file(str(checkpoint / "adapter_model.safetensors"))
    validate_tensors(weights, get_peft_model_state_dict(engine.engine.model))
    head = load_file(str(checkpoint / "decision_readout.safetensors"))
    validate_tensors(head, {"weight": engine.engine.readout.weight})
    result = set_peft_model_state_dict(engine.engine.model, weights, adapter_name="default")
    if result.unexpected_keys:
        raise ValueError("Unexpected image checkpoint weights")
    import torch

    with torch.no_grad():
        engine.engine.readout.weight.copy_(head["weight"].to(engine.device))
    return meta


def image_path(images, binding):
    """Content-addressed local cache; model inputs never contain source filenames."""
    digest = binding.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
        raise ValueError("Invalid frozen image hash")
    path = Path(images) / (digest + ".png")
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 10 * 1024**2
        or sha(path) != digest
    ):
        raise ValueError("Frozen image bytes changed or are unavailable")
    return path


def train(reference, training, images, out, seed, device):
    import gc

    import torch

    from . import settings
    from .image_contract import validate_image_request

    started = time.perf_counter()
    verify_starting_checkpoint(reference)
    rows = [json.loads(line) for line in Path(training).read_text().splitlines() if line.strip()]
    if not 1 <= len(rows) <= 1024:
        raise ValueError("Image training requires 1–1024 examples")
    for row in rows:
        if set(row) != {"state", "questions", "images"} or set(row["questions"]) != {"decision"}:
            raise ValueError("Invalid image training export")
        question = {k: v for k, v in row["questions"]["decision"].items() if k != "label"}
        validate_image_request(
            {
                "model": models.IMAJEV,
                "state": row["state"],
                "questions": {"decision": question},
                "images": [{"asset_id": item["asset_id"]} for item in row["images"]],
            }
        )
        if row["questions"]["decision"]["label"] not in question["criteria"]:
            raise ValueError("Training label is not a known outcome")
        image_path(images, row["images"][0])
    random.Random(seed).shuffle(rows)
    torch.manual_seed(seed)
    engine = ImageEngine(Path(settings.required("ZILS_IMAGE_REFERENCE")), device, trainable=True)
    parameters = trainable_parameters(engine.engine)
    base = engine.engine.model.get_base_model()
    base.config.use_cache = False
    if hasattr(base.config, "text_config"):
        base.config.text_config.use_cache = False

    def prepared(row):
        question = {k: v for k, v in row["questions"]["decision"].items() if k != "label"}
        return engine.prepare(image_path(images, row["images"][0]), row["state"], question)

    def probe():
        with torch.inference_mode():
            return engine.logits(prepared(rows[0])).cpu()

    before = probe()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    engine.engine.model.train()
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0, foreach=False)
    losses, token_counts = [], []

    def loss_for_example(row):
        item = prepared(row)
        token_counts.append(item["input_tokens"])
        target = item["keys"].index(row["questions"]["decision"]["label"])
        return torch.nn.functional.cross_entropy(
            engine.logits(item)[None], torch.tensor([target], device=device)
        )

    loop_started = time.perf_counter()
    updates = 0
    for batch in accumulation_batches(rows, 4):
        optimizer.zero_grad(set_to_none=True)
        losses.extend(accumulate_gradients(batch, loss_for_example))
        for group in (parameters[:-1], parameters[-1:]):
            if not any(p.grad is not None and bool((p.grad != 0).any()) for p in group) or any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in group
            ):
                raise ValueError("Image LoRA/head gradients must be finite and nonzero")
        torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
        optimizer.step()
        updates += 1
    torch.cuda.synchronize()
    loop_seconds = time.perf_counter() - loop_started
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    base.gradient_checkpointing_disable()
    engine.engine.model.eval()
    after = probe()
    change = float((after - before).abs().max())
    if not math.isfinite(change) or change <= 0:
        raise ValueError("Training did not change image predictions")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    engine.engine.model.save_pretrained(out, safe_serialization=True)
    engine.engine.save_readout(out)
    models.write_metadata(out, model=models.IMAJEV)
    # Perturb the active state so equality proves a reload, not merely another forward pass.
    with torch.no_grad():
        for parameter in parameters:
            parameter.zero_()
    load_checkpoint(engine, out)
    delta = float((probe() - after).abs().max())
    if not math.isfinite(delta) or delta > 1e-5:
        raise ValueError("Reloaded image checkpoint changed predictions")
    verify_starting_checkpoint(reference)
    report = {
        "examples": len(rows),
        "optimizer_steps": updates,
        "seed": seed,
        "training_seconds": loop_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "mean_loss": sum(losses) / len(losses),
        "input_tokens": token_counts,
        "probe_logit_change": change,
        "reload_max_logit_difference": delta,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "artifact_sha256": {name: sha(out / name) for name in models.IMAJEV_FILES},
        **models.profile_identity(models.IMAJEV),
    }
    return report
