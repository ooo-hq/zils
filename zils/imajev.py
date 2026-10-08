"""Pinned Imajev reference, processor and full-native decision engine."""

import hashlib
import importlib
import json
import math
import sys
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
    def __init__(self, reference, device):
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
            self.engine.model, str(stock), is_trainable=False
        )
        if not self.engine.enable_readout(stock, trainable=False, codes=256):
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
