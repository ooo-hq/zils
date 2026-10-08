"""Pinned model identities and checkpoint formats; legacy Kev artifacts stay readable."""

import json
import math
from copy import deepcopy
from pathlib import Path

KEV = "kev-0.8b-v1"
JEVK5 = "jevk5-4b-v0.3"
IMAJEV = "imajev-4b-v1"
SPECS = {
    KEV: {
        "id": KEV,
        "name": "Kev 0.8B",
        "base": "Qwen/Qwen3.5-0.8B-Base",
        "base_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    },
    JEVK5: {
        "id": JEVK5,
        "name": "JevK5 4B",
        "base": "alibiserikbay/JevK5",
        "base_revision": "c4f7fdb3aeab5582336406e78d3bef11bf98833d",
    },
    IMAJEV: {
        "id": IMAJEV,
        "name": "Imajev 4B",
        "base": "Qwen/Qwen3.5-4B",
        "base_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "adapter": "mohit67890/imajev-4b",
        "adapter_revision": "f8d8234cebc6c99065c07731e59716dc0a6e27ab",
        "runtime_revision": "ccf586d43d2a580319b6535c893668904d909eb9",
        "preprocessor": "zils-image-rgb-png/v1",
        "prompt": "imajev-readout/v1",
        "option_order": "declared_then_unknown",
        "min_pixels": 65536,
        "max_pixels": 400000,
        "max_input_tokens": 4096,
        "recipe": "imajev-lora64-readout-adamw/v1",
        "calibration": "full-native-temperature/v1",
    },
}
KEV_FILES = ("adapter_config.json", "adapter_model.safetensors", "head.pt")
JEVK5_FILES = ("adapter_config.json", "adapter_model.safetensors", "model.json")
IMAJEV_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "decision_readout.json",
    "decision_readout.safetensors",
    "model.json",
)
VERSION = "zils-checkpoint/v1"


def spec(model):
    if model not in SPECS:
        raise ValueError("unsupported model identity")
    return deepcopy(SPECS[model])


def validate_spec(value):
    if not isinstance(value, dict) or value != spec(value.get("id")):
        raise ValueError("model specification differs from its pinned version")
    return value["id"]


def job_model(job):
    value = (job.get("manifest") or {}).get("model")
    return validate_spec(value) if value is not None else KEV


def candidate_files(model):
    spec(model)
    return {KEV: KEV_FILES, JEVK5: JEVK5_FILES, IMAJEV: IMAJEV_FILES}[model]


def metadata(checkpoint):
    path = Path(checkpoint) / "model.json"
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16384:
        raise ValueError("model metadata must be a regular file of at most 16 KiB")
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or set(value) - {"version", "model", "kind", "temperature", "calibration", "profile"}
        or value.get("version") != VERSION
        or value.get("model") not in (JEVK5, IMAJEV)
        or value.get("kind") not in ("base", "adapter")
    ):
        raise ValueError("invalid model metadata")
    if value["model"] == IMAJEV:
        if value.get("profile") != spec(IMAJEV):
            raise ValueError("image metadata differs from the pinned profile")
    elif "profile" in value:
        raise ValueError("legacy metadata cannot contain an image profile")
    temperature = value.get("temperature")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("checkpoint temperature must be positive and finite")
    return value


def checkpoint_model(checkpoint):
    value = metadata(checkpoint)
    return value["model"] if value is not None else KEV


def artifact_files(checkpoint):
    value = metadata(checkpoint)
    if value is None:
        if any((Path(checkpoint) / name).exists() for name in IMAJEV_FILES[2:4]):
            raise ValueError("checkpoint mixes model formats")
        return KEV_FILES
    if (Path(checkpoint) / "head.pt").exists():
        raise ValueError("checkpoint mixes model formats")
    if value["model"] == JEVK5 and any(
        (Path(checkpoint) / name).exists() for name in IMAJEV_FILES[2:4]
    ):
        raise ValueError("checkpoint mixes model formats")
    return ("model.json",) if value["kind"] == "base" else candidate_files(value["model"])


def write_metadata(checkpoint, *, kind="adapter", temperature=1.0, calibration=None, model=JEVK5):
    value = {"version": VERSION, "model": model, "kind": kind, "temperature": temperature}
    if model == IMAJEV:
        value["profile"] = spec(model)
    if calibration is not None:
        value["calibration"] = calibration
    (Path(checkpoint) / "model.json").write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n"
    )
    metadata(checkpoint)


def temperature(checkpoint):
    value = metadata(checkpoint)
    if value is not None:
        return value["temperature"]
    from kev.checkpoint import read_meta

    return read_meta(checkpoint).temperature


def set_temperature(checkpoint, value, calibration=None):
    meta = metadata(checkpoint)
    path = Path(checkpoint) / ("model.json" if meta is not None else "head.pt")
    path.chmod(0o600)
    try:
        if meta is not None:
            write_metadata(
                checkpoint,
                kind=meta["kind"],
                temperature=value,
                calibration=calibration,
                model=meta["model"],
            )
        else:
            from kev.checkpoint import read_meta, write_meta

            legacy = read_meta(checkpoint)
            legacy.temperature = value
            if calibration is not None:
                legacy.extra["fez_temperature_fit"] = calibration
            write_meta(checkpoint, legacy)
    finally:
        path.chmod(0o444)
