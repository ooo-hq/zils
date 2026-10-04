"""Pinned model identities and checkpoint formats; legacy Kev artifacts stay readable."""

import json
import math
from pathlib import Path

KEV = "kev-0.8b-v1"
JEVK5 = "jevk5-4b-v0.3"
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
}
KEV_FILES = ("adapter_config.json", "adapter_model.safetensors", "head.pt")
JEVK5_FILES = ("adapter_config.json", "adapter_model.safetensors", "model.json")
VERSION = "zils-checkpoint/v1"


def spec(model):
    if model not in SPECS:
        raise ValueError("unsupported model identity")
    return dict(SPECS[model])


def validate_spec(value):
    if not isinstance(value, dict) or value != spec(value.get("id")):
        raise ValueError("model specification differs from its pinned version")
    return value["id"]


def job_model(job):
    value = (job.get("manifest") or {}).get("model")
    return validate_spec(value) if value is not None else KEV


def candidate_files(model):
    spec(model)
    return JEVK5_FILES if model == JEVK5 else KEV_FILES


def metadata(checkpoint):
    path = Path(checkpoint) / "model.json"
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16384:
        raise ValueError("model metadata must be a regular file of at most 16 KiB")
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or set(value) - {"version", "model", "kind", "temperature", "calibration"}
        or value.get("version") != VERSION
        or value.get("model") != JEVK5
        or value.get("kind") not in ("base", "adapter")
    ):
        raise ValueError("invalid model metadata")
    temperature = value.get("temperature")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("checkpoint temperature must be positive and finite")
    return value


def checkpoint_model(checkpoint):
    return JEVK5 if metadata(checkpoint) is not None else KEV


def artifact_files(checkpoint):
    value = metadata(checkpoint)
    if value is None:
        return KEV_FILES
    if (Path(checkpoint) / "head.pt").exists():
        raise ValueError("checkpoint mixes model formats")
    return ("model.json",) if value["kind"] == "base" else JEVK5_FILES


def write_metadata(checkpoint, *, kind="adapter", temperature=1.0, calibration=None):
    value = {"version": VERSION, "model": JEVK5, "kind": kind, "temperature": temperature}
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
                checkpoint, kind=meta["kind"], temperature=value, calibration=calibration
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
