"""Pinned model identities and checkpoint formats; legacy Kev artifacts stay readable."""

import hashlib
import importlib
import json
import math
from copy import deepcopy
from pathlib import Path

KEV = "kev-0.8b-v1"
# Retired; kept because the hashed d3 trainer (zils/jevk5.py) still compares against it.
H2O = "h2o-lightning-4b-v1.2.3"
D3 = "d3-mini-4.5b-v1"
DEFAULT_TRAINING_MODEL = D3
TEXT_ADAPTER_MODELS = (D3,)
# Archived identities are readable in history, but never enter the active catalog.
RETIRED_SPECS = {
    "jevk5-4b-v0.3": {
        "id": "jevk5-4b-v0.3",
        "name": "JevK5 4B",
        "base": "alibiserikbay/JevK5",
        "base_revision": "c4f7fdb3aeab5582336406e78d3bef11bf98833d",
    },
    "h2o-lightning-4b-v1.2.3": {
        "id": "h2o-lightning-4b-v1.2.3",
        "name": "H2O Lightning 4B",
        "base": "h2oai/h2o-lightning-4b",
        "base_revision": "acaf0d4ea251e54de928c75ef4352670d33192d3",
        "runtime_revision": "h2o-torch210-cu130-fla052/v1",
        "prompt": "h2o-native-v1.2.3",
        "max_input_tokens": 2048,
        "recipe": "h2o-language-lora16-adamw/v1",
        "calibration": "native-then-heldout-temperature/v1",
    },
    "imajev-4b-v1": {
        "id": "imajev-4b-v1",
        "name": "Imajev 4B",
        "base": "Qwen/Qwen3.5-4B",
        "base_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "adapter": "mohit67890/imajev-4b",
        "adapter_revision": "f8d8234cebc6c99065c07731e59716dc0a6e27ab",
        "starting_adapter_sha256": "88c2c44361e0c469352495abcfee789ff73a4deae0811168d9402cfc2b6e749c",
        "starting_head_sha256": "52ceafd7d824bf3ea5ce55276b48cc08ba9dd6b2c98a55d9bd6a72ed1643a427",
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
RETIRED_MODELS = tuple(RETIRED_SPECS)
TEXT_RUNTIME_MODULES = {D3: ".d3"}
RUNTIME_PYTHON_ENV = {D3: "ZILS_D3_RUNTIME_PYTHON"}
CANDIDATE_BASE_ENV = {D3: "ZILS_D3_CANDIDATE_BASE"}
PINS_FILES = {D3: "d3-pins.json"}
SPECS = {
    KEV: {
        "id": KEV,
        "name": "Kev 0.8B",
        "base": "Qwen/Qwen3.5-0.8B-Base",
        "base_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    },
}
SPECS[D3] = {
    "id": D3,
    "name": "d3-mini 4.5B",
    "base": "zils-ai/d3-mini-4.5b-v1",
    "base_revision": "b54402a963373cc44713f18c9596c580e31cc68b",
    "runtime_revision": "d3-vendor-runtime-a475367/v1",
    "prompt": "d3-native-v1",
    "max_input_tokens": 4096,
    "recipe": "d3-language-lora16-adamw/v1",
    "calibration": "native-then-heldout-temperature/v1",
    "inputs": {"images": 4, "video": True},
}
D3_POOL_MODELS = {
    "d3-lite": "d3-lite-0.8b-v1",
    "d3-nano": "d3-nano-2.2b-v1",
    "d3-mini": D3,
    "d3-flash": "d3-flash-8.4b-v1",
    # z3 27B pool: vllm-sr/d3-ultra replaced vllm-sr/d3 as the base (higher Decision Index and Exam v3).
    "d3": "d3-ultra-26b-v1",
}
for _pool, _model in D3_POOL_MODELS.items():
    if _model == D3:
        continue
    _pins_file = "d3-ultra-pins.json" if _pool == "d3" else f"{_pool}-pins.json"
    _pins = json.loads(Path(__file__).with_name(_pins_file).read_text())
    SPECS[_model] = {
        **SPECS[D3],
        "id": _model,
        "name": _pool,
        "base": _pins["vendor"]["repo"],
        "base_revision": _pins["revision"],
        "runtime_revision": f"d3-vendor-runtime-{_pins['files']['d3_runtime.py'][:12]}/v1",
    }
    TEXT_ADAPTER_MODELS = (*TEXT_ADAPTER_MODELS, _model)
    TEXT_RUNTIME_MODULES[_model] = ".d3"
    RUNTIME_PYTHON_ENV[_model] = "ZILS_D3_RUNTIME_PYTHON"
    CANDIDATE_BASE_ENV[_model] = "ZILS_" + _pool.upper().replace("-", "_") + "_CANDIDATE_BASE"
    PINS_FILES[_model] = _pins_file
KEV_FILES = ("adapter_config.json", "adapter_model.safetensors", "head.pt")
TEXT_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "model.json")
D3_CONTRACT_FILES = (
    "config.json",
    "decision_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "readout.safetensors",
    "d3_runtime.py",
    "d3_format.py",
    "d3_kernels.py",
    "d3_fast.py",
    "d3_engine.py",
    "modeling_d3.py",
    "pipeline_d3.py",
    "LICENSE",
    "NOTICE",
)
D3_CANDIDATE_FILES = (
    "model.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    *D3_CONTRACT_FILES,
)
MAX_ADAPTER_BYTES = 512 * 1024 * 1024
MAX_CANDIDATE_BYTES = 10 * 1024**3
VERSION = "zils-checkpoint/v1"


def d3_family():
    """Profiles the d3 runtime serves: the vendor base and every Zils base promoted from it."""
    return tuple(m for m, module in TEXT_RUNTIME_MODULES.items() if module == ".d3")


def pool_id(model):
    """Resolve a released profile's lineage without changing its immutable identity."""
    seen = set()
    while model not in seen:
        seen.add(model)
        profile = spec(model)
        for pool, root in D3_POOL_MODELS.items():
            if model == root:
                return pool
        model = (profile.get("parent") or {}).get("id")
        if model is None:
            break
    raise ValueError("profile has no supported model pool")


def pool_catalog():
    """Known lineages; membership is not a hardware or deployment certificate."""
    return [
        {"id": pool, "name": SPECS[model]["name"], "model": model, "qualified": False}
        for pool, model in D3_POOL_MODELS.items()
    ]


def base_candidate_models():
    return d3_family()


def spec(model):
    if model not in SPECS:
        raise ValueError("unsupported model identity")
    return deepcopy(SPECS[model])


def pins(model):
    return json.loads(Path(__file__).with_name(PINS_FILES[model]).read_text())


def runtime_python_env(model):
    return RUNTIME_PYTHON_ENV.get(model)


def candidate_base_env(model):
    if model not in CANDIDATE_BASE_ENV:
        raise ValueError("profile has no base-candidate support")
    return CANDIDATE_BASE_ENV[model]


def profile_identity(model):
    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    profile = spec(model)
    runtime = {"base_revision": profile["base_revision"]}
    if model in PINS_FILES:
        runtime = pins(model)
    return {"profile_sha256": digest(profile), "runtime_sha256": digest(runtime)}


def validate_spec(value):
    if not isinstance(value, dict) or value != spec(value.get("id")):
        raise ValueError("model specification differs from its pinned version")
    return value["id"]


def job_model(job):
    value = (job.get("manifest") or {}).get("model")
    profile = job.get("model_profile")
    if profile is not None:
        model = validate_spec(profile)
        if value is not None and value != profile:
            raise ValueError("job profile differs from its frozen manifest")
        return model
    return validate_spec(value) if value is not None else KEV


def job_history_profile(job):
    """Read a pinned job profile without making retired models operational."""
    frozen = (job.get("manifest") or {}).get("model")
    profile = job.get("model_profile")
    value = profile if profile is not None else frozen
    if not isinstance(value, dict) or value.get("id") not in RETIRED_SPECS:
        return spec(job_model(job))
    expected = RETIRED_SPECS[value["id"]]
    if value != expected:
        raise ValueError("model specification differs from its pinned version")
    if profile is not None and frozen is not None and frozen != profile:
        raise ValueError("job profile differs from its frozen manifest")
    return deepcopy(expected)


def candidate_files(model, kind="adapter"):
    spec(model)
    if kind == "candidate":
        if model not in base_candidate_models():
            raise ValueError("base candidates are supported for the d3 profiles only")
        files = pins(model)["files"]
        weights = sorted(
            n for n in files if n.endswith(".safetensors") and n != "readout.safetensors"
        )
        index = ("model.safetensors.index.json",) if "model.safetensors.index.json" in files else ()
        return ("model.json", *weights, *index, *D3_CONTRACT_FILES)
    return {
        KEV: KEV_FILES,
        **{m: TEXT_ADAPTER_FILES for m in d3_family()},
    }[model]


def artifact_budget(kind="adapter", model=None):
    if kind == "candidate" and model in d3_family():
        return pins(model).get("candidate_max_bytes", MAX_CANDIDATE_BYTES)
    return MAX_CANDIDATE_BYTES if kind == "candidate" else MAX_ADAPTER_BYTES


def artifact_kind(checkpoint):
    value = metadata(checkpoint)
    return value["kind"] if value is not None else "adapter"


def metadata(checkpoint):
    path = Path(checkpoint) / "model.json"
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16384:
        raise ValueError("model metadata must be a regular file of at most 16 KiB")
    return validate_metadata(json.loads(path.read_text()))


def validate_metadata(value):
    """Validate the same contract before a remote candidate's weights are downloaded."""
    if (
        not isinstance(value, dict)
        or set(value) - {"version", "model", "kind", "temperature", "calibration", "profile"}
        or value.get("version") != VERSION
        or not isinstance(value.get("model"), str)
        or value.get("model") not in (*TEXT_ADAPTER_MODELS, *RETIRED_MODELS)
        or value.get("kind") not in ("base", "adapter", "candidate")
    ):
        raise ValueError("invalid model metadata")
    if value["kind"] == "candidate" and value["model"] not in base_candidate_models():
        raise ValueError("base candidates are supported for the d3 profiles only")
    if value["model"] in PINS_FILES:
        if value.get("profile") != spec(value["model"]):
            raise ValueError("profile metadata differs from the pinned profile")
    elif "profile" in value and value["profile"] != RETIRED_SPECS.get(value["model"]):
        raise ValueError("profile metadata differs from the pinned profile")
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
        return KEV_FILES
    if (Path(checkpoint) / "head.pt").exists():
        raise ValueError("checkpoint mixes model formats")
    if value["kind"] == "candidate":
        if (Path(checkpoint) / "adapter_model.safetensors").exists():
            raise ValueError("checkpoint mixes model formats")
        return candidate_files(value["model"], "candidate")
    return ("model.json",) if value["kind"] == "base" else candidate_files(value["model"])


def write_metadata(checkpoint, *, model, kind="adapter", temperature=1.0, calibration=None):
    spec(model)  # retired profiles stay readable but are never written
    value = {"version": VERSION, "model": model, "kind": kind, "temperature": temperature}
    if model in PINS_FILES:
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


def text_runtime(model):
    if model not in TEXT_ADAPTER_MODELS:
        raise ValueError("unsupported text adapter runtime")
    return importlib.import_module(TEXT_RUNTIME_MODULES[model], __package__)
