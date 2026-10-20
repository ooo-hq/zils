"""Pinned box adapters, with a separate commitment namespace from merged base candidates."""

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from urllib.parse import quote

import zils

from . import commitments, d3, models

PREFIX = "zilsbox1"
VERSION = "zils-feature-adapter/v1"
DECODER = "box-edges-101-presence/v1"
TRACKS = ("box/d3-mini", "box/d3-lite")
MANIFEST = "feature.json"
METADATA_BYTES = 16384


def validate_manifest(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "track_id", "model", "decoder", "checkpoint_sha256"}
        or value.get("version") != VERSION
        or value.get("track_id") not in TRACKS
        or value.get("decoder") != DECODER
        or not isinstance(value.get("model"), dict)
        or not isinstance(value["model"].get("id"), str)
        or not isinstance(value.get("checkpoint_sha256"), str)
        or not re.fullmatch("[a-f0-9]{64}", value["checkpoint_sha256"])
    ):
        raise ValueError("invalid feature manifest")
    model = models.validate_spec(value["model"])
    if model not in models.d3_family() or value["track_id"] != "box/" + models.pool_id(model):
        raise ValueError("feature track differs from its parent model")
    return value


def _directory(path):
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_dir() or any(p.is_symlink() for p in path.iterdir()):
        raise ValueError("feature artifacts require a directory without symlinks")
    return path


def _adapter(checkpoint, model):
    path = checkpoint / "adapter_config.json"
    if not path.is_file() or path.stat().st_size > METADATA_BYTES:
        raise ValueError("adapter configuration must be a bounded regular file")
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("invalid adapter configuration")
    # PEFT metadata is never executed; accept only the saved native LoRA recipe.
    defaults = {
        "peft_type": "LORA",
        "use_dora": False,
        "use_rslora": False,
        "use_qalora": False,
        "fan_in_fan_out": False,
        "lora_bias": False,
        "ensure_weight_tying": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "loftq_config": {},
        "init_lora_weights": True,
        "inference_mode": True,
        "task_type": None,
        "layers_to_transform": None,
        "layers_pattern": None,
        "layer_replication": None,
        "target_parameters": None,
        "exclude_modules": None,
        "trainable_token_indices": None,
        "alora_invocation_tokens": None,
        "arrow_config": None,
        "corda_config": None,
        "eva_config": None,
        "kasa_config": None,
        "lora_ga_config": None,
        "megatron_config": None,
        "monteclora_config": None,
        "use_bdlora": None,
        "velora_config": None,
        "megatron_core": "megatron.core",
        "qalora_group_size": 16,
    }
    required = {
        "r",
        "lora_alpha",
        "lora_dropout",
        "bias",
        "base_model_name_or_path",
        "revision",
        "modules_to_save",
        "target_modules",
    }
    if set(config) - required - set(defaults) - {"auto_mapping", "peft_version"} or any(
        config[key] != expected for key, expected in defaults.items() if key in config
    ):
        raise ValueError("adapter configuration differs from the pinned LoRA recipe")
    mapping = config.get("auto_mapping")
    if mapping not in (
        None,
        {
            "base_model_class": "Qwen3_5Model",
            "parent_library": "transformers.models.qwen3_5.modeling_qwen3_5",
        },
    ):
        raise ValueError("adapter auto_mapping differs from the pinned model")
    targets = config.get("target_modules")
    if (
        not isinstance(targets, list)
        or any(not isinstance(t, str) for t in targets)
        or len(targets) != len(set(targets))
        or len(targets) != models.pins(model).get("target_modules", d3.TARGET_MODULES)
        or any(
            not re.fullmatch(
                r"language_model\.layers\.\d+\.(self_attn|linear_attn)\."
                + "("
                + "|".join(d3.TARGETS)
                + ")",
                t,
            )
            for t in targets
        )
    ):
        raise ValueError("adapter target_modules differ from the language LoRA recipe")
    d3.validate_adapter_config(checkpoint, targets)


def _checkpoint(path, allow_base):
    path = _directory(path)
    meta = models.metadata(path)
    if (
        meta is None
        or meta["model"] not in models.d3_family()
        or meta["kind"] not in (("adapter", "base") if allow_base else ("adapter",))
    ):
        raise ValueError("feature entry requires a d3 adapter checkpoint")
    if meta["temperature"] != 1:
        raise ValueError("feature checkpoint temperature must be 1")
    if meta["kind"] == "adapter":
        _adapter(path, meta["model"])
    return meta, zils.checkpoint_hash(path)


def validate_bundle(path, allow_base=False):
    path = _directory(path)
    manifest = path / MANIFEST
    if not manifest.is_file() or manifest.stat().st_size > METADATA_BYTES:
        raise ValueError("feature manifest must be a regular file of at most 16 KiB")
    value = validate_manifest(json.loads(manifest.read_text()))
    meta, sha256 = _checkpoint(path, allow_base)
    if value["model"] != models.spec(meta["model"]):
        raise ValueError("feature parent differs from its checkpoint")
    if value["checkpoint_sha256"] != sha256:
        raise ValueError("feature checkpoint hash differs from its manifest")
    return value


def pack(checkpoint, out, track_id, allow_base=False):
    checkpoint = _directory(checkpoint)
    meta, sha256 = _checkpoint(checkpoint, allow_base)
    manifest = validate_manifest(
        {
            "version": VERSION,
            "track_id": track_id,
            "model": models.spec(meta["model"]),
            "decoder": DECODER,
            "checkpoint_sha256": sha256,
        }
    )
    out = Path(out).expanduser()
    out.mkdir()
    for name in models.artifact_files(checkpoint):
        shutil.copyfile(checkpoint / name, out / name)
    (out / MANIFEST).write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return validate_bundle(out, allow_base=allow_base)


def bundle_hash(path):
    path = _directory(path)
    validate_bundle(path, allow_base=True)
    records = []
    for name in sorted((*models.artifact_files(path), MANIFEST)):
        with (path / name).open("rb") as stream:
            records.append([name, hashlib.file_digest(stream, "sha256").hexdigest()])
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def format_commitment(sha256, revision, repo):
    return PREFIX + commitments.format_commitment(sha256, revision, repo)[len(commitments.PREFIX) :]


def parse_commitment(text):
    if not isinstance(text, str) or not text.startswith(PREFIX + ":"):
        raise ValueError("commitment is not a feature bundle")
    return commitments.parse_commitment(commitments.PREFIX + text[len(PREFIX) :])


def candidates(snapshot):
    selected = {
        uid: {**item, "data": commitments.PREFIX + item["data"][len(PREFIX) :]}
        for uid, item in snapshot.get("commitments", {}).items()
        if isinstance(item.get("data"), str) and item["data"].startswith(PREFIX + ":")
    }
    return commitments.candidates({"commitments": selected}, None)


def _manifest(claim, endpoint, listed):
    size = listed.get(MANIFEST, {}).get("size")
    if type(size) is not int or not 0 < size <= METADATA_BYTES:
        raise ValueError("feature manifest must be at most 16 KiB")
    url = f"{endpoint}/{claim['repo']}/resolve/{claim['revision']}/{MANIFEST}"
    with commitments.opener().open(url, timeout=30) as response:
        raw = response.read(METADATA_BYTES + 1)
    if len(raw) != size:
        raise ValueError("feature manifest size differs from the pinned revision")
    return raw, validate_manifest(json.loads(raw))


def parent(claim, endpoint=None):
    """Untrusted routing metadata only: fetch verifies the entire committed bundle."""
    endpoint = endpoint or commitments.HF_ENDPOINT
    listed = commitments.revision_files(claim["repo"], claim["revision"], endpoint)
    return _manifest(claim, endpoint, listed)[1]


def fetch(claim, destination, *, track_id, model, endpoint=None):
    """Download only bounded adapter artifacts at the pinned revision and verify all bytes."""
    format_commitment(claim["sha256"], claim["revision"], claim["repo"])
    model = models.validate_spec(model) if isinstance(model, dict) else model
    endpoint = endpoint or commitments.HF_ENDPOINT
    started = time.monotonic()
    listed = commitments.revision_files(claim["repo"], claim["revision"], endpoint)
    raw, manifest = _manifest(claim, endpoint, listed)
    if manifest["track_id"] != track_id or manifest["model"] != models.spec(model):
        raise ValueError("feature track or parent differs from the required profile")
    files = models.candidate_files(model, "adapter")
    budget = models.artifact_budget("adapter", model)
    if any(name not in listed for name in files):
        raise ValueError("feature bundle is missing an adapter artifact")
    sizes = [listed[name]["size"] for name in files]
    if any(type(size) is not int or size <= 0 for size in sizes) or sum(sizes) > budget:
        raise ValueError("invalid artifact size or feature bundle exceeds byte budget")
    if any(listed[name]["size"] > METADATA_BYTES for name in ("model.json", "adapter_config.json")):
        raise ValueError("feature adapter metadata must be at most 16 KiB")
    destination = Path(destination)
    destination.mkdir()
    (destination / MANIFEST).write_bytes(raw)
    total = 0
    for name, expected in zip(files, sizes, strict=True):
        url = f"{endpoint}/{claim['repo']}/resolve/{claim['revision']}/{quote(name)}"
        with commitments.opener().open(url, timeout=30) as response:
            size = int(response.headers.get("Content-Length", "-1"))
            if size != expected or size > budget - total:
                raise ValueError("artifact size differs from the pinned revision or byte budget")
            received = 0
            with (destination / name).open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > size or time.monotonic() - started > commitments.DOWNLOAD_SECONDS:
                        raise ValueError("artifact exceeds declared size or download deadline")
                    output.write(chunk)
            if received != size:
                raise ValueError("truncated artifact")
            total += received
    validate_bundle(destination)
    if bundle_hash(destination) != claim["sha256"]:
        raise ValueError("downloaded feature bundle hash differs from the commitment")
    return {
        "bytes": total + len(raw),
        "download_ms": (time.monotonic() - started) * 1000,
        "manifest": manifest,
    }
