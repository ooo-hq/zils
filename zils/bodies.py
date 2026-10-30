"""Soup bodies: a miner or scout uploads a full model body (the chassis's language and vision tensors)
to Hugging Face and commits `zilsb2:<body sha256>:<revision>:<repo id>` with its hotkey.

The body hash covers only the chassis tensors after name mapping (name, shape, raw bf16 bytes, sorted
by name), so the same weights hash the same whatever the shard layout or `model.` prefix. Tensors are
read straight from safetensors headers and byte ranges, one at a time, so a 4B body never sits in
memory whole."""

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

from . import commitments, models

PREFIX = "zilsb2"
MAX_BYTES = models.MAX_CANDIDATE_BYTES
LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause"}
KEEP = ("language_model.", "visual.")
DROP = ("lm_head", "mtp.")
HEADER_BYTES = 8 * 1024**2  # a 4B body's header is ~100 KB; a bigger one is hostile, not a model
CHUNK = 64 * 1024**2


def format_commitment(sha256, revision, repo):
    if not isinstance(sha256, str) or not re.fullmatch("[a-f0-9]{64}", sha256):
        raise ValueError("invalid body hash")
    revision = commitments.check_revision(revision)
    return f"{PREFIX}:{sha256}:{revision}:{commitments.check_repo(repo)}"


def parse_commitment(text):
    parts = text.split(":") if isinstance(text, str) else []
    if len(parts) != 4 or parts[0] != PREFIX:
        raise ValueError(f"commitment is not {PREFIX}:<sha256>:<revision>:<repo>")
    _, sha256, revision, repo = parts
    if format_commitment(sha256, revision, repo) != text:
        raise ValueError("commitment is not canonical")
    return {"sha256": sha256, "revision": revision, "repo": repo}


def map_names(keys, chassis_names):
    """chassis name -> source name. One leading `model.` is stripped; `lm_head*` and `mtp.*` are dropped;
    every other name must be a chassis tensor under `language_model.` or `visual.`, and all must be present."""
    chassis, mapped = set(chassis_names), {}
    for key in keys:
        name = key.removeprefix("model.")
        if name.startswith(DROP):
            continue
        if not name.startswith(KEEP) or name not in chassis:
            raise ValueError(f"body has a tensor outside the chassis: {key}")
        if name in mapped:
            raise ValueError(f"body names {mapped[name]} and {key} map to the same tensor")
        mapped[name] = key
    if missing := sorted(chassis - set(mapped)):
        raise ValueError(f"body is missing chassis tensor {missing[0]}")
    return mapped


def load_index(folder):
    """name -> (file, key) for a sharded (`model.safetensors.index.json`) or single-file checkpoint."""
    folder = Path(folder)
    index = folder / "model.safetensors.index.json"
    if not index.exists():
        return {key: ("model.safetensors", key) for key in _header(folder / "model.safetensors")}
    value = json.loads(index.read_text())
    weight_map = value.get("weight_map") if isinstance(value, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("body index has no valid weight_map")
    for key, file in weight_map.items():
        if not isinstance(key, str) or not isinstance(file, str) or Path(file).name != file:
            raise ValueError("body index entries must name a shard in the checkpoint folder")
    return {key: (file, key) for key, file in weight_map.items()}


def _header(path):
    """A safetensors header: key -> {"dtype", "shape", "data_offsets"} with offsets made absolute."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"body requires a regular {path.name}")
    size = path.stat().st_size
    with path.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 0 < length <= min(HEADER_BYTES, size - 8):
            raise ValueError(f"{path.name} is not a safetensors file")
        try:
            header = json.loads(stream.read(length))
        except (ValueError, RecursionError):  # RecursionError: a deeply nested hostile header
            raise ValueError(f"{path.name} has an unreadable safetensors header") from None
    if not isinstance(header, dict):
        raise ValueError(f"{path.name} has an unreadable safetensors header")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(isinstance(v, str) for v in metadata.values()):
        raise ValueError(f"{path.name} metadata must map names to strings")
    out = {}
    for key, info in header.items():
        try:
            start, end = info["data_offsets"]
            valid = (
                isinstance(info["dtype"], str)
                and isinstance(info["shape"], list)
                and all(type(d) is int and d >= 0 for d in info["shape"])
                and type(start) is int
                and type(end) is int
                and 0 <= start <= end <= size - 8 - length
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"{path.name} tensor {key} is malformed or outside the file")
        out[key] = {**info, "data_offsets": (8 + length + start, 8 + length + end)}
    return out


def _tensors(folder, chassis_names):
    """chassis name -> (path, header entry) for the mapped source tensors."""
    folder = Path(folder)
    index = load_index(folder)
    mapped = map_names(index, chassis_names)
    headers, out = {}, {}
    for name, source in mapped.items():
        file, key = index[source]
        if file not in headers:
            headers[file] = _header(folder / file)
        if key not in headers[file]:
            raise ValueError(f"{file} has no tensor {key}")
        out[name] = (folder / file, headers[file][key])
    return out


def _bf16(name, info):
    shape = info["shape"]
    start, end = info["data_offsets"]
    if info["dtype"] != "BF16":
        raise ValueError(f"body tensor {name} must be bfloat16, not {info['dtype']}")
    if end - start != 2 * math.prod(shape):
        raise ValueError(f"body tensor {name} size differs from its shape")
    return shape


def _chunks(path, info):
    start, end = info["data_offsets"]
    with path.open("rb") as stream:
        stream.seek(start)
        while start < end:
            chunk = stream.read(min(CHUNK, end - start))
            if not chunk:
                raise ValueError(f"{path.name} is truncated")
            start += len(chunk)
            yield chunk


def body_hash(folder, chassis_names):
    """sha256 over each mapped chassis tensor in sorted name order: name, shape, raw bf16 bytes.
    Every value must be finite (a bf16 NaN or infinity has all exponent bits set)."""
    import numpy as np

    digest = hashlib.sha256()
    for name, (path, info) in sorted(_tensors(folder, chassis_names).items()):
        shape = _bf16(name, info)
        digest.update(f"{name}\0{','.join(map(str, shape))}\0".encode())
        for chunk in _chunks(path, info):
            if ((np.frombuffer(chunk, dtype="<u2") & 0x7F80) == 0x7F80).any():
                raise ValueError(f"body tensor {name} is not finite")
            digest.update(chunk)
    return digest.hexdigest()


def relative_change(folder, base_folder, chassis_names, *, prefix="language_model.layers."):
    """‖body − base‖ / ‖base‖ over the chassis tensors whose names start with `prefix`, streamed."""
    import torch

    names = [n for n in sorted(chassis_names) if n.startswith(prefix)]
    if not names:
        raise ValueError(f"no chassis tensors start with {prefix!r}")
    body, base = _tensors(folder, chassis_names), _tensors(base_folder, chassis_names)
    delta = norm = 0.0
    for name in names:
        if _bf16(name, body[name][1]) != _bf16(name, base[name][1]):
            raise ValueError(f"body tensor {name} shape differs from the base")
        for a, b in zip(_chunks(*body[name]), _chunks(*base[name]), strict=True):
            x = torch.frombuffer(bytearray(a), dtype=torch.bfloat16).double()
            y = torch.frombuffer(bytearray(b), dtype=torch.bfloat16).double()
            delta += float((x - y).square().sum())
            norm += float(y.square().sum())
    if norm == 0:
        raise ValueError("base tensors are all zero")
    return math.sqrt(delta / norm)


def _license(folder):
    path = Path(folder) / "README.md"
    if path.is_symlink() or not path.is_file():
        raise ValueError("body repo needs a README.md model card with a license")
    lines = path.read_text(errors="replace")[:65536].splitlines()
    if not lines or lines[0].strip() != "---" or "---" not in (line.strip() for line in lines[1:]):
        raise ValueError("README.md has no YAML front matter with a license")
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if match := re.fullmatch(r"license:\s*(.*?)\s*", line):
            return match.group(1).strip("\"'").lower()
    raise ValueError("README.md front matter has no license")


def _owner(folder):
    path = Path(folder) / "zils-owner.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4096:
        raise ValueError("body repo needs a small zils-owner.json naming the committing hotkey")
    try:
        value = json.loads(path.read_text())
    except ValueError:
        raise ValueError("zils-owner.json is not JSON") from None
    if (
        not isinstance(value, dict)
        or set(value) != {"hotkey"}
        or not isinstance(value["hotkey"], str)
    ):
        raise ValueError('zils-owner.json must be exactly {"hotkey": "<ss58>"}')
    return value["hotkey"]


def _shapes(chassis):
    if isinstance(chassis, Mapping):
        return {name: tuple(shape) for name, shape in chassis.items()}
    folder = Path(chassis)
    headers, out = {}, {}
    for name, (file, key) in load_index(folder).items():
        if file not in headers:
            headers[file] = _header(folder / file)
        out[name] = tuple(headers[file][key]["shape"])
    return out


def validate_body(folder, chassis, *, hotkey, base=None, max_change=0.15):
    """Check a downloaded body against the chassis and return its body hash. `chassis` is the chassis
    checkpoint folder or a mapping name -> shape; `base`, when given, is the parent (Qwen) folder whose
    language layers the body may move by at most `max_change` (relative Frobenius norm)."""
    folder = Path(folder)
    owner = _owner(folder)
    if owner != hotkey:
        raise ValueError(
            f"zils-owner.json names hotkey {owner}, not the committing hotkey {hotkey}"
        )
    license = _license(folder)
    if license not in LICENSES:
        raise ValueError(f"body license {license!r} is not one of {', '.join(sorted(LICENSES))}")
    total = sum(p.stat().st_size for p in folder.rglob("*.safetensors"))
    if total > MAX_BYTES:
        raise ValueError(f"body safetensors total {total} bytes, over the {MAX_BYTES} byte limit")
    shapes = _shapes(chassis)
    for name, (_, info) in sorted(_tensors(folder, shapes).items()):
        if info["dtype"] != "BF16":
            raise ValueError(f"body tensor {name} must be bfloat16, not {info['dtype']}")
        if tuple(info["shape"]) != shapes[name]:
            raise ValueError(f"body tensor {name} shape {info['shape']} differs from the chassis")
    digest = body_hash(folder, shapes)
    if base is not None:
        change = relative_change(folder, base, shapes)
        if not change <= max_change:  # NaN fails too
            raise ValueError(f"body change from the base {change:.4f} exceeds {max_change}")
    return digest
