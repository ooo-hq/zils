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
DTYPES = {
    "BF16": 2,
    "F32": 4,
}  # stored dtypes a body may use; every tensor is canonicalised to bf16


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


def _regular(path, trusted=False):
    """`path` as a regular file. A trusted reference (the chassis or base from our own snapshot
    download, or a miner's own folder) may be a symlink into the Hugging Face blob store and is
    resolved; a body the validator downloaded must be made of regular files."""
    path = Path(path)
    if trusted and path.is_symlink():
        path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"body requires a regular {path.name}")
    return path


def load_index(folder, *, trusted=False):
    """name -> (file, key) for a sharded (`model.safetensors.index.json`) or single-file checkpoint."""
    folder = Path(folder)
    index = folder / "model.safetensors.index.json"
    if not index.exists():
        single = _header(folder / "model.safetensors", trusted=trusted)
        return {key: ("model.safetensors", key) for key in single}
    value = json.loads(index.read_text())
    weight_map = value.get("weight_map") if isinstance(value, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("body index has no valid weight_map")
    for key, file in weight_map.items():
        if not isinstance(key, str) or not isinstance(file, str) or Path(file).name != file:
            raise ValueError("body index entries must name a shard in the checkpoint folder")
    return {key: (file, key) for key, file in weight_map.items()}


def _header(path, *, trusted=False):
    """A safetensors header: key -> {"dtype", "shape", "data_offsets"} with offsets made absolute."""
    path = _regular(path, trusted)
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


def _tensors(folder, chassis_names, *, trusted=False):
    """chassis name -> (path, header entry) for the mapped source tensors."""
    folder = Path(folder)
    index = load_index(folder, trusted=trusted)
    mapped = map_names(index, chassis_names)
    headers, out = {}, {}
    for name, source in mapped.items():
        file, key = index[source]
        if file not in headers:
            headers[file] = _header(folder / file, trusted=trusted)
        if key not in headers[file]:
            raise ValueError(f"{file} has no tensor {key}")
        out[name] = (_regular(folder / file, trusted), headers[file][key])
    return out


def _canonical(name, info):
    """The tensor's shape, after checking it is stored as bf16 or f32 with a matching byte size."""
    shape = info["shape"]
    start, end = info["data_offsets"]
    if info["dtype"] not in DTYPES:
        raise ValueError(f"body tensor {name} must be bfloat16 or float32, not {info['dtype']}")
    if end - start != DTYPES[info["dtype"]] * math.prod(shape):
        raise ValueError(f"body tensor {name} size differs from its shape")
    return shape


def read_bf16(path, info, offset, count):
    """`count` elements from `offset` of one tensor as bf16 bits (uint16), whatever the stored
    dtype: f32 is rounded to the nearest bf16 (ties to even), as a cast to bfloat16 does."""
    import numpy as np

    width = DTYPES[info["dtype"]]
    with Path(path).open("rb") as stream:
        stream.seek(info["data_offsets"][0] + width * offset)
        raw = stream.read(width * count)
    if len(raw) != width * count:
        raise ValueError(f"{Path(path).name} is truncated")
    if width == 2:
        return np.frombuffer(raw, dtype="<u2")
    bits = np.frombuffer(raw, dtype="<u4").astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    return np.where(nan, 0x7FC0, rounded).astype(np.uint16)


def _bf16_blocks(path, info):
    """The tensor's canonical bf16 bits, in blocks of a fixed element count."""
    total, step = math.prod(info["shape"]), CHUNK // 4
    for offset in range(0, total, step):
        yield read_bf16(path, info, offset, min(step, total - offset))


def _float64(bits):
    import numpy as np

    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


class WeightError(ValueError):
    """The body's own weights (hash verified) fail a check: its hash is refused for good."""


def _digest(folder, chassis_names, *, trusted=False):
    """(body hash, the first tensor holding a NaN or infinity, or None), in one streamed pass over
    the canonical bf16 values, so a tensor stored as f32 hashes as its bf16 cast. A bf16 NaN or
    infinity has all exponent bits set."""
    digest, bad = hashlib.sha256(), None
    for name, (path, info) in sorted(_tensors(folder, chassis_names, trusted=trusted).items()):
        shape = _canonical(name, info)
        digest.update(f"{name}\0{','.join(map(str, shape))}\0".encode())
        for block in _bf16_blocks(path, info):
            if bad is None and ((block & 0x7F80) == 0x7F80).any():
                bad = name
            digest.update(block.astype("<u2").tobytes())
    return digest.hexdigest(), bad


def body_hash(folder, chassis_names, *, trusted=False):
    """sha256 over each mapped chassis tensor in sorted name order: name, shape, canonical bf16
    bytes. Every value must be finite."""
    digest, bad = _digest(folder, chassis_names, trusted=trusted)
    if bad is not None:
        raise ValueError(f"body tensor {bad} is not finite")
    return digest


def relative_change(
    folder, base_folder, chassis_names, *, prefix="language_model.layers.", trusted=False
):
    """‖body − base‖ / ‖base‖ over the chassis tensors whose names start with `prefix`, streamed,
    on canonical bf16 values. The base is a trusted reference; `trusted` covers the body."""
    names = [n for n in sorted(chassis_names) if n.startswith(prefix)]
    if not names:
        raise ValueError(f"no chassis tensors start with {prefix!r}")
    body = _tensors(folder, chassis_names, trusted=trusted)
    base = _tensors(base_folder, chassis_names, trusted=True)
    delta = norm = 0.0
    for name in names:
        if _canonical(name, body[name][1]) != _canonical(name, base[name][1]):
            raise ValueError(f"body tensor {name} shape differs from the base")
        for a, b in zip(_bf16_blocks(*body[name]), _bf16_blocks(*base[name]), strict=True):
            x, y = _float64(a), _float64(b)
            delta += float(((x - y) ** 2).sum())
            norm += float((y**2).sum())
    if norm == 0:
        raise ValueError("base tensors are all zero")
    return math.sqrt(delta / norm)


def _license(folder, trusted=False):
    path = Path(folder) / "README.md"
    if trusted and path.is_symlink():
        path = path.resolve(strict=True)
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


def _owner(folder, trusted=False):
    path = Path(folder) / "zils-owner.json"
    if trusted and path.is_symlink():
        path = path.resolve(strict=True)
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
    folder = Path(chassis)  # a trusted reference
    headers, out = {}, {}
    for name, (file, key) in load_index(folder, trusted=True).items():
        if file not in headers:
            headers[file] = _header(folder / file, trusted=True)
        out[name] = tuple(headers[file][key]["shape"])
    return out


def check_layout(folder, chassis):
    """The cheap checks: every chassis tensor is named in a readable safetensors header within the
    file. Reads headers only."""
    _tensors(folder, _shapes(chassis))


def validate_body(
    folder, chassis, *, hotkey, base=None, max_change=0.15, sha256=None, trusted=False
):
    """Check a downloaded body against the chassis and return its body hash. `chassis` is the chassis
    checkpoint folder or a mapping name -> shape; `base`, when given, is the parent (Qwen) folder whose
    language layers the body may move by at most `max_change` (relative Frobenius norm).

    Failures of the repo (owner file, licence, size, unreadable files, a hash other than `sha256`)
    raise ValueError; failures of weights that hash to `sha256` (shape, finiteness, change) raise
    `WeightError`, so only the latter condemn the hash itself. The chassis and base are trusted
    references (symlinks into a snapshot are followed); the body's files must be regular unless
    `trusted` (a miner checking its own folder or its own snapshot download)."""
    folder = Path(folder)
    owner = _owner(folder, trusted)
    if owner != hotkey:
        raise ValueError(
            f"zils-owner.json names hotkey {owner}, not the committing hotkey {hotkey}"
        )
    license = _license(folder, trusted)
    if license not in LICENSES:
        raise ValueError(f"body license {license!r} is not one of {', '.join(sorted(LICENSES))}")
    total = sum(p.stat().st_size for p in folder.rglob("*.safetensors"))
    if total > MAX_BYTES:
        raise ValueError(f"body safetensors total {total} bytes, over the {MAX_BYTES} byte limit")
    shapes = _shapes(chassis)
    tensors = sorted(_tensors(folder, shapes, trusted=trusted).items())
    for name, (_, info) in tensors:
        _canonical(name, info)
    digest, bad = _digest(folder, shapes, trusted=trusted)
    if sha256 is not None and digest != sha256:
        raise ValueError("downloaded body hash differs from the commitment")
    for name, (_, info) in tensors:
        if tuple(info["shape"]) != shapes[name]:
            raise WeightError(f"body tensor {name} shape {info['shape']} differs from the chassis")
    if bad is not None:
        raise WeightError(f"body tensor {bad} is not finite")
    if base is not None:
        change = relative_change(folder, base, shapes, trusted=trusted)
        if not change <= max_change:  # NaN fails too
            raise WeightError(f"body change from the base {change:.4f} exceeds {max_change}")
    return digest
