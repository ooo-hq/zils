"""Soup builder: the uniform average of bodies written on a pinned d3 chassis.

The chassis is the pinned parent folder. The soup takes the parent's shard layout and tensor names, an
FP32 average of each tensor over the bodies cast to the parent's dtype, and every other candidate file
from the parent byte for byte, so it runs in the parent's runtime with the parent's answer layer.
A body named with ``strip-model`` drops a leading ``model.`` (Hugging Face names); tensors the chassis
lacks (``lm_head*``, ``mtp.*``) are ignored.
"""

import contextlib
import hashlib
import json
import shutil
import struct
from pathlib import Path

from . import d3, models

RENAMES = {None: lambda key: key, "strip-model": lambda key: key.removeprefix("model.")}
READOUT = "readout.safetensors"


def _sources(root, rename):
    if rename not in RENAMES:
        raise ValueError(f"unknown body rename: {rename}")
    root = Path(root)
    return {RENAMES[rename](key): (root / shard, key) for key, shard in d3.weight_map(root).items()}


def _sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(out, chassis, bodies, *, model):
    """Write the soup of `bodies` ([(folder, rename)]) on `chassis` as a `model` base candidate at `out`."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    out, chassis = Path(out), Path(chassis)
    files = models.candidate_files(model, "candidate")
    if not bodies:
        raise ValueError("a soup needs at least one body")
    layout = d3.weight_map(chassis)
    sources = [_sources(root, rename) for root, rename in bodies]
    for (root, _), source in zip(bodies, sources, strict=True):  # fail before writing anything
        if missing := sorted(set(layout) - set(source)):
            raise KeyError(f"body {root} lacks chassis tensor {missing[0]}")
    with contextlib.ExitStack() as stack:
        streams = {}

        def stream(path):
            if path not in streams:
                streams[path] = stack.enter_context(safe_open(str(path), framework="pt"))
            return streams[path]

        for name, shard in layout.items():
            shape = stream(chassis / shard).get_slice(name).get_shape()
            for source in sources:
                path, key = source[name]
                if stream(path).get_slice(key).get_shape() != shape:
                    raise ValueError(f"body tensor {key} in {path} differs in shape from {name}")
        out.mkdir(parents=True, exist_ok=False)
        for shard in sorted(set(layout.values())):
            tensors = {}
            for name in stream(chassis / shard).keys():
                dtype = stream(chassis / shard).get_tensor(name).dtype
                # Each body tensor is canonicalised to bf16 first (a body may store some as f32).
                total = sum(
                    stream(p).get_tensor(k).bfloat16().float()
                    for p, k in (s[name] for s in sources)
                )
                tensors[name] = (total / len(sources)).to(dtype).contiguous()
            save_file(tensors, str(out / shard), metadata={"format": "pt"})
            del tensors
    for name in files:
        if name != "model.json" and name not in layout.values():
            shutil.copyfile(chassis / name, out / name)
    models.write_metadata(out, kind="candidate", model=model)
    if _sha256(out / READOUT) != _sha256(chassis / READOUT):
        raise ValueError("soup readout.safetensors differs from the chassis answer layer")
    if _sha256(out / READOUT) != d3.pins(model)["files"][READOUT]:
        raise ValueError("soup readout.safetensors differs from the pinned parent answer layer")
    return out


def soup_hash(folder):
    """SHA-256 over the shard tensors' sorted names, shapes and raw bytes (layout-independent)."""
    folder = Path(folder)
    headers = {}
    for shard in set(d3.weight_map(folder).values()):
        with open(folder / shard, "rb") as stream:
            (size,) = struct.unpack("<Q", stream.read(8))
            headers[shard] = (8 + size, json.loads(stream.read(size)))
    digest = hashlib.sha256()
    for name, shard in sorted(d3.weight_map(folder).items()):
        start, header = headers[shard]
        begin, end = header[name]["data_offsets"]
        digest.update(json.dumps([name, header[name]["shape"], end - begin]).encode() + b"\n")
        with open(folder / shard, "rb") as stream:
            stream.seek(start + begin)
            for _ in range((end - begin) // (1 << 24)):
                digest.update(stream.read(1 << 24))
            digest.update(stream.read((end - begin) % (1 << 24)))
    return digest.hexdigest()
