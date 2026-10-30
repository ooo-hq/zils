"""Soup copy checks: reject a body that copies an earlier commitment, re-uploads the soup or release, or
behaves as a twin of a pool member.

Weight checks run on a sketch of the body's change from the base (Qwen) over the language-layer tensors.
The sketch is a seeded sparse random projection (count sketch): element `i` of tensor `name` adds
`±delta[i]` to one of `dims` buckets, with bucket and sign taken from a 64-bit hash of (seed, name, i).
It is linear in the change, so a weight average of bodies sketches to the same average of their sketches,
and it depends only on tensor names and values, never on shard layout. Tensors are read in fixed slices,
so a 4B body never sits in memory whole."""

import hashlib
import math
from pathlib import Path

import numpy as np

from . import bodies

try:
    from scipy.optimize import nnls as _scipy_nnls
except ImportError:  # pragma: no cover - the coordinator env has scipy
    _scipy_nnls = None

PREFIX = "language_model.layers."
READ_ELEMENTS = 4 * 1024**2
TEMPERATURES = np.exp(np.linspace(math.log(0.5), math.log(2.5), 161))
FLOOR = 1e-12


class Sketch:
    """A fixed random projection of a body's language-layer change from the base to `dims` numbers."""

    def __init__(self, seed: bytes, dims: int = 4096):
        if not isinstance(seed, bytes) or not seed:
            raise ValueError("sketch seed must be non-empty bytes")
        if not isinstance(dims, int) or dims < 1:
            raise ValueError("sketch dims must be a positive integer")
        self.seed, self.dims = seed, dims

    def _key(self, name):
        digest = hashlib.sha256(self.seed + b"\0" + name.encode()).digest()
        return np.uint64(int.from_bytes(digest[:8], "little"))

    def sketch(self, folder, base_folder, chassis_names) -> np.ndarray:
        names = [n for n in sorted(chassis_names) if n.startswith(PREFIX)]
        if not names:
            raise ValueError(f"no chassis tensors start with {PREFIX!r}")
        body = bodies._tensors(folder, chassis_names)
        base = bodies._tensors(base_folder, chassis_names)
        out = np.zeros(self.dims, dtype=np.float64)
        for name in names:
            shape = bodies._bf16(name, body[name][1])
            if shape != bodies._bf16(name, base[name][1]):
                raise ValueError(f"body tensor {name} shape differs from the base")
            key, part = self._key(name), np.zeros(self.dims, dtype=np.float64)
            total = math.prod(shape)
            for offset in range(0, total, READ_ELEMENTS):
                count = min(READ_ELEMENTS, total - offset)
                delta = _read(*body[name], offset, count) - _read(*base[name], offset, count)
                index = np.arange(offset, offset + count, dtype=np.uint64)
                bucket, sign = _hash(index, key, self.dims)
                part += np.bincount(bucket, weights=sign * delta, minlength=self.dims)
            out += part
        return out


def sketch(folder, base_folder, chassis_names, *, seed: bytes, dims: int = 4096) -> np.ndarray:
    return Sketch(seed, dims).sketch(folder, base_folder, chassis_names)


def _read(path, info, offset, count):
    """`count` bf16 elements from `offset` of one tensor, as float64."""
    start = info["data_offsets"][0] + 2 * offset
    with Path(path).open("rb") as stream:
        stream.seek(start)
        raw = stream.read(2 * count)
    if len(raw) != 2 * count:
        raise ValueError(f"{Path(path).name} is truncated")
    bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return bits.view(np.float32).astype(np.float64)


def _hash(index, key, dims):
    """splitmix64 of (index + key): bucket from the low bits, sign from the top bit."""
    with np.errstate(over="ignore"):
        x = index + key
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    bucket = (x % np.uint64(dims)).astype(np.intp)
    sign = 1.0 - 2.0 * (x >> np.uint64(63)).astype(np.float64)
    return bucket, sign


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / norm if norm > 0 else 0.0


def _nnls_fallback(matrix, target):
    """Lawson–Hanson non-negative least squares, for when scipy is not importable."""
    matrix, target = np.asarray(matrix, dtype=np.float64), np.asarray(target, dtype=np.float64)
    n = matrix.shape[1]
    x, active = np.zeros(n), np.zeros(n, dtype=bool)
    tol = 10 * np.finfo(float).eps * np.abs(matrix).sum(axis=0).max(initial=0) * max(matrix.shape)
    for _ in range(3 * n + 10):
        gradient = matrix.T @ (target - matrix @ x)
        if active.all() or gradient[~active].max(initial=-np.inf) <= tol:
            break
        active[np.argmax(np.where(active, -np.inf, gradient))] = True
        while True:
            trial = np.zeros(n)
            trial[active] = np.linalg.lstsq(matrix[:, active], target, rcond=None)[0]
            if trial[active].min() > 0:
                break
            blocked = active & (trial <= 0)
            step = np.min(x[blocked] / (x[blocked] - trial[blocked]))
            x = x + step * (trial - x)
            active &= x > tol
            x[~active] = 0.0
        x = trial
    return x


def explained_share(target, others) -> float:
    """‖P t‖² / ‖t‖², where P t is the non-negative least-squares fit of `target` on the columns `others`."""
    target = np.asarray(target, dtype=np.float64)
    norm = float(target @ target)
    if norm == 0 or len(others) == 0:
        return 0.0
    matrix = np.column_stack([np.asarray(o, dtype=np.float64) for o in others])
    if _scipy_nnls is not None:
        weights = _scipy_nnls(matrix, target)[0]
    else:
        weights = _nnls_fallback(matrix, target)
    fit = matrix @ weights
    return min(1.0, max(0.0, float(fit @ fit) / norm))


def is_copy(
    target,
    earlier,
    soup=None,
    release=None,
    *,
    house=(),
    explained=None,
    cosine_max=0.999,
    explained_max=0.8,
    others_max=0.3,
):
    """Why a body sketch is a copy, or None. `earlier` are sketches of earlier commitments by other
    hotkeys: a near-exact copy of one, or a body mostly explained by them, is a copy. `house`
    (house bodies) and `soup` and `release` (the current soup and release bodies) reject only
    near-exact re-uploads (cosine at least `cosine_max`): fine-tuning them is allowed.

    The body is also fitted on the earlier and house sketches jointly. The share the earlier
    (other-hotkey) bodies add to that fit beyond the house bodies alone must stay under
    `others_max`, so merging another hotkey's body with a house body is a copy, while a body
    explained by house bodies alone (a house fine-tune) is not. `explained`, when given, replaces
    `earlier` in both explained-share rules (for example only the earlier bodies still in the pool);
    near-exact copies are always checked against all of `earlier`. Only "explained by earlier" is a
    judgement on the current pool rather than on the weights themselves."""
    target = np.asarray(target, dtype=np.float64)
    if not np.any(target):
        return "no change from the base"
    if any(cosine(target, e) >= cosine_max for e in earlier):
        return "cosine"
    if any(cosine(target, h) >= cosine_max for h in house):
        return "house re-upload"
    for label, other in (("soup re-upload", soup), ("release re-upload", release)):
        if other is not None and cosine(target, other) >= cosine_max:
            return label
    explained = list(earlier if explained is None else explained)
    if explained_share(target, explained) >= explained_max:
        return "explained by earlier"
    if explained:
        joint = explained_share(target, [*explained, *house])
        if joint - explained_share(target, list(house)) >= others_max:
            return "explained by earlier"
    return None


def _table(rows):
    table = {}
    for row in rows:
        item, probabilities = row.get("id"), row.get("probabilities")
        if not isinstance(item, str) or not isinstance(probabilities, dict) or not probabilities:
            raise ValueError("prediction rows need an id and a non-empty probabilities mapping")
        if item in table:
            raise ValueError(f"prediction rows repeat item {item}")
        values = list(probabilities.values())
        if not all(isinstance(v, int | float) and math.isfinite(v) and v >= 0 for v in values):
            raise ValueError(f"item {item} has an invalid probability")
        if sum(values) <= 0:
            raise ValueError(f"item {item} probabilities sum to zero")
        table[item] = probabilities
    return table


def _jsd_bits(p, q):
    m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(p > 0, p * np.log2(p / m), 0.0)
        right = np.where(q > 0, q * np.log2(q / m), 0.0)
    return 0.5 * (left.sum(axis=1) + right.sum(axis=1))


def is_twin(rows_a, rows_b, *, jsd_max=0.005, min_items=50) -> bool:
    """True when b, after one fitted temperature, predicts like a: mean Jensen–Shannon divergence in bits
    under `jsd_max` over at least `min_items` shared items. Fewer shared items are never judged twins."""
    a, b = _table(rows_a), _table(rows_b)
    shared = sorted(set(a) & set(b))
    if len(shared) < min_items:
        return False
    options = [sorted(set(a[i]) | set(b[i])) for i in shared]
    width = max(len(o) for o in options)
    p, q = np.zeros((len(shared), width)), np.zeros((len(shared), width))
    mask = np.zeros((len(shared), width), dtype=bool)
    for row, (item, names) in enumerate(zip(shared, options, strict=True)):
        p[row, : len(names)] = [a[item].get(n, 0.0) for n in names]
        q[row, : len(names)] = [b[item].get(n, 0.0) for n in names]
        mask[row, : len(names)] = True
    p /= p.sum(axis=1, keepdims=True)
    logits = np.where(mask, np.log(np.maximum(q / q.sum(axis=1, keepdims=True), FLOOR)), -np.inf)
    best = math.inf
    for temperature in np.append(TEMPERATURES, 1.0):
        scaled = logits / temperature
        scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        best = min(best, float(_jsd_bits(p, scaled / scaled.sum(axis=1, keepdims=True)).mean()))
    return best < jsd_max
