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

    def sketch(
        self, folder, base_folder, chassis_names, *, trusted=False, any_layout=False
    ) -> np.ndarray:
        """``trusted`` reads a pinned reference folder (the chassis), whose shards may be links.
        ``any_layout`` reads only the language layers of a known public checkpoint in any usual
        layout (``public_layers``); the sketch is the same as the full body's would be."""
        names = [n for n in sorted(chassis_names) if n.startswith(PREFIX)]
        if not names:
            raise ValueError(f"no chassis tensors start with {PREFIX!r}")
        if any_layout:
            body = public_layers(folder, chassis_names)
        else:
            body = bodies._tensors(folder, chassis_names, trusted=trusted)
        base = bodies._tensors(base_folder, chassis_names, trusted=True)
        out = np.zeros(self.dims, dtype=np.float64)
        for name in names:
            shape = bodies._canonical(name, body[name][1])
            if shape != bodies._canonical(name, base[name][1]):
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


def sketch(
    folder,
    base_folder,
    chassis_names,
    *,
    seed: bytes,
    dims: int = 4096,
    trusted=False,
    any_layout=False,
) -> np.ndarray:
    return Sketch(seed, dims).sketch(
        folder, base_folder, chassis_names, trusted=trusted, any_layout=any_layout
    )


def public_layers(folder, chassis_names):
    """chassis language-layer name -> (path, header entry) for a known public checkpoint in any usual
    layout: the chassis's own (``language_model.layers.*``, optionally under ``model.``) or a
    text-only one (``layers.*`` or ``model.layers.*``). Every other tensor (embeddings, norm, vision
    tower, heads) is ignored; each chassis language layer must be present exactly once."""
    folder = Path(folder)
    index = bodies.load_index(folder)
    wanted, mapped = {n for n in chassis_names if n.startswith(PREFIX)}, {}
    for key in index:
        name = key.removeprefix("model.")
        name = "language_model." + name if name.startswith("layers.") else name
        if name not in wanted:
            continue
        if name in mapped:
            raise ValueError(f"checkpoint names {mapped[name]} and {key} map to the same tensor")
        mapped[name] = key
    if missing := sorted(wanted - set(mapped)):
        raise ValueError(f"checkpoint is missing chassis tensor {missing[0]}")
    headers, out = {}, {}
    for name, source in mapped.items():
        file, key = index[source]
        if file not in headers:
            headers[file] = bodies._header(folder / file)
        if key not in headers[file]:
            raise ValueError(f"{file} has no tensor {key}")
        out[name] = (bodies._regular(folder / file), headers[file][key])
    return out


def _read(path, info, offset, count):
    """`count` elements from `offset` of one tensor, canonicalised to bf16, as float64."""
    return bodies._float64(bodies.read_bf16(path, info, offset, count))


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


def _nnls(matrix, target):
    if _scipy_nnls is not None:
        return _scipy_nnls(matrix, target)[0]
    return _nnls_fallback(matrix, target)


def explained_share(target, others) -> float:
    """‖P t‖² / ‖t‖², where P t is the non-negative least-squares fit of `target` on the columns `others`."""
    target = np.asarray(target, dtype=np.float64)
    norm = float(target @ target)
    if norm == 0 or len(others) == 0:
        return 0.0
    matrix = np.column_stack([np.asarray(o, dtype=np.float64) for o in others])
    fit = matrix @ _nnls(matrix, target)
    return min(1.0, max(0.0, float(fit @ fit) / norm))


def base_shares(target, bases) -> dict:
    """{name: share} of a body's change explained by each of `bases` ({name: sketch}): the
    non-negative fit's weight times that base's overlap with the target, over ‖t‖². At the fit's
    optimum the shares add up to ``explained_share``; what is left is the body's own change."""
    target = np.asarray(target, dtype=np.float64)
    names = list(bases)
    norm = float(target @ target)
    if not names or norm == 0:
        return {name: 0.0 for name in names}
    columns = [np.asarray(bases[name], dtype=np.float64) for name in names]
    weights = _nnls(np.column_stack(columns), target)
    return {
        name: max(0.0, float(w * (column @ target)) / norm)
        for name, w, column in zip(names, weights, columns)
    }


def _allowed(house, soup, release):
    return [*house, *(base for base in (soup, release) if base is not None)]


def base_residual(target, *, house=(), soup=None, release=None) -> float:
    """The share of a body's change (squared norm) left after fitting it on the allowed bases: the
    house bodies, the current soup and the release (non-negative least squares). A body that is
    almost all allowed bases has little of its own; a real fine-tune of one keeps about 10%."""
    return 1.0 - explained_share(target, _allowed(house, soup, release))


def is_copy(
    target,
    earlier,
    soup=None,
    release=None,
    *,
    house=(),
    public=(),
    explained=None,
    cosine_max=0.999,
    others_max=0.3,
    scout_cosine=0.9,
):
    """Why a body sketch is a copy, or None. `earlier` are sketches of earlier commitments by other
    hotkeys: a near-exact copy of one is a copy. `house` (house bodies), `soup` and `release` (the
    current soup and release bodies) are the allowed bases: they reject only near-exact re-uploads
    (cosine at least `cosine_max`), and fine-tuning them is allowed.

    What a body adds beyond the allowed bases is its own work, unless other hotkeys' bodies explain
    it. The body is fitted on the allowed bases alone and on the allowed bases plus the earlier
    bodies. If the earlier bodies explain `others_max` or more of the part the allowed bases leave
    unexplained, it is a copy. So a fine-tune of the soup or release is not a copy even though both
    are mostly seated miners' bodies, and neither are two independent fine-tunes of one house body;
    a near-copy of another hotkey's body, a merge of it with a house body, or a re-weighted mix of
    the seated bodies is. `explained`, when given, replaces `earlier` in the joint fit (for example
    only the earlier bodies still in the pool); near-exact copies are always checked against all of
    `earlier`. Only "explained by earlier" is a judgement on the current pool rather than on the
    weights themselves.

    `public` (sketches of the pool's listed known-public models) are allowed bases too, so a real
    fine-tune of one is never a copy of another hotkey's re-upload of it. Unlike the other bases,
    re-uploading one is allowed (it is a scout's work, paid at the scout rate): a body within
    `scout_cosine` of a listed model is refused only when an earlier body still in the pool is a
    near re-upload of the same model, as explained by that earlier body (the first scout wins)."""
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
    for listed in public:
        if cosine(target, listed) >= scout_cosine and any(
            cosine(e, listed) >= scout_cosine for e in explained
        ):
            return "explained by earlier"
    if explained:
        allowed = _allowed([*house, *public], soup, release)
        own = 1.0 - explained_share(target, allowed)  # the share beyond the allowed bases
        if own > 1e-9:
            beyond = explained_share(target, [*explained, *allowed]) - (1.0 - own)
            if beyond / own >= others_max:
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
