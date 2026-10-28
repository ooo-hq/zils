"""Compare a miner's answer probabilities with our own runtime's answer."""

import math
from dataclasses import dataclass

FLOOR = 1e-6
CONSIDER = 0.01


@dataclass(frozen=True)
class Thresholds:
    # 28 Oct 2026 calibration: honest d3-mini on an H100 vs the RTX 4000 Ada reference drifted at most
    # 0.166 on one item (p99 0.044) over 1,000 serving items; these keep a >=3x margin.
    mean_pass: float = 0.10
    mean_fail: float = 0.25
    item_fail: float = 0.5


@dataclass(frozen=True)
class Verdict:
    outcome: str  # "pass" | "marginal" | "fail" | "mismatch"
    mean: float
    worst: float


def _clean(probs: dict[str, float]) -> dict[str, float]:
    if not isinstance(probs, dict):
        raise ValueError("answer must be a dict")
    clean = {}
    for k, p in probs.items():
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ValueError("probability must be a number")
        try:
            p = float(p)  # huge ints overflow here
        except OverflowError:
            raise ValueError("probability out of range") from None
        if not math.isfinite(p) or p < 0:
            raise ValueError("probability must be finite and non-negative")
        clean[k] = p
    return clean


def _centred(probs: dict[str, float], keys: list[str]) -> dict[str, float]:
    logs = {k: math.log(max(probs[k], FLOOR)) for k in keys}
    mean = sum(logs.values()) / len(logs)
    return {k: v - mean for k, v in logs.items()}


def _top(probs: dict[str, float]) -> set[str]:
    best = max(probs.values())
    return {k for k, v in probs.items() if v == best}


def distance(ours: dict[str, float], theirs: dict[str, float]) -> float:
    """Largest absolute gap between the two sides' centred log-probabilities, over the options
    either side gives at least CONSIDER; 0.0 if fewer than two options qualify."""
    if not isinstance(ours, dict) or not isinstance(theirs, dict):
        raise ValueError("answer must be a dict")
    if ours.keys() != theirs.keys() or not ours:
        raise ValueError("answers must have the same options")
    ours, theirs = _clean(ours), _clean(theirs)
    keys = [k for k in ours if ours[k] >= CONSIDER or theirs[k] >= CONSIDER]
    if len(keys) < 2:
        return 0.0
    a, b = _centred(ours, keys), _centred(theirs, keys)
    return max(abs(a[k] - b[k]) for k in keys)


def compare(
    ours: dict[str, dict[str, float]],
    theirs: dict[str, dict[str, float]],
    thresholds: Thresholds = Thresholds(),
) -> Verdict:
    mismatch = Verdict("mismatch", math.inf, math.inf)
    if not isinstance(ours, dict) or not isinstance(theirs, dict):
        return mismatch
    if ours.keys() != theirs.keys() or not ours:
        return mismatch
    try:
        ds = [distance(ours[q], theirs[q]) for q in ours]
    except ValueError:
        return mismatch
    mean, worst = sum(ds) / len(ds), max(ds)
    if mean > thresholds.mean_fail or worst > thresholds.item_fail:
        outcome = "fail"
    elif mean <= thresholds.mean_pass:
        outcome = "pass"
    else:
        outcome = "marginal"
    if outcome == "pass" and any(not (_top(ours[q]) & _top(theirs[q])) for q in ours):
        outcome = "marginal"
    return Verdict(outcome, mean, worst)
