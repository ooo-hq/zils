"""Public soup round record: aggregates and commitments, never exam items, labels or families."""

import hashlib
import json
import math

from .base_metrics import CATEGORY_WEIGHTS

MEMBER_FIELDS = ("hotkey", "body_sha256", "rate", "seat_weight")
CROWN_FIELDS = ("hotkey", "body_sha256", "rate", "weight")


def canonical_json(value):
    """Sorted keys, no whitespace, no NaN: the bytes every commitment hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _half_point(value):
    return math.floor(float(value) * 2 + 0.5) / 2


def public_record(private_round):
    """Build the publishable round record from the validator's private round record.

    Category scores are on the 0-100 point scale and published to 0.1 point; member
    contributions are published to the nearest 0.5 point (halves round up).
    """
    r = private_round
    categories = r["category_scores"]
    unknown = set(categories) - set(CATEGORY_WEIGHTS)
    if unknown:
        raise ValueError(f"unknown categories: {len(unknown)}")
    crown = r.get("crown")
    seed = r["seed"]
    record = {
        "round_id": r["round_id"],
        "pool_id": r["pool_id"],
        "seed": seed.hex() if isinstance(seed, bytes) else seed,
        "members": [
            {**{k: m[k] for k in MEMBER_FIELDS}, "contribution": _half_point(m["contribution"])}
            for m in r["members"]
        ],
        "crown": None if crown is None else {k: crown[k] for k in CROWN_FIELDS if k in crown},
        "category_scores": {c: round(float(v), 1) for c, v in sorted(categories.items())},
        "soup_sha256": r["soup_sha256"],
        "subset_sha256": _sha256(sorted(r["subset_ids"])),
        "predictions_sha256": _sha256(r["predictions"]),
    }
    if "temperature" in r:  # the released soup's fitted temperature, which serving applies
        record["scoring"] = r["scoring"]
        record["temperature"] = r["temperature"]
    if "decision" in r:  # the rounds whose items the seats and crown were scored on
        record["decision"] = {k: r["decision"][k] for k in ("rounds", "items")}
    return record


def verify_public(public, private_round):
    """Raise ValueError unless the private record reproduces the published one exactly."""
    expected = public_record(private_round)
    keys = set(expected) | set(public)
    bad = sorted(
        k for k in keys if canonical_json(expected.get(k)) != canonical_json(public.get(k))
    )
    if bad:
        raise ValueError(f"public record mismatch: {', '.join(bad)}")
