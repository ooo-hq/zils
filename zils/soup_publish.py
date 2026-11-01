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


FEED_ROUNDS = 60  # newest public round records served per pool by GET /soup


def _read_object(path):
    """The JSON object at `path`, or None when it is missing, unreadable or not an object."""
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _feed_rounds(state, limit):
    """The newest `limit` readable public.json records under `state`/rounds, newest first, each with
    the UTC time its public.json was written as ``finished_at``. Only public.json is read."""
    from datetime import UTC, datetime

    found = []
    try:
        folders = list((state / "rounds").iterdir())
    except OSError:
        return []
    for folder in folders:
        path = folder / "public.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        found.append((mtime, folder.name, path))
    rounds = []
    for mtime, _, path in sorted(found, key=lambda f: (f[0], f[1]), reverse=True):
        record = _read_object(path)
        if record is None or not isinstance(record.get("round_id"), str):
            continue
        finished = datetime.fromtimestamp(mtime, UTC).isoformat()
        rounds.append({**record, "finished_at": finished})
        if len(rounds) == limit:
            break
    return rounds


def _round_bodies(record):
    out = []
    for member in record.get("members") or []:
        if isinstance(member, dict) and isinstance(member.get("body_sha256"), str):
            out.append(member["body_sha256"])
    crown = record.get("crown")
    if isinstance(crown, dict) and isinstance(crown.get("body_sha256"), str):
        out.append(crown["body_sha256"])
    return out


def feed(config, directory, *, limit=FEED_ROUNDS):
    """The payload of the validator's ``GET /soup``: each enabled soup pool's newest public round
    records and, for every body those rounds seat or crown, its house name and repo when known.

    House names and repos come from the pool's house registry (by pinned hash, or by the hash this
    validator resolved an unpinned entry to) and repos otherwise from the intake record in
    ``pool.json``; anything else is null. A missing or malformed file is skipped, never raised.
    """
    from pathlib import Path

    from . import model_pools, soup_pool

    pools = {}
    if "base_pools" not in config:
        return {"kind": "soup", "pools": pools}
    normalized = model_pools.validate(config["base_pools"])["pools"]
    for pool, entry in normalized.items():
        if not (entry["enabled"] and model_pools.is_soup(entry)):
            continue
        state = soup_pool.state_dir(Path(directory), pool)
        rounds = _feed_rounds(state, limit)
        pool_state = _read_object(state / "pool.json") or {}
        known = pool_state.get("bodies") if isinstance(pool_state.get("bodies"), dict) else {}
        resolved = pool_state.get("house") if isinstance(pool_state.get("house"), dict) else {}
        try:
            house = soup_pool.house(entry, directory)
        except (OSError, ValueError):
            house = []
        by_sha = {}
        for item in house:
            sha = item["body_sha256"] or resolved.get(f"{item['repo']}@{item['revision']}")
            if isinstance(sha, str):
                by_sha.setdefault(sha, item)
        bodies = {}
        for record in rounds:
            for sha in _round_bodies(record):
                if sha in bodies:
                    continue
                item = by_sha.get(sha)
                info = known.get(sha) if isinstance(known.get(sha), dict) else {}
                name = item["name"] if item else None
                if name is None and info.get("house") and isinstance(info.get("name"), str):
                    name = info["name"]
                repo = item["repo"] if item else info.get("repo")
                bodies[sha] = {"house": name, "repo": repo if isinstance(repo, str) else None}
        pools[pool] = {"bodies": bodies, "rounds": rounds}
    return {"kind": "soup", "pools": pools}
