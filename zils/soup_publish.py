"""Public soup round record: aggregates and commitments, never exam items, labels or families."""

import hashlib
import json
import math
import re

from .base_metrics import CATEGORY_WEIGHTS

HEX64 = re.compile("[a-f0-9]{64}")
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


def _iso(timestamp):
    from datetime import UTC, datetime

    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def live(state):
    """The pool's round in progress, or None: its stage, whether its process is alive, and one public
    state per committed claim. Reasons for a refusal stay in the private records."""
    from . import soup_pool

    job = _read_object(state / "job.json") or {}
    round_id = job.get("round_id")
    if not isinstance(round_id, str) or not isinstance(job.get("started"), (int, float)):
        return None
    folder = state / "rounds" / round_id
    if (folder / "public.json").exists():
        return None
    request = _read_object(state / f"request-{round_id}.json") or {}
    snapshot = request.get("snapshot") if isinstance(request.get("snapshot"), dict) else {}
    commits = snapshot.get("commitments") if isinstance(snapshot.get("commitments"), dict) else {}
    members = snapshot.get("members") if isinstance(snapshot.get("members"), dict) else {}
    progress = _read_object(folder / "intake-progress.json")
    stage = "starting" if progress is None else "intake"
    if (folder / "decision-items.json").exists():
        stage = "scoring"
    progress = progress or {}
    pool = _read_object(state / "pool.json") or {}
    known = pool.get("bodies") if isinstance(pool.get("bodies"), dict) else {}
    new = progress.get("new") if isinstance(progress.get("new"), dict) else {}
    admitted = {str(v.get("uid")) for v in new.values() if isinstance(v, dict)}
    refused = progress.get("refused") if isinstance(progress.get("refused"), dict) else {}
    rejected = progress.get("rejected") if isinstance(progress.get("rejected"), dict) else {}
    done = {str(u) for u in progress.get("done") or []}

    claims = []
    for uid, item in commits.items():
        if not (str(uid).isdigit() and isinstance(item, dict)):
            continue
        parts = str(item.get("data") or "").split(":")
        sha = next((p for p in parts if HEX64.fullmatch(p)), None)
        claims.append((int(item.get("block") or 0), str(members.get(uid) or ""), int(uid), sha))
    out, checking = [], stage != "scoring"
    for _, _, uid, sha in sorted(claims):  # the order intake works through them
        key = str(uid)
        if key in admitted:
            state_name = "admitted"
        elif key in refused:
            state_name = "deferred" if "retried next round" in str(refused[key]) else "refused"
        elif sha in rejected or key in done:
            state_name = "refused"
        elif sha in known:
            state_name = "in_pool"
        elif checking:
            state_name, checking = "checking", False
        else:
            state_name = "waiting"
        out.append({"uid": uid, "state": state_name})
    out.sort(key=lambda c: c["uid"])
    started = float(job["started"])
    return {
        "round_id": round_id,
        "attempt": job.get("attempts") if type(job.get("attempts")) is int else None,
        "running": soup_pool._alive(job.get("pid"), job.get("pid_started")),
        "stage": stage,
        "started_at": _iso(started),
        "deadline_at": _iso(started + soup_pool.ROUND_SECONDS),
        "claims": out,
    }


DEFERRED = "retried next round"  # the only refusal reasons published: the round's own limits


def intake(state, max_new):
    """The newest finished intake of the pool: the hashes this validator has graded or rejected and,
    by uid and committed hash, each claim that round passed over, so a page can rank waiting bodies
    against chain. A reason is published only for a deferral (``DEFERRED``); others stay private."""
    pool = _read_object(state / "pool.json") or {}
    known = pool.get("bodies") if isinstance(pool.get("bodies"), dict) else {}
    rejected = pool.get("rejected") if isinstance(pool.get("rejected"), dict) else {}
    try:
        folders = sorted(
            f.name for f in (state / "rounds").iterdir() if (f / "intake.json").exists()
        )
    except OSError:
        folders = []
    round_id, claims = (folders[-1] if folders else None), []
    if round_id is not None:
        record = _read_object(state / "rounds" / round_id / "intake.json") or {}
        refused = record.get("refused") if isinstance(record.get("refused"), dict) else {}
        request = _read_object(state / f"request-{round_id}.json") or {}
        snapshot = request.get("snapshot") if isinstance(request.get("snapshot"), dict) else {}
        commits = (
            snapshot.get("commitments") if isinstance(snapshot.get("commitments"), dict) else {}
        )
        for uid, reason in refused.items():
            item = commits.get(uid)
            parts = str(item.get("data") or "").split(":") if isinstance(item, dict) else []
            sha = next((p for p in parts if HEX64.fullmatch(p)), None)
            if not (str(uid).isdigit() and sha):
                continue
            deferred = DEFERRED in str(reason)
            claims.append({
                "uid": int(uid), "sha256": sha, "state": "deferred" if deferred else "refused",
                "reason": str(reason)[:200] if deferred else None,
            })  # fmt: skip
        claims.sort(key=lambda c: c["uid"])
    return {
        "round_id": round_id,
        "max_new_per_round": max_new,
        "graded": sorted(
            s
            for s, info in known.items()
            if HEX64.fullmatch(s) and isinstance(info, dict) and not info.get("house")
        ),  # fmt: skip
        "rejected": sorted(s for s in rejected if HEX64.fullmatch(s)),
        "claims": claims,
    }


def feed(config, directory, *, limit=FEED_ROUNDS):
    """The payload of the validator's ``GET /soup``: each enabled soup pool's newest public round
    records and, for every body those rounds seat or crown, its house name and repo when known.

    House names and repos come from the pool's house registry (by pinned hash, or by the hash this
    validator resolved an unpinned entry to) and repos otherwise from the intake record in
    ``pool.json``; anything else is null. A missing or malformed file is skipped, never raised.
    """
    from pathlib import Path

    from . import model_pools, soup_pool, soup_rates

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
            house = soup_rates.house(pool)
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
        pools[pool] = {
            "bodies": bodies,
            "rounds": rounds,
            "intake": intake(state, entry["max_new_per_round"]),
        }
        if (now := live(state)) is not None:  # only while a round is in progress
            pools[pool]["live"] = now
    return {"kind": "soup", "pools": pools}
