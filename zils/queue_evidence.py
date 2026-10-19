"""The adapter track from hosted-queue customer work (signed platform evidence), not frozen-job rounds.

The platform exports facts per finished attempt (zils_platform.adapter_evidence); this module applies
the payout rule. Per hotkey over the window:

    score       = sum(total_tokens * skill) over delivered customer attempts (not operator-owned)
    reliability = (valid + 1) / (valid + failures + 2) over all its attempts, as graded routing uses
    weight      = score * reliability, paid at the uid the hotkey holds in the round's chain snapshot

Qualification runs and undelivered work earn nothing. Unregistered hotkeys earn nothing.
"""

import json
import math
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

from .miner_grading import FAILURES
from .runtime import verified

SCHEMA = "zils-adapter-evidence/v1"
MAX_BYTES = 8 * 1024**2
SS58 = re.compile("[1-9A-HJ-NP-Za-km-z]{40,64}")
HEX64 = re.compile("[a-f0-9]{64}")
ROW = {
    "hotkey",
    "job_sha256",
    "purpose",
    "internal",
    "owner_sha256",
    "outcome",
    "total_tokens",
    "finished_at",
    "sha256",
    "accepted",
    "skill",
}


def validate_track(track):
    if not isinstance(track, dict) or set(track) != {"url", "signer", "window_hours"}:
        raise ValueError("adapter_evidence needs exactly url, signer and window_hours")
    url, signer, hours = track["url"], track["signer"], track["window_hours"]
    if not isinstance(url, str) or not (
        url.startswith("https://") or url.startswith("http://127.0.0.1:")
    ):
        raise ValueError("adapter_evidence.url must be https")
    if not isinstance(signer, str) or not SS58.fullmatch(signer):
        raise ValueError("adapter_evidence.signer must be the platform evidence ss58 hotkey")
    if type(hours) is not int or not 1 <= hours <= 720:
        raise ValueError("adapter_evidence.window_hours must be an integer in 1..720")
    return track


def window_end(now=None):
    """The current whole UTC hour: every validator asking in that hour gets the same document."""
    return (now or datetime.now(UTC)).astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


def fetch(track, until):
    from .protocol import opener

    query = urlencode({"until": iso(until), "hours": track["window_hours"]})
    with opener().open(f"{track['url']}?{query}", timeout=30) as response:
        body = response.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError("adapter evidence is too large")
    message = json.loads(body)
    verify(message, track, until)
    return message


def verify(message, track, until=None):
    """The pinned signer's payload for exactly this track's window, or ValueError."""
    payload = verified(message, track["signer"])
    if payload.get("schema") != SCHEMA or payload.get("signer") != track["signer"]:
        raise ValueError("adapter evidence schema or signer differs")
    try:
        end = datetime.fromisoformat(payload["until"].replace("Z", "+00:00"))
        start = datetime.fromisoformat(payload["since"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError):
        raise ValueError("adapter evidence window is invalid") from None
    if (until is not None and end != until) or end - start != timedelta(
        hours=track["window_hours"]
    ):
        raise ValueError("adapter evidence is for another window")
    rows = payload.get("attempts")
    if not isinstance(rows, list):
        raise ValueError("adapter evidence has no attempts")
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != ROW
            or not isinstance(row["hotkey"], str)
            or not SS58.fullmatch(row["hotkey"])
            or row["purpose"] not in ("customer", "qualification")
            or type(row["internal"]) is not bool
            or row["accepted"] not in (True, False, None)
            or not isinstance(row["outcome"], str)
            or type(row["total_tokens"]) is not int
            or row["total_tokens"] <= 0
            or not (row["sha256"] is None or HEX64.fullmatch(str(row["sha256"])))
            or not (
                row["skill"] is None
                or (
                    type(row["skill"]) in (int, float)
                    and math.isfinite(row["skill"])
                    and 0 <= row["skill"] <= 1
                )
            )
        ):
            raise ValueError("adapter evidence row is invalid")
    return payload


def earns(row):
    return (
        row["purpose"] == "customer"
        and not row["internal"]
        and row["outcome"] == "valid"
        and row["accepted"] is True
        and row["skill"] is not None
        and row["sha256"] is not None
    )


def hotkey_scores(payload):
    """Per-hotkey breakdown before uid mapping; refuses an artifact that earns twice."""
    seen, stats = set(), {}
    for row in payload["attempts"]:
        item = stats.setdefault(
            row["hotkey"], {"valid": 0, "failures": 0, "earning": 0, "score": 0.0}
        )
        item["valid"] += row["outcome"] == "valid"
        item["failures"] += row["outcome"] in FAILURES
        if earns(row):
            if row["sha256"] in seen:
                raise ValueError("one adapter artifact earns twice in the evidence")
            seen.add(row["sha256"])
            item["earning"] += 1
            item["score"] += row["total_tokens"] * row["skill"]
    for item in stats.values():
        item["reliability"] = (item["valid"] + 1) / (item["valid"] + item["failures"] + 2)
        item["weight"] = item["score"] * item["reliability"]
    return stats


def track_weights(payload, members):
    """{uid: weight} for registered members; the hotkey at the snapshot block decides the uid."""
    uid_of = {item["hotkey"]: int(uid) for uid, item in members.items()}
    weights = {
        uid_of[hotkey]: item["weight"]
        for hotkey, item in hotkey_scores(payload).items()
        if hotkey in uid_of and item["weight"] > 0
    }
    total = sum(weights.values())
    return {uid: w / total for uid, w in sorted(weights.items())} if total else {}
