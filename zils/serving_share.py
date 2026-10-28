"""Serving share: frozen serving scores become a uid weight vector."""

import json
import math
import os
from datetime import datetime, timedelta

from .queue_evidence import HEX64, SS58
from .runtime import verified

SCHEMA = "zils-serving-receipts/v1"
HOP = {"hotkey", "model", "weights_sha256", "answered", "input_tokens", "failed"}
CHECK = {"hotkey", "pass", "marginal", "fail", "mismatch"}
MAX_SCORES_BYTES = 8 << 20


def _number(x):
    try:
        return type(x) in (int, float) and math.isfinite(float(x))
    except (OverflowError, ValueError):
        return False


def validate(config):
    base = {"share", "scores", "switch_on_miners", "max_age_seconds"}
    if not isinstance(config, dict) or set(config) not in (
        base,
        base | {"receipts", "volume_share"},
    ):
        raise ValueError(
            "serving_share requires share, scores, switch_on_miners and max_age_seconds, "
            "and receipts only together with volume_share"
        )
    if "receipts" in config:
        receipts, volume = config["receipts"], config["volume_share"]
        if not _number(volume) or not 0 <= volume <= 1:
            raise ValueError("serving_share volume_share must be finite and in [0, 1]")
        if not isinstance(receipts, dict) or set(receipts) != {"url", "signer", "window_hours"}:
            raise ValueError("serving_share receipts needs exactly url, signer and window_hours")
        if not isinstance(receipts["url"], str) or not (
            receipts["url"].startswith("https://")
            or receipts["url"].startswith("http://127.0.0.1:")
        ):
            raise ValueError("serving_share receipts url must be https")
        if not isinstance(receipts["signer"], str) or not SS58.fullmatch(receipts["signer"]):
            raise ValueError("serving_share receipts signer must be an ss58 hotkey")
        hours = receipts["window_hours"]
        if type(hours) is not int or not 1 <= hours <= 720:
            raise ValueError("serving_share receipts window_hours must be an integer in 1..720")
    share, path = config["share"], config["scores"]
    if not _number(share) or not 0 <= share <= 1:
        raise ValueError("serving_share share must be finite and in [0, 1]")
    if not isinstance(path, str) or not path:
        raise ValueError("serving_share scores must be a nonempty path")
    if type(config["switch_on_miners"]) is not int or config["switch_on_miners"] < 1:
        raise ValueError("serving_share switch_on_miners must be an integer of at least 1")
    if type(config["max_age_seconds"]) is not int or not 60 <= config["max_age_seconds"] <= 86400:
        raise ValueError("serving_share max_age_seconds must be an integer in [60, 86400]")
    return dict(config)


def validate_budget(config):
    if "serving_share" not in config:
        return
    from . import feature_tracks, model_pools

    if "base_pools" not in config or "base_tier" in config:
        raise ValueError("serving_share requires base_pools, not legacy base_tier")
    total = validate(config["serving_share"])["share"]
    total += model_pools.validate(config["base_pools"])["share"]
    if "feature_tracks" in config:
        total += feature_tracks.validate(config["feature_tracks"])["share"]
    if total > 1 + 1e-9:
        raise ValueError("base, feature and serving shares exceed the reward budget")


def freeze(config, now):
    config = validate(config)
    written, scores = None, {}
    try:
        stat = os.stat(config["scores"])
        written = stat.st_mtime
        if stat.st_size > MAX_SCORES_BYTES:
            raise ValueError("serving scores file too large")
        with open(config["scores"]) as f:
            data = json.load(f)
    except (OSError, ValueError, RecursionError, MemoryError):
        data = {}
    if (
        isinstance(data, dict)
        and written is not None
        and -60 <= now - written <= config["max_age_seconds"]  # a future mtime is a skewed clock
    ):
        for hotkey, models in data.items():
            if not isinstance(models, dict):
                continue
            kept = {  # only what pays: other fields could carry NaN the frozen JSON refuses
                m: {"score": float(s["score"])}
                for m, s in models.items()
                if isinstance(s, dict) and _number(s.get("score")) and s["score"] >= 0
            }
            if kept:
                scores[hotkey] = kept
    return {"written_at": written, "frozen_at": now, "scores": scores}


def _count(x):
    return type(x) is int and x >= 0


def verify_receipts(message, receipts, until=None):
    """The pinned signer's receipts for exactly this window, or ValueError."""
    payload = verified(message, receipts["signer"])
    if payload.get("schema") != SCHEMA or payload.get("signer") != receipts["signer"]:
        raise ValueError("serving receipts schema or signer differs")
    try:
        end = datetime.fromisoformat(payload["until"].replace("Z", "+00:00"))
        start = datetime.fromisoformat(payload["since"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError):
        raise ValueError("serving receipts window is invalid") from None
    if (until is not None and end != until) or end - start != timedelta(
        hours=receipts["window_hours"]
    ):
        raise ValueError("serving receipts are for another window")
    hops, checks = payload.get("hops"), payload.get("checks")
    if not isinstance(hops, list) or not isinstance(checks, list):
        raise ValueError("serving receipts need hops and checks")
    for row in hops:
        if (
            not isinstance(row, dict)
            or set(row) != HOP
            or not isinstance(row["hotkey"], str)
            or not SS58.fullmatch(row["hotkey"])
            or not isinstance(row["model"], str)
            or not isinstance(row["weights_sha256"], str)
            or not HEX64.fullmatch(row["weights_sha256"])
            or not _count(row["answered"])
            or not _count(row["input_tokens"])
            or not isinstance(row["failed"], dict)
            or not all(isinstance(k, str) and _count(v) for k, v in row["failed"].items())
        ):
            raise ValueError("serving receipts hop is invalid")
    for row in checks:
        if (
            not isinstance(row, dict)
            or set(row) != CHECK
            or not isinstance(row["hotkey"], str)
            or not SS58.fullmatch(row["hotkey"])
            or not all(_count(row[k]) for k in CHECK - {"hotkey"})
        ):
            raise ValueError("serving receipts check is invalid")
    return payload


def weights(config, frozen, members, receipts=None):
    config = validate(config)
    uids = {m["hotkey"]: int(uid) for uid, m in members.items()}
    struck = sorted(
        {r["hotkey"] for r in receipts["checks"] if r["fail"] + r["mismatch"] > 0}
        if receipts
        else ()
    )
    totals = {}
    for hotkey, models in frozen["scores"].items():
        total = sum(float(s["score"]) for s in models.values())
        if hotkey in uids and hotkey not in struck and math.isfinite(total) and total > 0:
            totals[uids[hotkey]] = total
    on = len(totals) >= config["switch_on_miners"]
    vector = totals if on else {}
    volume = {}
    if on and receipts:
        for hop in receipts["hops"]:
            uid = uids.get(hop["hotkey"])
            if uid in vector and hop["input_tokens"]:
                volume[uid] = volume.get(uid, 0) + hop["input_tokens"]
    record = {
        "share": config["share"],
        "switch_on_miners": config["switch_on_miners"],
        "earners": len(totals),
        "switched_on": on,
        "written_at": frozen["written_at"],
        "frozen_at": frozen["frozen_at"],
        "weights": {str(uid): w for uid, w in vector.items()},
    }
    if receipts:
        record |= {
            "volume_share": config["volume_share"],
            "struck": struck,
            "volume": {str(uid): t for uid, t in sorted(volume.items())},
        }
    return vector, volume, record
