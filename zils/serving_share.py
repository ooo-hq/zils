"""Serving share: frozen serving scores become a uid weight vector."""

import json
import math
import os


def _number(x):
    return type(x) in (int, float) and math.isfinite(x)


def validate(config):
    if not isinstance(config, dict) or set(config) != {
        "share",
        "scores",
        "switch_on_miners",
        "max_age_seconds",
    }:
        raise ValueError("serving_share requires share, scores, switch_on_miners and max_age_seconds")
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
        written = os.stat(config["scores"]).st_mtime
        with open(config["scores"]) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    if isinstance(data, dict) and written is not None and now - written <= config["max_age_seconds"]:
        for hotkey, models in data.items():
            if not isinstance(models, dict):
                continue
            kept = {
                m: s
                for m, s in models.items()
                if isinstance(s, dict) and _number(s.get("score")) and s["score"] >= 0
            }
            if kept:
                scores[hotkey] = kept
    return {"written_at": written, "frozen_at": now, "scores": scores}


def weights(config, frozen, members):
    config = validate(config)
    uids = {m["hotkey"]: int(uid) for uid, m in members.items()}
    totals = {}
    for hotkey, models in frozen["scores"].items():
        total = sum(s["score"] for s in models.values())
        if hotkey in uids and total > 0:
            totals[uids[hotkey]] = total
    on = len(totals) >= config["switch_on_miners"]
    vector = totals if on else {}
    return vector, {
        "share": config["share"],
        "switch_on_miners": config["switch_on_miners"],
        "earners": len(totals),
        "switched_on": on,
        "written_at": frozen["written_at"],
        "frozen_at": frozen["frozen_at"],
        "weights": {str(uid): w for uid, w in vector.items()},
    }
