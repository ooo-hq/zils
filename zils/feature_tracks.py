"""Pinned feature competitions, independent king histories and conserved rewards."""

import math

from . import base_round, box, core, feature_commitments, models

SCOPE = "synthetic-only"


def validate(config):
    if not isinstance(config, dict) or set(config) != {"share", "burn_uid", "tracks"}:
        raise ValueError("feature_tracks requires share, burn_uid and tracks")
    share, burn = config["share"], config["burn_uid"]
    if type(share) not in (int, float) or not math.isfinite(share) or not 0 <= share <= 1:
        raise ValueError("feature share must be finite and in [0, 1]")
    if type(burn) is not int or not 0 <= burn <= 65535:
        raise ValueError("feature burn_uid must be a UID")
    entries = config["tracks"]
    if not isinstance(entries, dict) or not entries:
        raise ValueError("feature tracks must be a nonempty mapping")
    tracks = {}
    for name, entry in entries.items():
        if not isinstance(entry, dict) or set(entry) - {
            "model",
            "enabled",
            "share",
            "incumbent_sha256",
            "kings",
            "draws",
            "policy",
        }:
            raise ValueError("unsupported feature track fields")
        model = entry.get("model")
        model = models.validate_spec(model) if isinstance(model, dict) else model
        if (
            name not in feature_commitments.TRACKS
            or model not in models.d3_family()
            or name != f"box/{models.pool_id(model)}"
        ):
            raise ValueError("feature track differs from its model pool")
        if type(entry.get("enabled")) is not bool:
            raise ValueError("feature enabled must be a boolean")
        common = base_round.validate_track(
            {k: v for k, v in entry.items() if k not in ("model", "enabled", "policy")}
            | {"burn_uid": burn},
            allow_empty_draws=not entry["enabled"],
        )
        if not math.isfinite(common["share"]):
            raise ValueError("feature track share must be finite")
        draws = common["draws"]
        if entry["enabled"] and len(draws) != 2:
            raise ValueError("enabled box tracks require two disjoint draws")
        for field in ("manifest_sha256", "dataset_sha256"):
            if len({d[field] for d in draws}) != len(draws):
                raise ValueError("feature draws must be distinct")
        common.pop("burn_uid")
        tracks[name] = {
            **common,
            "model": models.spec(model),
            "enabled": entry["enabled"],
            "policy": box.validate_policy(entry.get("policy", {})),
        }
    if not math.isclose(sum(t["share"] for t in tracks.values()), 1.0, abs_tol=1e-9):
        raise ValueError("feature track shares must sum to one")
    return {"share": share, "burn_uid": burn, "tracks": tracks}


def validate_budget(config):
    if "feature_tracks" not in config:
        return
    from . import model_pools

    if "base_pools" not in config or "base_tier" in config:
        raise ValueError("feature_tracks requires base_pools, not legacy base_tier")
    features, base = validate(config["feature_tracks"]), model_pools.validate(config["base_pools"])
    if features["burn_uid"] != base["burn_uid"]:
        raise ValueError("base and feature burn UIDs must match")
    if features["share"] + base["share"] > 1:
        raise ValueError("base and feature shares exceed the reward budget")


def validate_decision(name, entry, decision):
    expected = {
        "track_id": name,
        "model": entry["model"],
        "incumbent_sha256": entry["incumbent_sha256"],
        "draws": entry["draws"],
        "policy": entry["policy"],
        "decoder": box.DECODER,
        "scope": SCOPE,
    }
    if not isinstance(decision, dict) or any(decision.get(k) != v for k, v in expected.items()):
        raise ValueError("feature decision differs from its configured track, parent or draws")
    timestamp = decision.get("evaluated_at")
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("feature evaluation time must be finite and nonnegative")
    return decision


def weights(config, decisions, members):
    config = validate(config)
    enabled = {name for name, entry in config["tracks"].items() if entry["enabled"]}
    if not isinstance(decisions, dict) or set(decisions) != enabled:
        raise ValueError("feature decisions must cover each enabled track exactly")
    vectors, shares, records, burned = {}, {}, {}, 0.0
    for name, entry in config["tracks"].items():
        earned = {}
        if entry["enabled"]:
            validate_decision(name, entry, decisions[name])
            common = {k: v for k, v in entry.items() if k not in ("model", "enabled", "policy")}
            earned = base_round.king_weights({**common, "burn_uid": config["burn_uid"]}, members)
        vectors[name], shares[name] = earned, entry["share"]
        if not earned:
            burned += entry["share"]
        records[name] = {**entry, "weights": {str(uid): weight for uid, weight in earned.items()}}
    return core.blend_weights(vectors, shares, burn_uid=config["burn_uid"]), {
        "tracks": records,
        "burned": burned,
        "scope": SCOPE,
        "decoder": box.DECODER,
    }
