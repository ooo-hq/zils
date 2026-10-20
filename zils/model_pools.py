"""Independent base competitions composed under the existing base reward budget."""

import math

from . import base_round, core, models


def validate(config):
    if not isinstance(config, dict) or set(config) != {"share", "burn_uid", "pools"}:
        raise ValueError("base_pools requires share, burn_uid and pools")
    share, burn, pools = config["share"], config["burn_uid"], config["pools"]
    if type(share) not in (int, float) or not math.isfinite(share) or not 0 <= share <= 1:
        raise ValueError("base_pools share must be finite and in [0, 1]")
    if type(burn) is not int or not 0 <= burn <= 65535:
        raise ValueError("base_pools burn_uid must be a uid")
    if not isinstance(pools, dict) or not pools:
        raise ValueError("base_pools requires at least one allocated pool")
    result = {}
    for pool, entry in pools.items():
        if pool not in models.D3_POOL_MODELS or not isinstance(entry, dict):
            raise ValueError("unsupported model pool")
        if type(entry.get("enabled")) is not bool:
            raise ValueError("pool enabled must be a boolean")
        model = entry.get("model")
        if isinstance(model, dict):
            model = models.validate_spec(model)
        if not isinstance(model, str) or models.pool_id(model) != pool:
            raise ValueError("incumbent profile belongs to a different model pool")
        track = base_round.validate_track(
            {
                **{k: v for k, v in entry.items() if k not in ("model", "enabled")},
                "burn_uid": burn,
            }
        )
        if not math.isfinite(track["share"]):
            raise ValueError("pool share must be finite")
        track.pop("burn_uid")
        result[pool] = {"model": models.spec(model), "enabled": entry["enabled"], **track}
    if not math.isclose(sum(p["share"] for p in result.values()), 1.0, abs_tol=1e-9):
        raise ValueError("model pool allocations must sum to one")
    return {"share": share, "burn_uid": burn, "pools": result}


def weights(config, decisions, members):
    config = validate(config)
    if not isinstance(decisions, dict) or set(decisions) - set(config["pools"]):
        raise ValueError("decisions name an unconfigured model pool")
    vectors, shares, records = {}, {}, {}
    burned = 0.0
    for pool, entry in config["pools"].items():
        track = {k: v for k, v in entry.items() if k not in ("model", "enabled")}
        track["burn_uid"] = config["burn_uid"]
        decision = decisions.get(pool)
        if decision is not None:
            if (
                not isinstance(decision, dict)
                or decision.get("pool_id") != pool
                or decision.get("model") != entry["model"]
            ):
                raise ValueError("base decision differs from its pool or incumbent profile")
            earned = base_round.track_weights(track, decision, members)
        else:
            earned = base_round.king_weights(track, members)
        if not entry["enabled"]:
            earned = {}
        vectors[pool], shares[pool] = earned, entry["share"]
        if not earned:
            burned += entry["share"]
        records[pool] = {**entry, "weights": {str(uid): w for uid, w in earned.items()}}
    return core.blend_weights(vectors, shares, burn_uid=config["burn_uid"]), {
        "pools": records,
        "burned": burned,
        "burn_uid": config["burn_uid"],
    }
