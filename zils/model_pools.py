"""Independent base competitions composed under the existing base reward budget."""

import math
import re

from . import base_round, core, models

# Pool-entry keys that are not part of the base-tier payout track.
NOT_TRACK = ("model", "enabled", "cooldown_seconds")
SOUP_KEYS = {"type", "model", "enabled", "share", "house", "schedule"}
SOUP_SECRET = "sketch_seed"  # optional, local to each validator, never echoed into a report
SOUP_OPTIONAL = {
    "subset_size": 3000,
    "screen_size": 300,
    "max_pool": 20,
    "max_new_per_round": 6,
    "max_claims_per_round": 12,
    "max_download_bytes_per_round": 60 * 1024**3,
    "max_decision_age_days": 3,
    "others_max": 0.3,
    "house_mix_max": 0.05,
    "failed_claim_backoff_rounds": 3,
}


def track(entry):
    """The base-tier track inside a pool entry, as base_round.validate_track accepts it."""
    return {k: v for k, v in entry.items() if k not in NOT_TRACK}


def is_soup(entry):
    return entry.get("type", "king") == "soup"


def validate_soup(entry):
    """A soup pool: daily soup rounds (``zils.soup_round.run``) pay seats and a crown, not kings.

    ``house`` is the house registry inline (a list, as ``soup_rates.load_house`` reads it) or a path
    to it relative to the validator directory; ``schedule.deadline_utc`` is the daily commit deadline
    ("HH:MM", UTC). An optional ``sketch_seed`` (64 hex characters) fixes this validator's private
    copy-check sketch seed; it is checked here but left out of the result, which reports publish.
    Without it each validator generates and keeps its own (``soup_round.sketch_seed``).
    """
    from . import soup_rates

    if set(entry) - SOUP_KEYS - set(SOUP_OPTIONAL) - {SOUP_SECRET} or SOUP_KEYS - set(entry):
        raise ValueError("soup pool requires " + ", ".join(sorted(SOUP_KEYS)))
    share = entry["share"]
    if type(share) not in (int, float) or not math.isfinite(share) or not 0 <= share <= 1:
        raise ValueError("pool share must be finite and in [0, 1]")
    house = entry["house"]
    if isinstance(house, str):
        if not house:
            raise ValueError("soup house path must not be empty")
    else:
        soup_rates.validate_house(house)
    schedule = entry["schedule"]
    deadline = schedule.get("deadline_utc") if isinstance(schedule, dict) else None
    if (
        set(schedule if isinstance(schedule, dict) else ()) != {"deadline_utc"}
        or not isinstance(deadline, str)
        or not re.fullmatch(r"([01][0-9]|2[0-3]):[0-5][0-9]", deadline)
    ):
        raise ValueError('soup schedule must be {"deadline_utc": "HH:MM"}')
    if SOUP_SECRET in entry and (
        not isinstance(entry[SOUP_SECRET], str)
        or not re.fullmatch("[a-f0-9]{64}", entry[SOUP_SECRET])
    ):
        raise ValueError("soup sketch_seed must be 64 lowercase hex characters")
    sizes = {k: entry.get(k, v) for k, v in SOUP_OPTIONAL.items()}
    others = sizes.pop("others_max")
    if type(others) not in (int, float) or not 0.1 <= others <= 0.9:
        raise ValueError("soup others_max must be a number in [0.1, 0.9]")
    house_mix = sizes.pop("house_mix_max")
    if type(house_mix) not in (int, float) or not 0 <= house_mix <= 0.5:
        raise ValueError("soup house_mix_max must be a number in [0, 0.5]")
    if any(type(v) is not int for v in sizes.values()):
        raise ValueError(
            "soup subset_size, screen_size, max_pool and the per-round caps must be integers"
        )
    if not 1 <= sizes["screen_size"] <= sizes["subset_size"]:
        raise ValueError("soup screen_size must be in [1, subset_size]")
    if not 5 <= sizes["max_pool"] <= 20:
        raise ValueError("soup max_pool must be in [5, 20]")
    if not 1 <= sizes["max_new_per_round"] <= 20:
        raise ValueError("soup max_new_per_round must be in [1, 20]")
    if not 1 <= sizes["max_claims_per_round"] <= 100:
        raise ValueError("soup max_claims_per_round must be in [1, 100]")
    if sizes["max_download_bytes_per_round"] < 1:
        raise ValueError("soup max_download_bytes_per_round must be positive")
    if not 0 <= sizes["failed_claim_backoff_rounds"] <= 30:
        raise ValueError("soup failed_claim_backoff_rounds must be in [0, 30]")
    if not 1 <= sizes["max_decision_age_days"] <= 30:
        raise ValueError("soup max_decision_age_days must be in [1, 30]")
    return {
        "type": "soup",
        "enabled": entry["enabled"],
        "share": share,
        "house": house,
        "schedule": {"deadline_utc": deadline},
        **sizes,
        "others_max": others,
        "house_mix_max": house_mix,
    }


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
        if "type" in entry and entry["type"] not in ("king", "soup"):
            raise ValueError("pool type must be king or soup")
        if is_soup(entry):
            result[pool] = {"model": models.spec(model), **validate_soup(entry)}
            continue
        if "type" in entry:
            raise ValueError("a king pool omits type")
        checked = base_round.validate_track(
            {
                **track(entry),
                "burn_uid": burn,
            },
            allow_empty_draws=not entry["enabled"],
        )
        if not math.isfinite(checked["share"]):
            raise ValueError("pool share must be finite")
        if any(not math.isfinite(v) for v in checked["policy"].values()):
            raise ValueError("pool policy values must be finite")
        checked.pop("burn_uid")
        result[pool] = {"model": models.spec(model), "enabled": entry["enabled"], **checked}
        if "cooldown_seconds" in entry:
            # Minimum spacing between one hotkey's distinct submissions to this pool.
            cooldown = entry["cooldown_seconds"]
            if type(cooldown) is not int or not 0 <= cooldown <= 30 * 86400:
                raise ValueError("pool cooldown_seconds must be an integer in [0, 30 days]")
            result[pool]["cooldown_seconds"] = cooldown
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
        decision = decisions.get(pool)
        paid = 1.0
        if not entry["enabled"]:
            if decision is not None:
                raise ValueError("disabled model pool cannot have a decision")
            earned = {}
        elif is_soup(entry):
            earned = {}
            if decision is not None:
                if not isinstance(decision, dict) or decision.get("pool_id") != pool:
                    raise ValueError("soup decision differs from its pool")
                from . import soup_round

                earned = soup_round.soup_weights(decision, members)
            # blend_weights normalizes each pool over its own earners, so a soup paying less than
            # its whole share must name the burn uid for the rest or the earners would be scaled up.
            paid = sum(earned.values())
            if earned and paid < 1:
                earned[config["burn_uid"]] = earned.get(config["burn_uid"], 0.0) + 1 - paid
        elif decision is not None:
            pool_track = {**track(entry), "burn_uid": config["burn_uid"]}
            if (
                not isinstance(decision, dict)
                or decision.get("pool_id") != pool
                or decision.get("model") != entry["model"]
            ):
                raise ValueError("base decision differs from its pool or incumbent profile")
            earned = base_round.track_weights(pool_track, decision, members)
        else:
            pool_track = {**track(entry), "burn_uid": config["burn_uid"]}
            earned = base_round.king_weights(pool_track, members)
        vectors[pool], shares[pool] = earned, entry["share"]
        if not earned:
            burned += entry["share"]
        elif paid < 1:
            burned += entry["share"] * (1 - paid)
        records[pool] = {**entry, "weights": {str(uid): w for uid, w in earned.items()}}
    return core.blend_weights(vectors, shares, burn_uid=config["burn_uid"]), {
        "pools": records,
        "burned": burned,
        "burn_uid": config["burn_uid"],
    }
