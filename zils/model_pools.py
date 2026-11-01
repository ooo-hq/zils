"""Independent base competitions composed under the existing base reward budget.

A soup pool may carry an emission schedule (``opens_block`` and ``boost``): it pays nothing and
burns its share until its opening block, then pays each boost step's share for that step's blocks,
then its own ``share``. Shares are a function of the chain block alone, so every validator computes
the same vector for a round frozen at the same block. Pool shares may total less than one; the
unassigned rest burns. A config with no schedule pays exactly as before.
"""

import math
import re

from . import base_round, core, models

# Pool-entry keys that are not part of the base-tier payout track.
NOT_TRACK = ("model", "enabled", "cooldown_seconds")
SOUP_KEYS = {"type", "model", "enabled", "share", "house", "schedule"}
SOUP_KNOWN = "known_public"  # optional: public models whose re-uploads earn the scout rate
SOUP_KNOWN_BYTES = "max_known_public_bytes_per_round"  # optional; soup_round default otherwise
SOUP_KNOWN_SECONDS = "max_known_public_seconds_per_round"  # optional, at most one 6 h round
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
    "decision_rounds": 3,
}


SCHEDULE_KEYS = ("opens_block", "boost")
MAX_BOOST_STEPS = 8
UNASSIGNED = "unassigned"  # the blend track for shares no pool is assigned


def _share(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def validate_opening(entry):
    """The pool's validated schedule keys ({} without one). ``opens_block`` is a block number, or
    null for a pool that stays closed until a block is set; ``boost`` is a list of
    {"share", "blocks"} steps paid in order from the opening block."""
    out = {}
    if "opens_block" in entry:
        block = entry["opens_block"]
        if block is not None and (type(block) is not int or block < 0):
            raise ValueError("pool opens_block must be a block number or null")
        out["opens_block"] = block
    if "boost" in entry:
        if "opens_block" not in entry:
            raise ValueError("a pool boost needs an opens_block")
        steps = entry["boost"]
        if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_BOOST_STEPS:
            raise ValueError(f"pool boost must be a list of 1 to {MAX_BOOST_STEPS} steps")
        for step in steps:
            if not isinstance(step, dict) or set(step) != {"share", "blocks"}:
                raise ValueError("each boost step is exactly {share, blocks}")
            _share(step["share"], "boost share")
            if type(step["blocks"]) is not int or step["blocks"] < 1:
                raise ValueError("boost step blocks must be a positive integer")
        out["boost"] = [{"share": s["share"], "blocks": s["blocks"]} for s in steps]
    return out


def is_open(entry, block):
    """True once the pool's opening block is reached (always, for a pool with no schedule)."""
    if "opens_block" not in entry:
        return True
    return entry["opens_block"] is not None and block is not None and block >= entry["opens_block"]


def _scheduled_share(entry, block):
    """The share an open pool is paid at `block`: its boost step's while one runs, else its own."""
    if "boost" not in entry:
        return entry["share"]
    elapsed = block - entry["opens_block"]
    for step in entry["boost"]:
        if elapsed < step["blocks"]:
            return step["share"]
        elapsed -= step["blocks"]
    return entry["share"]


def scheduled(config):
    """True when any pool of a validated config has a schedule."""
    return any(k in e for e in config["pools"].values() for k in SCHEDULE_KEYS)


def shares_at(config, block):
    """{pool: share paid at `block`} for a validated config; a closed pool pays 0 (its share burns)."""
    return {
        p: _scheduled_share(e, block) if is_open(e, block) else 0.0
        for p, e in config["pools"].items()
    }


def _check_budget(pools):
    """The pools' own shares, and the shares paid at every block (a boost borrows from the budget
    closed pools leave unpaid), must each total at most one. Paid shares change only at opening
    blocks and step ends, so those blocks are checked."""
    total = sum(e["share"] for e in pools.values())
    if total > 1 + 1e-9:
        raise ValueError(f"model pool shares exceed one ({total:.4f})")
    points = {0}
    for entry in pools.values():
        start = entry.get("opens_block")
        if start is None:
            continue
        points.add(start)
        for step in entry.get("boost", ()):
            start += step["blocks"]
            points.add(start)
    for block in sorted(points):
        total = sum(shares_at({"pools": pools}, block).values())
        if total > 1 + 1e-9:
            raise ValueError(f"model pool shares exceed one at block {block} ({total:.4f})")


def track(entry):
    """The base-tier track inside a pool entry, as base_round.validate_track accepts it."""
    return {k: v for k, v in entry.items() if k not in NOT_TRACK}


def is_soup(entry):
    return entry.get("type", "king") == "soup"


def validate_soup(entry):
    """A soup pool: daily soup rounds (``zils.soup_round.run``) pay seats and a crown, not kings.

    ``house`` is the house registry inline (a list, as ``soup_rates.load_house`` reads it) or a path
    to it relative to the validator directory; the optional ``known_public`` list (same format and
    forms, empty by default) names public models that are not house ingredients, whose unchanged
    or near re-uploads earn the scout rate (``soup_round._known_public``), resolved within
    ``max_known_public_bytes_per_round``; ``schedule.deadline_utc`` is the daily commit deadline
    ("HH:MM", UTC). An optional ``sketch_seed`` (64 hex characters) fixes this validator's private
    copy-check sketch seed; it is checked here but left out of the result, which reports publish.
    Without it each validator generates and keeps its own (``soup_round.sketch_seed``).
    An optional ``exam`` picks where round items come from (``soup_current.validate``): the fixed
    round bank (the default) or Zils Current dated exports. An optional ``scoring`` (one of
    ``soup_grade.SCORINGS``, default ``soup_grade.DEFAULT_SCORING``) picks how graded models are
    scored.
    """
    from . import soup_current, soup_grade, soup_rates

    known_keys = {SOUP_KNOWN, SOUP_KNOWN_BYTES, SOUP_KNOWN_SECONDS}
    allowed = SOUP_KEYS | set(SOUP_OPTIONAL) | known_keys | {SOUP_SECRET, "exam", "scoring"}
    if set(entry) - allowed or SOUP_KEYS - set(entry):
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
    known = entry.get(SOUP_KNOWN, [])
    if isinstance(known, str):
        if not known:
            raise ValueError("soup known_public path must not be empty")
    else:
        soup_rates.validate_house(known)
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
    known_bytes = entry.get(SOUP_KNOWN_BYTES, 1)
    if type(known_bytes) is not int or known_bytes < 1:
        raise ValueError("soup max_known_public_bytes_per_round must be a positive integer")
    known_seconds = entry.get(SOUP_KNOWN_SECONDS, 1)
    if type(known_seconds) is not int or not 1 <= known_seconds <= 6 * 3600:
        raise ValueError("soup max_known_public_seconds_per_round must be an integer in [1, 21600]")
    if not 0 <= sizes["failed_claim_backoff_rounds"] <= 30:
        raise ValueError("soup failed_claim_backoff_rounds must be in [0, 30]")
    if not 1 <= sizes["decision_rounds"] <= 7:
        raise ValueError("soup decision_rounds must be in [1, 7]")
    if not 1 <= sizes["max_decision_age_days"] <= 30:
        raise ValueError("soup max_decision_age_days must be in [1, 30]")
    exam = soup_current.validate(entry["exam"]) if "exam" in entry else None
    scoring = entry.get("scoring", soup_grade.DEFAULT_SCORING)
    if not isinstance(scoring, str) or scoring not in soup_grade.SCORINGS:
        raise ValueError("soup scoring must be one of " + ", ".join(soup_grade.SCORINGS))
    return {
        **({"exam": exam} if exam and exam["source"] != "bank" else {}),
        "type": "soup",
        "enabled": entry["enabled"],
        "share": share,
        "house": house,
        # Echoed only when set, so a pool without a list validates (and reports) as before.
        **({SOUP_KNOWN: known} if SOUP_KNOWN in entry else {}),
        **({SOUP_KNOWN_BYTES: known_bytes} if SOUP_KNOWN_BYTES in entry else {}),
        **({SOUP_KNOWN_SECONDS: known_seconds} if SOUP_KNOWN_SECONDS in entry else {}),
        "schedule": {"deadline_utc": deadline},
        **sizes,
        "others_max": others,
        "house_mix_max": house_mix,
        # Echoed only when set, so a pool without it validates (and hashes) as before; rounds then
        # use soup_grade.DEFAULT_SCORING.
        **({"scoring": scoring} if "scoring" in entry else {}),
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
        opening = validate_opening(entry)
        if opening and not is_soup(entry):
            raise ValueError("only soup pools take an opens_block or boost")
        if is_soup(entry):
            entry = {k: v for k, v in entry.items() if k not in SCHEDULE_KEYS}
            result[pool] = {"model": models.spec(model), **validate_soup(entry), **opening}
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
    # Pools may leave part of the budget unassigned (it burns); they may never exceed it.
    _check_budget(result)
    return {"share": share, "burn_uid": burn, "pools": result}


def weights(config, decisions, members, block=None):
    """(chain vector, record) for the pools. A config with a schedule needs `block`, the chain block
    the round froze at; without one, `block` is ignored and the result is as before schedules."""
    config = validate(config)
    if not isinstance(decisions, dict) or set(decisions) - set(config["pools"]):
        raise ValueError("decisions name an unconfigured model pool")
    timed = scheduled(config)
    if timed and (type(block) is not int or block < 0):
        raise ValueError("a pool schedule needs the round's chain block")
    paid_shares = shares_at(config, block) if timed else {}
    vectors, shares, records = {}, {}, {}
    burned = 0.0
    for pool, entry in config["pools"].items():
        decision = decisions.get(pool)
        paid = 1.0
        share = paid_shares.get(pool, entry["share"])
        if timed and not is_open(entry, block):
            if decision is not None:
                raise ValueError(f"model pool {pool} is not open at block {block}")
            earned = {}  # its share is 0 until it opens; the budget it would take burns
        elif not entry["enabled"]:
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
        vectors[pool], shares[pool] = earned, share
        if not earned:
            burned += share
        elif paid < 1:
            burned += share * (1 - paid)
        records[pool] = {**entry, "weights": {str(uid): w for uid, w in earned.items()}}
        if timed:
            records[pool].update(open=is_open(entry, block), paid_share=paid_shares[pool])
    extra = {}
    unassigned = 1 - sum(shares.values())
    if unassigned > 1e-12:  # shares under one: the rest burns, never flows to the pools
        vectors[UNASSIGNED], shares[UNASSIGNED] = {}, unassigned
        burned += unassigned
        extra["unassigned"] = unassigned
    if timed:
        extra["block"] = block
    return core.blend_weights(vectors, shares, burn_uid=config["burn_uid"]), {
        "pools": records,
        "burned": burned,
        "burn_uid": config["burn_uid"],
        **extra,
    }
