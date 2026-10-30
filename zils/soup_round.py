"""Soup rounds: greedy soup selection, leave-one-out contributions, seat and crown pay.

Each round, validators build a fresh uniform-weight average ("soup") of up to five model bodies from the
round's pool, chosen greedily (Wortsman et al. 2022, with a minimum gain per added member). The pool's
emission share splits 80% among soup seats and 20% to the round's best single body (the crown); any share
nobody earns burns. These functions are pure: grading is passed in as ``evaluate`` and never imported here.

A body is any hashable id (a body sha256 in production). ``evaluate(members)`` grades the uniform soup of a
sorted tuple of bodies and returns its exam score in points.
"""

import math

MARGIN = 0.3  # minimum soup score gain (points) for greedy to add a body; above GPU-to-GPU drift
MAX_MEMBERS = 5
SEAT_SHARE = 0.8
CROWN_SHARE = 0.2
FLOOR = 0.3  # minimum leave-one-out contribution (points) for a paid seat
HOUSE_RATE = 0.0  # open models known at launch: seated but unpaid
SCOUT_RATE = 0.25  # open models released after launch, submitted by a scout
TRAINER_RATE = 1.0
RATES = (HOUSE_RATE, SCOUT_RATE, TRAINER_RATE)
_EPS = 1e-9  # score differences are floats; 3.3 - 3.0 must clear a 0.3 margin


def greedy(solo, evaluate, *, margin=MARGIN, max_members=MAX_MEMBERS, order_key):
    """Greedy soup over ``solo`` ({body: solo score}). Returns (members, trace).

    Bodies are tried in solo-score order, ties broken by ``order_key(body)`` (earlier commitment first).
    The soup starts as the best body alone, which is not re-evaluated; each next body joins only if the
    soup's score rises by at least ``margin``. Selection stops once the soup has ``max_members``.
    ``members`` is a sorted tuple; each trace row is {"try", "score", "best_before", "accepted"}.
    """
    if not solo:
        raise ValueError("greedy soup needs at least one body")
    if type(max_members) is not int or max_members < 1:
        raise ValueError("max_members must be a positive integer")
    order = sorted(solo, key=lambda body: (-solo[body], order_key(body)))
    soup, best, trace = (order[0],), solo[order[0]], []
    for body in order[1:]:
        if len(soup) >= max_members:
            break
        trial = tuple(sorted((*soup, body)))
        score = evaluate(trial)
        accepted = score - best >= margin - _EPS
        trace.append(
            {"try": list(trial), "score": score, "best_before": best, "accepted": accepted}
        )
        if accepted:
            soup, best = trial, score
    return soup, trace


def leave_one_out(members, evaluate, *, baseline=None):
    """Each member's contribution: the soup's score minus the score of the soup without that member.

    A one-member soup has no smaller soup, so its contribution is its score minus ``baseline``, the
    chassis (stock) score, which the caller must pass in that case.
    """
    members = tuple(sorted(members))
    if not members:
        raise ValueError("leave-one-out needs at least one member")
    full = evaluate(members)
    if len(members) == 1:
        if baseline is None:
            raise ValueError("a one-member soup needs the chassis baseline score")
        return {members[0]: full - baseline}
    return {m: full - evaluate(tuple(b for b in members if b != m)) for m in members}


def _check_rate(rate):
    if rate not in RATES:
        raise ValueError("rate must be one of " + ", ".join(map(str, RATES)))
    return rate


def seat_weights(members, contributions, rates, *, share=SEAT_SHARE, floor=FLOOR):
    """Seat pay as fractions of the pool's share.

    Members whose contribution is at least ``floor`` split ``share`` equally; each slice is multiplied by
    the member's rate, so a house ingredient keeps its seat at 0.0 and a scout gets a quarter. Members
    below the floor are omitted. Whatever is not paid burns; if nobody clears the floor the result is {}.
    """
    paid = [m for m in members if contributions[m] >= floor - _EPS]
    return {m: share / len(paid) * _check_rate(rates[m]) for m in paid}


def crown_weight(solo, rates, house, *, share=CROWN_SHARE, order_key):
    """The crown: the best solo body that is not a house ingredient gets ``share`` times its rate.

    Ties go to the earlier commitment (``order_key``). With no eligible body the result is {} and the
    crown share burns.
    """
    eligible = [body for body in solo if body not in house]
    if not eligible:
        return {}
    best = min(eligible, key=lambda body: (-solo[body], order_key(body)))
    return {best: share * _check_rate(rates[best])}


def _weight(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("soup weights must be finite and non-negative")
    return value


def soup_weights(decision, members):
    """{uid: weight} for one soup pool, from a round record and the metagraph roster.

    ``decision`` is {"pool_id", "members": [{"hotkey", "body_sha256", "rate", "contribution",
    "seat_weight"}], "crown": {"hotkey", "body_sha256", "rate", "weight"} | None}; ``members`` is
    {uid: {"hotkey": ...}}. Seat and crown weights are summed per hotkey; hotkeys no longer registered
    are dropped, so their share burns. Weights are fractions of the pool's share and total at most one;
    the caller burns the remainder.
    """
    if not isinstance(decision, dict) or not isinstance(decision.get("members"), list):
        raise ValueError("soup decision needs a members list")
    rows = [(row["hotkey"], _weight(row["seat_weight"])) for row in decision["members"]]
    crown = decision.get("crown")
    if crown is not None:
        rows.append((crown["hotkey"], _weight(crown["weight"])))
    if sum(w for _, w in rows) > 1 + _EPS:
        raise ValueError("soup decision pays more than the pool's share")
    uid_by_hotkey = {item["hotkey"]: int(uid) for uid, item in members.items()}
    result = {}
    for hotkey, weight in rows:
        if hotkey in uid_by_hotkey and weight > 0:
            uid = uid_by_hotkey[hotkey]
            result[uid] = result.get(uid, 0.0) + weight
    return result
