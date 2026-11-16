"""The weekly soup judge: which soup becomes the week's soup (pure functions, ZIL-204).

The nightly rounds keep building a soup (``soup_round``). Once a week the judge takes the last
nightly soup, tries to add the week's best left-out bodies, drops a member the soup is better
without, and compares the result with the current weekly soup. Every score here comes from
``evaluate(members)`` on a sealed set no nightly round draws from, so a pick that was lucky on one
night's items does not carry over.
"""

import hashlib
from pathlib import Path

from zils.soup_round import _EPS, MARGIN, MAX_MEMBERS, leave_one_out, seat_weights

JOIN = MARGIN  # a left-out body joins if it lifts the soup by this much, as at night
DROP = MARGIN  # a member goes if the soup is this much better without it
TRIES = 2  # left-out bodies tried each week
SEAT_BAR = 0.5  # minimum contribution (points) on the sealed set for a weekly seat


def add_left_out(members, left_out, evaluate, *, margin=JOIN, tries=TRIES, max_members=MAX_MEMBERS):
    """Try to add up to ``tries`` of ``left_out`` (best first) to the soup. Returns (members, trace).

    Each body is tried on the soup as it stands after the tries before it and joins only if the
    soup's score rises by at least ``margin``. A full soup takes no more. Trace rows are
    ``soup_round.greedy``'s: {"try", "score", "best_before", "accepted"}.
    """
    soup, best, trace = tuple(sorted(members)), None, []
    for body in [b for b in left_out if b not in soup][:tries]:
        if len(soup) >= max_members:
            break
        best = evaluate(soup) if best is None else best
        trial = tuple(sorted((*soup, body)))
        score = evaluate(trial)
        accepted = score - best >= margin - _EPS
        trace.append(
            {"try": list(trial), "score": score, "best_before": best, "accepted": accepted}
        )
        if accepted:
            soup, best = trial, score
    return soup, trace


def drop_members(members, evaluate, *, margin=DROP):
    """Drop members the soup is better without. Returns (members, trace).

    Each pass scores the soup without each member and removes the one whose removal helps most,
    if the soup rises by at least ``margin``; it stops at the first pass that removes nothing.
    The last member is never dropped. Trace rows are {"drop", "score", "best_before", "accepted"}.
    """
    soup, trace = tuple(sorted(members)), []
    while len(soup) > 1:
        best = evaluate(soup)
        score, body = max((evaluate(tuple(m for m in soup if m != b)), b) for b in soup)
        accepted = score - best >= margin - _EPS
        trace.append({"drop": body, "score": score, "best_before": best, "accepted": accepted})
        if not accepted:
            break
        soup = tuple(m for m in soup if m != body)
    return soup, trace


def judge(candidate, left_out, evaluate, *, current, bar, stock=None, **settings):
    """The week's soup. ``candidate`` is the last nightly soup, ``left_out`` the week's bodies outside
    it (best first), ``current`` the weekly soup now serving (None in the first week, when ``stock``,
    the release's own score, stands in for it).

    The candidate, after ``add_left_out`` and ``drop_members``, replaces the current soup only if it
    scores at least ``bar`` above it. Returns {"members", "candidate", "score", "current_score",
    "passed", "trace"}; ``members`` is the soup for the coming week (None: the release stays).
    """
    soup, trace = add_left_out(candidate, left_out, evaluate, **settings)
    soup, dropped = drop_members(soup, evaluate)
    score = evaluate(soup)
    current = tuple(sorted(current)) if current else None
    if current is None and stock is None:
        raise ValueError("the first weekly soup needs the stock score")
    current_score = stock if current is None else evaluate(current)
    passed = soup != current and score - current_score >= bar - _EPS
    return {
        "members": soup if passed else current,
        "candidate": soup,
        "score": score,
        "current_score": current_score,
        "passed": passed,
        "trace": trace + dropped,
    }


def left_out(private):
    """The bodies the judge may try: in the last nightly round's pool, outside its soup, and paid
    (a house body is never added by the judge). Best solo score first, ties by hash."""
    members = {m["body_sha256"] for m in private["members"]}
    solo, rates = private["solo"], private["rates"]
    bodies = [b for b in private["pool"] if b not in members and b in solo and rates.get(b, 0) > 0]
    return sorted(bodies, key=lambda b: (-solo[b], b))


class Missing(Exception):
    """A soup the judge needs has sealed items with no cached answer yet."""

    def __init__(self, key, members):
        super().__init__(key)
        self.key, self.members = key, members


def soup_key(members, scope):
    """The prediction-cache key of a soup of ``members``, as ``soup_round.run`` names it."""
    members = tuple(sorted(members))
    if len(members) == 1:
        return f"{members[0][:32]}-{scope}"
    return f"soup-{hashlib.sha256('+'.join(members).encode()).hexdigest()[:32]}-{scope}"


def cached_scores(cases, cache, *, scope, score):
    """``evaluate(members)`` for the judge, from cached answers only: ``score(cases, rows)`` of the
    soup's rows on the sealed ``cases``, or ``Missing`` when any case is not graded yet. The caller
    grades what is missing and asks again, so the GPU is held one model at a time."""
    ids, scores = [c["id"] for c in cases], {}

    def evaluate(members):
        members = tuple(sorted(members))
        if members not in scores:
            key = soup_key(members, scope)
            rows = cache.get(key, ids)
            if len(rows) < len(ids):
                raise Missing(key, members)
            scores[members] = score(cases, [rows[i] for i in ids])
        return scores[members]

    return evaluate


def decide(
    week_id, private, known, weekly, evaluate, *, bar, stock=None, floor=SEAT_BAR, **settings
):
    """The week's record. ``private`` is the last nightly round's private record, ``known`` the
    pool's {body: {"hotkey"}}, ``weekly`` last week's record (None before the first weekly soup).

    A candidate that passes (``judge``) is the new weekly soup: each member's contribution is the
    soup's sealed score minus the score without it (``soup_round.leave_one_out``), and a
    contribution of at least ``floor`` holds a seat. Otherwise last week's soup and seats carry
    over unchanged ("kept_from"); with no weekly soup yet, nobody is seated.
    """
    candidate = tuple(sorted(m["body_sha256"] for m in private["members"]))
    current = tuple(m["body_sha256"] for m in weekly["members"]) if weekly else None
    result = judge(
        candidate, left_out(private), evaluate, current=current or None, bar=bar, stock=stock,
        **settings,
    )  # fmt: skip
    record = {
        "week_id": week_id,
        "round_id": private["round_id"],
        "passed": result["passed"],
        "candidate": list(result["candidate"]),
        "score": result["score"],
        "current_score": result["current_score"],
        "trace": result["trace"],
    }
    if not result["passed"]:
        kept = {"kept_from": weekly.get("kept_from", weekly["week_id"])} if weekly else {}
        return {**record, "members": weekly["members"] if weekly else [], **kept}
    members, rates = result["members"], private["rates"]
    contributions = leave_one_out(members, evaluate, baseline=stock)
    seats = seat_weights(members, contributions, {m: rates[m] for m in members}, floor=floor)
    record["members"] = [
        {
            "hotkey": known[m]["hotkey"],
            "body_sha256": m,
            "rate": rates[m],
            "contribution": contributions[m],
            "seat_weight": seats.get(m, 0.0),
        }
        for m in members
    ]
    return record


RELEASE = "release"  # the pool's served model as a judge "member": its score is the stock score


def run_week(week_id, private, known, weekly, cases, cache, *, scope, score, bar, grade, stop, **settings):  # fmt: skip
    """Decide the week, grading what the judge still needs one model at a time. Returns the record
    (``decide``), or None when ``stop()`` says to give way (a nightly round, the end of the window);
    a later call resumes from the cache.

    ``grade(key, members)`` grades a chunk of the sealed items ``members``' soup has no answer for
    and stores them in ``cache`` under ``key``. It is called until the soup is complete, with
    ``stop`` checked before each call, so the GPU is never held for more than a chunk."""
    ids = [c["id"] for c in cases]
    evaluate = cached_scores(cases, cache, scope=scope, score=score)
    while not stop():
        try:
            stock = evaluate((RELEASE,))
            return decide(week_id, private, known, weekly, evaluate, bar=bar, stock=stock, **settings)  # fmt: skip
        except Missing as missing:
            before = len(cache.get(missing.key, ids))
            grade(missing.key, missing.members)
            if len(cache.get(missing.key, ids)) <= before:
                raise RuntimeError(f"grading {missing.key} made no progress") from None
    return None


def sources(members, state_dir, chassis, *, rename):
    """``soup.build`` sources for a judge soup: each body's folder under the pool's state with its
    tensor rename, and the chassis itself for the release."""
    out = []
    for member in members:
        if member == RELEASE:
            out.append((Path(chassis), None))
        else:
            folder = Path(state_dir) / "bodies" / member
            out.append((folder, rename(folder)))
    return out
