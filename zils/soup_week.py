"""The weekly soup judge: which soup becomes the week's soup (pure functions, ZIL-204).

The nightly rounds keep building a soup (``soup_round``). Once a week the judge takes the last
nightly soup, tries to add the week's best left-out bodies, drops a member the soup is better
without, and compares the result with the current weekly soup. Every score here comes from
``evaluate(members)`` on a sealed set no nightly round draws from, so a pick that was lucky on one
night's items does not carry over.
"""

from zils.soup_round import _EPS, MARGIN, MAX_MEMBERS

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
