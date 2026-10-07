"""Paired, conversation-grouped evidence; no labels or examples leave this module."""

import random
from collections import defaultdict

import zils


def chosen(row):
    return row.get("choice") or max(sorted(row["probabilities"]), key=row["probabilities"].get)


def summarize(cases, trained, comparator):
    # Also checks complete case coverage, option names, probability sums and finite values.
    if any(not isinstance(c.get("group_id"), str) or not c["group_id"] for c in cases):
        raise ValueError("Every example needs a source group")
    zils.score(cases, trained)
    zils.score(cases, comparator)
    trained = {r["id"]: r for r in trained}
    comparator = {r["id"]: r for r in comparator}
    rows, groups, losses = [], defaultdict(list), [[], []]
    correct = [0, 0]
    for case in cases:
        pair = [trained[case["id"]], comparator[case["id"]]]
        answers = [chosen(p) for p in pair]
        if any(answer not in zils.options(case["question"]) for answer in answers):
            raise ValueError("Returned choice is outside the allowed outcomes")
        wins = [answer == case["label"] for answer in answers]
        for i, prediction in enumerate(pair):
            correct[i] += wins[i]
            losses[i].append(
                sum(
                    (p - (key == case["label"])) ** 2
                    for key, p in prediction["probabilities"].items()
                )
            )
        groups[case["group_id"]].append(int(wins[0]) - int(wins[1]))
        rows.append(
            {
                "id": case["id"],
                "expected": case["label"],
                "trained": answers[0],
                "jev": answers[1],
                "trained_correct": wins[0],
                "jev_correct": wins[1],
            }
        )
    units = list(groups.values())
    rng = random.Random(20261008)
    differences = []
    for _ in range(2000):
        sample = [units[rng.randrange(len(units))] for _ in units]
        differences.append(sum(sum(unit) for unit in sample) / sum(map(len, sample)))
    differences.sort()
    interval = [differences[49], differences[1949]]
    supported = len(units) >= 30
    verdict = (
        "more_accurate"
        if supported and interval[0] > 0
        else "less_accurate"
        if supported and interval[1] < 0
        else "no_clear_difference"
    )
    n = len(cases)
    return {
        "cases": n,
        "groups": len(units),
        "verdict": verdict,
        "trained": {"correct": correct[0], "accuracy": correct[0] / n, "brier": sum(losses[0]) / n},
        "jev": {"correct": correct[1], "accuracy": correct[1] / n, "brier": sum(losses[1]) / n},
        "accuracy_difference": (correct[0] - correct[1]) / n,
        "accuracy_difference_95_ci": interval,
        "wins": sum(r["trained_correct"] and not r["jev_correct"] for r in rows),
        "losses": sum(r["jev_correct"] and not r["trained_correct"] for r in rows),
        "rows": rows,
        "method": "Same new inputs and questions; paired bootstrap of source groups, 2,000 resamples. At least 30 independent groups required for a directional verdict. Brier is mean per-example squared probability error. No tuning on these examples.",
    }
