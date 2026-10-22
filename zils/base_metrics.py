"""Fixed decision-quality scores for base elections; customer scoring is unchanged."""

import random
import statistics
from collections import defaultdict

import zils

VERSION = "zils-base-decision/v1"
CATEGORY_WEIGHTS = {
    "reasoning": 0.2585,
    "language": 0.2585,
    "retrieval": 0.2002,
    "tools": 0.1828,
    "arts": 0.10,
}
BLOCK_WEIGHTS = {"known": 0.6, "new": 0.4}


def _groups(cases):
    zils.validate_cases(cases)
    groups = defaultdict(list)
    for case in sorted(cases, key=lambda c: c["id"]):
        if "target_probabilities" in case:
            raise ValueError("base election requires hard labels")
        if (
            not isinstance(case.get("category"), str)
            or case["category"] not in CATEGORY_WEIGHTS
            or not isinstance(case.get("block"), str)
            or case["block"] not in BLOCK_WEIGHTS
        ):
            raise ValueError("base case requires a supported category and block")
        groups[case["family"]].append(case)
    cells = set()
    for rows in groups.values():
        tags = {(c["category"], c["block"], c["question"]["type"]) for c in rows}
        if len(tags) != 1:
            raise ValueError("family must have one category, block and question type")
        category, block, _ = tags.pop()
        cells.add((category, block))
    if cells != {(c, b) for c in CATEGORY_WEIGHTS for b in BLOCK_WEIGHTS}:
        raise ValueError("base exam is missing a category/block")
    return dict(sorted(groups.items()))


def baselines(cases):
    result = {}
    for family, rows in _groups(cases).items():
        kind = rows[0]["question"]["type"]
        options = [zils.options(c["question"]) for c in rows]
        labels = [c["label"] for c in rows]
        if kind == "noul":
            if set(labels) != {"false", "true"}:
                raise ValueError("yes/no family requires both classes")
            baseline = 0.5
        elif kind == "choice":
            uniform = statistics.mean(1 / len(keys) for keys in options)
            fixed_ids = [
                statistics.mean(
                    float(label == key) if key in keys else 1 / len(keys)
                    for label, keys in zip(labels, options, strict=True)
                )
                for key in sorted({k for keys in options for k in keys})
            ]
            positions = [
                statistics.mean(
                    float(label == keys[pos]) if pos < len(keys) else 1 / len(keys)
                    for label, keys in zip(labels, options, strict=True)
                )
                for pos in range(max(map(len, options)))
            ]
            baseline = max(uniform, *fixed_ids, *positions)
            if baseline >= 1 - 1e-12:
                raise ValueError("degenerate Choice baseline")
        else:
            targets = [
                int(label) / (len(keys) - 1) for label, keys in zip(labels, options, strict=True)
            ]
            median = statistics.median(targets)
            baseline = statistics.mean(abs(v - median) for v in targets)
            if baseline <= 1e-12:
                raise ValueError("degenerate Score baseline")
        result[family] = baseline
    return result


def _values(groups, predictions):
    by_id = {r["id"]: r["probabilities"] for r in predictions}
    values = {}
    for family, cases in groups.items():
        rows = []
        for case in cases:
            p = by_id[case["id"]]
            if case["question"]["type"] == "score":
                expectation = sum(int(k) * value for k, value in p.items()) / (len(p) - 1)
                value = abs(expectation - int(case["label"]) / (len(p) - 1))
            else:
                value = float(max(sorted(p), key=p.get) == case["label"])
            rows.append((case["label"], value))
        values[family] = rows
    return values


def _score(groups, values, fixed, sample=None):
    families, cells = {}, defaultdict(list)
    for family, cases in groups.items():
        category, block = cases[0]["category"], cases[0]["block"]
        kind, baseline = cases[0]["question"]["type"], fixed[family]
        rows = values[family]
        if sample is not None:
            rows = [rows[i] for i in sample[family]]
        if kind == "noul":
            raw = statistics.mean(
                statistics.mean(value for label, value in rows if label == target)
                for target in ("false", "true")
            )
        else:
            raw = statistics.mean(value for _, value in rows)
        skill = 1 - raw / baseline if kind == "score" else (raw - baseline) / (1 - baseline)
        clipped = max(0.0, min(1.0, skill))
        families[family] = {
            "raw_metric": raw,
            "baseline": baseline,
            "unclipped_skill": skill,
            "skill": clipped,
            "category": category,
            "block": block,
            "type": kind,
            "count": len(rows),
        }
        cells[category, block].append(clipped)
    categories = {
        category: sum(
            weight * statistics.mean(cells[category, block])
            for block, weight in BLOCK_WEIGHTS.items()
        )
        for category in CATEGORY_WEIGHTS
    }
    return {
        "score": min(
            100.0, max(0.0, 100 * sum(CATEGORY_WEIGHTS[c] * v for c, v in categories.items()))
        ),
        "families": families,
        "categories": categories,
    }


def score(cases, predictions):
    fixed = baselines(cases)
    zils.score(cases, predictions)
    groups = _groups(cases)
    return _score(groups, _values(groups, predictions), fixed)


def paired_improvement(cases, incumbent, candidate, *, resamples, seed):
    if type(resamples) is not int or resamples < 1 or type(seed) is not int:
        raise ValueError("resamples must be a positive integer and seed an integer")
    fixed = baselines(cases)
    for predictions in (incumbent, candidate):
        zils.score(cases, predictions)
    groups = _groups(cases)
    inc, cand = (_values(groups, p) for p in (incumbent, candidate))

    def difference(sample=None):
        return (
            _score(groups, cand, fixed, sample)["score"]
            - _score(groups, inc, fixed, sample)["score"]
        )

    strata = {
        family: [
            [i for i, c in enumerate(rows) if label is None or c["label"] == label]
            for label in (("false", "true") if rows[0]["question"]["type"] == "noul" else (None,))
        ]
        for family, rows in groups.items()
    }
    rng = random.Random(seed)
    draws = sorted(
        difference(
            {
                f: [i for group in parts for i in rng.choices(group, k=len(group))]
                for f, parts in strata.items()
            }
        )
        for _ in range(resamples)
    )
    return {
        "improvement": difference(),
        "lower_95": draws[int(0.025 * resamples)],
        "upper_95": draws[min(resamples - 1, int(0.975 * resamples))],
        "bootstrap_variance": statistics.pvariance(draws),
        "resamples": resamples,
        "seed": seed,
    }
