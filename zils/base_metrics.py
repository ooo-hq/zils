"""Fixed decision-quality scores for base elections; customer scoring is unchanged.

Two family metrics share one weighting (cells, blocks, categories, clipping):

- ``argmax`` (the default): yes/no and Choice score the top answer; Score questions score the
  distance of the expected value from the label.
- ``argmax_rps``: as ``argmax``, but Score questions score the ranked probability score (RPS)
  against the best constant distribution. The expected-value distance is not a proper scoring
  rule: sharpening probabilities moves it without changing any answer.
- ``brier``: yes/no and Choice score the multi-class Brier loss, Score questions the RPS; each
  against the best constant distribution for the family. Both are proper scoring rules, so
  confidence counts and no answer-free change of confidence can help in expectation.

``fit_temperature``, ``temper`` and ``calibration_split`` let a validator fit one temperature per
model on a held-apart part of an exam and score the rest (``soup_grade.evaluate``).
"""

import hashlib
import hmac
import math
import random
import statistics
from collections import defaultdict

import zils

VERSION = "zils-base-decision/v1"
METRICS = ("argmax", "argmax_rps", "brier")
# 121 log-spaced temperatures from 0.1 to 10 (1.0 is on the grid): wide enough for bodies far
# from calibrated, which the older 0.25-4 grid clipped.
TEMPERATURES = tuple(math.exp(math.log(0.1) + i * math.log(100) / 120) for i in range(121))
TEMPERATURES = tuple(1.0 if i == 60 else t for i, t in enumerate(TEMPERATURES))
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


def _metric(metric):
    if metric not in METRICS:
        raise ValueError("base metric must be one of " + ", ".join(METRICS))
    return metric


def _onehot(keys, label):
    return [float(k == label) for k in keys]


def _brier(p, label):
    return sum((q - y) ** 2 for q, y in zip(p, _onehot(range(len(p)), label), strict=True))


def _rps(p, label):
    total, cumulative = 0.0, 0.0
    for k in range(len(p) - 1):
        cumulative += p[k]
        total += (cumulative - float(k >= label)) ** 2
    return total / (len(p) - 1)


def _constant_losses(options, labels, kind):
    """Mean loss of the best constant distribution: uniform, by key or by position frequency."""
    loss = _rps if kind == "score" else _brier
    positions = [keys.index(label) for keys, label in zip(options, labels, strict=True)]
    by_key = defaultdict(float)
    by_position = defaultdict(float)
    for keys, label, pos in zip(options, labels, positions, strict=True):
        by_key[label] += 1
        by_position[pos] += 1

    def constant(weights):
        values = []
        for keys, pos in zip(options, positions, strict=True):
            q = [weights(keys, j) for j in range(len(keys))]
            total = sum(q)
            q = [1 / len(keys)] * len(keys) if total <= 0 else [v / total for v in q]
            values.append(loss(q, pos))
        return statistics.mean(values)

    return min(
        constant(lambda keys, j: 1.0),
        constant(lambda keys, j: by_key[keys[j]]),
        constant(lambda keys, j: by_position[j]),
    )


def baselines(cases, metric="argmax"):
    """Each family's reference: chance-level accuracy or expected-value error for ``argmax``,
    the best constant distribution's Brier (yes/no: 0.5, per-label) or RPS for ``brier``."""
    metric = _metric(metric)
    result = {}
    for family, rows in _groups(cases).items():
        kind = rows[0]["question"]["type"]
        options = [zils.options(c["question"]) for c in rows]
        labels = [c["label"] for c in rows]
        if metric == "brier" or metric == "argmax_rps" and kind == "score":
            if kind == "noul" and set(labels) != {"false", "true"}:
                raise ValueError("yes/no family requires both classes")
            baseline = 0.5 if kind == "noul" else _constant_losses(options, labels, kind)
            if baseline <= 1e-12:
                raise ValueError("degenerate Brier baseline")
            result[family] = baseline
            continue
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


def _values(groups, predictions, metric="argmax", only=None):
    by_id = {r["id"]: r["probabilities"] for r in predictions}
    values = {}
    for family, cases in groups.items():
        rows = []
        for case in cases:
            if only is not None and case["id"] not in only:
                continue
            p = by_id[case["id"]]
            if metric == "brier" or metric == "argmax_rps" and case["question"]["type"] == "score":
                keys = zils.options(case["question"])
                vector = [p[k] for k in keys]
                label = keys.index(case["label"])
                loss = _rps if case["question"]["type"] == "score" else _brier
                rows.append((case["label"], loss(vector, label)))
                continue
            if case["question"]["type"] == "score":
                expectation = sum(int(k) * value for k, value in p.items()) / (len(p) - 1)
                value = abs(expectation - int(case["label"]) / (len(p) - 1))
            else:
                value = float(max(sorted(p), key=p.get) == case["label"])
            rows.append((case["label"], value))
        values[family] = rows
    return values


def _score(groups, values, fixed, sample=None, metric="argmax"):
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
        loss = metric == "brier" or kind == "score"
        skill = 1 - raw / baseline if loss else (raw - baseline) / (1 - baseline)
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


def score(cases, predictions, *, metric="argmax", only=None):
    """The exam score (0-100). ``only`` (a set of case ids) scores just those cases, against the
    baselines of the whole of ``cases``; every family must keep at least one of each label."""
    metric = _metric(metric)
    fixed = baselines(cases, metric)
    zils.score(cases, predictions)
    groups = _groups(cases)
    values = _values(groups, predictions, metric, None if only is None else set(only))
    for family, rows in values.items():
        kinds = {label for label, _ in rows}
        if (
            not rows
            or groups[family][0]["question"]["type"] == "noul"
            and kinds != {"false", "true"}
        ):
            raise ValueError("a scored part must keep every family and both yes/no labels")
    return _score(groups, values, fixed, metric=metric)


def paired_improvement(cases, incumbent, candidate, *, resamples, seed, metric="argmax"):
    if type(resamples) is not int or resamples < 1 or type(seed) is not int:
        raise ValueError("resamples must be a positive integer and seed an integer")
    metric = _metric(metric)
    fixed = baselines(cases, metric)
    for predictions in (incumbent, candidate):
        zils.score(cases, predictions)
    groups = _groups(cases)
    inc, cand = (_values(groups, p, metric) for p in (incumbent, candidate))

    def difference(sample=None):
        return (
            _score(groups, cand, fixed, sample, metric)["score"]
            - _score(groups, inc, fixed, sample, metric)["score"]
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


def temper(cases, predictions, temperature):
    """Predictions with every case's probabilities raised to 1/temperature and renormalised. A
    positive temperature never changes the top answer (ties stay ties); zeros stay zero."""
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    by_id = {p["id"]: p for p in predictions}
    result = []
    for case in cases:
        row = by_id[case["id"]]
        p = row["probabilities"]
        keys = zils.options(case["question"])
        logs = {k: math.log(p[k]) / temperature for k in keys if p[k] > 0}
        top = max(logs.values())
        weights = {k: math.exp(logs[k] - top) if k in logs else 0.0 for k in keys}
        total = sum(weights.values())
        result.append({**row, "probabilities": {k: weights[k] / total for k in keys}})
    return result


def fit_temperature(cases, predictions):
    """The temperature on ``TEMPERATURES`` that minimises the equal-family mean Brier loss of
    ``cases`` (yes/no families: the mean of each label's mean). Ties go to the temperature
    closest to 1, so a temperature that changes nothing is 1."""
    groups = defaultdict(list)
    for case in cases:
        groups[case["family"]].append(case)
    by_id = {p["id"]: p["probabilities"] for p in predictions}
    rows = []  # (family index, label stratum, keys' probabilities, label index)
    for f, family in enumerate(sorted(groups)):
        for case in groups[family]:
            keys = zils.options(case["question"])
            stratum = case["label"] if case["question"]["type"] == "noul" else None
            rows.append(
                (f, stratum, [by_id[case["id"]][k] for k in keys], keys.index(case["label"]))
            )
    if not rows:
        return 1.0
    logs = [[math.log(v) if v > 0 else None for v in p] for _, _, p, _ in rows]
    best = None
    for temperature in TEMPERATURES:
        cells = defaultdict(list)
        for (f, stratum, _, label), log in zip(rows, logs, strict=True):
            top = max(v for v in log if v is not None)
            w = [0.0 if v is None else math.exp((v - top) / temperature) for v in log]
            total = sum(w)
            cells[f, stratum].append(_brier([v / total for v in w], label))
        per_family = defaultdict(list)
        for (f, _), values in cells.items():
            per_family[f].append(statistics.mean(values))
        loss = statistics.mean(statistics.mean(v) for v in per_family.values())
        key = (round(loss, 12), abs(math.log(temperature)))
        if best is None or key < best[0]:
            best = (key, temperature)
    return best[1]


def calibration_split(cases, secret, share):
    """(calibration ids, scored ids): in each family and yes/no label, the ``floor(share * n)``
    cases first in HMAC-SHA256(secret, id) order are held apart for fitting, never all of them.
    ``secret`` is private to the validator and changes every round, so miners cannot know which
    cases calibrate."""
    if not isinstance(secret, bytes) or not secret:
        raise ValueError("calibration split needs a non-empty secret")
    if type(share) is not float or not 0 < share <= 0.5:
        raise ValueError("calibration share must be a float in (0, 0.5]")
    strata = defaultdict(list)
    for case in cases:
        label = case["label"] if case["question"]["type"] == "noul" else None
        strata[case["family"], label].append(case["id"])
    held = set()
    for ids in strata.values():
        ordered = sorted(ids, key=lambda i: hmac.new(secret, i.encode(), hashlib.sha256).digest())
        held.update(ordered[: min(math.floor(share * len(ordered)), len(ordered) - 1)])
    return held, {c["id"] for c in cases} - held
