"""Private full-native image scoring; unknown remains an incorrect answer."""

import math
import statistics
from collections import defaultdict

from .image_contract import UNKNOWN, native_probabilities


def full_brier(gold, probabilities):
    if gold == UNKNOWN:
        raise ValueError("This profile evaluates answerable labelled cases")
    return math.fsum((p - float(key == gold)) ** 2 for key, p in probabilities.items())


def score_rows(labels, distributions, families, outcome_order):
    if (
        not labels
        or len(labels) != len(distributions)
        or len(labels) != len(families)
        or not 2 <= len(outcome_order) <= 16
        or len(set(outcome_order)) != len(outcome_order)
        or UNKNOWN in outcome_order
        or any(not isinstance(k, str) or not k.strip() for k in outcome_order)
        or any(label not in outcome_order for label in labels)
        or any(not isinstance(f, str) or not f for f in families)
    ):
        raise ValueError("Invalid image score population")
    keys = [*outcome_order, UNKNOWN]
    confusion = {label: {key: 0 for key in keys} for label in outcome_order}
    losses = defaultdict(list)
    correct = unknown = confident_errors = 0
    for label, distribution, family in zip(labels, distributions, families, strict=True):
        probabilities = native_probabilities(distribution, outcome_order)
        choice = max(keys, key=probabilities.get)
        confusion[label][choice] += 1
        correct += choice == label
        unknown += choice == UNKNOWN
        confident_errors += choice != label and probabilities[choice] >= 0.9
        losses[family].append(
            (full_brier(label, probabilities), -math.log(max(1e-9, probabilities[label])))
        )
    brier = statistics.mean(statistics.mean(row[0] for row in group) for group in losses.values())
    nll = statistics.mean(statistics.mean(row[1] for row in group) for group in losses.values())
    per_class = {}
    for label in outcome_order:
        support = sum(confusion[label].values())
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in outcome_order if other != label)
        negatives = len(labels) - support
        per_class[label] = {
            "support": support,
            "true_positives": tp,
            "false_negatives": support - tp,
            "false_positives": fp,
            "negatives": negatives,
            "recall": tp / support if support else None,
            "false_positive_rate": fp / negatives if negatives else None,
        }
    uniform = 1 - 1 / len(outcome_order)
    return {
        "brier": brier,
        "nll": nll,
        "uniform_brier": uniform,
        "skill": max(0.0, 1 - brier / uniform),
        "accuracy": correct / len(labels),
        "confident_errors": confident_errors,
        "cases": len(labels),
        "unknown_rate": unknown / len(labels),
        "unknown_count": unknown,
        "confusion": confusion,
        "per_class": per_class,
        "outcome_order": list(outcome_order),
    }


def probabilities_at_temperature(logits, temperature):
    if (
        not logits
        or type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits)
    ):
        raise ValueError("Invalid image logits or temperature")
    maximum = max(logits)
    weights = [math.exp((x - maximum) / temperature) for x in logits]
    total = math.fsum(weights)
    return [x / total for x in weights]


def fit_image_temperature(logits, target_indices, families):
    if (
        not logits
        or len(logits) != len(target_indices)
        or len(logits) != len(families)
        or any(not isinstance(f, str) or not f for f in families)
    ):
        raise ValueError("Invalid image calibration population")
    for row, index in zip(logits, target_indices, strict=True):
        if not 3 <= len(row) <= 17 or type(index) is not int or not 0 <= index < len(row) - 1:
            raise ValueError("Calibration target must be a known outcome")
        probabilities_at_temperature(row, 1.0)

    def nll(temperature):
        groups = defaultdict(list)
        for row, index, family in zip(logits, target_indices, families, strict=True):
            p = probabilities_at_temperature(row, temperature)[index]
            groups[family].append(-math.log(max(1e-9, p)))
        return statistics.mean(statistics.mean(group) for group in groups.values())

    grid = [0.25 * 16 ** (i / 80) for i in range(81)]
    return min(grid, key=nll)


def image_policy_passes(metrics, policy):
    from .image_jobs import validate_policy

    order = metrics["outcome_order"]
    validate_policy(policy, order)
    if metrics["skill"] <= 0 or metrics["accuracy"] < policy["min_accuracy"]:
        return False
    if len(order) == 2:
        positive = metrics["per_class"][policy["positive_class"]]
        if (
            not positive["support"]
            or not positive["negatives"]
            or positive["recall"] < policy["min_positive_recall"]
            or positive["false_positive_rate"] > policy["max_false_positive_rate"]
        ):
            return False
    for label, target in policy.get("min_class_recall", {}).items():
        row = metrics["per_class"][label]
        if not row["support"] or row["recall"] < target:
            return False
    return True


def score(cases, predictions):
    if (
        not cases
        or not isinstance(predictions, list)
        or [r.get("id") for r in predictions] != [c["id"] for c in cases]
    ):
        raise ValueError("Image predictions must use the identical frozen case IDs and order")
    order = list(cases[0]["question"]["criteria"])
    if any(list(c["question"]["criteria"]) != order for c in cases):
        raise ValueError("Image outcome order changed")
    result = score_rows(
        [c["label"] for c in cases],
        [r["probabilities"] for r in predictions],
        [c["family"] for c in cases],
        order,
    )
    latencies = [r.get("elapsed_ms") for r in predictions]
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in latencies):
        raise ValueError("Invalid image inference latency")
    return {
        **result,
        "median_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1],
    }


PUBLIC_FIELDS = (
    "nll",
    "uniform_brier",
    "cases",
    "unknown_rate",
    "unknown_count",
    "confusion",
    "per_class",
    "outcome_order",
)


def public_metrics(row):
    return {k: row[k] for k in ("accuracy", "brier", "skill", *PUBLIC_FIELDS) if k in row}
