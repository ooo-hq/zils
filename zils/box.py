"""Fixed five-question box decoder and paired gates for the synthetic-only pilot."""

import json
import math
import random
import re
import statistics
from collections import defaultdict

from .core import validate_targets

DECODER = "box-edges-101-presence/v1"
EDGES = ("top", "bottom", "left", "right")
DEFAULT_POLICY = {
    "min_iou_gain": 0.02,
    "bootstrap_resamples": 2000,
    "min_mean_iou": 0.40,
    "max_family_iou_drop": 0.02,
    "max_false_positive_rate": 0.10,
    "max_presence_brier_increase": 0.005,
}


def validate_policy(policy=None):
    if policy is not None and (not isinstance(policy, dict) or set(policy) - DEFAULT_POLICY.keys()):
        raise ValueError("unknown box policy fields")
    policy = {**DEFAULT_POLICY, **(policy or {})}
    for key, value in policy.items():
        if key == "bootstrap_resamples":
            if type(value) is not int or not 1 <= value <= 100_000:
                raise ValueError("bootstrap_resamples must be an integer in [1, 100000]")
        elif type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be finite and in [0, 1]")
    return policy


def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("box cases must be a nonempty list")
    seen, groups, images = set(), {}, {}
    present = 0
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "id",
            "group_id",
            "family",
            "state",
            "target",
            "image",
            "box",
        }:
            raise ValueError(
                "box case fields must be id, group_id, family, state, target, image, box"
            )
        for key in ("id", "group_id", "family", "target"):
            if not isinstance(case[key], str) or not case[key].strip():
                raise ValueError(f"box case {key} must be a nonempty string")
        if case["id"] in seen:
            raise ValueError("duplicate box case id")
        seen.add(case["id"])
        if not isinstance(case["image"], str) or not re.fullmatch(
            r"[a-f0-9]{64}\.png", case["image"]
        ):
            raise ValueError("box image must be a SHA256-addressed PNG filename")
        group = (case["image"], case["family"])
        if groups.setdefault(case["group_id"], group) != group:
            raise ValueError("a box group must identify one image and family")
        if images.setdefault(case["image"], case["group_id"]) != case["group_id"]:
            raise ValueError("an image must belong to one box group")
        try:
            encoded = json.dumps(case, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise ValueError("box cases must contain finite JSON values") from error
        if len(encoded.encode()) > 128 * 1024:
            raise ValueError("box case exceeds 128 KiB")
        truth = case["box"]
        if truth is None:
            continue
        if not isinstance(truth, dict) or set(truth) != set(EDGES):
            raise ValueError("box must contain exactly top, bottom, left, right")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in truth.values()
        ):
            raise ValueError("box edges must be finite numbers in [0, 1]")
        if truth["top"] >= truth["bottom"] or truth["left"] >= truth["right"]:
            raise ValueError("ground-truth box must have ordered edges and positive area")
        present += 1
    if not 0 < present < len(cases):
        raise ValueError("box evaluation requires both present and absent targets")


def expand_cases(cases):
    """Ordinary evaluation rows; only context and target enter inference state."""
    validate_cases(cases)
    rows = []
    for case in cases:
        for edge in (*EDGES, "presence"):
            presence = edge == "presence"
            rows.append(
                {
                    "id": case["id"] + ":" + edge,
                    "family": case["family"],
                    "state": {"context": case["state"], "target": case["target"]},
                    "image": case["image"],
                    "question": {
                        "type": "choice",
                        "instructions": (
                            "Is the target present in the image?"
                            if presence
                            else f"Locate the {edge} edge of the target's bounding box. Coordinates run from "
                            "0 at the top/left to 100 at the bottom/right of the image. "
                            "Choose the nearest integer; choose 0 if the target is absent."
                        ),
                        "criteria": {"false": "Absent", "true": "Present"}
                        if presence
                        else {str(i): f"{i}%" for i in range(101)},
                    },
                    "label": (
                        str(case["box"] is not None).lower()
                        if presence
                        else str(round(100 * case["box"][edge]))
                        if case["box"]
                        else "0"
                    ),
                }
            )
    return rows


def decode(cases, predictions):
    """Argmax edges, sorted into a box; confidence is P(target present), not IoU."""
    expanded = expand_cases(cases)
    if not isinstance(predictions, list) or len(predictions) != len(expanded):
        raise ValueError("exactly five predictions per box case are required")
    by_id = {}
    for row in predictions:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in by_id:
            raise ValueError("invalid or duplicate box prediction id")
        by_id[row["id"]] = row
    if set(by_id) != {row["id"] for row in expanded}:
        raise ValueError("prediction ids differ from expanded box cases")
    for row in expanded:
        validate_targets(by_id[row["id"]].get("probabilities"), row["question"]["criteria"])
    decoded = []
    for case in cases:
        confidence = by_id[case["id"] + ":presence"]["probabilities"]["true"]
        values = {}
        for edge in EDGES:
            probabilities = by_id[case["id"] + ":" + edge]["probabilities"]
            values[edge] = int(max(sorted(probabilities), key=probabilities.get)) / 100
        values["top"], values["bottom"] = sorted((values["top"], values["bottom"]))
        values["left"], values["right"] = sorted((values["left"], values["right"]))
        decoded.append(
            {
                "id": case["id"],
                "box": values if confidence >= 0.5 else None,
                "presence_confidence": confidence,
            }
        )
    return decoded


def _iou(truth, prediction):
    if prediction is None:
        return 0.0
    overlap = max(
        0, min(truth["right"], prediction["right"]) - max(truth["left"], prediction["left"])
    ) * max(0, min(truth["bottom"], prediction["bottom"]) - max(truth["top"], prediction["top"]))

    def area(rectangle):
        return (rectangle["right"] - rectangle["left"]) * (rectangle["bottom"] - rectangle["top"])

    return overlap / (area(truth) + area(prediction) - overlap)


def _measure(cases, predictions):
    decoded = decode(cases, predictions)
    families, ious, briers = defaultdict(list), {}, []
    false_positives, absent = 0, 0
    for case, row in zip(cases, decoded, strict=True):
        present = case["box"] is not None
        briers.append((row["presence_confidence"] - present) ** 2)
        if present:
            ious[case["id"]] = _iou(case["box"], row["box"])
            families[case["family"]].append(ious[case["id"]])
        else:
            absent += 1
            false_positives += row["box"] is not None
    family_iou = {family: statistics.mean(values) for family, values in sorted(families.items())}
    return {
        "cases": len(cases),
        "present_cases": len(ious),
        "absent_cases": absent,
        "mean_iou": statistics.mean(family_iou.values()),
        "family_iou": family_iou,
        "false_positive_rate": false_positives / absent,
        "presence_brier": statistics.mean(briers),
    }, ious


def score(cases, predictions):
    """Equal-family positive IoU, absent FPR, and binary presence Brier."""
    return _measure(cases, predictions)[0]


def compare(cases, incumbent_predictions, candidate_predictions, policy=None):
    """Paired, family-stratified image-group bootstrap; all entry gates must pass."""
    policy = validate_policy(policy)
    incumbent, incumbent_iou = _measure(cases, incumbent_predictions)
    candidate, candidate_iou = _measure(cases, candidate_predictions)
    families = defaultdict(lambda: defaultdict(list))
    for case in cases:
        if case["box"] is not None:
            families[case["family"]][case["group_id"]].append(
                candidate_iou[case["id"]] - incumbent_iou[case["id"]]
            )
    rng, draws = random.Random(0), []
    # Resample images together when an image has several target questions.
    groups = [list(families[family].values()) for family in sorted(families)]
    for _ in range(policy["bootstrap_resamples"]):
        draws.append(
            statistics.mean(
                statistics.mean(
                    value for group in rng.choices(family, k=len(family)) for value in group
                )
                for family in groups
            )
        )
    draws.sort()
    gain = candidate["mean_iou"] - incumbent["mean_iou"]
    lower = draws[int(0.025 * len(draws))]
    checks = {
        "gain": gain >= policy["min_iou_gain"],
        "lower_95": lower > 0,
        "mean_iou": candidate["mean_iou"] >= policy["min_mean_iou"],
        "family_iou": all(
            candidate["family_iou"][family] >= value - policy["max_family_iou_drop"]
            for family, value in incumbent["family_iou"].items()
        ),
        "false_positive_rate": candidate["false_positive_rate"]
        <= policy["max_false_positive_rate"],
        "presence_brier": candidate["presence_brier"]
        <= incumbent["presence_brier"] + policy["max_presence_brier_increase"],
    }
    return {
        "eligible": all(checks.values()),
        "gain": gain,
        "lower_95": lower,
        "incumbent": incumbent,
        "candidate": candidate,
        "checks": checks,
    }
