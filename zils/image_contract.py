"""Versioned image wire contract and the pinned Imajev unknown-answer conversion."""

import math
from uuid import UUID

from .decisions import DecisionError, invalid, validate_request

UNKNOWN = "__unknown__"
TEXT_CAPABILITIES = {"modalities": ["text"], "question_types": ["noul", "choice", "score"]}
IMAGE_CAPABILITIES = {
    "modalities": ["image", "text"],
    "question_types": ["choice"],
    "max_images": 1,
    "max_questions": 1,
    "max_options": 16,
    "max_input_tokens": 4096,
    "min_pixels": 65536,
    "max_pixels": 400000,
    "unknown_key": UNKNOWN,
    "option_order": "declared_then_unknown",
}


def validate_image_request(body):
    if not isinstance(body, dict) or set(body) != {"model", "state", "questions", "images"}:
        raise invalid(["body"])
    validate_request({key: value for key, value in body.items() if key != "images"})
    images = body["images"]
    if not isinstance(images, list) or len(images) != 1:
        raise invalid(["body", "images"])
    image = images[0]
    if not isinstance(image, dict) or set(image) != {"asset_id"}:
        raise invalid(["body", "images"])
    try:
        if (
            not isinstance(image["asset_id"], str)
            or str(UUID(image["asset_id"])) != image["asset_id"]
        ):
            raise ValueError()
    except ValueError:
        raise invalid(["body", "images", "asset_id"]) from None
    questions = body["questions"]
    if len(questions) != 1:
        raise invalid(["body", "questions"])
    question = next(iter(questions.values()))
    if (
        question["type"] != "choice"
        or len(question["criteria"]) > 16
        or UNKNOWN in question["criteria"]
        or any(not name.strip() for name in question["criteria"])
    ):
        raise invalid(["body", "questions"])
    return body


def native_probabilities(probabilities, outcomes):
    """Validate without losing unknown mass or the declared candidate ordering."""
    keys = [*outcomes, UNKNOWN]
    if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
        raise ValueError("invalid image probability labels")
    values = [probabilities[key] for key in keys]
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("invalid image probability")
    if not math.isclose(math.fsum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("image probabilities must sum to one")
    return dict(zip(keys, values, strict=True))


def make_image_response(release_id, body, predictions):
    validate_image_request(body)
    bad = DecisionError(502, "invalid_model_response", "Model returned an invalid result.")
    if not isinstance(predictions, dict) or set(predictions) != set(body["questions"]):
        raise bad
    qid, question = next(iter(body["questions"].items()))
    row = predictions[qid]
    if not isinstance(row, dict) or set(row) != {"probabilities", "input_tokens"}:
        raise bad
    count = row["input_tokens"]
    if type(count) is not int or not 0 <= count <= 4096:
        raise bad
    outcomes = list(question["criteria"])
    try:
        full = native_probabilities(row["probabilities"], outcomes)
    except ValueError:
        raise bad from None
    mass = math.fsum(full[k] for k in outcomes)
    known = {k: full[k] / mass if mass else 1 / len(outcomes) for k in outcomes}
    choice = max(known, key=known.get)
    unknown = full[UNKNOWN]
    concentration = max(0.0, (len(known) * known[choice] - 1) / (len(known) - 1))
    return {
        "model": release_id,
        "answers": {
            qid: {
                "type": "choice",
                "choice": choice,
                "probabilities": known,
                "confidence": min(1.0, concentration * (1 - unknown)),
                "unknown_probability": unknown,
                "abstained": max(full, key=full.get) == UNKNOWN,
            }
        },
        "usage": {"input_tokens": count, "output_tokens": 0},
    }
