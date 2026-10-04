"""Zils' TypeSafe-style wire contract, independent of model and storage libraries."""

import json
import math

MAX_BODY = 1024 * 1024
MAX_DEPTH = 32


class DecisionError(Exception):
    def __init__(self, status, code, message, field=None, retry_after=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.field = field
        self.retry_after = retry_after


def invalid(field=None):
    return DecisionError(422, "invalid_request", "Request field is missing or unsupported.", field)


def render(value):
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    )


def _tree(value):
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > MAX_DEPTH:
            raise invalid()
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise invalid()
            pending.extend((v, depth + 1) for v in item.values())
            pending.extend((k, depth + 1) for k in item)
        elif isinstance(item, list):
            pending.extend((v, depth + 1) for v in item)
        elif isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeEncodeError:
                raise invalid() from None
        elif type(item) is float:
            if not math.isfinite(item):
                raise invalid()
        elif item is not None and type(item) not in (bool, int):
            raise invalid()


def decode_body(raw, limit=MAX_BODY):
    if len(raw) > limit:
        raise DecisionError(413, "body_too_large", "Request exceeds the body limit.")

    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate key")
            obj[key] = value
        return obj

    def constant(value):
        raise ValueError("non-finite number")

    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=constant)
    except (UnicodeError, ValueError, RecursionError):
        raise DecisionError(
            400, "invalid_json", "Body must be valid UTF-8 JSON with unique keys."
        ) from None
    _tree(result)
    return result


def _description(value, nullable=False):
    return isinstance(value, (str, dict, list)) or (nullable and value is None)


def validate_request(body):
    _tree(body)
    if not isinstance(body, dict) or set(body) != {"state", "model", "questions"}:
        raise invalid(["body"])
    if not _description(body["state"]):
        raise invalid(["body", "state"])
    if not isinstance(body["model"], str) or not 1 <= len(body["model"]) <= 128:
        raise invalid(["body", "model"])
    questions = body["questions"]
    if not isinstance(questions, dict) or not questions:
        raise invalid(["body", "questions"])
    for qid, q in questions.items():
        loc = ["body", "questions", qid]
        if not qid or not isinstance(q, dict) or set(q) - {"type", "instructions", "criteria"}:
            raise invalid(loc)
        if not _description(q.get("instructions"), nullable=True):
            raise invalid([*loc, "instructions"])
        kind, criteria = q.get("type"), q.get("criteria")
        if kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict)
                or set(criteria) - {"true", "false"}
                or not all(_description(v, True) for v in criteria.values())
            ):
                raise invalid([*loc, "criteria"])
        elif kind == "choice":
            if (
                not isinstance(criteria, dict)
                or not 2 <= len(criteria) <= 255
                or any(not k or not _description(v, True) for k, v in criteria.items())
            ):
                raise invalid([*loc, "criteria"])
        elif kind == "score":
            if (
                not isinstance(criteria, list)
                or not 2 <= len(criteria) <= 10
                or not all(_description(v) for v in criteria)
            ):
                raise invalid([*loc, "criteria"])
        else:
            raise invalid([*loc, "type"])
    try:
        size = len(json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    except (ValueError, RecursionError):
        raise invalid() from None
    if size > MAX_BODY:
        raise DecisionError(413, "body_too_large", "Request exceeds the body limit.")
    return body


def option_descriptions(question):
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "noul":
        return {
            k: render(
                (criteria or {}).get(k)
                if (criteria or {}).get(k) is not None
                else f"The proposition is {k}."
            )
            for k in ("true", "false")
        }
    if kind == "choice":
        return {k: render(v) if v is not None else k for k, v in criteria.items()}
    return {str(i): render(value) for i, value in enumerate(criteria)}


def make_response(release_id, body, predictions):
    bad = DecisionError(502, "invalid_model_response", "Model returned an invalid result.")
    if not isinstance(predictions, dict) or set(predictions) != set(body["questions"]):
        raise bad
    answers, tokens = {}, 0
    for qid, question in body["questions"].items():
        row = predictions[qid]
        keys = list(option_descriptions(question))
        if not isinstance(row, dict) or set(row) != {"probabilities", "input_tokens"}:
            raise bad
        probs = row["probabilities"]
        count = row["input_tokens"]
        if (
            not isinstance(probs, dict)
            or set(probs) != set(keys)
            or type(count) is not int
            or count < 0
        ):
            raise bad
        values = [probs[k] for k in keys]
        if any(
            type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in values
        ):
            raise bad
        total = math.fsum(values)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise bad
        values = [float(p / total) for p in values]
        kind = question["type"]
        answer = {"type": kind}
        if kind == "noul":
            answer["noul"] = values[0]
        else:
            n = len(values)
            mode = values.index(max(values))
            answer["probabilities"] = dict(zip(keys, values, strict=True))
            if kind == "choice":
                answer["choice"] = keys[mode]
                confidence = (values[mode] - 1 / n) / (1 - 1 / n)
            else:
                answer["score"] = sum(i * p for i, p in enumerate(values))
                answer["legend"] = dict(enumerate(question["criteria"]))
                answer["legend"] = {str(k): v for k, v in answer["legend"].items()}
                spread = sum(p * abs(i - mode) for i, p in enumerate(values))
                uniform = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
                confidence = 1 - spread / uniform
            answer["confidence"] = max(0.0, min(1.0, confidence))
        answers[qid] = answer
        tokens += count
    return {
        "model": release_id,
        "answers": answers,
        "usage": {"input_tokens": tokens, "output_tokens": 0},
    }
