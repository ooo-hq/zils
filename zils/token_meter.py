"""Logical input accounting, independent of repeated model execution."""

import json

from .decisions import DecisionError


def billable_tokens(body, tokenizer):
    """Tokenize one canonical JSON input with the release's actual tokenizer.

    Call after request validation. Model routing, prompt templates, generated
    text, and repeated/refinement passes do not belong to the logical input.
    """
    text = json.dumps(
        {"state": body["state"], "questions": body["questions"]},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, list) or not ids or not all(type(i) is int and i >= 0 for i in ids):
        raise DecisionError(503, "tokenizer_error", "Model tokenizer is unavailable.")
    return len(ids)
