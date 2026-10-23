"""Versioned decision wire capabilities for text and image-capable profiles."""

from . import models

TEXT_CAPABILITIES = {"modalities": ["text"], "question_types": ["noul", "choice", "score"]}


def capabilities(model):
    if model not in models.d3_family():
        return TEXT_CAPABILITIES
    spec = models.spec(model)
    return {
        "modalities": ["text", "image", "video"],
        "question_types": ["noul", "choice", "score"],
        "max_images": spec["inputs"]["images"],
        "max_videos": 1,
        "max_questions": 8,
        "max_options": 255,
        "max_input_tokens": spec["max_input_tokens"],
        "unknown_key": None,
    }
