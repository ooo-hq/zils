"""Capabilities each pinned profile advertises: text, plus d3's images and one video."""

from . import models

TEXT_CAPABILITIES = {"modalities": ["text"], "question_types": ["noul", "choice", "score"]}


def capabilities(model):
    if model != models.D3:
        return TEXT_CAPABILITIES
    spec = models.spec(models.D3)
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
