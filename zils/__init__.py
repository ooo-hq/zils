"""Zils checkpoint scoring and subnet services."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ROOT must exist before core imports it. Explicit aliases preserve the public API.
from .core import (  # noqa: E402
    ARTIFACT_FILES as ARTIFACT_FILES,
    BASE as BASE,
    MAX_ARTIFACT_BYTES as MAX_ARTIFACT_BYTES,
    RUBRIC as RUBRIC,
    checkpoint_hash as checkpoint_hash,
    evaluate as evaluate,
    options as options,
    score as score,
    stage as stage,
    submission as submission,
    validate_cases as validate_cases,
    validate_submissions as validate_submissions,
    weight_vector as weight_vector,
)
