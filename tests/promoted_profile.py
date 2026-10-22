"""A test-only base promoted from d3-mini, registered the way scripts/promote_base.py proposes."""

import contextlib
from unittest import mock

from zils import models

PROMOTED = "d3-mini-4.5b-test-promoted"


@contextlib.contextmanager
def promoted_profile():
    """Register PROMOTED in models' registries for the duration; nothing leaks afterwards."""
    parent = models.spec(models.D3)
    entries = {
        "SPECS": {
            **parent,
            "id": PROMOTED,
            "name": "Test promoted (d3-mini 4.5B)",
            "base": f"zils-ai/{PROMOTED}",
            "base_revision": "0" * 40,
            "parent": {"id": models.D3, "base_revision": parent["base_revision"]},
        },
        "TEXT_RUNTIME_MODULES": models.TEXT_RUNTIME_MODULES[models.D3],
        "RUNTIME_PYTHON_ENV": models.RUNTIME_PYTHON_ENV[models.D3],
        "CANDIDATE_BASE_ENV": "ZILS_D3_TEST_PROMOTED_CANDIDATE_BASE",
        # Reuses the parent's pins file: the fixture has no published weights of its own.
        "PINS_FILES": models.PINS_FILES[models.D3],
    }
    with contextlib.ExitStack() as stack:
        for name, value in entries.items():
            stack.enter_context(mock.patch.dict(getattr(models, name), {PROMOTED: value}))
        stack.enter_context(
            mock.patch.object(
                models, "TEXT_ADAPTER_MODELS", (*models.TEXT_ADAPTER_MODELS, PROMOTED)
            )
        )
        yield PROMOTED
