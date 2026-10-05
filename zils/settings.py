"""Canonical Zils settings with compatibility for existing Fez deployments."""

import os


def get(name, default=None):
    """A present ZILS_* setting wins, including an explicitly empty value."""
    legacy = "FEZ_" + name.removeprefix("ZILS_")
    return os.environ.get(name, os.environ.get(legacy, default))


def required(name):
    value = get(name)
    if not value:
        raise ValueError(f"Set {name} to a nonempty value")
    return value
