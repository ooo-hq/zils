"""Frozen customer version lineage; shared/base models are never replaced."""

import re
import uuid

from .model_names import valid_customer_aliases

VERSION = "zils-version-selection/v1"


class StaleVersion(ValueError):
    """The selected incumbent is no longer the task's active version."""


def job_id(value):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError("previous_job_id must be a canonical job UUID")
    return value


def validate(selection, previous_job_id=None, *, current_job_id=None):
    if (
        not isinstance(selection, dict)
        or set(selection) != {"version", "root_job_id", "previous"}
        or selection["version"] != VERSION
    ):
        raise ValueError("Invalid frozen version selection")
    job_id(selection["root_job_id"])
    previous = selection["previous"]
    if previous_job_id is None:
        if previous is not None or (
            current_job_id is not None and selection["root_job_id"] != current_job_id
        ):
            raise ValueError("First version must compare with the base")
    elif (
        not isinstance(previous, dict)
        or set(previous) != {"job_id", "model_id", "sha256"}
        or previous["job_id"] != job_id(previous_job_id)
        or not isinstance(previous["sha256"], str)
        or not re.fullmatch("[a-f0-9]{64}", previous["sha256"])
        or previous["model_id"] != f"zils-adapter-{previous_job_id}-{previous['sha256']}"
        or previous_job_id == current_job_id
    ):
        raise ValueError("Upgrade must compare with its frozen previous model")
    return selection


def check_promotion(current, entry, selection):
    """Called under the registry lock immediately before moving the task alias."""
    previous = selection.get("previous")
    match = re.fullmatch(r"zils-adapter-([a-f0-9-]{36})-([a-f0-9]{64})", entry["id"])
    if not match:
        raise ValueError("Version release requires an immutable customer model ID")
    validate(selection, previous["job_id"] if previous else None, current_job_id=match[1])
    alias = "zils-task-" + selection["root_job_id"]
    if not valid_customer_aliases(entry, selection):
        raise ValueError("A version release must use its task alias")
    active = next((row for row in current["models"] if alias in row["aliases"]), None)
    if active and active["owners"] != entry["owners"]:
        raise ValueError("Task alias belongs to another customer")
    if active and active["id"] == entry["id"]:
        return  # Retry after a registry write succeeded but its acknowledgement was lost.
    if previous:
        incumbent = next(
            (row for row in current["models"] if row["id"] == previous["model_id"]), None
        )
        if incumbent is not None and incumbent.get("profile") != entry.get("profile"):
            raise ValueError("A task version cannot move between model profiles")
        if incumbent is None or incumbent["owners"] != entry["owners"]:
            raise ValueError("Previous model is unavailable to this customer")
        if active is None and (
            selection["root_job_id"] != previous["job_id"]
            or any(a.startswith("zils-task-") for a in incumbent["aliases"])
        ):
            raise StaleVersion("The task's active model cannot be established")
        # Legacy accepted models had no task alias. Their first upgrade establishes one.
        if active and active["id"] != previous["model_id"]:
            raise StaleVersion("A newer version is active; evaluate against that version")
    elif active:
        raise StaleVersion("This task already has an active model")
