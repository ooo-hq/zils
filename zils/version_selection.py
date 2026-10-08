"""Frozen customer version lineage; shared/base models are never replaced."""

import re
import uuid

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


def freeze(store, job):
    from .adapter_releases import _accepted

    previous_id = job["acceptance"].get("previous_job_id")
    selection = {"version": VERSION, "root_job_id": job["id"], "previous": None}
    if previous_id is not None:
        rows = store.rows("fez_training_jobs", f"id=eq.{job_id(previous_id)}")
        if len(rows) != 1 or rows[0]["owner_id"] != job["owner_id"]:
            raise ValueError("Previous model is unavailable to this customer")
        parent = rows[0]
        from . import models

        # Historical callers without a frozen profile retain their existing path.
        if job.get("model_profile") is not None and models.job_model(job) != models.job_model(
            parent
        ):
            raise ValueError("Previous model belongs to a different profile family")
        delivery = _accepted(parent)
        model_id = f"zils-adapter-{previous_id}-{delivery['sha256']}"
        workflow = parent["result"].get("workflow") or {}
        if workflow.get("state") != "ready" or workflow.get("model_id") != model_id:
            raise ValueError("Previous model must have completed API activation")
        lineage = parent["manifest"].get("selection")
        selection["root_job_id"] = lineage["root_job_id"] if lineage else previous_id
        selection["previous"] = {
            "job_id": previous_id,
            "model_id": model_id,
            "sha256": delivery["sha256"],
        }
    return validate(selection, previous_id, current_job_id=job["id"])


def check_promotion(current, entry, selection):
    """Called under the registry lock immediately before moving the task alias."""
    previous = selection.get("previous")
    match = re.fullmatch(r"zils-adapter-([a-f0-9-]{36})-([a-f0-9]{64})", entry["id"])
    if not match:
        raise ValueError("Version release requires an immutable customer model ID")
    validate(selection, previous["job_id"] if previous else None, current_job_id=match[1])
    alias = "zils-task-" + selection["root_job_id"]
    if entry["aliases"] != [alias]:
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
