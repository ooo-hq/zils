"""Readable per-run aliases, distinct from movable task aliases and release IDs."""

import re
import uuid


def model_name(run_name, job_id):
    job_id = str(uuid.UUID(job_id))
    slug = re.sub(r"[^a-z0-9]+", "-", run_name.lower()).strip("-") or "model"
    if slug.startswith("zils-task-"):
        slug = "model-" + slug.removeprefix("zils-task-")
    if len(slug) > 24:
        slug = slug[:25].rsplit("-", 1)[0] if "-" in slug[:25] else slug[:24]
    return f"{slug.rstrip('-')}-{job_id[:8]}"


def is_model_name(name, release_id):
    match = re.fullmatch(r"zils-adapter-([a-f0-9-]{36})-[a-f0-9]{64}", release_id)
    return bool(
        match
        and isinstance(name, str)
        and not name.startswith("zils-task-")
        and re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,22}[a-z0-9])?-" + match[1][:8], name)
    )


def valid_customer_aliases(entry, selection):
    """Registration may add one readable name bound to this run, never a shared alias."""
    aliases = entry.get("aliases")
    required = ["zils-task-" + selection["root_job_id"]] if selection else []
    return aliases == required or (
        isinstance(aliases, list)
        and len(aliases) == len(required) + 1
        and aliases[:-1] == required
        and is_model_name(aliases[-1], entry.get("id", ""))
    )
