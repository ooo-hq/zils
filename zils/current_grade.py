"""Zils Current grading: score a dated export with a checkpoint under the shared GPU lock.

    python -m zils.current grade --items BANK/2026-10-30/items.jsonl --model CHECKPOINT

Predictions use the soup-round path (`soup_grade.grade` with its per-key prediction cache and
`soup_grade.production_runner`), so a stopped run resumes and a graded item is never run twice.
GPU work holds the training/validator lock (`ZILS_COMPUTE_LOCK`, default
/var/lock/zils-training/cuda.lock); if it stays busy, nothing runs and the caller may retry.
"""

import fcntl
import hashlib
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import zils

from . import settings, soup_grade

DEFAULT_LOCK = "/var/lock/zils-training/cuda.lock"
BUSY = 75  # EX_TEMPFAIL: the GPU lock stayed busy; try again later


def lock_path():
    return settings.get("ZILS_COMPUTE_LOCK") or DEFAULT_LOCK


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def compute_lock(path, wait_seconds, poll=0.5):
    """Yield True while holding the lock, or False if it stayed busy for `wait_seconds`."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    yield False
                    return
                time.sleep(poll)
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _metrics(cases, rows):
    result = zils.score(cases, rows)
    return {
        k: result[k]
        for k in ("cases", "accuracy", "brier", "uniform_brier", "skill", "confident_errors")
    }


def grade(items, runner, *, key, cache, lock, wait_seconds=600):
    """Per-stream accuracy and Brier for one model key on one export."""
    items = Path(items)
    cases = [json.loads(line) for line in items.read_text().split("\n") if line]
    zils.validate_cases(cases)
    result = {"key": key, "items": items.parent.name, "items_sha256": sha256(items)}
    with compute_lock(lock, wait_seconds) as held:
        if not held:
            return {**result, "status": "busy"}
        rows = soup_grade.grade(key, cases, soup_grade.PredictionCache(cache), runner)
    by_id = {r["id"]: r for r in rows}
    streams = defaultdict(list)
    for case in cases:
        streams[case["family"]].append(case)
    return {
        **result,
        "status": "graded",
        "streams": {
            s: _metrics(group, [by_id[c["id"]] for c in group])
            for s, group in sorted(streams.items())
        },
        "all": _metrics(cases, rows),
    }
