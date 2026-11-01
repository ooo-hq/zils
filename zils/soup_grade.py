"""Soup-round grading: per-(body or soup, item) prediction cache, scores and the new-body screen.

A body or soup is graded on each bank item at most once; rows live in one append-only JSONL file
per key, so a crashed round resumes where it stopped. Model code loads only in
`production_runner`, so the coordinator can import this module without torch.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

from zils import base_metrics

_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class PredictionCache:
    def __init__(self, root):
        self.root = Path(root)
        self._rows = {}

    def _path(self, key):
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError(f"unsafe cache key: {key!r}")
        return self.root / f"{key}.jsonl"

    def _load(self, key):
        if key not in self._rows:
            rows, path = {}, self._path(key)
            if path.exists():
                for line in path.read_bytes().split(b"\n"):
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue  # empty line or a tail cut off by a crash
                    rows.setdefault(row["id"], row)
            self._rows[key] = rows
        return self._rows[key]

    def get(self, key, ids):
        rows = self._load(key)
        return {i: rows[i] for i in ids if i in rows}

    def put(self, key, rows):
        cached, path = self._load(key), self._path(key)
        new = [r for r in rows if r["id"] not in cached]
        if not new:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            if handle.tell():  # drop a partial last line left by a crash before appending
                handle.truncate(path.read_bytes().rfind(b"\n") + 1)
            for row in new:
                handle.write(json.dumps(row, sort_keys=True).encode() + b"\n")
                cached[row["id"]] = row
            handle.flush()
            os.fsync(handle.fileno())


def grade(key, cases, cache, runner):
    """Rows for `cases` in case order; only cases missing from the cache go to `runner`."""
    ids = [c["id"] for c in cases]
    have = cache.get(key, ids)
    missing = [c for c in cases if c["id"] not in have]
    if missing:
        fresh = {r["id"]: r for r in runner(missing)}
        if set(fresh) != {c["id"] for c in missing}:
            raise ValueError("runner must return exactly one row per requested case")
        cache.put(key, [fresh[c["id"]] for c in missing])
        have.update(fresh)
    return [have[i] for i in ids]


# How a soup pool scores bodies and soups (the pool's ``scoring`` setting):
# - "argmax": the base exam as before (top answers; expected value for Score questions);
# - "calibrated_argmax": one temperature per graded model fitted on a held-apart part of the
#   round's items, then top answers on the rest (Score questions: RPS, a proper score);
# - "calibrated_brier": the same fit, then Brier skill (RPS for Score questions) on the rest.
SCORINGS = ("argmax", "calibrated_argmax", "calibrated_brier")
DEFAULT_SCORING = "calibrated_brier"
CALIBRATION_SHARE = 0.2  # of each family (and yes/no label), held apart to fit the temperature


def evaluate(cases, rows, *, scoring=DEFAULT_SCORING, secret=None):
    """{"score", "temperature", "metrics", "calibration_ids"} for one graded model.

    Calibrated scorings split ``cases`` with ``base_metrics.calibration_split(cases, secret,
    CALIBRATION_SHARE)``, fit the model's temperature on the held-apart cases and score the others
    at that temperature against the whole exam's baselines. The rows are the model's raw
    temperature-1 probabilities, as graded and cached."""
    if scoring not in SCORINGS:
        raise ValueError("soup scoring must be one of " + ", ".join(SCORINGS))
    if scoring == "argmax":
        metrics = base_metrics.score(cases, rows)
        return {"score": metrics["score"], "temperature": 1.0, "metrics": metrics,
                "calibration_ids": []}  # fmt: skip
    if not isinstance(secret, bytes) or not secret:
        raise ValueError("calibrated soup scoring needs the round's private split secret")
    held, scored = base_metrics.calibration_split(cases, secret, CALIBRATION_SHARE)
    temperature = base_metrics.fit_temperature([c for c in cases if c["id"] in held], rows)
    tempered = base_metrics.temper(cases, rows, temperature)
    metric = "brier" if scoring == "calibrated_brier" else "argmax_rps"
    metrics = base_metrics.score(cases, tempered, metric=metric, only=scored)
    return {"score": metrics["score"], "temperature": temperature, "metrics": metrics,
            "calibration_ids": sorted(held)}  # fmt: skip


def score(cases, rows, *, scoring="argmax", secret=None):
    return evaluate(cases, rows, scoring=scoring, secret=secret)["score"]


def screen_score(cases, rows):
    """Mean top-choice accuracy, for comparing new bodies on the screen subset."""
    by_id = {r["id"]: r["probabilities"] for r in rows}
    hits = [max(sorted(by_id[c["id"]]), key=by_id[c["id"]].get) == c["label"] for c in cases]
    return sum(hits) / len(hits)


def _rank(seed, text):
    return hashlib.sha256(seed + b"\0" + text.encode()).digest()


def screen_subset(cases, seed: bytes, size=300):
    """Deterministic seeded case ids, spread as evenly over families as their sizes allow."""
    families = {}
    for case in cases:
        families.setdefault(case["family"], []).append(case["id"])
    queues = [
        sorted(ids, key=lambda i: _rank(seed, i))
        for _, ids in sorted(families.items(), key=lambda item: _rank(seed, item[0]))
    ]
    chosen, depth = [], 0
    while len(chosen) < size and any(depth < len(q) for q in queues):
        for queue in queues:
            if depth < len(queue) and len(chosen) < size:
                chosen.append(queue[depth])
        depth += 1
    return chosen


def production_runner(checkpoint, device="cuda"):
    """A runner that loads the d3 checkpoint once, on first use, and predicts case by case."""
    model = None

    def run(cases):
        nonlocal model
        if model is None:
            from zils import d3

            model = d3.DecisionModel(checkpoint, device)
        rows = []
        for case in cases:
            start = time.perf_counter()
            probabilities = model.predict(case["state"], case["question"])
            elapsed = round((time.perf_counter() - start) * 1000, 3)
            rows.append({"id": case["id"], "probabilities": probabilities, "elapsed_ms": elapsed})
        return rows

    return run
