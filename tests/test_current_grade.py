"""Zils Current grading: shared GPU lock, cached predictions and per-stream scores (no GPU)."""

import contextlib
import fcntl
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import zils
from tests.test_current import resolved_record, wiki_data
from zils import current, current_grade, current_items


def items():
    pr = {"repo": "w", "head": "h", "base": "m", "title": "T", "body": ""}
    records = [
        resolved_record("wikipedia-revert", 1, "true", {**wiki_data("A")}),
        resolved_record("wikipedia-revert", 2, "false", {**wiki_data("B")}),
        resolved_record("github-pr-merge", 3, "true", {**pr, "title": "C"}),
        resolved_record("github-pr-merge", 4, "false", {**pr, "title": "D"}),
    ]
    return [current_items.build(r) for r in records]


class Runner:
    """Right with p=0.9 on Wikipedia edits, wrong with p=0.9 on pull requests."""

    def __init__(self, lock=None):
        self.calls, self.lock, self.held = 0, lock, []

    def __call__(self, cases):
        self.calls += 1
        if self.lock:
            fd = os.open(self.lock, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.held.append(False)
            except BlockingIOError:
                self.held.append(True)
            finally:
                os.close(fd)
        rows = []
        for case in cases:
            right = (
                case["label"]
                if case["family"] == "wikipedia-revert"
                else ("false" if case["label"] == "true" else "true")
            )
            probabilities = {k: 0.9 if k == right else 0.1 for k in zils.options(case["question"])}
            rows.append({"id": case["id"], "probabilities": probabilities, "elapsed_ms": 1.0})
        return rows


class GradeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.items = self.root / "2026-10-30" / "items.jsonl"
        self.items.parent.mkdir()
        self.items.write_text("".join(json.dumps(c) + "\n" for c in items()))
        self.lock = self.root / "cuda.lock"

    def tearDown(self):
        self.tmp.cleanup()

    def grade(self, runner, wait=0):
        return current_grade.grade(
            self.items,
            runner,
            key="stock-lite",
            cache=self.root / "cache",
            lock=self.lock,
            wait_seconds=wait,
        )

    def test_scores_per_stream_while_holding_the_shared_lock(self):
        runner = Runner(self.lock)
        result = self.grade(runner)
        self.assertEqual(runner.held, [True])
        self.assertEqual(result["status"], "graded")
        wiki, pr = result["streams"]["wikipedia-revert"], result["streams"]["github-pr-merge"]
        self.assertEqual((wiki["accuracy"], pr["accuracy"]), (1.0, 0.0))
        self.assertAlmostEqual(wiki["brier"], 0.02)
        self.assertAlmostEqual(pr["brier"], 1.62)
        self.assertEqual(wiki["cases"], 2)
        self.assertEqual(result["all"]["cases"], 4)
        self.assertEqual(result["key"], "stock-lite")
        self.assertEqual(result["items_sha256"], current_grade.sha256(self.items))

    def test_cached_predictions_are_not_recomputed(self):
        first = Runner()
        self.grade(first)
        again = Runner()
        result = self.grade(again)
        self.assertEqual(again.calls, 0)
        self.assertEqual(result["streams"]["wikipedia-revert"]["accuracy"], 1.0)

    def test_a_busy_gpu_means_no_grading(self):
        fd = os.open(self.lock, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            runner = Runner()
            result = self.grade(runner, wait=0)
        finally:
            os.close(fd)
        self.assertEqual(result["status"], "busy")
        self.assertEqual(runner.calls, 0)

    def test_the_default_lock_is_the_shared_training_lock(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(current_grade.lock_path(), "/var/lock/zils-training/cuda.lock")
        with patch.dict(os.environ, {"ZILS_COMPUTE_LOCK": "/tmp/other.lock"}):
            self.assertEqual(current_grade.lock_path(), "/tmp/other.lock")

    def test_command_line_writes_results_and_signals_busy(self):
        out = self.root / "grades.json"
        argv = [
            "grade",
            "--items",
            str(self.items),
            "--model",
            str(self.root / "ckpt"),
            "--lock",
            str(self.lock),
            "--lock-wait",
            "0",
            "--out",
            str(out),
        ]
        with patch("zils.soup_grade.production_runner", return_value=Runner()) as factory:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(current.main(argv), 0)
        factory.assert_called_once_with(str(self.root / "ckpt"), "cuda")
        result = json.loads(out.read_text())
        self.assertEqual(result["key"], "ckpt")
        fd = os.open(self.lock, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with patch("zils.soup_grade.production_runner", return_value=Runner()):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        current.main(
                            [*argv[:-2], "--out", str(self.root / "b.json"), "--key", "x"]
                        ),
                        75,
                    )
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
