"""Base-tier rounds: sealed audit, calibrated comparison with a bootstrap margin, two-draw confirmation."""

import json
import random
import tempfile
import unittest
from pathlib import Path

import zils
from zils import base_round, benchmark

FAMILIES = ("alpha", "beta")


def make_cases(prefix, per_family=120):
    cases = []
    for family in FAMILIES:
        for index in range(per_family):
            keys = ["a", "b", "c", "d"]
            cases.append(
                {
                    "id": f"{prefix}-{family}-{index}",
                    "family": family,
                    "state": {"text": f"{prefix} {family} {index}"},
                    "question": {
                        "type": "choice",
                        "instructions": "Pick.",
                        "criteria": {k: k for k in keys},
                    },
                    "label": keys[index % 4],
                }
            )
    return cases


def predict(cases, accuracy, seed, confidence=0.7):
    """A model right on about `accuracy` of cases, putting `confidence` on its chosen option."""
    rng = random.Random(seed)
    rows = []
    for case in cases:
        keys = zils.options(case["question"])
        chosen = (
            case["label"]
            if rng.random() < accuracy
            else rng.choice([k for k in keys if k != case["label"]])
        )
        p = {k: (1 - confidence) / (len(keys) - 1) for k in keys}
        p[chosen] = confidence
        p = {k: max(v + rng.uniform(-0.02, 0.02), 1e-3) for k, v in p.items()}
        total = sum(p.values())
        rows.append(
            {
                "id": case["id"],
                "probabilities": {k: v / total for k, v in p.items()},
                "elapsed_ms": 1.0,
            }
        )
    return rows


def report(cases, predictions_by_uid):
    return {
        "dataset_sha256": base_round.dataset_hash(cases),
        "miners": [
            {
                "uid": uid,
                "sha256": f"{uid:064x}",
                "status": "evaluated",
                "predictions": preds,
                **zils.score(cases, preds),
            }
            for uid, preds in predictions_by_uid.items()
        ],
    }


def sealed(root, calibration, test):
    root.mkdir()
    for name, cases in (("calibration.jsonl", calibration), ("test.jsonl", test)):
        (root / name).write_text("".join(json.dumps(c) + "\n" for c in cases))
    manifest = {
        "version": base_round.VERSION,
        "files": {n: benchmark.file_hash(root / n) for n in ("calibration.jsonl", "test.jsonl")},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


class BaseRoundTest(unittest.TestCase):
    def test_audit_rejects_tampering_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "round"
            sealed(root, make_cases("cal", 5), make_cases("test", 5))
            base_round.audit(root)
            (root / "test.jsonl").write_text(
                (root / "test.jsonl").read_text().replace('"label": "a"', '"label": "b"', 1)
            )
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                base_round.audit(root)
            overlap = Path(tmp) / "overlap"
            sealed(overlap, make_cases("same", 5), make_cases("same", 5))
            with self.assertRaisesRegex(ValueError, "overlap"):
                base_round.audit(overlap)

    def test_decision_margin_bootstrap_and_confirmation(self):
        calibration, test = make_cases("cal"), make_cases("test")
        cases = {"calibration": calibration, "test": test}
        # uid 0 incumbent; 1 clearly better; 2 marginally better (fails the bootstrap); 3 worse.
        masses = {0: 0.55, 1: 0.80, 2: 0.57, 3: 0.40}
        cal_rows = base_round.rows_for(
            report(calibration, {u: predict(calibration, m, 100 + u) for u, m in masses.items()}),
            calibration,
        )
        test_rows = base_round.rows_for(
            report(test, {u: predict(test, m, 200 + u) for u, m in masses.items()}), test
        )
        decision = base_round.decide(cases, 0, cal_rows, test_rows, {"bootstrap_resamples": 300})
        by_uid = {c["uid"]: c for c in decision["candidates"]}
        self.assertTrue(by_uid[1]["eligible"])
        self.assertFalse(by_uid[2]["eligible"], by_uid[2])
        self.assertFalse(by_uid[3]["eligible"])
        self.assertLess(by_uid[3]["improvement"], 0)
        self.assertEqual(decision["pending_winner"]["uid"], 1)
        self.assertEqual(set(decision["weights"]), {1})
        self.assertTrue(0.25 <= decision["incumbent"]["temperature"] <= 4)
        # Confirmation needs a disjoint draw where the same checkpoint stays eligible.
        second_test = make_cases("second")
        second_rows = base_round.rows_for(
            report(second_test, {u: predict(second_test, m, 300 + u) for u, m in masses.items()}),
            second_test,
        )
        second = base_round.decide(
            {"calibration": calibration, "test": second_test},
            0,
            cal_rows,
            second_rows,
            {"bootstrap_resamples": 300},
        )
        self.assertEqual(base_round.confirm(decision, second)["uid"], 1)
        regressed = base_round.rows_for(
            report(
                second_test, {0: predict(second_test, 0.55, 1), 1: predict(second_test, 0.50, 2)}
            ),
            second_test,
        )
        failed = base_round.decide(
            {"calibration": calibration, "test": second_test},
            0,
            cal_rows,
            regressed,
            {"bootstrap_resamples": 300},
        )
        self.assertIsNone(base_round.confirm(decision, failed))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            base_round.confirm(decision, decision)

    def test_track_describes_draws_and_pays_only_matching_decisions(self):
        import contextlib
        import io
        from unittest.mock import patch

        calibration, test = make_cases("cal", 20), make_cases("test", 20)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "draw-1"
            sealed(root, calibration, test)
            draw = base_round.draw(root)
            self.assertEqual(draw["dataset_sha256"], base_round.dataset_hash(test))
            self.assertEqual(draw["manifest_sha256"], benchmark.file_hash(root / "manifest.json"))
            out = io.StringIO()
            with (
                patch("sys.argv", ["base_round", "--round", str(root), "--describe-draw"]),
                contextlib.redirect_stdout(out),
            ):
                base_round.main()
            self.assertEqual(json.loads(out.getvalue()), draw)
        incumbent = "c" * 64
        track = {"share": 0.7, "incumbent_sha256": incumbent, "draws": [draw]}
        self.assertEqual(base_round.validate_track(track)["policy"], base_round.DEFAULT_POLICY)
        for bad in (
            {**track, "share": 1.5},
            {**track, "share": "0.7"},
            {**track, "incumbent_sha256": "not-a-hash"},
            {**track, "draws": []},
            {**track, "draws": [{"manifest_sha256": draw["manifest_sha256"]}]},
            {**track, "policy": {"unknown": 1}},
            {**track, "policy": {"min_brier_improvement": -1}},
            {**track, "extra": True},
        ):
            with self.assertRaises(ValueError):
                base_round.validate_track(bad)
        members = {"1": {"hotkey": "a"}, "2": {"hotkey": "b"}}
        decision = {
            **draw,
            "policy": dict(base_round.DEFAULT_POLICY),
            "incumbent": {"uid": 65535, "sha256": incumbent},
            "candidates": [
                {"uid": 2, "sha256": "d" * 64, "eligible": True, "skill": 0.4},
                {"uid": 1, "sha256": "e" * 64, "eligible": False, "skill": 0.5},
            ],
            "weights": {"2": 1.0},
        }
        self.assertEqual(base_round.track_weights(track, decision, members), {2: 1.0})
        shared = [
            {"uid": 1, "sha256": "d" * 64, "eligible": True, "skill": 0.4},
            {"uid": 2, "sha256": "d" * 64, "eligible": True, "skill": 0.4},
        ]
        for message, bad in (
            ("draw", {**decision, "dataset_sha256": "f" * 64}),
            ("draw", {**decision, "manifest_sha256": "f" * 64}),
            ("policy", {**decision, "policy": {**decision["policy"], "min_brier_improvement": 0}}),
            ("incumbent", {**decision, "incumbent": {"uid": 65535, "sha256": "f" * 64}}),
            ("roster", {**decision, "incumbent": {"uid": 1, "sha256": incumbent}}),
            ("distinct", {**decision, "candidates": shared, "weights": {"1": 0.5, "2": 0.5}}),
            (
                "incumbent",
                {
                    **decision,
                    "candidates": [{"uid": 2, "sha256": incumbent, "eligible": True, "skill": 0.4}],
                },
            ),
            (
                "outside",
                {
                    **decision,
                    "candidates": [{"uid": 7, "sha256": "d" * 64, "eligible": True, "skill": 0.4}],
                    "weights": {"7": 1.0},
                },
            ),
            ("differ", {**decision, "weights": {"2": 0.5, "1": 0.5}}),
            ("missing", {k: v for k, v in decision.items() if k != "candidates"}),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                base_round.track_weights(track, bad, members)
        # Rotation: a decision for a draw no longer listed stops paying.
        with self.assertRaisesRegex(ValueError, "draw"):
            base_round.track_weights(
                {**track, "draws": [{**draw, "dataset_sha256": "0" * 64}]}, decision, members
            )

    def test_rows_reject_foreign_reports(self):
        cases = make_cases("x", 3)
        with self.assertRaisesRegex(ValueError, "sealed split"):
            base_round.rows_for({"dataset_sha256": "0" * 64, "miners": []}, cases)


if __name__ == "__main__":
    unittest.main()
