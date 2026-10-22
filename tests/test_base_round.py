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
        decision = base_round.archival_decide(
            cases, 0, cal_rows, test_rows, {"bootstrap_resamples": 300}
        )
        by_uid = {c["uid"]: c for c in decision["candidates"]}
        self.assertTrue(by_uid[1]["eligible"])
        self.assertFalse(by_uid[2]["eligible"], by_uid[2])
        self.assertFalse(by_uid[3]["eligible"])
        self.assertLess(by_uid[3]["improvement"], 0)
        self.assertEqual(decision["pending_winner"]["uid"], 1)
        # The exam names a challenger; it carries no payout of its own (king of the hill).
        self.assertNotIn("weights", decision)
        self.assertTrue(0.25 <= decision["incumbent"]["temperature"] <= 4)
        # Confirmation needs a disjoint draw where the same checkpoint stays eligible.
        second_test = make_cases("second")
        second_rows = base_round.rows_for(
            report(second_test, {u: predict(second_test, m, 300 + u) for u, m in masses.items()}),
            second_test,
        )
        second = base_round.archival_decide(
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
        failed = base_round.archival_decide(
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
            "pending_winner": {"uid": 2, "sha256": "d" * 64},
        }
        # Unowned stock base: an eligible challenger is not paid, and nobody else is either.
        self.assertEqual(base_round.track_weights(track, decision, members), {})
        self.assertEqual(base_round.king_weights(track, members), {})
        nobody = {**decision, "pending_winner": None}
        self.assertEqual(base_round.track_weights(track, nobody, members), {})
        # A crowned king is paid with no challenger, and keeps the whole share while a challenger
        # (pending, or even confirmed) waits for promotion.
        crowned = {**track, "kings": ["a"]}
        self.assertEqual(base_round.track_weights(crowned, nobody, members), {1: 1.0})
        self.assertEqual(base_round.track_weights(crowned, decision, members), {1: 1.0})
        confirmed = {**decision, "confirmed_winner": decision["pending_winner"]}
        self.assertEqual(base_round.track_weights(crowned, confirmed, members), {1: 1.0})
        self.assertEqual(base_round.king_weights(crowned, members), {1: 1.0})
        # A decision written before king of the hill still carries payout weights; they are ignored.
        legacy = {**decision, "weights": {"2": 1.0}}
        self.assertEqual(base_round.track_weights(crowned, legacy, members), {1: 1.0})
        # Promotion crowns the challenger: the owner's PR sets the track to its hash and prepends its
        # hotkey to the history. Decisions scored against the old incumbent stop counting; the new
        # king and the previous one share the base.
        promoted = {**track, "incumbent_sha256": "d" * 64, "kings": ["b", "a"]}
        self.assertEqual(base_round.king_weights(promoted, members), {2: 0.5, 1: 0.5})
        with self.assertRaisesRegex(ValueError, "different incumbent"):
            base_round.track_weights(promoted, decision, members)
        rescored = {
            **decision,
            "incumbent": {"uid": 65535, "sha256": "d" * 64},
            "candidates": [{"uid": 1, "sha256": "e" * 64, "eligible": False, "skill": 0.3}],
            "pending_winner": None,
        }
        self.assertEqual(base_round.track_weights(promoted, rescored, members), {2: 0.5, 1: 0.5})
        # A copy of the king is never an eligible challenger.
        copy = {
            **rescored,
            "candidates": [{"uid": 1, "sha256": "d" * 64, "eligible": True, "skill": 0.4}],
            "pending_winner": {"uid": 1, "sha256": "d" * 64},
        }
        with self.assertRaisesRegex(ValueError, "differ from the incumbent"):
            base_round.track_weights(promoted, copy, members)
        for bad in ({"kings": "a"}, {"kings": [""]}, {"burn_uid": "0"}, {"incumbent_uid": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                base_round.validate_track({**track, **bad})
        self.assertEqual(base_round.validate_track(track)["burn_uid"], 0)
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
            ("distinct", {**decision, "candidates": shared}),
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
                    "pending_winner": {"uid": 7, "sha256": "d" * 64},
                },
            ),
            ("not an eligible", {**decision, "pending_winner": {"uid": 1, "sha256": "e" * 64}}),
            ("missing", {k: v for k, v in decision.items() if k != "candidates"}),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                base_round.track_weights(track, bad, members)
        # Rotation: a decision for a draw no longer listed stops paying.
        with self.assertRaisesRegex(ValueError, "draw"):
            base_round.track_weights(
                {**track, "draws": [{**draw, "dataset_sha256": "0" * 64}]}, decision, members
            )

    def test_king_history_pays_the_last_five_distinct_kings(self):
        draw = {"manifest_sha256": "a" * 64, "dataset_sha256": "b" * 64}
        track = {"share": 0.7, "incumbent_sha256": "c" * 64, "draws": [draw]}
        members = {str(u): {"hotkey": f"k{u}"} for u in range(1, 8)}
        # Newest first, de-duplicated by hotkey, five slots; an unregistered king's slot is shared
        # by the rest, and kings past the fifth distinct one are not paid (Teutonic MAIN_SHARES[0]).
        kings = ["k7", "k6", "k6", "gone", "k5", "k4", "k3", "k2"]
        paid = base_round.king_weights({**track, "kings": kings}, members)
        self.assertEqual(paid, {7: 0.25, 6: 0.25, 5: 0.25, 4: 0.25})
        three = base_round.king_weights({**track, "kings": ["k3", "k2", "k1"]}, members)
        self.assertEqual(three, {3: 1 / 3, 2: 1 / 3, 1: 1 / 3})
        # Paid by hotkey: a king re-registered at another uid is paid there; its old uid is not.
        moved = {**members, "9": {"hotkey": "k1"}, "1": {"hotkey": "newcomer"}}
        self.assertEqual(base_round.king_weights({**track, "kings": ["k1"]}, moved), {9: 1.0})
        # Every king gone, or no king at all (the stock base): nobody is paid.
        self.assertEqual(base_round.king_weights({**track, "kings": ["gone"]}, members), {})
        self.assertEqual(base_round.king_weights(track, members), {})

    def test_rows_reject_foreign_reports(self):
        cases = make_cases("x", 3)
        with self.assertRaisesRegex(ValueError, "sealed split"):
            base_round.rows_for({"dataset_sha256": "0" * 64, "miners": []}, cases)


if __name__ == "__main__":
    unittest.main()


class DecisionElectionTest(unittest.TestCase):
    def rows(self, cases, uid, value):
        from tests.test_base_metrics import predictions

        return {
            "uid": uid,
            "sha256": f"{uid:064x}",
            "predictions": predictions(cases, confidence=value),
        }

    def test_fixed_production_policy_and_boundary_gates(self):
        from unittest.mock import patch

        from tests.test_base_exam import inputs
        from zils import base_metrics

        cases = inputs()[0]
        rows = {s: {u: self.rows(cases[s], u, 0.8) for u in (0, 1, 2)} for s in cases}
        claims = {
            u: {"sha256": f"{u:064x}", "block": 100 - u, "hotkey": f"miner-{u}"} for u in (1, 2)
        }
        draw = {
            "manifest_sha256": "a" * 64,
            "dataset_sha256": base_round.dataset_hash(cases["test"]),
        }
        with (
            patch.object(base_round, "fit_temperature", return_value=1),
            patch.object(
                base_metrics, "score", return_value={"score": 51, "families": {}, "categories": {}}
            ),
            patch.object(
                base_metrics,
                "paired_improvement",
                return_value={"improvement": 3.0, "lower_95": 0.01, "upper_95": 4.0},
            ) as paired,
        ):
            decision = base_round.decide(
                cases, 0, rows["calibration"], rows["test"], draw_identity=draw, commitments=claims
            )
            self.assertTrue(all(c["eligible"] for c in decision["candidates"]))
            self.assertEqual(decision["pending_winner"]["uid"], 2)
            self.assertEqual(len({call.kwargs["seed"] for call in paired.call_args_list}), 1)
            paired.return_value["lower_95"] = 0
            decision = base_round.decide(
                cases, 0, rows["calibration"], rows["test"], draw_identity=draw, commitments=claims
            )
            self.assertIsNone(decision["pending_winner"])
        for policy in (
            {"min_brier_improvement": 0.01},
            {"score_version": 2},
            {"score_version": True},
            {"bootstrap_resamples": 1},
            {"bootstrap_resamples": 0},
            {"bootstrap_resamples": 2000.0},
            {"max_brier_increase": float("nan")},
            {"min_score_improvement": 0.1},
        ):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                base_round.validate_track(
                    {"share": 1, "incumbent_sha256": "0" * 64, "draws": [draw], "policy": policy}
                )

    def test_bad_candidate_is_isolated_and_public_fields_are_aggregates(self):
        from unittest.mock import patch

        from tests.test_base_exam import inputs
        from zils import base_metrics

        cases = inputs()[0]
        rows = {s: {u: self.rows(cases[s], u, 0.8) for u in (0, 1, 2)} for s in cases}
        rows["test"][1]["predictions"] = []
        claims = {u: {"sha256": f"{u:064x}", "block": u, "hotkey": f"miner-{u}"} for u in (1, 2)}
        draw = {
            "manifest_sha256": "a" * 64,
            "dataset_sha256": base_round.dataset_hash(cases["test"]),
        }
        with (
            patch.object(base_round, "fit_temperature", return_value=1),
            patch.object(
                base_metrics,
                "paired_improvement",
                return_value={"improvement": 4.0, "lower_95": 1.0, "upper_95": 5.0},
            ),
        ):
            decision = base_round.decide(
                cases, 0, rows["calibration"], rows["test"], draw_identity=draw, commitments=claims
            )
            self.assertFalse(decision["candidates"][0]["eligible"])
            self.assertTrue(decision["candidates"][1]["eligible"])
            decision["incumbent"]["evaluation"] = {"path": "/private/sealed"}
            public = json.dumps(base_round.public_decision(decision))
            for private in (
                "reasoning-known",
                "/private/sealed",
                "predictions",
                cases["test"][0]["id"],
            ):
                self.assertNotIn(private, public)
            rows["test"][0]["predictions"] = []
            with self.assertRaises(ValueError):
                base_round.decide(
                    cases,
                    0,
                    rows["calibration"],
                    rows["test"],
                    draw_identity=draw,
                    commitments=claims,
                )

    def test_confirmation_uses_each_candidates_time_and_hotkey(self):
        import copy
        from unittest.mock import patch

        from tests.test_base_exam import inputs
        from zils import base_metrics

        def row(uid, sha, start, end, hotkey="miner"):
            return {
                "uid": uid,
                "sha256": sha,
                "eligible": True,
                "hotkey": hotkey,
                "commitment_block": 20,
                "evaluation": {
                    s: {"started_at": start, "completed_at": end} for s in base_round.SPLITS
                },
            }

        active = {
            "pool_id": "d3-mini",
            "model": {},
            "policy": base_round.DEFAULT_POLICY,
            "incumbent_sha256": "a" * 64,
        }
        first = {
            **active,
            "metric_version": base_metrics.VERSION,
            "incumbent": row(0, "a" * 64, 90, 100),
            "candidates": [row(1, "b" * 64, 190, 200), row(2, "c" * 64, 290, 300)],
        }
        second = copy.deepcopy(first)
        second["incumbent"] = row(0, "a" * 64, 43400, 43401)
        second["candidates"] = [row(9, "b" * 64, 43400, 43401), row(2, "c" * 64, 43400, 43401)]
        audits = [({}, inputs(p)[0]) for p in ("first", "second")]
        with patch.object(
            base_round, "verify_decision", side_effect=[(audits[0], first), (audits[1], second)]
        ):
            self.assertEqual(
                [c["uid"] for c in base_round.confirm_candidates(first, second, active=active)], [9]
            )
        second["candidates"][0]["hotkey"] = "different-miner"
        with patch.object(
            base_round, "verify_decision", side_effect=[(audits[0], first), (audits[1], second)]
        ):
            self.assertEqual(base_round.confirm_candidates(first, second, active=active), [])
        with self.assertRaisesRegex(ValueError, "active"):
            base_round.confirm_candidates(
                first, second, active={**active, "incumbent_sha256": "d" * 64}
            )
