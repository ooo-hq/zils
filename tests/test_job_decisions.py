"""A model that learns several decisions is delivered only if every decision clears the bar on its own."""

import random
import unittest

import zils
from zils import jobs

POLICY = {"min_accuracy": 0.75, "min_brier_improvement": 0.01}


def case(family, i, label, kind="choice"):
    question = (
        {"type": kind}
        if kind == "noul"
        else {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}}
    )
    return {
        "id": f"{family}-{i}",
        "group_id": f"{family}-{i}",
        "family": family,
        "state": {"i": i},
        "question": question,
        "label": label,
    }


def answer(c, right, confidence=0.9):
    keys = ["false", "true"] if c["question"]["type"] == "noul" else ["a", "b", "c"]
    chosen = c["label"] if right else next(k for k in keys if k != c["label"])
    rest = (1 - confidence) / (len(keys) - 1)
    return {
        "id": c["id"],
        "probabilities": {k: confidence if k == chosen else rest for k in keys},
        "elapsed_ms": 1,
    }


def support(team_right, urgent_right, n=100):
    """`n` tickets asked two questions: team (choice) and urgent (yes or no)."""
    cases = [case("team", i, "abc"[i % 3]) for i in range(n)] + [
        case("urgent", i, ["false", "true"][i % 2], "noul") for i in range(n)
    ]
    predictions = [
        answer(
            c,
            int(c["id"].rsplit("-", 1)[1])
            < (team_right if c["family"] == "team" else urgent_right),
        )
        for c in cases
    ]
    return cases, predictions


def row(uid, cases, predictions, **extra):
    return {
        "uid": uid,
        "sha256": f"{uid:x}" * 64,
        "status": "evaluated",
        **zils.score(cases, predictions),
        "decisions": jobs.decisions(cases, predictions),
        **extra,
    }


class DecisionScoresTest(unittest.TestCase):
    def test_each_decision_is_scored_on_its_own_examples(self):
        cases, predictions = support(95, 60)
        scores = jobs.decisions(cases, predictions)
        self.assertEqual(set(scores), {"team", "urgent"})
        self.assertAlmostEqual(scores["team"]["accuracy"], 0.95)
        self.assertAlmostEqual(scores["urgent"]["accuracy"], 0.60)
        self.assertEqual(scores["team"]["cases"], 100)
        team = [c for c in cases if c["family"] == "team"]
        alone = zils.score(team, [p for p in predictions if p["id"].startswith("team-")])
        self.assertAlmostEqual(scores["team"]["brier"], alone["brier"])
        # Pooled, the two decisions look fine together: 155 of 200.
        self.assertAlmostEqual(zils.score(cases, predictions)["accuracy"], 0.775)


class PerDecisionGateTest(unittest.TestCase):
    def setUp(self):
        cases, starting = support(40, 50)  # the starting model, before training
        self.cases, self.baseline = (
            cases,
            {
                "status": "evaluated",
                **zils.score(cases, starting),
                "decisions": jobs.decisions(cases, starting),
            },
        )

    def test_a_weak_decision_is_not_hidden_by_a_strong_one(self):
        candidate = row(1, *support(95, 60))
        self.assertGreaterEqual(
            candidate["accuracy"], POLICY["min_accuracy"]
        )  # pooled would accept it
        self.assertEqual(
            jobs.select(
                {k: v for k, v in self.baseline.items() if k != "decisions"},
                [{k: v for k, v in candidate.items() if k != "decisions"}],
                POLICY,
            )["status"],
            "accepted",
        )
        result = jobs.select(self.baseline, [candidate], POLICY)
        self.assertEqual(result["status"], "no_qualifying_model")
        self.assertEqual(result["failed_decisions"], ["urgent"])

    def test_every_decision_passing_delivers_the_model(self):
        result = jobs.select(self.baseline, [row(1, *support(95, 85))], POLICY)
        self.assertEqual(result["status"], "accepted")
        self.assertNotIn("failed_decisions", result)

    def test_the_winner_is_the_best_candidate_that_passes_every_decision(self):
        strong_but_uneven = row(1, *support(100, 70))
        even = row(2, *support(85, 80))
        result = jobs.select(self.baseline, [strong_but_uneven, even], POLICY)
        self.assertEqual((result["status"], result["uid"]), ("accepted", 2))

    def test_each_decision_must_improve_on_the_starting_model(self):
        # Urgent clears the accuracy bar but is no better than before training.
        cases, starting = support(40, 90)
        baseline = {
            "status": "evaluated",
            **zils.score(cases, starting),
            "decisions": jobs.decisions(cases, starting),
        }
        result = jobs.select(baseline, [row(1, *support(100, 90))], POLICY)
        self.assertEqual(
            (result["status"], result["failed_decisions"]), ("no_qualifying_model", ["urgent"])
        )

    def test_failed_decisions_names_every_decision_that_held_a_model_back(self):
        result = jobs.select(
            self.baseline, [row(1, *support(60, 90)), row(2, *support(95, 60))], POLICY
        )
        self.assertEqual(
            (result["status"], result["failed_decisions"]),
            ("no_qualifying_model", ["team", "urgent"]),
        )
        weak = row(3, *support(60, 60))
        weak["status"] = "rejected"
        self.assertEqual(
            jobs.select(self.baseline, [weak], POLICY),
            {"status": "no_qualifying_model", "acceptance": POLICY},
        )

    def test_one_decision_is_judged_exactly_as_before(self):
        rng = random.Random(7)
        for _ in range(200):
            cases = [case("routing", i, "abc"[i % 3]) for i in range(30)]
            starting = [answer(c, rng.random() < 0.5, 0.5 + rng.random() / 2) for c in cases]
            trained = [answer(c, rng.random() < 0.85, 0.5 + rng.random() / 2) for c in cases]
            baseline = {"status": "evaluated", **zils.score(cases, starting)}
            pooled = {
                "uid": 1,
                "sha256": "1" * 64,
                "status": "evaluated",
                **zils.score(cases, trained),
            }
            policy = {
                "min_accuracy": rng.choice([0.6, 0.8, 0.9]),
                "min_brier_improvement": rng.choice([0, 0.01, 0.05]),
            }
            with_decisions = jobs.select(
                {**baseline, "decisions": jobs.decisions(cases, starting)},
                [{**pooled, "decisions": jobs.decisions(cases, trained)}],
                policy,
            )
            self.assertEqual(with_decisions, jobs.select(baseline, [pooled], policy))

    def test_migrations_hold_each_decision_to_the_same_rule(self):
        cases, before = support(90, 90)
        baseline = {
            "status": "evaluated",
            **zils.score(cases, before),
            "decisions": jobs.decisions(cases, before),
        }
        upgrade = {
            **POLICY,
            "previous_job_id": "00000000-0000-4000-8000-000000000001",
            "base_change": True,
            "migration": "no_regression",
        }
        same = row(1, cases, before)
        self.assertEqual(jobs.select(baseline, [same], upgrade)["status"], "accepted")
        # Better overall, but urgent regressed: a migration must not make any decision worse.
        shifted = row(2, *support(100, 85))
        self.assertLess(shifted["brier"], baseline["brier"])
        result = jobs.select(baseline, [shifted], upgrade)
        self.assertEqual(
            (result["status"], result["failed_decisions"]), ("no_qualifying_model", ["urgent"])
        )

    def test_decision_scores_are_checked(self):
        candidate = row(1, *support(95, 85))
        for bad in (
            {},
            {"team": {"accuracy": 2, "brier": 0.1, "cases": 1}},
            {"team": {"accuracy": 0.9, "brier": float("nan"), "cases": 1}},
            [1],
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "decision"):
                jobs.select(self.baseline, [{**candidate, "decisions": bad}], POLICY)


if __name__ == "__main__":
    unittest.main()
