import math
import unittest

from zils.answer_check import CONSIDER, Thresholds, Verdict, compare, distance

OURS = {"q": {"yes": 0.80, "no": 0.20}}


def one(yes: float, no: float) -> dict:
    return {"q": {"yes": yes, "no": no}}


class AnswerCheckTests(unittest.TestCase):
    def test_identical_passes_with_zero_distance(self):
        self.assertEqual(compare(OURS, OURS), Verdict("pass", 0.0, 0.0))

    def test_tiny_gpu_drift_passes(self):
        ours = one(0.9970513893, 1 - 0.9970513893)
        theirs = one(0.9970486909, 1 - 0.9970486909)
        v = compare(ours, theirs)
        self.assertEqual(v.outcome, "pass")
        self.assertLess(v.worst, 1e-3)

    def test_fp8_like_drift_is_marginal(self):
        v = compare(OURS, one(0.88, 0.12))
        self.assertTrue(0.20 < v.mean < 0.50, v.mean)
        self.assertEqual(v.outcome, "marginal")

    def test_swapped_model_fails(self):
        v = compare(one(0.95, 0.05), one(0.05, 0.95))
        self.assertEqual(v.outcome, "fail")
        self.assertGreater(v.worst, 2.0)

    def test_zero_and_one_are_clamped(self):
        edge = one(1, 0)
        self.assertEqual(distance(edge["q"], edge["q"]), 0.0)
        d = distance(edge["q"], {"yes": 0.0, "no": 1})
        self.assertAlmostEqual(d, math.log(1e6))
        self.assertEqual(compare(edge, one(0.0, 1)).outcome, "fail")

    def test_item_fail_beats_a_low_mean(self):
        ours = {f"q{i}": {"a": 0.5, "b": 0.5} for i in range(20)}
        theirs = dict(ours, q0={"a": 0.001, "b": 0.999})
        v = compare(ours, theirs)
        self.assertLess(v.mean, 0.50)
        self.assertEqual(v.outcome, "fail")

    def test_different_keys(self):
        with self.assertRaises(ValueError):
            distance({"a": 0.5, "b": 0.5}, {"a": 0.5, "c": 0.5})
        self.assertEqual(compare(OURS, {"x": OURS["q"]}).outcome, "mismatch")
        v = compare(OURS, {"q": {"yes": 0.8, "maybe": 0.2}})
        self.assertEqual(v, Verdict("mismatch", math.inf, math.inf))

    def test_bad_probabilities_are_mismatch(self):
        for bad in (float("nan"), float("inf"), -0.1, True):
            with self.assertRaises(ValueError):
                distance({"a": 0.5, "b": 0.5}, {"a": bad, "b": 0.5})
            self.assertEqual(compare(OURS, one(bad, 0.2)).outcome, "mismatch")

    def test_top_answer_flip_is_at_least_marginal(self):
        v = compare(one(0.51, 0.49), one(0.49, 0.51))
        self.assertLess(v.mean, Thresholds().mean_pass)
        self.assertEqual(v.outcome, "marginal")

    def test_malformed_input_is_mismatch_never_raises(self):
        mismatch = Verdict("mismatch", math.inf, math.inf)
        for bad in (
            {"q": [0.8, 0.2]},
            {"q": None},
            None,
            one(10**400, 0.2),
            one("0.8", 0.2),
        ):
            self.assertEqual(compare(OURS, bad), mismatch)
            self.assertEqual(compare(bad, OURS), mismatch)

    def test_distance_raises_only_value_error(self):
        for bad in (10**400, "0.8", None, [1]):
            with self.assertRaises(ValueError):
                distance({"a": 0.5, "b": 0.5}, {"a": bad, "b": 0.5})
        with self.assertRaises(ValueError):
            distance(None, {"a": 0.5})

    def test_tie_in_ours_is_not_a_flip(self):
        tie = one(0.5, 0.5)
        for theirs in (one(0.51, 0.49), one(0.49, 0.51)):
            self.assertEqual(compare(tie, theirs).outcome, "pass")
        self.assertEqual(compare(one(0.51, 0.49), tie).outcome, "pass")

    def test_boundaries(self):
        d = distance(OURS["q"], one(0.88, 0.12)["q"])
        edge = one(0.88, 0.12)
        self.assertEqual(compare(OURS, edge, Thresholds(mean_pass=d)).outcome, "pass")
        self.assertEqual(
            compare(OURS, edge, Thresholds(mean_pass=0.0, mean_fail=d)).outcome,
            "marginal",
        )
        self.assertEqual(
            compare(OURS, edge, Thresholds(mean_pass=0.0, mean_fail=9, item_fail=d)).outcome,
            "marginal",
        )

    def test_top_answer_flip_keeps_fail(self):
        self.assertEqual(compare(one(0.95, 0.05), one(0.05, 0.95)).outcome, "fail")

    def test_tail_drift_on_many_options_is_ignored(self):
        tail = 0.2 / 253
        ours = {"top": 0.5, "second": 0.3, **{f"o{i}": tail for i in range(253)}}
        theirs = {"top": 0.505, "second": 0.297}
        rest = 1 - theirs["top"] - theirs["second"]
        weights = [0.5 if i % 2 else 2.0 for i in range(253)]  # tail drift of log(2) each way
        theirs.update({f"o{i}": rest * w / sum(weights) for i, w in enumerate(weights)})
        self.assertEqual(len(ours), 255)
        self.assertEqual(compare({"q": ours}, {"q": theirs}).outcome, "pass")
        swapped = dict(theirs, top=0.297, second=0.505)
        self.assertIn(compare({"q": ours}, {"q": swapped}).outcome, ("marginal", "fail"))

    def test_options_below_consider_on_both_sides_are_ignored(self):
        self.assertEqual(CONSIDER, 0.01)
        a = {"a": 0.7, "b": 0.295, "c": 0.005}
        b = {"a": 0.7, "b": 0.2999, "c": 0.0001}
        self.assertAlmostEqual(distance(a, b), abs(math.log(0.295 / 0.2999)) / 2, places=12)
        self.assertGreater(distance(a, dict(b, c=0.05)), 0.5)  # one side >= CONSIDER counts

    def test_fewer_than_two_considered_options_is_zero_distance(self):
        many = {"a": 0.995, **{f"o{i}": 0.005 / 9 for i in range(9)}}
        drift = {"a": 0.995, **{f"o{i}": (0.005 / 9) * (2 if i % 2 else 0.5) for i in range(9)}}
        self.assertEqual(distance(many, drift), 0.0)
        with self.assertRaises(ValueError):  # invalid tail values still raise
            distance(many, dict(drift, o1=float("nan")))


if __name__ == "__main__":
    unittest.main()
