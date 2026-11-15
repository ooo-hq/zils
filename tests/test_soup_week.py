"""Weekly soup judge: which soup becomes the week's soup (pure functions, ZIL-204)."""

import unittest

from zils import soup_round, soup_week


def scripted(scores, seen=None):
    def evaluate(members):
        if seen is not None:
            seen.append(members)
        return scores[members]

    return evaluate


class AddLeftOutTest(unittest.TestCase):
    def test_a_left_out_body_joins_when_it_lifts_the_soup_by_the_margin(self):
        scores = {("a", "b"): 40.0, ("a", "b", "x"): 40.5}
        members, trace = soup_week.add_left_out(("a", "b"), ["x"], scripted(scores))
        self.assertEqual(members, ("a", "b", "x"))
        self.assertEqual(
            trace,
            [{"try": ["a", "b", "x"], "score": 40.5, "best_before": 40.0, "accepted": True}],
        )

    def test_a_body_that_lifts_the_soup_by_less_stays_out(self):
        scores = {("a", "b"): 40.0, ("a", "b", "x"): 40.4}
        members, trace = soup_week.add_left_out(("a", "b"), ["x"], scripted(scores))
        self.assertEqual(members, ("a", "b"))
        self.assertFalse(trace[0]["accepted"])

    def test_tries_are_capped_and_each_builds_on_the_soup_so_far(self):
        scores = {("a",): 40.0, ("a", "x"): 41.0, ("a", "x", "y"): 41.2}
        seen = []
        members, trace = soup_week.add_left_out(
            ("a",), ["x", "y", "z"], scripted(scores, seen), tries=2
        )
        self.assertEqual(members, ("a", "x"))
        self.assertEqual(seen, [("a",), ("a", "x"), ("a", "x", "y")])  # z is never tried
        self.assertEqual([t["accepted"] for t in trace], [True, False])

    def test_a_full_soup_and_a_member_are_not_tried(self):
        full = ("a", "b", "c", "d", "e")
        self.assertEqual(soup_week.add_left_out(full, ["x"], scripted({full: 40.0})), (full, []))
        scores = {("a", "b"): 40.0}
        self.assertEqual(soup_week.add_left_out(("a", "b"), ["a"], scripted(scores))[1], [])

    def test_owner_settings(self):
        self.assertEqual(soup_week.JOIN, soup_round.MARGIN)
        self.assertEqual(soup_week.SEAT_BAR, 0.5)
        self.assertEqual(soup_week.TRIES, 2)


class DropTest(unittest.TestCase):
    def test_a_member_goes_when_the_soup_is_better_without_it_by_the_margin(self):
        scores = {("a", "b", "c"): 40.0, ("b", "c"): 40.6, ("a", "c"): 39.0, ("a", "b"): 39.5}
        scores.update({("b",): 38.0, ("c",): 38.5})
        members, trace = soup_week.drop_members(("a", "b", "c"), scripted(scores))
        self.assertEqual(members, ("b", "c"))
        self.assertEqual(
            trace[0], {"drop": "a", "score": 40.6, "best_before": 40.0, "accepted": True}
        )
        self.assertEqual([t["accepted"] for t in trace], [True, False])  # then nothing more goes

    def test_a_member_that_adds_nothing_stays_when_the_gain_is_under_the_margin(self):
        scores = {("a", "b"): 40.0, ("a",): 40.4, ("b",): 39.0}
        members, trace = soup_week.drop_members(("a", "b"), scripted(scores))
        self.assertEqual(members, ("a", "b"))
        self.assertEqual(
            trace, [{"drop": "b", "score": 40.4, "best_before": 40.0, "accepted": False}]
        )

    def test_the_last_member_is_never_dropped(self):
        self.assertEqual(soup_week.drop_members(("a",), scripted({("a",): 40.0})), (("a",), []))


class JudgeTest(unittest.TestCase):
    def test_the_candidate_replaces_the_weekly_soup_when_it_clears_the_bar(self):
        scores = {("a", "b"): 41.0, ("a",): 40.0, ("b",): 39.0, ("p", "q"): 40.2}
        result = soup_week.judge(("a", "b"), [], scripted(scores), current=("p", "q"), bar=0.5)
        self.assertEqual(result["members"], ("a", "b"))
        self.assertTrue(result["passed"])
        self.assertEqual((result["score"], result["current_score"]), (41.0, 40.2))
        self.assertEqual(result["candidate"], ("a", "b"))

    def test_the_weekly_soup_stays_when_the_candidate_does_not_clear_the_bar(self):
        scores = {("a", "b"): 40.6, ("a",): 40.0, ("b",): 39.0, ("p", "q"): 40.2}
        result = soup_week.judge(("a", "b"), [], scripted(scores), current=("p", "q"), bar=0.5)
        self.assertEqual(result["members"], ("p", "q"))
        self.assertFalse(result["passed"])
        self.assertEqual(result["candidate"], ("a", "b"))

    def test_the_first_week_compares_with_the_stock_score(self):
        scores = {("a", "b"): 41.0, ("a",): 40.0, ("b",): 39.0}
        result = soup_week.judge(
            ("a", "b"), [], scripted(scores), current=None, bar=0.5, stock=30.0
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["current_score"], 30.0)
        weak = soup_week.judge(("a", "b"), [], scripted(scores), current=None, bar=0.5, stock=40.8)
        self.assertIsNone(weak["members"])  # no weekly soup yet: the release stays
        with self.assertRaises(ValueError):
            soup_week.judge(("a", "b"), [], scripted(scores), current=None, bar=0.5)

    def test_the_same_soup_as_last_week_stays_without_a_comparison(self):
        scores = {("a", "b"): 41.0, ("a",): 40.0, ("b",): 39.0}
        result = soup_week.judge(("a", "b"), [], scripted(scores), current=("a", "b"), bar=0.5)
        self.assertEqual(result["members"], ("a", "b"))
        self.assertFalse(result["passed"])

    def test_left_out_bodies_are_added_then_weak_members_dropped_before_the_comparison(self):
        scores = {
            ("a", "b"): 40.0,
            ("a", "b", "x"): 41.0,  # x joins
            ("b", "x"): 41.6,  # a goes: the soup is 0.6 better without it
            ("a", "x"): 40.0,
            ("b",): 39.0,
            ("x",): 41.0,
        }
        result = soup_week.judge(
            ("a", "b"), ["x"], scripted(scores), current=None, bar=0.5, stock=30.0
        )
        self.assertEqual(result["members"], ("b", "x"))
        self.assertEqual([t["accepted"] for t in result["trace"]], [True, True, False])


if __name__ == "__main__":
    unittest.main()
