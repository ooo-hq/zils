"""Soup round selection and pay: greedy soup, leave-one-out, seat and crown weights (pure functions)."""

import unittest

from zils import soup_round


def scripted(scores, seen=None):
    def evaluate(members):
        if seen is not None:
            seen.append(members)
        return scores[members]

    return evaluate


def commit_order(*bodies):
    return {body: index for index, body in enumerate(bodies)}.__getitem__


class GreedyTest(unittest.TestCase):
    def test_adds_in_solo_order_and_keeps_only_non_decreasing_steps(self):
        # Ported from the 29 Oct prototype (margin 0 reproduces its "do not drop" rule).
        solo = {"a": 3.0, "b": 1.0, "c": 2.0, "d": -1.0}
        scores = {("a", "c"): 3.5, ("a", "b", "c"): 3.2, ("a", "c", "d"): 3.5}
        seen = []
        soup, trace = soup_round.greedy(
            solo, scripted(scores, seen), margin=0, order_key=commit_order("a", "b", "c", "d")
        )
        self.assertEqual(soup, ("a", "c", "d"))
        self.assertEqual(seen, [("a", "c"), ("a", "b", "c"), ("a", "c", "d")])
        self.assertEqual([t["accepted"] for t in trace], [True, False, True])
        self.assertEqual(trace[0]["score"], 3.5)

    def test_owner_constants(self):
        self.assertEqual(soup_round.MARGIN, 0.3)
        self.assertEqual(soup_round.MAX_MEMBERS, 5)
        self.assertEqual(soup_round.SEAT_SHARE, 0.8)
        self.assertEqual(soup_round.CROWN_SHARE, 0.2)
        self.assertEqual(soup_round.FLOOR, 0.3)
        self.assertEqual(
            (soup_round.HOUSE_RATE, soup_round.SCOUT_RATE, soup_round.TRAINER_RATE),
            (0.0, 0.25, 1.0),
        )

    def test_margin_blocks_a_small_gain_and_accepts_a_full_one(self):
        solo = {"a": 3.0, "b": 2.0, "c": 1.0}
        scores = {("a", "b"): 3.2, ("a", "c"): 3.3}
        soup, trace = soup_round.greedy(
            solo, scripted(scores), order_key=commit_order("a", "b", "c")
        )
        self.assertEqual(soup, ("a", "c"))
        self.assertEqual([t["accepted"] for t in trace], [False, True])

    def test_margin_is_measured_from_the_current_soup(self):
        solo = {"a": 3.0, "b": 2.0, "c": 1.0}
        scores = {("a", "b"): 3.4, ("a", "b", "c"): 3.6}
        soup, _ = soup_round.greedy(solo, scripted(scores), order_key=commit_order("a", "b", "c"))
        self.assertEqual(soup, ("a", "b"))

    def test_stops_at_five_members(self):
        bodies = "abcdefg"
        solo = {b: 10.0 - i for i, b in enumerate(bodies)}
        seen = []

        def evaluate(members):
            seen.append(members)
            return 10.0 + len(members)

        soup, trace = soup_round.greedy(solo, evaluate, order_key=commit_order(*bodies))
        self.assertEqual(soup, tuple("abcde"))
        self.assertEqual(len(trace), 4)
        self.assertNotIn("f", {b for members in seen for b in members})

    def test_solo_ties_break_by_commitment_order(self):
        solo = {"late": 3.0, "early": 3.0, "low": 1.0}
        key = commit_order("early", "late", "low")
        seen = []
        soup, _ = soup_round.greedy(
            solo, scripted({("early", "late"): 2.0, ("early", "low"): 2.0}, seen), order_key=key
        )
        self.assertEqual(soup, ("early",))
        self.assertEqual(seen, [("early", "late"), ("early", "low")])
        again, _ = soup_round.greedy(
            dict(reversed(solo.items())),
            scripted({("early", "late"): 2.0, ("early", "low"): 2.0}),
            order_key=key,
        )
        self.assertEqual(again, soup)

    def test_empty_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            soup_round.greedy({}, scripted({}), order_key=str)


class LeaveOneOutTest(unittest.TestCase):
    def test_contribution_is_soup_minus_soup_without_member(self):
        scores = {("a", "b", "c"): 5.0, ("b", "c"): 4.0, ("a", "c"): 4.8, ("a", "b"): 5.1}
        got = soup_round.leave_one_out(("a", "b", "c"), scripted(scores))
        self.assertEqual(set(got), {"a", "b", "c"})
        self.assertAlmostEqual(got["a"], 1.0)
        self.assertAlmostEqual(got["b"], 0.2)
        self.assertAlmostEqual(got["c"], -0.1)

    def test_single_member_is_measured_against_the_chassis_baseline(self):
        got = soup_round.leave_one_out(("a",), scripted({("a",): 3.0}), baseline=2.5)
        self.assertEqual(got, {"a": 0.5})
        with self.assertRaises(ValueError):
            soup_round.leave_one_out(("a",), scripted({("a",): 3.0}))


class SeatWeightsTest(unittest.TestCase):
    T, S, H = soup_round.TRAINER_RATE, soup_round.SCOUT_RATE, soup_round.HOUSE_RATE

    def test_members_above_the_floor_split_the_seat_share_equally(self):
        got = soup_round.seat_weights(
            ("a", "b", "c"),
            {"a": 1.0, "b": 0.3, "c": 0.29},
            {"a": self.T, "b": self.T, "c": self.T},
        )
        self.assertEqual(set(got), {"a", "b"})
        self.assertAlmostEqual(got["a"], 0.4)
        self.assertAlmostEqual(got["b"], 0.4)

    def test_scout_is_paid_a_quarter_of_a_trainer_seat(self):
        got = soup_round.seat_weights(("a", "b"), {"a": 1.0, "b": 1.0}, {"a": self.T, "b": self.S})
        self.assertAlmostEqual(got["a"], 0.4)
        self.assertAlmostEqual(got["b"], 0.1)
        self.assertLess(sum(got.values()), soup_round.SEAT_SHARE)

    def test_house_ingredient_takes_a_seat_but_earns_nothing(self):
        got = soup_round.seat_weights(("h", "a"), {"h": 2.0, "a": 1.0}, {"h": self.H, "a": self.T})
        self.assertEqual(got["h"], 0.0)
        self.assertAlmostEqual(got["a"], 0.4)

    def test_nobody_above_the_floor_pays_nothing(self):
        got = soup_round.seat_weights(("a", "b"), {"a": 0.2, "b": -1.0}, {"a": self.T, "b": self.T})
        self.assertEqual(got, {})

    def test_rejects_an_unknown_rate(self):
        with self.assertRaises(ValueError):
            soup_round.seat_weights(("a",), {"a": 1.0}, {"a": 0.5})


class CrownWeightTest(unittest.TestCase):
    def test_best_non_house_solo_body_wins_at_its_rate(self):
        solo = {"house": 3.0, "scout": 4.0, "trainer": 3.5}
        rates = {"house": 0.0, "scout": 0.25, "trainer": 1.0}
        got = soup_round.crown_weight(
            solo, rates, {"house"}, order_key=commit_order("house", "scout", "trainer")
        )
        self.assertEqual(set(got), {"scout"})
        self.assertAlmostEqual(got["scout"], 0.05)

    def test_ties_go_to_the_earlier_commitment(self):
        got = soup_round.crown_weight(
            {"late": 4.0, "early": 4.0}, {"late": 1.0, "early": 1.0}, set(),
            order_key=commit_order("early", "late"),
        )  # fmt: skip
        self.assertEqual(got, {"early": 0.2})

    def test_the_crown_must_beat_every_house_body_by_the_margin(self):
        rates = {"h1": 0.0, "h2": 0.0, "t": 1.0}
        order = commit_order("h1", "h2", "t")
        for best, paid in ((5.29, False), (5.3, True), (6.0, True), (4.0, False)):
            with self.subTest(best=best):
                solo = {"h1": 4.5, "h2": 5.0, "t": best}
                got = soup_round.crown_weight(solo, rates, {"h1", "h2"}, order_key=order)
                self.assertEqual(got, {"t": 0.2} if paid else {})

    def test_only_house_bodies_means_no_crown(self):
        self.assertEqual(soup_round.crown_weight({"h": 5.0}, {"h": 0.0}, {"h"}, order_key=str), {})


class SoupWeightsTest(unittest.TestCase):
    ROSTER = {"3": {"hotkey": "hk-a"}, "7": {"hotkey": "hk-b"}, "9": {"hotkey": "hk-c"}}

    def decision(self, members, crown=None):
        return {
            "pool_id": "z3-mini",
            "members": [
                {
                    "hotkey": hk,
                    "body_sha256": sha,
                    "rate": rate,
                    "contribution": c,
                    "seat_weight": w,
                }
                for hk, sha, rate, c, w in members
            ],
            "crown": crown,
        }

    def test_maps_hotkeys_to_uids(self):
        got = soup_round.soup_weights(
            self.decision([("hk-a", "1" * 64, 1.0, 1.0, 0.4), ("hk-b", "2" * 64, 1.0, 0.5, 0.4)]),
            self.ROSTER,
        )
        self.assertEqual(got, {3: 0.4, 7: 0.4})

    def test_merges_seat_and_crown_for_one_hotkey(self):
        crown = {"hotkey": "hk-a", "body_sha256": "1" * 64, "rate": 1.0, "weight": 0.2}
        got = soup_round.soup_weights(
            self.decision(
                [("hk-a", "1" * 64, 1.0, 1.0, 0.4), ("hk-a", "3" * 64, 0.25, 0.5, 0.1)], crown
            ),
            self.ROSTER,
        )
        self.assertEqual(set(got), {3})
        self.assertAlmostEqual(got[3], 0.7)

    def test_drops_unregistered_hotkeys_and_keeps_the_total_at_most_one(self):
        crown = {"hotkey": "hk-gone", "body_sha256": "4" * 64, "rate": 1.0, "weight": 0.2}
        got = soup_round.soup_weights(
            self.decision(
                [
                    ("hk-a", "1" * 64, 1.0, 1.0, 0.4),
                    ("hk-gone", "4" * 64, 1.0, 1.0, 0.4),
                    ("house", "5" * 64, 0.0, 2.0, 0.0),
                ],
                crown,
            ),
            self.ROSTER,
        )
        self.assertEqual(got, {3: 0.4})
        self.assertLessEqual(sum(got.values()), 1.0)

    def test_empty_decision_pays_nobody(self):
        self.assertEqual(soup_round.soup_weights(self.decision([]), self.ROSTER), {})

    def test_rejects_a_decision_paying_more_than_the_pool(self):
        with self.assertRaises(ValueError):
            soup_round.soup_weights(
                self.decision([("hk-a", "1" * 64, 1.0, 1.0, 0.8)], {
                    "hotkey": "hk-b", "body_sha256": "2" * 64, "rate": 1.0, "weight": 0.3,
                }),
                self.ROSTER,
            )  # fmt: skip


if __name__ == "__main__":
    unittest.main()
