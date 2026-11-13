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
        self.assertEqual(soup_round.MARGIN, 0.5)
        self.assertEqual(soup_round.MAX_MEMBERS, 5)
        self.assertEqual(soup_round.SEAT_SHARE, 0.8)
        self.assertEqual(soup_round.CROWN_SHARE, 0.2)
        self.assertEqual(soup_round.FLOOR, 0.8)
        self.assertEqual(soup_round.CROWN_MARGIN, 1.2)
        self.assertEqual(
            (soup_round.HOUSE_RATE, soup_round.SCOUT_RATE, soup_round.TRAINER_RATE),
            (0.0, 0.25, 1.0),
        )

    def test_margin_blocks_a_small_gain_and_accepts_a_full_one(self):
        solo = {"a": 3.0, "b": 2.0, "c": 1.0}
        scores = {("a", "b"): 3.4, ("a", "c"): 3.5}
        soup, trace = soup_round.greedy(
            solo, scripted(scores), order_key=commit_order("a", "b", "c")
        )
        self.assertEqual(soup, ("a", "c"))
        self.assertEqual([t["accepted"] for t in trace], [False, True])

    def test_margin_is_measured_from_the_current_soup(self):
        solo = {"a": 3.0, "b": 2.0, "c": 1.0}
        scores = {("a", "b"): 3.6, ("a", "b", "c"): 4.0}
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


class BaseContributionsTest(unittest.TestCase):
    """ZIL-117: a fine-tune is credited with what it adds over its base, not the base's value."""

    FULL = {("a", "t"): 60.0, ("a",): 55.0}

    def credit(self, scores, bases, members=("a", "t"), **options):
        score = scripted({**self.FULL, **scores})
        options = {"base_score": score, **options}
        return soup_round.base_contributions(members, score, bases, **options)

    def test_a_member_without_bases_keeps_its_leave_one_out_contribution(self):
        got, unscored = self.credit({("t",): 50.0}, {})
        self.assertEqual((got, unscored), ({"a": 10.0, "t": 5.0}, set()))

    def test_a_fine_tune_is_credited_with_the_soup_minus_the_soup_with_its_base_in_its_place(self):
        got, _ = self.credit({("t",): 50.0, ("a", "b"): 58.5}, {"t": ["b"]})
        self.assertAlmostEqual(got["t"], 1.5)
        self.assertAlmostEqual(got["a"], 10.0)  # other members are not affected

    def test_a_base_that_would_hurt_the_soup_does_not_raise_the_credit(self):
        got, _ = self.credit({("t",): 50.0, ("a", "b"): 54.0}, {"t": ["b"]})
        self.assertAlmostEqual(got["t"], 5.0)  # leave-one-out: the soup would not have taken b

    def test_the_hardest_of_several_bases_counts(self):
        scores = {("t",): 50.0, ("a", "b"): 58.5, ("a", "release"): 59.5}
        got, _ = self.credit(scores, {"t": ["b", "release"]})
        self.assertAlmostEqual(got["t"], 0.5)

    def test_a_base_already_seated_leaves_the_leave_one_out_contribution(self):
        got, _ = self.credit({("t",): 50.0}, {"t": ["a"]})
        self.assertAlmostEqual(got["t"], 5.0)

    def test_a_base_that_cannot_be_scored_is_reported(self):
        got, unscored = self.credit({("t",): 50.0}, {"t": [None]})
        self.assertEqual(unscored, {"t"})
        got, unscored = self.credit({("t",): 50.0}, {"t": ["b"]}, base_score=lambda members: None)
        self.assertEqual(unscored, {"t"})

    def test_a_one_member_soup_of_a_fine_tune_is_compared_with_its_base_alone(self):
        score = scripted({("t",): 60.0, ("b",): 58.0})
        got, _ = soup_round.base_contributions(
            ("t",), score, {"t": ["b"]}, base_score=score, baseline=40.0
        )
        self.assertAlmostEqual(got["t"], 2.0)

    def test_bases_of_bodies_outside_the_soup_are_ignored(self):
        got, unscored = self.credit({("t",): 50.0}, {"z": ["b"]})
        self.assertEqual((set(got), unscored), ({"a", "t"}, set()))


class SeatWeightsTest(unittest.TestCase):
    T, S, H = soup_round.TRAINER_RATE, soup_round.SCOUT_RATE, soup_round.HOUSE_RATE

    def test_each_seat_is_a_fixed_fifth_of_the_seat_share(self):
        # A seat pays SEAT_SHARE / MAX_MEMBERS whoever else is seated; unfilled seats burn.
        got = soup_round.seat_weights(
            ("a", "b", "c"),
            {"a": 1.0, "b": 0.8, "c": 0.79},
            {"a": self.T, "b": self.T, "c": self.T},
        )
        self.assertEqual(set(got), {"a", "b"})
        self.assertAlmostEqual(got["a"], 0.16)
        self.assertAlmostEqual(got["b"], 0.16)

    def test_a_lone_seat_is_still_one_fifth(self):
        got = soup_round.seat_weights(("a",), {"a": 1.0}, {"a": self.T})
        self.assertAlmostEqual(got["a"], soup_round.SEAT_SHARE / soup_round.MAX_MEMBERS)

    def test_a_full_soup_pays_the_whole_seat_share(self):
        names = tuple("abcde")
        got = soup_round.seat_weights(
            names, dict.fromkeys(names, 1.0), dict.fromkeys(names, self.T)
        )
        self.assertAlmostEqual(sum(got.values()), soup_round.SEAT_SHARE)

    def test_scout_is_paid_a_quarter_of_a_trainer_seat(self):
        got = soup_round.seat_weights(("a", "b"), {"a": 1.0, "b": 1.0}, {"a": self.T, "b": self.S})
        self.assertAlmostEqual(got["a"], 0.16)
        self.assertAlmostEqual(got["b"], 0.04)

    def test_house_ingredient_takes_a_seat_but_earns_nothing(self):
        got = soup_round.seat_weights(("h", "a"), {"h": 2.0, "a": 1.0}, {"h": self.H, "a": self.T})
        self.assertEqual(got["h"], 0.0)
        self.assertAlmostEqual(got["a"], 0.16)

    def test_nobody_above_the_floor_pays_nothing(self):
        got = soup_round.seat_weights(("a", "b"), {"a": 0.7, "b": -1.0}, {"a": self.T, "b": self.T})
        self.assertEqual(got, {})

    def test_rejects_an_unknown_rate(self):
        with self.assertRaises(ValueError):
            soup_round.seat_weights(("a",), {"a": 1.0}, {"a": 0.5})


class CrownWeightTest(unittest.TestCase):
    def test_best_non_house_solo_body_wins_at_its_rate(self):
        solo = {"house": 3.0, "scout": 4.5, "trainer": 3.5}
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
        for best, paid in ((6.19, False), (6.2, True), (7.0, True), (4.0, False)):
            with self.subTest(best=best):
                solo = {"h1": 4.5, "h2": 5.0, "t": best}
                got = soup_round.crown_weight(solo, rates, {"h1", "h2"}, order_key=order)
                self.assertEqual(got, {"t": 0.2} if paid else {})

    def test_a_soup_fine_tune_needs_the_margin_over_the_current_soup(self):
        rates, order = {"h": 0.0, "t": 1.0}, commit_order("h", "t")
        for best, paid in ((6.1, False), (6.2, True)):
            with self.subTest(best=best):
                got = soup_round.crown_weight(
                    {"h": 4.0, "t": best}, rates, {"h"}, soup_score=5.0, order_key=order
                )
                self.assertEqual(got, {"t": 0.2} if paid else {})

    def test_first_round_without_a_soup_uses_the_house_bodies_only(self):
        got = soup_round.crown_weight(
            {"h": 4.0, "t": 5.2}, {"h": 0.0, "t": 1.0}, {"h"}, soup_score=None,
            order_key=commit_order("h", "t"),
        )  # fmt: skip
        self.assertEqual(got, {"t": 0.2})

    def test_only_house_bodies_means_no_crown(self):
        self.assertEqual(soup_round.crown_weight({"h": 5.0}, {"h": 0.0}, {"h"}, order_key=str), {})


class FineTuneBasesTest(unittest.TestCase):
    """ZIL-117: which members are fine-tunes, and of what. A fine-tune of a listed public model is
    paid as a scout, since listed models are not graded."""

    def bases(self, shares):
        return soup_round.fine_tune_bases(shares, house={"h"})

    def test_a_body_mostly_made_of_a_base_names_it(self):
        self.assertEqual(self.bases({"b": {"house:h": 0.9}}), ({"b": ["house:h"]}, set()))

    def test_every_base_explaining_enough_is_named(self):
        shares = {"b": {"soup": 0.4, "house:h": 0.4, "release": 0.05}}
        self.assertEqual(self.bases(shares), ({"b": ["soup", "house:h"]}, set()))

    def test_a_body_mostly_its_own_has_no_bases(self):
        self.assertEqual(self.bases({"b": {"house:h": 0.45}}), ({}, set()))

    def test_a_fine_tune_of_listed_models_is_a_scout(self):
        self.assertEqual(self.bases({"b": {"public:open/x": 0.6, "house:h": 0.3}}), ({}, {"b"}))
        # A listed model that is only part of the base still makes the body a scout's.
        mixed = {"b": {"house:h": 0.56, "public:open/x": 0.32}}
        self.assertEqual(self.bases(mixed), ({}, {"b"}))
        # A listed model under BASE_MIN of a mostly-house body does not.
        slight = {"b": {"house:h": 0.85, "public:open/x": 0.05}}
        self.assertEqual(self.bases(slight), ({"b": ["house:h"]}, set()))

    def test_house_bodies_have_no_bases(self):
        self.assertEqual(self.bases({"h": {"house:h": 1.0}}), ({}, set()))


class KeepCurrentTest(unittest.TestCase):
    """ZIL-193: the current soup stays unless greedy's soup beats it by the greedy margin, so a light
    re-tune of the soup cannot take its place (and its members' seats) for a gain within noise."""

    SOLO = {"retune": 56.26, "v2": 55.24, "s1": 54.0, "new": 60.0}

    def keep(self, members, current=("s1", "v2"), soup_bar=56.12, **kwargs):
        scores = {("s1", "v2"): 56.12, ("new",): 60.0, ("retune",): 56.26, ("new", "v2"): 60.7}
        options = {"solo": self.SOLO, "evaluate": scores.__getitem__, **kwargs}
        return soup_round.keep_current(members, current=current, soup_bar=soup_bar, **options)

    def test_a_soup_within_the_margin_of_the_current_one_keeps_the_current(self):
        # d3-mini-2026-11-09: a re-tune of qhoenix v2 + Charlie320 s1 alone, 56.26 against 56.12.
        members, row = self.keep(("retune",))
        self.assertEqual(members, ("s1", "v2"))
        self.assertEqual(row["try"], ["retune"])
        self.assertEqual((row["score"], row["best_before"], row["accepted"]), (56.26, 56.12, False))

    def test_a_soup_better_by_the_margin_replaces_the_current(self):
        self.assertEqual(self.keep(("new",)), (("new",), {
            "try": ["new"], "score": 60.0, "best_before": 56.12, "accepted": True, "versus": "current",
        }))  # fmt: skip
        self.assertEqual(self.keep(("new",), soup_bar=59.5)[0], ("new",))  # exactly the margin

    def test_greedy_soup_equal_to_the_current_needs_no_comparison(self):
        self.assertEqual(self.keep(("s1", "v2")), (("s1", "v2"), None))

    def test_without_a_gradable_current_soup_greedy_stands(self):
        self.assertEqual(self.keep(("retune",), current=()), (("retune",), None))
        self.assertEqual(self.keep(("retune",), soup_bar=None), (("retune",), None))
        # A member left the pool and was not graded: the current soup cannot be served again.
        self.assertEqual(self.keep(("retune",), current=("gone", "v2")), (("retune",), None))


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
