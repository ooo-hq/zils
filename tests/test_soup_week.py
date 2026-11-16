"""Weekly soup judge: which soup becomes the week's soup (pure functions, ZIL-204)."""

import hashlib
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


def private(members, solo, rates=None):
    return {
        "round_id": "d3-mini-2026-11-22",
        "members": [{"body_sha256": m} for m in members],
        "pool": sorted(solo),
        "solo": solo,
        "rates": rates or dict.fromkeys(solo, 1.0),
    }


KNOWN = {name: {"hotkey": f"hk-{name}"} for name in "abxyh"}


class LeftOutTest(unittest.TestCase):
    def test_pool_bodies_outside_the_soup_best_solo_first(self):
        record = private(["a", "b"], {"a": 40.0, "b": 39.0, "x": 41.0, "y": 41.5})
        self.assertEqual(soup_week.left_out(record), ["y", "x"])

    def test_unpaid_bodies_are_never_tried(self):
        record = private(["a"], {"a": 40.0, "h": 45.0, "x": 39.0}, {"a": 1.0, "h": 0.0, "x": 0.25})
        self.assertEqual(soup_week.left_out(record), ["x"])


class CachedScoresTest(unittest.TestCase):
    def setUp(self):
        self.cases = [{"id": "i1"}, {"id": "i2"}]
        self.rows = {}
        self.cache = type("Cache", (), {"get": lambda _, key, ids: self.rows.get(key, {})})()
        self.evaluate = soup_week.cached_scores(
            self.cases, self.cache, scope="s", score=lambda cases, rows: float(len(rows))
        )

    def test_a_graded_soup_is_scored_from_its_cached_rows_in_case_order(self):
        key = soup_week.soup_key(("a", "b"), "s")
        self.rows[key] = {"i2": {"id": "i2"}, "i1": {"id": "i1"}}
        self.assertEqual(self.evaluate(("b", "a")), 2.0)

    def test_a_soup_with_ungraded_items_is_reported_missing(self):
        self.rows[soup_week.soup_key(("a",), "s")] = {"i1": {"id": "i1"}}
        with self.assertRaises(soup_week.Missing) as caught:
            self.evaluate(("a",))
        self.assertEqual(caught.exception.key, soup_week.soup_key(("a",), "s"))
        self.assertEqual(caught.exception.members, ("a",))

    def test_keys_match_the_nightly_rounds(self):
        # soup_round.run: a body is "<sha[:32]>-<scope>", a soup "soup-<sha256 of a+b...>[:32]-<scope>".
        self.assertEqual(soup_week.soup_key(("a" * 64,), "s"), "a" * 32 + "-s")
        digest = hashlib.sha256(b"a+b").hexdigest()[:32]
        self.assertEqual(soup_week.soup_key(("b", "a"), "s"), f"soup-{digest}-s")


class DecideTest(unittest.TestCase):
    def test_a_passing_candidate_is_recorded_with_seats_for_members_over_the_bar(self):
        scores = {("a", "b"): 41.0, ("a",): 40.6, ("b",): 39.0, ("a", "b", "x"): 41.1}
        record = soup_week.decide(
            "2026-w48",
            private(["a", "b"], {"a": 40.0, "b": 39.0, "x": 38.0}),
            KNOWN,
            None,
            scripted(scores),
            bar=0.5,
            stock=30.0,
        )
        self.assertEqual(
            (record["week_id"], record["round_id"]), ("2026-w48", "d3-mini-2026-11-22")
        )
        self.assertTrue(record["passed"])
        self.assertEqual((record["score"], record["current_score"]), (41.0, 30.0))
        a, b = record["members"]
        self.assertEqual((a["hotkey"], a["body_sha256"], a["rate"]), ("hk-a", "a", 1.0))
        self.assertAlmostEqual(a["contribution"], 2.0)  # 41.0 - the soup without a (39.0)
        self.assertAlmostEqual(a["seat_weight"], 0.16)
        self.assertAlmostEqual(b["contribution"], 0.4)
        self.assertEqual(b["seat_weight"], 0.0)  # under the 0.5 bar

    def test_a_one_member_soup_is_credited_against_stock(self):
        record = soup_week.decide(
            "2026-w48", private(["a"], {"a": 40.0}), KNOWN, None, scripted({("a",): 40.0}),
            bar=0.5, stock=30.0,
        )  # fmt: skip
        self.assertAlmostEqual(record["members"][0]["contribution"], 10.0)

    def test_when_the_candidate_fails_last_weeks_soup_and_seats_carry_over(self):
        last = {
            "week_id": "2026-w47",
            "members": [{"hotkey": "hk-x", "body_sha256": "x", "rate": 1.0, "contribution": 0.9,
                         "seat_weight": 0.16}],
        }  # fmt: skip
        scores = {("a", "b"): 40.2, ("a",): 40.0, ("b",): 39.0, ("x",): 40.0}
        record = soup_week.decide(
            "2026-w48", private(["a", "b"], {"a": 40.0, "b": 39.0}), KNOWN, last, scripted(scores),
            bar=0.5,
        )  # fmt: skip
        self.assertFalse(record["passed"])
        self.assertEqual(record["members"], last["members"])
        self.assertEqual(record["kept_from"], "2026-w47")
        self.assertEqual(record["candidate"], ["a", "b"])

    def test_no_weekly_soup_yet_and_a_failing_candidate_seats_nobody(self):
        scores = {("a", "b"): 30.2, ("a",): 30.0, ("b",): 29.0}
        record = soup_week.decide(
            "2026-w48", private(["a", "b"], {"a": 30.0, "b": 29.0}), KNOWN, None, scripted(scores),
            bar=0.5, stock=30.0,
        )  # fmt: skip
        self.assertEqual((record["passed"], record["members"]), (False, []))


class FakeCache:
    def __init__(self):
        self.rows = {}

    def get(self, key, ids):
        return {i: self.rows[key][i] for i in ids if i in self.rows.get(key, {})}


class RunWeekTest(unittest.TestCase):
    """The judge grades one missing model at a time and decides once everything is cached."""

    SCORES = {("release",): 30.0, ("a", "b"): 41.0, ("a",): 40.0, ("b",): 39.0}

    def run_week(self, *, chunk=2, stop=lambda: False, known_keys=None):
        cases = [{"id": f"i{n}"} for n in range(3)]
        cache, graded = FakeCache(), []
        by_key = {soup_week.soup_key(m, "s"): v for m, v in self.SCORES.items()}

        def grade(key, members):  # a chunk of the items the model has no answer for yet
            graded.append((key, members))
            have = cache.rows.setdefault(key, {})
            for case in [c for c in cases if c["id"] not in have][:chunk]:
                have[case["id"]] = {"id": case["id"], "key": key}

        record = soup_week.run_week(
            "2026-w48",
            private(["a", "b"], {"a": 40.0, "b": 39.0}),
            KNOWN,
            None,
            cases,
            cache,
            scope="s",
            score=lambda cases, rows: by_key[rows[0]["key"]],
            bar=0.5,
            grade=grade,
            stop=stop,
        )
        return record, graded

    def test_every_needed_model_is_graded_in_chunks_then_the_week_is_decided(self):
        record, graded = self.run_week()
        self.assertTrue(record["passed"])
        self.assertEqual([m["body_sha256"] for m in record["members"]], ["a", "b"])
        self.assertEqual(record["current_score"], 30.0)
        order = [members for _, members in graded]
        self.assertEqual(order[:2], [("release",), ("release",)])  # 3 items in chunks of 2
        self.assertEqual(sorted(set(order)), [("a",), ("a", "b"), ("b",), ("release",)])
        self.assertEqual(len(order), 8)

    def test_it_stops_between_models_when_asked_and_decides_nothing(self):
        calls = iter([False, False, True])
        record, graded = self.run_week(stop=lambda: next(calls))
        self.assertIsNone(record)
        self.assertEqual(len(graded), 2)

    def test_a_grading_that_makes_no_progress_is_an_error_not_a_loop(self):
        with self.assertRaises(RuntimeError):
            self.run_week(chunk=0)


class SourcesTest(unittest.TestCase):
    def test_the_release_is_the_chassis_and_a_body_is_its_folder_with_its_rename(self):
        got = soup_week.sources(
            ("b1", "release"), "/state", "/chassis", rename=lambda folder: f"r:{folder.name}"
        )
        self.assertEqual(
            [(str(folder), rename) for folder, rename in got],
            [("/state/bodies/b1", "r:b1"), ("/chassis", None)],
        )


class WindowTest(unittest.TestCase):
    def test_the_judge_gives_way_at_its_utc_stop_time(self):
        self.assertTrue(soup_week.past("23:30", now=(23, 30)))
        self.assertFalse(soup_week.past("23:30", now=(9, 5)))
        self.assertFalse(soup_week.past(None, now=(23, 59)))


if __name__ == "__main__":
    unittest.main()
