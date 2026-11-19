"""Weekly soup judge: which soup becomes the week's soup (pure functions, ZIL-204)."""

import datetime
import hashlib
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path

from zils import soup_grade, soup_pool, soup_round, soup_week


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

    def test_a_fine_tune_is_credited_against_the_soup_with_its_base_in_its_place(self):
        # a is a tune of h (the round's record says so): leave-one-out would credit it 2.0
        record = private(["a", "b"], {"a": 40.0, "b": 39.0, "h": 40.0})
        record["fine_tunes"] = {"a": ["h"]}
        scores = {("a", "b"): 41.0, ("a",): 40.6, ("b",): 39.0, ("b", "h"): 40.7,
                  ("a", "b", "h"): 41.0}  # fmt: skip
        week = soup_week.decide(
            "2026-w48", record, KNOWN, None, scripted(scores), bar=0.5, stock=30.0
        )
        a, b = week["members"]
        self.assertAlmostEqual(a["contribution"], 0.3)
        self.assertEqual(a["seat_weight"], 0.0)
        self.assertAlmostEqual(b["contribution"], 0.4)
        self.assertEqual(week["gated"], {})

    def test_a_fine_tune_whose_base_the_round_could_not_build_holds_no_seat(self):
        record = private(["a", "b"], {"a": 40.0, "b": 39.0})
        record["fine_tunes"] = {"a": [None]}
        scores = {("a", "b"): 41.0, ("a",): 40.6, ("b",): 39.0}
        week = soup_week.decide(
            "2026-w48", record, KNOWN, None, scripted(scores), bar=0.5, stock=30.0
        )
        a, _ = week["members"]
        self.assertAlmostEqual(a["contribution"], 2.0)
        self.assertEqual(a["seat_weight"], 0.0)
        self.assertEqual(week["gated"], {"a": "its base could not be scored"})

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
            "soup_sha256": "f" * 64,
            "temperature": 1.25,
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
        # the kept soup's files stay where the week that made them saved them
        self.assertEqual((record["soup_sha256"], record["temperature"]), ("f" * 64, 1.25))
        self.assertEqual(str(soup_week.soup_folder("/w", record)), "/w/2026-w47/soup")

    def test_no_weekly_soup_yet_and_a_failing_candidate_seats_nobody(self):
        scores = {("a", "b"): 30.2, ("a",): 30.0, ("b",): 29.0}
        record = soup_week.decide(
            "2026-w48", private(["a", "b"], {"a": 30.0, "b": 29.0}), KNOWN, None, scripted(scores),
            bar=0.5, stock=30.0,
        )  # fmt: skip
        self.assertEqual((record["passed"], record["members"]), (False, []))
        self.assertIsNone(soup_week.soup_folder("/w", record))

    def test_a_passing_weeks_soup_is_saved_under_its_own_week(self):
        record = soup_week.decide(
            "2026-w48", private(["a"], {"a": 40.0}), KNOWN, None, scripted({("a",): 40.0}),
            bar=0.5, stock=30.0,
        )  # fmt: skip
        self.assertEqual(str(soup_week.soup_folder("/w", record)), "/w/2026-w48/soup")


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


class BaseKeyTest(unittest.TestCase):
    def test_a_kept_copy_of_an_admission_soup_has_its_own_key_and_folder(self):
        copy = "admission-" + "c" * 32
        other = "admission-" + "c" * 31 + "d"
        self.assertNotEqual(soup_week.soup_key((copy,), "s"), soup_week.soup_key((other,), "s"))
        self.assertTrue(soup_week.soup_key((copy,), "s").startswith("soup-"))
        got = soup_week.sources((copy,), "/state", "/chassis", rename=lambda folder: "r")
        self.assertEqual([(str(f), r) for f, r in got], [("/state/admission/" + "c" * 32, None)])


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


class WeekOfTest(unittest.TestCase):
    def test_the_week_is_named_after_its_monday_round(self):
        self.assertEqual(soup_week.week_of("2026-11-16"), ("2026-w47", "2026-11-16"))
        self.assertEqual(soup_week.week_of("2026-11-22"), ("2026-w47", "2026-11-16"))
        self.assertEqual(soup_week.week_of("2026-11-23"), ("2026-w48", "2026-11-23"))
        self.assertEqual(soup_week.week_of("2027-01-01"), ("2026-w53", "2026-12-28"))


class WeeksSettingTest(unittest.TestCase):
    def test_no_setting_means_no_weekly_judge(self):
        self.assertEqual(soup_week.weeks({}, "/v"), {})

    def test_a_pool_names_its_sealed_set_and_pass_bar(self):
        config = {"soup_weeks": {"d3-mini": {"sealed": "sealed/set.jsonl", "bar": 0.5}}}
        self.assertEqual(
            soup_week.weeks(config, "/v"),
            {"d3-mini": {"sealed": str(Path("/v/sealed/set.jsonl").resolve()), "bar": 0.5}},
        )

    def test_anything_else_is_refused(self):
        for value in (
            [],
            {"d3-mini": {"sealed": "s.jsonl"}},
            {"d3-mini": {"sealed": "s.jsonl", "bar": -1}},
            {"d3-mini": {"sealed": "", "bar": 0.5}},
            {"d3-mini": {"sealed": "s.jsonl", "bar": 0.5, "weekday": 2}},
        ):
            with self.assertRaises(ValueError):
                soup_week.weeks({"soup_weeks": value}, "/v")


class EpochTest(unittest.TestCase):
    """The validator's pass starts Monday's judge once the Monday round is in, and only then."""

    POOL, MONDAY = "d3-mini", "d3-mini-2026-11-16"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)
        self.config = {"soup_weeks": {self.POOL: {"sealed": "sealed.jsonl", "bar": 0.5}}}
        self.started = []
        self.finish(self.POOL, self.MONDAY)

    def state(self, pool):
        return self.directory / "base-pools" / pool / "soup"

    def finish(self, pool, round_id):
        """A finished nightly round: its request, the pool's decision and the ledger line."""
        state = self.state(pool)
        state.mkdir(parents=True, exist_ok=True)
        nightly = {"round_id": round_id, "config": {"pool_id": pool}, "state_dir": str(state),
                   "runner": {"device": "cpu", "poll": 1, "runtime_python": "python"}}  # fmt: skip
        (state / f"request-{round_id}.json").write_text(json.dumps(nightly))
        (state / "decision.json").write_text(json.dumps({"round_id": round_id}))
        with (state / "uses.jsonl").open("a") as ledger:
            ledger.write(json.dumps({"round": round_id}) + "\n")

    def job(self, path, **job):
        me = os.getpid()  # a process that is certainly running: this one
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"pid": me, "pid_started": soup_pool._started(me), **job}))

    def epoch(self, start=None):
        def record(pool, path):
            self.started.append((pool, path))
            return None

        return soup_week.epoch(self.config, self.directory, start=start or record)

    def test_the_judge_starts_on_the_monday_round_with_a_request_pinned_to_it(self):
        summary = self.epoch()
        self.assertEqual(summary, {self.POOL: {"week_id": "2026-w47", "started": True}})
        ((pool, path),) = self.started
        self.assertEqual(pool, self.POOL)
        self.assertEqual(path, self.state(self.POOL) / "weeks" / "request-2026-w47.json")
        request = json.loads(path.read_text())
        self.assertEqual(
            (request["week_id"], request["round_id"], request["bar"]),
            ("2026-w47", self.MONDAY, 0.5),
        )
        self.assertEqual(request["sealed"], str((self.directory / "sealed.jsonl").resolve()))
        self.assertEqual(request["runner"]["runtime_python"], "python")
        job = json.loads((self.state(self.POOL) / "weeks" / "job.json").read_text())
        self.assertEqual((job["week_id"], job["starts"]), ("2026-w47", 1))

    def test_a_recorded_week_is_not_judged_again(self):
        record = self.state(self.POOL) / "weeks" / "2026-w47" / "record.json"
        record.parent.mkdir(parents=True)
        record.write_text("{}")
        self.assertEqual(self.epoch(), {self.POOL: {"week_id": "2026-w47", "recorded": True}})
        self.assertEqual(self.started, [])

    def test_no_setting_starts_nothing(self):
        self.config = {}
        self.assertEqual(self.epoch(), {})
        self.assertEqual(self.started, [])

    def test_a_later_round_means_the_week_was_missed_and_last_weeks_soup_stays(self):
        self.finish(self.POOL, "d3-mini-2026-11-17")
        summary = self.epoch()
        self.assertIn("missed", summary[self.POOL])
        self.assertEqual(self.started, [])

    def test_a_round_whose_ledger_line_is_not_written_yet_is_not_judged(self):
        (self.state(self.POOL) / "uses.jsonl").write_text("")
        self.assertIn("waiting", self.epoch()[self.POOL])
        self.assertEqual(self.started, [])

    def test_a_running_judge_is_only_reported(self):
        self.job(self.state(self.POOL) / "weeks" / "job.json", week_id="2026-w47", starts=1)
        self.assertEqual(self.epoch()[self.POOL], {"week_id": "2026-w47", "running": True})
        self.assertEqual(self.started, [])

    def test_the_judge_waits_while_a_nightly_round_runs(self):
        self.job(self.state("d3-lite") / "job.json", round_id="d3-lite-2026-11-17")
        self.assertIn("waiting", self.epoch()[self.POOL])
        self.assertEqual(self.started, [])

    def test_the_judge_waits_for_another_pools_judge(self):
        self.job(self.state("d3-lite") / "weeks" / "job.json", week_id="2026-w47", starts=1)
        self.assertIn("waiting", self.epoch()[self.POOL])
        self.assertEqual(self.started, [])

    def test_a_judge_that_gave_way_is_started_again_up_to_a_limit(self):
        for starts in range(1, soup_week.MAX_STARTS + 1):
            self.epoch()
            self.assertEqual(len(self.started), starts)
        summary = self.epoch()
        self.assertIn("skipped", summary[self.POOL]["error"])
        self.assertEqual(len(self.started), soup_week.MAX_STARTS)

    def test_a_failure_is_reported_and_never_raised(self):
        def broken(pool, path):
            raise OSError("no such python")

        self.assertIn("OSError", self.epoch(start=broken)[self.POOL]["error"])
        self.config = {"soup_weeks": {self.POOL: {"sealed": "s.jsonl"}}}
        self.assertIn("ValueError", self.epoch()["error"])


class PublicWeekTest(unittest.TestCase):
    RECORD = {
        "week_id": "2026-w47", "round_id": "d3-mini-2026-11-16", "passed": True,
        "candidate": ["a"], "score": 58.05, "current_score": 46.6, "trace": [{"try": ["a"]}],
        "members": [{"hotkey": "hk-a", "body_sha256": "a", "rate": 1.0, "contribution": 0.66,
                     "seat_weight": 0.16}],
        "soup_sha256": "f" * 64, "temperature": 1.17, "gated": {}, "scope": "secret-scope",
        "sealed_sha256": "s" * 64, "items": 25008, "pool_id": "d3-mini",
    }  # fmt: skip

    def test_only_the_soup_and_its_seats_are_public(self):
        public = soup_week.public_week(self.RECORD)
        self.assertEqual(
            public,
            {"week_id": "2026-w47", "round_id": "d3-mini-2026-11-16", "soup_sha256": "f" * 64,
             "kept_from": None,
             "members": [{"hotkey": "hk-a", "body_sha256": "a", "rate": 1.0, "seat_weight": 0.16,
                          "contribution": 0.5}]},
        )  # fmt: skip

    def test_a_contribution_under_the_seat_bar_never_shows_at_the_bar(self):
        member = {**self.RECORD["members"][0], "contribution": 0.3876, "seat_weight": 0.0}
        public = soup_week.public_week({**self.RECORD, "members": [member]})
        self.assertEqual(public["members"][0]["contribution"], 0.0)

    def attach(self, record, now):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "base-pools" / "d3-mini" / "soup" / "weeks"
            if record is not None:
                root.mkdir(parents=True)
                (root / "week.json").write_text(json.dumps(record))
            config = {"soup_weeks": {"d3-mini": {"sealed": "s.jsonl", "bar": 0.5}}}
            return soup_week.attach(config, tmp, "d3-mini", {"pool_id": "d3-mini"}, now=now)

    def test_the_current_week_is_attached_to_the_nightly_decision_for_pay(self):
        monday = datetime.datetime(2026, 11, 16, 18, tzinfo=datetime.UTC).timestamp()
        got = self.attach(self.RECORD, monday + 6 * 86400)
        self.assertEqual(got["week"], soup_week.public_week(self.RECORD))

    def test_before_the_first_week_the_nightly_rules_stand(self):
        self.assertEqual(self.attach(None, 0), {"pool_id": "d3-mini"})

    def test_a_week_left_too_long_seats_nobody(self):
        monday = datetime.datetime(2026, 11, 16, tzinfo=datetime.UTC).timestamp()
        got = self.attach(self.RECORD, monday + (soup_week.MAX_WEEK_DAYS + 1) * 86400)
        self.assertEqual(got["week"]["members"], [])

    def test_a_pool_without_a_weekly_judge_is_untouched(self):
        decision = {"pool_id": "d3-lite"}
        self.assertIs(soup_week.attach({}, "/v", "d3-lite", decision, now=0), decision)


class ClosedSeatsTest(unittest.TestCase):
    """A seated body must stay public for its week (ZIL-205): each nightly round checks."""

    WEEK = {"week_id": "2026-w47", "members": [
        {"hotkey": "hk-a", "body_sha256": "a", "seat_weight": 0.16},
        {"hotkey": "hk-b", "body_sha256": "b", "seat_weight": 0.16},
        {"hotkey": "hk-h", "body_sha256": "h", "seat_weight": 0.0},
    ]}  # fmt: skip
    BODIES = {m: {"repo": f"miner/{m}", "revision": "r" * 40} for m in "abh"}

    def listed(self, gone=(), error=None):
        asked = []

        def check(repo, revision):
            asked.append(repo)
            name = repo.split("/")[1]
            if name in gone:
                raise error or urllib.error.HTTPError(repo, 404, "Not Found", None, None)
            return {}

        return check, asked

    def test_a_seated_body_that_cannot_be_listed_loses_its_seat(self):
        check, asked = self.listed(gone="a")
        got = soup_week.closed_seats(self.WEEK, self.BODIES, check, None)
        self.assertEqual(got, {"week_id": "2026-w47", "bodies": ["a"]})
        self.assertEqual(asked, ["miner/a", "miner/b"])  # an unpaid seat is not checked

    def test_private_or_gone_both_count(self):
        for error in (
            urllib.error.HTTPError("u", 401, "Unauthorized", None, None),
            ValueError("revision is not a commit of this repo"),
        ):
            check, _ = self.listed(gone="b", error=error)
            got = soup_week.closed_seats(self.WEEK, self.BODIES, check, None)
            self.assertEqual(got["bodies"], ["b"])

    def test_a_network_failure_keeps_the_seat(self):
        for error in (
            urllib.error.URLError("timed out"),
            urllib.error.HTTPError("u", 503, "Unavailable", None, None),
        ):
            check, _ = self.listed(gone="a", error=error)
            got = soup_week.closed_seats(self.WEEK, self.BODIES, check, None)
            self.assertEqual(got["bodies"], [])

    def test_a_closed_seat_stays_closed_for_the_week_and_a_new_week_starts_clean(self):
        check, asked = self.listed()
        before = {"week_id": "2026-w47", "bodies": ["a"]}
        self.assertEqual(
            soup_week.closed_seats(self.WEEK, self.BODIES, check, before)["bodies"], ["a"]
        )
        self.assertEqual(asked, ["miner/b"])
        old = {"week_id": "2026-w46", "bodies": ["a"]}
        self.assertEqual(soup_week.closed_seats(self.WEEK, self.BODIES, check, old)["bodies"], [])


class RereadTest(unittest.TestCase):
    def test_answers_written_by_another_process_are_seen_on_the_next_read(self):
        # the grading child appends to the cache files; a PredictionCache held open would not see it
        with tempfile.TemporaryDirectory() as root:
            cache = soup_week.Reread(root)
            self.assertEqual(cache.get("k", ["i1", "i2"]), {})
            soup_grade.PredictionCache(root).put("k", [{"id": "i1"}])
            self.assertEqual(cache.get("k", ["i1", "i2"]), {"i1": {"id": "i1"}})
            soup_grade.PredictionCache(root).put("k", [{"id": "i2"}])
            self.assertEqual(sorted(cache.get("k", ["i1", "i2"])), ["i1", "i2"])


if __name__ == "__main__":
    unittest.main()
