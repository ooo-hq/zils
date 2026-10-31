"""Soup decisions scored on the pooled items of the last ``decision_rounds`` rounds."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_soup_run import Round, pools, soup_entry
from zils import d3, model_pools, soup_exam, soup_grade, soup_publish, soup_round

DAYS = [f"d3-mini-2026-11-0{day}" for day in range(3, 10)]
FIXTURE = Path(__file__).with_name("soup_fixtures") / "one-decision-round.json"


def block(day):
    return "0x" + f"{day + 1:02x}" * 32


class Recording:
    """Wraps a runner factory and records every (cache key, item id) sent to a runner."""

    def __init__(self, factory):
        self.factory, self.calls = factory, []

    def __call__(self, folder):
        run, key = self.factory(folder), Path(folder).name

        def recorded(cases):
            self.calls.extend((key, c["id"]) for c in cases)
            return run(cases)

        return recorded


class DecisionRoundsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.r = Round(self.tmp)
        pinned = hashlib.sha256((self.r.chassis / "readout.safetensors").read_bytes()).hexdigest()
        self.enterContext(patch.dict(d3.PINS["files"], {"readout.safetensors": pinned}))
        self.recording = Recording(self.r.runner)
        self.r.runner = self.recording

    def round(self, day, rounds=3, **kwargs):
        config = kwargs.pop("config", None) or self.r.config(decision_rounds=rounds)
        return self.r.run(
            round_id=DAYS[day], block_hash=kwargs.pop("block_hash", block(day)), config=config,
            **kwargs,
        )  # fmt: skip

    def rows(self, sha_or_key, ids):
        cache = soup_grade.PredictionCache(self.r.state / "cache")
        prefix = sha_or_key if sha_or_key.startswith("soup-") else sha_or_key[:32]
        paths = (self.r.state / "cache").glob(f"{prefix}-*.jsonl")
        key = max(paths, key=lambda p: p.stat().st_mtime_ns).stem  # the current scope
        have = cache.get(key, ids)
        self.assertEqual(set(have), set(ids), f"{key} lacks rows on the decision items")
        return [have[i] for i in ids]

    def cases(self, ids):
        by_id = {c["id"]: c for c in self.r.bank}
        return [by_id[i] for i in ids]

    def assert_decided_on(self, record, ids):
        """Every solo, greedy trial and the crown bar are scores on exactly `ids`."""
        self.assertEqual(record["decision_ids"], sorted(ids))
        self.assertEqual(record["decision"]["items"], len(set(ids)))
        cases = self.cases(sorted(ids))
        for sha, value in record["solo"].items():
            self.assertAlmostEqual(value, soup_grade.score(cases, self.rows(sha, sorted(ids))))
        for trial in record["trace"]:
            key = "soup-" + hashlib.sha256("+".join(trial["try"]).encode()).hexdigest()[:32]
            rows = self.rows(key, sorted(ids))
            self.assertAlmostEqual(trial["score"], soup_grade.score(cases, rows))

    def test_the_setting_defaults_to_three_and_is_validated(self):
        checked = model_pools.validate(pools())["pools"]["d3-mini"]
        self.assertEqual(checked["decision_rounds"], 3)
        self.assertEqual(soup_round.DECISION_ROUNDS, 3)
        for good in (1, 7):
            entry = soup_entry(decision_rounds=good)
            self.assertEqual(
                model_pools.validate(pools(entry))["pools"]["d3-mini"]["decision_rounds"], good
            )
        for bad in (0, 8, 2.0, True, "3", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model_pools.validate(pools(soup_entry(decision_rounds=bad)))

    def test_one_decision_round_is_exactly_todays_behaviour(self):
        """The same four rounds as origin/main before decision_rounds existed (fixture recorded
        from that code): every decision, score, trace and hash is unchanged."""
        expected = json.loads(FIXTURE.read_text())
        for day, want in enumerate(expected):
            if day == 2:
                self.r.commits[5] = ("tuned", 104)
            record = self.round(day, rounds=1)
            got = {k: record[k] for k in want if k in record}
            got["subset_sha256"] = hashlib.sha256(
                json.dumps(sorted(record["subset_ids"]), separators=(",", ":")).encode()
            ).hexdigest()
            got["predictions_sha256"] = hashlib.sha256(
                soup_publish.canonical_json(record["predictions"])
            ).hexdigest()
            self.assertEqual(got, want, DAYS[day])
            self.assertEqual(record["decision"]["rounds"], [DAYS[day]])
            self.assertEqual(record["decision_ids"], sorted(record["subset_ids"]))
        # Only each round's own items were ever graded.
        subsets = [
            set(json.loads((self.r.state / "rounds" / d / "subset.json").read_text()))
            for d in DAYS[:4]
        ]
        self.assertLessEqual({i for _, i in self.recording.calls}, set().union(*subsets))

    def test_decisions_pool_the_last_three_rounds_and_the_first_rounds_use_what_exists(self):
        records = [self.round(day) for day in range(4)]
        subsets = [r["subset_ids"] for r in records]
        self.assertEqual(len({tuple(s) for s in subsets}), 4)  # distinct seeds, distinct subsets
        windows = [[0], [0, 1], [0, 1, 2], [1, 2, 3]]
        for record, window in zip(records, windows, strict=True):
            self.assertEqual(record["decision"]["rounds"], [DAYS[i] for i in window])
            self.assertEqual(record["decision"]["requested"], 3)
            self.assertIsNone(record["decision"]["fallback"])
            self.assert_decided_on(record, set().union(*(subsets[i] for i in window)))
        # The ledger still counts each round's own subset only.
        ledger = [json.loads(line) for line in self.r.ledger.path.read_text().splitlines()]
        self.assertEqual([row["ids"] for row in ledger], [sorted(s) for s in subsets])
        # The crown bar is the current soup on the pooled items.
        last = records[3]
        members = tuple(sorted(m["body_sha256"] for m in records[2]["members"]))
        key = "soup-" + hashlib.sha256("+".join(members).encode()).hexdigest()[:32]
        ids = last["decision_ids"]
        self.assertAlmostEqual(
            last["soup_bar"], soup_grade.score(self.cases(ids), self.rows(key, ids))
        )

    def test_the_round_record_and_public_record_name_the_pooled_rounds(self):
        for day in range(3):
            record = self.round(day)
        public = json.loads((self.r.state / "rounds" / DAYS[2] / "public.json").read_text())
        self.assertEqual(
            public["decision"], {"rounds": DAYS[:3], "items": len(record["decision_ids"])}
        )
        soup_publish.verify_public(public, record)
        text = json.dumps(public)
        for case in self.r.bank:
            self.assertNotIn(case["id"], text)

    def test_a_cached_prediction_is_never_predicted_twice(self):
        for day in range(5):
            self.round(day)
        self.assertEqual(len(self.recording.calls), len(set(self.recording.calls)))

    def test_a_new_body_is_graded_on_every_pooled_item(self):
        for day in range(2):
            self.round(day)
        self.r.commits[5] = ("tuned", 104)
        record = self.round(2)
        self.assertIn(self.r.sha["tuned"], record["pool"])
        self.assert_decided_on(record, record["decision_ids"])
        tuned = {i for k, i in self.recording.calls if k.startswith(self.r.sha["tuned"][:32])}
        self.assertEqual(tuned, set(record["decision_ids"]))

    def test_house_predictions_missing_from_the_cache_are_graded(self):
        for day in range(2):
            self.round(day)
        house = self.r.sha["house"]
        (path,) = (self.r.state / "cache").glob(f"{house[:32]}-*.jsonl")
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[: len(lines) // 3]) + "\n")  # lose two thirds of its rows
        before = len(self.recording.calls)
        record = self.round(2)
        self.assert_decided_on(record, record["decision_ids"])
        regraded = {i for k, i in self.recording.calls[before:] if k.startswith(house[:32])}
        kept = {json.loads(line)["id"] for line in lines[: len(lines) // 3]}
        self.assertEqual(regraded, set(record["decision_ids"]) - kept)

    def test_a_failed_previous_round_is_not_pooled(self):
        self.round(0)
        self.recording.factory.fail_after = 10
        with self.assertRaises(RuntimeError):
            self.round(1)  # never finishes: no record, no ledger entry
        self.recording.factory.fail_after = None
        record = self.round(2)
        self.assertEqual(record["decision"]["rounds"], [DAYS[0], DAYS[2]])
        first = json.loads((self.r.state / "rounds" / DAYS[0] / "private.json").read_text())
        self.assert_decided_on(record, set(first["subset_ids"]) | set(record["subset_ids"]))

    def test_a_skipped_round_draws_no_items_and_is_not_pooled(self):
        self.round(0)
        exhausted = self.r.config(decision_rounds=3, subset_size=len(self.r.bank) + 1)
        self.assertIsNone(self.round(1, config=exhausted))
        record = self.round(2)
        self.assertEqual(record["decision"]["rounds"], [DAYS[0], DAYS[2]])

    def test_items_retired_since_they_were_drawn_are_skipped_for_everyone(self):
        seed = soup_round.round_seed(block(0), "d3-mini")
        first = [c["id"] for c in soup_exam.subset(self.r.bank, seed, 300)]
        tired = first[:7]
        for n in range(soup_exam.RETIRE_AFTER - 1):  # one use short of retirement
            self.r.ledger.record(f"old-{n}", tired)
        self.round(0)
        record = self.round(1)
        self.assertEqual(record["decision"]["rounds"], [DAYS[0], DAYS[1]])
        self.assertEqual(record["decision"]["dropped_items"], len(tired))
        self.assertFalse(set(tired) & set(record["decision_ids"]))
        self.assert_decided_on(record, (set(first) - set(tired)) | set(record["subset_ids"]))

    def test_each_round_records_the_cases_it_drew(self):
        record = self.round(0)
        path = self.r.state / "rounds" / DAYS[0] / "subset-cases.json"
        by_id = {c["id"]: c for c in self.r.bank}
        self.assertEqual(json.loads(path.read_text()), [by_id[i] for i in record["subset_ids"]])

    def test_previous_items_come_from_the_rounds_own_records_not_the_bank(self):
        first = self.round(0)["subset_ids"]
        gone, changed = set(first[::15]), first[7]  # aged out of the bank; edited in place
        bank = [
            {**c, "state": {"text": "edited"}} if c["id"] == changed else c
            for c in self.r.bank
            if c["id"] not in gone
        ]
        record = self.round(1, bank=bank)
        self.assertEqual(record["decision"]["rounds"], [DAYS[0], DAYS[1]])
        self.assertLessEqual(gone, set(record["decision_ids"]))  # still scored, from the record
        self.assertNotIn(changed, record["decision_ids"])  # its recorded text is out of date
        self.assertEqual(record["decision"]["dropped_items"], 1)
        self.assert_decided_on(record, (set(first) - {changed}) | set(record["subset_ids"]))

    def test_a_previous_round_without_its_record_is_skipped_and_recorded(self):
        self.round(0)
        self.round(1)
        (self.r.state / "rounds" / DAYS[0] / "subset-cases.json").unlink()
        record = self.round(2)
        self.assertEqual(record["decision"]["rounds"], [DAYS[1], DAYS[2]])
        self.assertEqual(record["decision"]["skipped"], {DAYS[0]: "no recorded items"})

    def test_the_deadline_falls_back_to_fewer_rounds_and_records_it(self):
        for day in range(2):
            self.round(day)
        copy = self.tmp / "copy"
        shutil.copytree(self.r.state, copy)
        roomy = self.round(2, deadline=10**12)
        estimates = roomy["decision"]["estimates"]
        self.assertEqual(roomy["decision"]["rounds"], DAYS[:3])
        self.assertLess(estimates["1"], estimates["2"])
        self.assertLess(estimates["2"], estimates["3"])
        for budget, rounds in (
            ((estimates["2"] + estimates["3"]) / 2, DAYS[1:3]),
            (estimates["1"] / 2, DAYS[2:3]),  # even one round is over: today's behaviour
        ):
            with self.subTest(budget=budget):
                shutil.rmtree(self.r.state)
                shutil.copytree(copy, self.r.state)
                deadline = 1000.0 + soup_round.RESERVE_SECONDS + budget
                record = self.round(2, deadline=deadline, clock=lambda: 1000.0)
                self.assertEqual(record["decision"]["rounds"], rounds)
                self.assertIn("deadline", record["decision"]["fallback"])
                self.assertEqual(record["decision"]["estimates"], estimates)

    def test_a_resumed_round_keeps_its_pooled_items(self):
        for day in range(2):
            self.round(day)
        self.recording.factory.fail_after = 900  # inside the round, after some grading
        with self.assertRaises(RuntimeError):
            self.round(2)
        self.recording.factory.fail_after = None
        # The resumed attempt has almost no time left, but the pooled items were already chosen.
        record = self.round(2, deadline=1.0, clock=lambda: 0.0)
        self.assertEqual(record["decision"]["rounds"], DAYS[:3])
        self.assert_decided_on(record, record["decision_ids"])

    def test_pooled_soup_predictions_are_kept_while_a_decision_can_still_use_them(self):
        weak = self.r.sha["weak"]
        first = self.round(0, rounds=5)
        self.assertIn(weak, first["pool"])
        small = self.r.config(decision_rounds=5, max_pool=3)  # weak's soups go unused
        for day in range(1, 5):
            self.round(day, config=small)
        soups = {p.stem for p in (self.r.state / "cache").glob("soup-*.jsonl")}
        unused = set(first["soup_keys"]) - {
            k for d in DAYS[1:5]
            for k in json.loads((self.r.state / "rounds" / d / "private.json").read_text())["soup_keys"]
        }  # fmt: skip
        self.assertTrue(unused)
        self.assertLessEqual(unused, soups)  # four rounds unused, inside five rounds: kept
        self.round(5, config=small)
        soups = {p.stem for p in (self.r.state / "cache").glob("soup-*.jsonl")}
        self.assertFalse(unused & soups)  # five rounds unused: dropped

    def test_pooling_works_on_a_zils_current_exam_whose_bank_changes_every_round(self):
        from tests.test_soup_current import STREAMS, Fixture, day_items, export
        from zils import soup_current

        f = Fixture(self.tmp / "exam")
        exam = f.config()["exam"]
        config = self.r.config(
            decision_rounds=3, subset_size=400, schedule={"deadline_utc": "03:00"}, exam=exam
        )
        records, cases = [], {}
        for day in range(3):
            if day:  # a new dated export each day: the round bank (and its hash) changes
                date = f"2026-11-0{day + 2}"
                export(f.current, date, day_items(date))
            bank, source = soup_current.prepare(
                config, self.r.state, round_id=DAYS[day], block_hash=block(day),
                bank=self.r.bank, ledger=self.r.ledger, log=lambda message: None,
            )  # fmt: skip
            self.assertIsNone(source["fallback"])
            cases.update((c["id"], c) for c in bank)
            records.append(self.round(day, config=config, bank=bank, bank_source=source))
        self.assertEqual(len({r["bank_sha256"] for r in records}), 3)
        last = records[2]
        self.assertEqual(last["decision"]["rounds"], DAYS[:3])
        self.assertEqual(last["decision"]["skipped"], {})
        drawn = {i for r in records for i in r["subset_ids"]}
        self.assertEqual(set(last["decision_ids"]), drawn)
        current = {i for i in records[0]["subset_ids"] if i.split("-2026")[0] in STREAMS}
        self.assertTrue(current)
        self.assertLessEqual(current, set(last["decision_ids"]))  # round 1's Current items count
        ids = last["decision_ids"]
        pooled = [cases[i] for i in ids]
        for sha, value in last["solo"].items():
            self.assertAlmostEqual(value, soup_grade.score(pooled, self.rows(sha, ids)))


class LedgerRoundsTest(unittest.TestCase):
    def test_rounds_in_recorded_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = soup_exam.UseLedger(Path(tmp) / "uses.jsonl")
            ledger.record("b", ["x", "y"])
            ledger.record("a", [])
            ledger.record("c", ["y"])
            ledger.record("b", ["z"])  # already recorded: ignored
            self.assertEqual(ledger.rounds(), [("b", ["x", "y"]), ("a", []), ("c", ["y"])])


if __name__ == "__main__":
    unittest.main()
