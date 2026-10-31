"""Soup round exams drawn from Zils Current dated exports, with the fixed bank as fallback."""

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tests.test_soup_exam import bank as exam_bank
from zils import base_metrics, model_pools, models, soup_current, soup_exam, soup_round

DEADLINE = datetime(2026, 11, 3, 3, 0, tzinfo=UTC)
ROUND = "d3-mini-2026-11-03"
HASH = "0x" + "ab" * 32
STREAMS = {
    "gh-close": ("reasoning", "noul", ["false", "true"]),
    "gh-merge": ("reasoning", "noul", ["false", "true"]),
    "gh-type": ("language", "choice", ["A", "B"]),
    "wiki-revert": ("language", "noul", ["false", "true"]),
}


def item(stream, index, resolved, *, label=None, text=None):
    category, kind, labels = STREAMS[stream]
    identity = f"{stream}-{resolved:%Y%m%d}-{index}"
    criteria = {"A": "Bug", "B": "Feature"} if kind == "choice" else None
    text = text if text is not None else f"{stream} state {index} {resolved:%F}"
    return {
        "id": identity,
        "source_id": identity,
        "family": stream,
        "category": category,
        "block": "new",
        "state": {"text": text},
        "question": {"type": kind, "instructions": "Decide.", "criteria": criteria},
        "label": label or labels[index % 2],
        "source": {"name": "test"},
        "asked_at": (resolved - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
        "resolved_at": resolved.isoformat().replace("+00:00", "Z"),
        "licence": "test",
        "content_sha256": hashlib.sha256(f"{stream}:{text}".encode()).hexdigest(),
    }


def export(root, day, items, **manifest_changes):
    """One dated folder the way ``current_bank.export_bank`` writes it."""
    folder = Path(root) / day
    folder.mkdir(parents=True)
    text = "".join(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in items)
    (folder / "items.jsonl").write_text(text)
    manifest = {
        "version": "zils-current-bank/v1",
        "date": day,
        "mode": "local",
        "overlap": {"checked": True, "references": [], "hits": []},
        "files": {
            "items.jsonl": {
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "bytes": len(text.encode()),
                "rows": len(items),
            }
        },
        **manifest_changes,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return folder


def day_items(day, per_stream=40, streams=STREAMS):
    resolved = datetime.fromisoformat(day).replace(hour=12, tzinfo=UTC)
    return [item(s, i, resolved) for s in streams for i in range(per_stream)]


class Fixture:
    def __init__(self, root, *, days=("2026-10-31", "2026-11-01", "2026-11-02"), **exam):
        self.root = Path(root)
        self.current = self.root / "current"
        self.current.mkdir(parents=True)
        for day in days:
            export(self.current, day, day_items(day))
        self.state = self.root / "state"
        self.ledger = soup_exam.UseLedger(self.state / "uses.jsonl")
        self.bank = exam_bank()
        self.exam = {
            "source": "current",
            "current_bank": str(self.current),
            "current_items": 120,
            "min_items": 40,
            **exam,
        }

    def config(self, **changes):
        return {
            "pool_id": "d3-mini",
            "model": models.D3,
            "schedule": {"deadline_utc": "03:00"},
            "sketch_seed": "5e" * 32,
            "subset_size": 400,
            "exam": soup_current.validate(self.exam),
            **changes,
        }

    def prepare(self, round_id=ROUND, **changes):
        return soup_current.prepare(
            self.config(**changes),
            self.state,
            round_id=round_id,
            block_hash=HASH,
            bank=self.bank,
            ledger=self.ledger,
            log=lambda message: None,
        )

    def subset(self, round_id=ROUND):
        path = self.state / "rounds" / round_id / "subset.json"
        return json.loads(path.read_text())


class ValidateTest(unittest.TestCase):
    def test_default_is_the_fixed_bank(self):
        self.assertEqual(soup_current.validate(None), {"source": "bank"})
        self.assertEqual(soup_current.validate({"source": "bank"}), {"source": "bank"})

    def test_current_fills_defaults(self):
        exam = soup_current.validate({"source": "current", "current_bank": "/b"})
        self.assertEqual(exam["max_age_days"], 14)
        self.assertEqual(exam["max_uses"], 3)
        self.assertEqual(exam["current_items"], 1000)
        self.assertEqual(exam["min_streams"], 3)
        self.assertEqual(exam["stream_max"], 0.4)

    def test_bad_settings_are_refused(self):
        bad = [
            {"source": "elsewhere"},
            {"source": "current"},
            {"source": "current", "current_bank": ""},
            {"source": "current", "current_bank": "/b", "max_age_days": 0},
            {"source": "current", "current_bank": "/b", "max_uses": 0},
            {"source": "current", "current_bank": "/b", "stream_max": 0},
            {"source": "current", "current_bank": "/b", "stream_max": 0.2, "min_streams": 3},
            {"source": "current", "current_bank": "/b", "min_items": 2000},
            {"source": "current", "current_bank": "/b", "surprise": 1},
            {"source": "bank", "max_uses": 3},
        ]
        for exam in bad:
            with self.subTest(exam=exam), self.assertRaises(ValueError):
                soup_current.validate(exam)

    def soup_entry(self, **changes):
        return {
            "type": "soup",
            "model": models.D3,
            "enabled": True,
            "share": 1.0,
            "house": "house.json",
            "schedule": {"deadline_utc": "03:00"},
            **changes,
        }

    def test_validate_soup_keeps_the_exam_setting(self):
        exam = {"source": "current", "current_bank": "/var/lib/zils-current/bank"}
        result = model_pools.validate_soup(self.soup_entry(exam=exam))
        self.assertEqual(result["exam"]["source"], "current")
        self.assertEqual(result["exam"]["max_age_days"], 14)

    def test_validate_soup_without_exam_is_unchanged(self):
        result = model_pools.validate_soup(self.soup_entry())
        self.assertNotIn("exam", result)
        self.assertNotIn(
            "exam", model_pools.validate_soup(self.soup_entry(exam={"source": "bank"}))
        )

    def test_validate_soup_refuses_a_bad_exam(self):
        with self.assertRaises(ValueError):
            model_pools.validate_soup(self.soup_entry(exam={"source": "current"}))


class LoadTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def load(self, **changes):
        args = {
            "deadline": DEADLINE,
            "max_age_days": 14,
            "max_tokens": 8192,
            "uses": {},
            "max_uses": 3,
            "log": lambda message: None,
            **changes,
        }
        return soup_current.load(self.root, **args)

    def test_a_good_folder_is_read_and_recorded(self):
        folder = export(self.root, "2026-11-01", day_items("2026-11-01", 5))
        result = self.load()
        self.assertEqual(len(result["items"]), 20)
        [entry] = result["folders"]
        manifest = (folder / "manifest.json").read_bytes()
        self.assertEqual(entry["manifest_sha256"], hashlib.sha256(manifest).hexdigest())
        self.assertEqual(entry["date"], "2026-11-01")
        self.assertEqual(entry["rows"], 20)

    def test_failing_manifests_are_skipped_and_logged(self):
        export(self.root, "2026-10-28", day_items("2026-10-28", 5))
        export(self.root, "2026-10-29", day_items("2026-10-29", 5), mode="draft")
        export(self.root, "2026-10-30", day_items("2026-10-30", 5), overlap={"checked": False})
        tampered = export(self.root, "2026-10-31", day_items("2026-10-31", 5))
        with (tampered / "items.jsonl").open("a") as handle:
            handle.write(json.dumps(item("gh-merge", 99, DEADLINE - timedelta(days=2))) + "\n")
        broken = self.root / "2026-11-01"
        broken.mkdir()
        (broken / "manifest.json").write_text("{not json")
        export(self.root, "2026-11-02", day_items("2026-11-02", 5), date="2026-11-09")
        (self.root / "notes").mkdir()
        logged = []
        result = self.load(log=logged.append)
        self.assertEqual([f["date"] for f in result["folders"]], ["2026-10-28"])
        self.assertEqual(
            set(result["skipped"]),
            {"2026-10-29", "2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02"},
        )
        self.assertIn("hash", result["skipped"]["2026-10-31"])
        self.assertIn("overlap", result["skipped"]["2026-10-30"])
        self.assertEqual(len(logged), 5)

    def test_a_missing_bank_directory_loads_nothing(self):
        result = soup_current.load(
            self.root / "absent",
            deadline=DEADLINE,
            max_age_days=14,
            max_tokens=8192,
            uses={},
            max_uses=3,
            log=lambda message: None,
        )
        self.assertEqual(result["items"], [])

    def test_only_items_resolved_in_the_window_before_the_deadline(self):
        old = DEADLINE - timedelta(days=15)
        late = DEADLINE + timedelta(hours=1)
        inside = DEADLINE - timedelta(days=2)
        export(self.root, f"{old:%F}", [item("gh-merge", 0, old)])
        export(self.root, f"{inside:%F}", [item("gh-merge", 1, inside)])
        export(self.root, f"{late:%F}", [item("gh-merge", 2, late)])
        result = self.load()
        self.assertEqual([c["id"] for c in result["items"]], [item("gh-merge", 1, inside)["id"]])
        self.assertEqual(result["excluded"].get("outside_window"), 1)

    def test_items_used_max_uses_times_are_excluded(self):
        rows = day_items("2026-11-01", 2)
        export(self.root, "2026-11-01", rows)
        result = self.load(uses={rows[0]["id"]: 3, rows[1]["id"]: 2})
        ids = {c["id"] for c in result["items"]}
        self.assertNotIn(rows[0]["id"], ids)
        self.assertIn(rows[1]["id"], ids)
        self.assertEqual(result["excluded"]["used_up"], 1)

    def test_items_over_the_model_input_cap_are_skipped(self):
        resolved = DEADLINE - timedelta(days=1)
        long = item("gh-merge", 0, resolved, text="x" * 13000)  # ~8,667 tokens + the question
        short = item("gh-merge", 1, resolved, text="x" * 3000)
        export(self.root, f"{resolved:%F}", [long, short])
        self.assertEqual([c["id"] for c in self.load()["items"]], [short["id"]])
        self.assertEqual(len(self.load(max_tokens=16384)["items"]), 2)

    def test_duplicates_and_invalid_rows_are_dropped(self):
        resolved = DEADLINE - timedelta(days=1)
        first = item("gh-merge", 0, resolved)
        twin = {**item("gh-merge", 1, resolved), "content_sha256": first["content_sha256"]}
        invalid = {"id": "x"}
        export(self.root, f"{resolved:%F}", [first, twin, invalid])
        result = self.load()
        self.assertEqual([c["id"] for c in result["items"]], [first["id"]])
        self.assertEqual(result["excluded"], {"duplicate": 1, "invalid": 1})


class DrawTest(unittest.TestCase):
    settings = {"current_items": 100, "min_items": 30, "min_streams": 3, "stream_max": 0.4}

    def test_no_stream_exceeds_its_cap_and_labels_are_balanced(self):
        items = day_items("2026-11-01", 200, streams=["gh-close", "gh-merge", "gh-type"])
        chosen, counts = soup_current.draw(items, b"k", self.settings)
        self.assertEqual(len(chosen), 100)
        per_stream = Counter(c["family"] for c in chosen)
        self.assertLessEqual(max(per_stream.values()), 40)
        for stream, n in counts.items():
            self.assertEqual(n["drawn"], per_stream[stream])
            self.assertLessEqual(max(n["labels"].values()) - min(n["labels"].values()), 1)

    def test_too_few_streams_cannot_draw(self):
        items = day_items("2026-11-01", 200, streams=["gh-close", "gh-merge"])
        with self.assertRaises(ValueError):
            soup_current.draw(items, b"k", self.settings)

    def test_a_one_label_stream_does_not_count(self):
        items = day_items("2026-11-01", 50, streams=["gh-close", "gh-merge"])
        resolved = DEADLINE - timedelta(days=1)
        items += [item("gh-type", i, resolved, label="A") for i in range(50)]
        with self.assertRaises(ValueError):
            soup_current.draw(items, b"k", self.settings)

    def test_too_few_items_under_the_cap_cannot_draw(self):
        items = day_items("2026-11-01", 4, streams=["gh-close", "gh-merge", "gh-type"])
        with self.assertRaises(ValueError):
            soup_current.draw(items, b"k", self.settings)

    def test_newest_days_come_first(self):
        items = day_items("2026-10-25", 60) + day_items("2026-11-02", 30)
        chosen, _ = soup_current.draw(items, b"k", self.settings)
        self.assertTrue(all(c["resolved_at"].startswith("2026-11-02") for c in chosen))

    def test_the_secret_picks_within_a_day(self):
        items = day_items("2026-11-01", 200)
        first = {c["id"] for c in soup_current.draw(items, b"one", self.settings)[0]}
        again = {c["id"] for c in soup_current.draw(items, b"one", self.settings)[0]}
        other = {c["id"] for c in soup_current.draw(items, b"two", self.settings)[0]}
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)


class PrepareTest(unittest.TestCase):
    def setUp(self):
        self.f = Fixture(self.enterContext(tempfile.TemporaryDirectory()))

    def test_the_subset_mixes_current_streams_into_their_cells(self):
        bank, record = self.f.prepare()
        ids = self.f.subset()
        self.assertEqual(len(ids), 400)
        by_id = {c["id"]: c for c in bank}
        cases = [by_id[i] for i in ids]
        families = Counter(c["family"] for c in cases)
        self.assertEqual(sum(families[s] for s in STREAMS), 120)
        # Current streams replace the fixed bank's families in the cells they cover.
        covered = {("reasoning", "new"), ("language", "new")}
        fixed = {c["family"] for c in cases if c["family"] not in STREAMS}
        self.assertFalse(
            {(c["category"], c["block"]) for c in cases if c["family"] in fixed} & covered
        )
        base_metrics.baselines(cases)  # every category/block cell is still there
        self.assertIsNone(record["fallback"])
        self.assertEqual(record["current_items"], 120)
        self.assertEqual(record["bank_items"], 280)
        self.assertEqual(record["replaced_cells"], ["language/new", "reasoning/new"])
        self.assertEqual(len(record["folders"]), 3)
        self.assertEqual(set(record["streams"]), set(STREAMS))
        self.assertTrue(all(len(f["manifest_sha256"]) == 64 for f in record["folders"]))
        saved = json.loads((self.f.state / "rounds" / ROUND / "bank-source.json").read_text())
        self.assertEqual(saved, record)

    def test_no_stream_takes_more_than_its_share_of_the_current_part(self):
        _, record = self.f.prepare()
        self.assertLessEqual(max(s["drawn"] for s in record["streams"].values()), 48)

    def test_the_draw_depends_on_the_local_secret(self):
        self.f.prepare()
        first = self.f.subset()
        other = Fixture(Path(self.f.root) / "other")
        other.prepare(sketch_seed="7a" * 32)
        self.assertNotEqual(set(first), set(other.subset()))

    def test_too_few_streams_falls_back_to_the_fixed_bank_and_records_it(self):
        f = Fixture(Path(self.f.root) / "narrow", min_streams=5, stream_max=0.4)
        bank, record = f.prepare()
        self.assertIn("streams", record["fallback"])
        seed = soup_round.round_seed(HASH, "d3-mini")
        expected = soup_exam.subset(f.bank, seed, 400, exclude=f.ledger.retired())
        self.assertEqual(f.subset(), [c["id"] for c in expected])
        self.assertEqual(record["current_items"], 0)
        self.assertEqual(record["bank_items"], 400)

    def test_an_unreadable_bank_falls_back_instead_of_crashing(self):
        f = Fixture(Path(self.f.root) / "gone", current_bank=str(Path(self.f.root) / "nowhere"))
        _, record = f.prepare()
        self.assertIsNotNone(record["fallback"])
        self.assertEqual(len(f.subset()), 400)

    def test_an_exhausted_fixed_bank_writes_nothing(self):
        f = Fixture(Path(self.f.root) / "tiny", min_streams=5)
        f.bank = f.bank[:50]
        bank, record = f.prepare()
        self.assertIsNotNone(record["fallback"])
        self.assertFalse((f.state / "rounds" / ROUND / "subset.json").exists())
        self.assertEqual(len(bank), 50)

    def test_a_resumed_round_reuses_its_draw(self):
        bank, record = self.f.prepare()
        ids = self.f.subset()
        export(self.f.current, "2026-11-02b", [])  # not a date: ignored
        again_bank, again = self.f.prepare()
        self.assertEqual(again, record)
        self.assertEqual(self.f.subset(), ids)
        self.assertEqual({c["id"] for c in again_bank}, {c["id"] for c in bank})

    def test_items_used_up_in_the_ledger_are_not_drawn_again(self):
        self.f.prepare()
        drawn = [i for i in self.f.subset() if i.split("-2026")[0] in STREAMS]
        self.f.ledger.record(ROUND, self.f.subset())
        self.f.exam = {**self.f.exam, "max_uses": 1}
        self.f.prepare("d3-mini-2026-11-04")
        again = set(self.f.subset("d3-mini-2026-11-04"))
        self.assertFalse(set(drawn) & again)

    def test_earlier_rounds_items_stay_in_the_bank_after_they_age_out(self):
        self.f.prepare()
        earlier = {i for i in self.f.subset() if i.split("-2026")[0] in STREAMS}
        self.f.ledger.record(ROUND, self.f.subset())
        export(self.f.current, "2026-11-20", day_items("2026-11-20"))
        bank, record = self.f.prepare("d3-mini-2026-11-21")
        self.assertIsNone(record["fallback"])
        self.assertEqual([f["date"] for f in record["folders"]], ["2026-11-20"])
        self.assertLessEqual(earlier, {c["id"] for c in bank})
        later = {i for i in self.f.subset("d3-mini-2026-11-21") if i.split("-2026")[0] in STREAMS}
        self.assertFalse(earlier & later)  # aged out of the draw, still loadable

    def test_the_input_cap_comes_from_the_pool_model(self):
        resolved = DEADLINE - timedelta(days=1)
        rows = [item(s, i, resolved, text="y" * 5000 + str(i)) for s in STREAMS for i in range(40)]
        lite = {**models.spec(models.D3), "max_input_tokens": 2048}
        f = Fixture(Path(self.f.root) / "lite", days=())
        export(f.current, f"{resolved:%F}", rows)
        _, record = f.prepare(model=lite)
        self.assertIsNotNone(record["fallback"])
        self.assertEqual(record["excluded"]["too_long"], 160)
        _, record = Fixture(Path(self.f.root) / "mini", days=()).prepare()
        self.assertIsNotNone(record["fallback"])  # empty bank: nothing to draw

    def test_prepare_before_run_is_what_the_round_grades(self):
        bank, record = self.f.prepare()
        by_id = {c["id"]: c for c in bank}
        self.assertTrue(all(i in by_id for i in self.f.subset()))


class RoundTest(unittest.TestCase):
    """A whole soup round (``tests.test_soup_run``'s synthetic bodies) on a Zils Current exam."""

    def setUp(self):
        from unittest.mock import patch

        from tests import test_soup_run
        from zils import d3

        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.r = test_soup_run.Round(self.tmp)
        pinned = hashlib.sha256((self.r.chassis / "readout.safetensors").read_bytes()).hexdigest()
        self.enterContext(patch.dict(d3.PINS["files"], {"readout.safetensors": pinned}))
        self.f = Fixture(self.tmp / "exam")
        self.f.state, self.f.ledger, self.f.bank = self.r.state, self.r.ledger, self.r.bank

    def test_the_round_grades_the_current_draw_and_records_its_source(self):
        config = self.r.config(
            subset_size=400, schedule={"deadline_utc": "03:00"}, exam=self.f.config()["exam"]
        )
        bank, source = soup_current.prepare(
            config,
            self.r.state,
            round_id=ROUND,
            block_hash=HASH,
            bank=self.r.bank,
            ledger=self.r.ledger,
            log=lambda message: None,
        )
        self.r.bank = bank
        private = self.r.run(config=config, bank_source=source)
        self.assertEqual(private["bank_source"], source)
        self.assertIsNone(source["fallback"])
        drawn = {i for i in private["subset_ids"] if i.split("-2026")[0] in STREAMS}
        self.assertEqual(len(drawn), 120)
        self.assertEqual(private["subset_ids"], self.f.subset())
        public = json.loads((self.r.state / "rounds" / ROUND / "public.json").read_text())
        self.assertNotIn("bank_source", public)

    def test_a_round_without_a_bank_source_keeps_its_old_record(self):
        private = self.r.run()
        self.assertNotIn("bank_source", private)


class MainTest(unittest.TestCase):
    def test_the_gpu_entry_point_draws_from_current_and_passes_the_record(self):
        from unittest.mock import patch

        from zils import d3, soup_pool

        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        f = Fixture(tmp)
        bank_path = tmp / "bank.jsonl"
        bank_path.write_text("".join(json.dumps(c) + "\n" for c in f.bank))
        request = {
            "round_id": ROUND,
            "block_hash": HASH,
            "snapshot": {"members": {}, "commitments": {}},
            "config": {**f.config(), "model": models.spec(models.D3)},
            "state_dir": str(f.state),
            "round_bank": str(bank_path),
            "base": str(tmp / "qwen"),
        }
        path = tmp / "request.json"
        path.write_text(json.dumps(request))
        seen = {}

        def run(config, root, **kwargs):
            seen.update(kwargs)

        with (
            patch.object(soup_round, "run", run),
            patch.object(d3, "pinned_base_path", lambda model: tmp / "chassis"),
            patch.object(soup_pool, "fit_bank", lambda bank, model, **kwargs: bank),
        ):
            soup_pool.main([str(path)])
        self.assertIsNone(seen["bank_source"]["fallback"])
        ids = {c["id"] for c in seen["bank"]}
        self.assertLessEqual(set(f.subset()), ids)

    def test_the_gpu_entry_point_without_an_exam_setting_is_unchanged(self):
        from unittest.mock import patch

        from zils import d3, soup_pool

        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        f = Fixture(tmp)
        bank_path = tmp / "bank.jsonl"
        bank_path.write_text("".join(json.dumps(c) + "\n" for c in f.bank))
        config = {k: v for k, v in f.config().items() if k != "exam"}
        request = {
            "round_id": ROUND,
            "block_hash": HASH,
            "snapshot": {"members": {}, "commitments": {}},
            "config": {**config, "model": models.spec(models.D3)},
            "state_dir": str(f.state),
            "round_bank": str(bank_path),
            "base": str(tmp / "qwen"),
        }
        path = tmp / "request.json"
        path.write_text(json.dumps(request))
        seen = {}
        with (
            patch.object(soup_round, "run", lambda config, root, **kwargs: seen.update(kwargs)),
            patch.object(d3, "pinned_base_path", lambda model: tmp / "chassis"),
            patch.object(soup_pool, "fit_bank", lambda bank, model, **kwargs: bank),
        ):
            soup_pool.main([str(path)])
        self.assertIsNone(seen["bank_source"])
        self.assertEqual(seen["bank"], f.bank)
        self.assertFalse((f.state / "rounds").exists())


class EpochTest(unittest.TestCase):
    def test_a_relative_current_bank_is_resolved_under_the_validator_directory(self):
        from tests import test_soup_run

        case = test_soup_run.SoupEpochTest("test_one_round_per_day_seeded_after_the_deadline")
        case.setUp()
        try:
            entry = case.config["base_pools"]["pools"]["d3-mini"]
            entry["exam"] = {"source": "current", "current_bank": "current-bank"}
            case.epoch()
            request = case.calls[0][1]
            exam = request["config"]["exam"]
            self.assertEqual(exam["current_bank"], str(case.directory / "current-bank"))
            self.assertEqual(exam["max_uses"], 3)
        finally:
            case.doCleanups()


if __name__ == "__main__":
    unittest.main()


CELL_STREAMS = {
    f"cell-{category}-{block}": (category, block)
    for category in ("reasoning", "language", "retrieval", "tools", "arts")
    for block in ("known", "new")
}


def cell_items(day, per_stream):
    resolved = datetime.fromisoformat(day).replace(hour=12, tzinfo=UTC)
    rows = []
    for stream, (category, block) in CELL_STREAMS.items():
        for i in range(per_stream):
            row = item("gh-merge", i, resolved)
            identity = f"{stream}-{resolved:%Y%m%d}-{i}"
            row.update(
                id=identity,
                source_id=identity,
                family=stream,
                category=category,
                block=block,
                state={"text": f"{identity} state"},
                content_sha256=hashlib.sha256(identity.encode()).hexdigest(),
            )
            rows.append(row)
    return rows


class AllCellsTest(unittest.TestCase):
    """Current streams in every category/block cell leave the fixed bank nothing to fill."""

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def fixture(self, per_stream, **exam):
        f = Fixture(self.root / str(per_stream), days=(), min_streams=3, **exam)
        export(f.current, "2026-11-02", cell_items("2026-11-02", per_stream))
        return f

    def test_the_whole_subset_comes_from_current_when_it_covers_every_cell(self):
        f = self.fixture(60)
        bank, record = f.prepare()
        self.assertIsNone(record["fallback"])
        ids = f.subset()
        self.assertEqual(len(ids), 400)
        by_id = {c["id"]: c for c in bank}
        self.assertTrue(all(by_id[i]["family"] in CELL_STREAMS for i in ids))
        self.assertEqual(record["current_items"], 400)
        self.assertEqual(record["bank_items"], 0)
        self.assertEqual(len(record["replaced_cells"]), 10)
        base_metrics.baselines([by_id[i] for i in ids])

    def test_too_few_current_items_for_every_cell_falls_back(self):
        f = self.fixture(20)  # 200 items cover every cell but cannot fill 400
        _, record = f.prepare()
        self.assertIsNotNone(record["fallback"])
        self.assertEqual(record["bank_items"], 400)

    def test_current_items_never_exceed_the_subset(self):
        f = self.fixture(60, current_items=1000, min_items=40)
        _, record = f.prepare()
        self.assertIsNone(record["fallback"])
        self.assertEqual(len(f.subset()), 400)


class ChoiceBalanceTest(unittest.TestCase):
    def test_choice_and_score_streams_are_not_cut_to_their_rarest_label(self):
        resolved = DEADLINE - timedelta(days=1)
        rows = [item("gh-type", i, resolved, label="A") for i in range(10)]
        rows += [item("gh-type", 10 + i, resolved, label="B") for i in range(2)]
        rows += [item("gh-merge", i, resolved, label="true") for i in range(10)]
        rows += [item("gh-merge", 10 + i, resolved, label="false") for i in range(2)]
        settings = {"current_items": 100, "min_items": 1, "min_streams": 2, "stream_max": 1.0}
        _, counts = soup_current.draw(rows, b"k", settings)
        self.assertEqual(counts["gh-type"]["drawn"], 12)  # a choice stream keeps every item
        self.assertEqual(counts["gh-merge"]["labels"], {"false": 2, "true": 2})  # yes/no balanced
