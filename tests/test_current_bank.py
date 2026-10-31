"""Zils Current dated exports: leak filters, licence gate, overlap checks and manifests."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_current import issue_data, resolved_record, wiki_data
from zils import current, current_bank, current_items

PASSAGE = (
    "The glacier retreated four kilometres between the survey of 1950 and the survey of 2010 "
    "and left a chain of small lakes behind it that local guides now call the Silver Steps "
    "because of how they shine in the low evening light during the late summer months"
)


def pr_data(body, text_at="2026-10-26T15:30:00Z"):
    return {
        "repo": "w",
        "head": "fix",
        "base": "main",
        "title": "Fix",
        "body": body,
        "text_at": text_at,
    }


def wiki(title, comment="copyedit", added=("New sentence.",)):
    return {**wiki_data(title), "comment": comment, "added": list(added)}


def window_records():
    rows = [
        resolved_record(
            "wikipedia-revert",
            i,
            "true" if i % 2 else "false",
            wiki(f"Page {i}"),
            resolved=f"2026-10-30T{10 + i:02d}:00:00Z",
        )
        for i in range(6)
    ]
    rows.append(
        resolved_record(
            "wikipedia-revert", 50, "true", wiki("Late page"), resolved="2026-10-31T01:00:00Z"
        )
    )
    rows.append(
        resolved_record(
            "github-issue-type", 60, "bug", issue_data("Crash"), resolved="2026-10-30T09:00:00Z"
        )
    )
    rows.append(
        resolved_record(
            "github-issue-type",
            61,
            "feature",
            issue_data("Dark mode"),
            resolved="2026-10-30T09:00:00Z",
        )
    )
    late = resolved_record(
        "github-pr-merge",
        70,
        "true",
        pr_data("Body", text_at="2026-10-31T00:00:00Z"),
        resolved="2026-10-30T09:00:00Z",
    )
    rows.append({**late, "outcome_at": "2026-10-27T00:00:00Z"})
    return rows


def store_with(tmp, records):
    store = current.Store(Path(tmp) / "state")
    store.add(records)
    store.save([], records, [])
    return store


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class LeakFilterTest(unittest.TestCase):
    def reason(self, record):
        return current_bank.leak_reason(record, current_items.build(record)["state"]["text"])

    def test_text_read_after_the_outcome_is_dropped(self):
        record = {
            **resolved_record("github-pr-merge", 1, "true", pr_data("b")),
            "outcome_at": "2026-10-26T16:00:00Z",
        }
        self.assertIsNone(self.reason(record))
        record["outcome_at"] = "2026-10-26T15:00:00Z"
        self.assertEqual(self.reason(record), "late-text")

    def test_type_labels_present_at_open_are_dropped(self):
        record = resolved_record(
            "github-issue-type", 1, "bug", {**issue_data("Crash"), "labels_at_open": ["Bug"]}
        )
        self.assertEqual(self.reason(record), "label-at-open")
        record["data"]["labels_at_open"] = ["good first issue"]
        self.assertIsNone(self.reason(record))
        close = resolved_record(
            "github-issue-close", 2, "true", {**issue_data("Crash"), "labels_at_open": ["bug"]}
        )
        self.assertIsNone(self.reason(close))

    def test_text_that_states_the_outcome_is_dropped(self):
        stated = [
            ("github-pr-merge", pr_data("Temporary probe. Closed without merging.")),
            ("github-pr-merge", pr_data("This PR was merged into main by the release bot.")),
            ("github-pr-merge", pr_data("Superseded by #12.")),
            ("github-issue-close", issue_data("Crash", "Duplicate of #3")),
            ("github-issue-close", issue_data("Crash", "Closing as not planned for now.")),
            ("wikipedia-revert", wiki("Lake", comment="Reverted vandalism")),
            ("wikipedia-revert", wiki("Lake", comment="Undid revision 123")),
        ]
        for stream, data in stated:
            with self.subTest(stream=stream, data=data):
                self.assertEqual(
                    self.reason(resolved_record(stream, 1, "true", data)), "states-outcome"
                )
        fine = [
            (
                "github-pr-merge",
                pr_data("Fixes merge conflicts; depends on #4 being merged first."),
            ),
            ("github-pr-merge", pr_data("Implements merge sort.")),
            (
                "github-issue-close",
                issue_data("Duplicate detection is broken", "It flags every file."),
            ),
            ("wikipedia-revert", wiki("Lake", comment="Added a source")),
        ]
        for stream, data in fine:
            with self.subTest(stream=stream, data=data):
                self.assertIsNone(self.reason(resolved_record(stream, 1, "true", data)))


class OverlapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        record = resolved_record("wikipedia-revert", 1, "true", wiki("Glacier"))
        self.case = current_items.build(record)

    def tearDown(self):
        self.tmp.cleanup()

    def index(self, *paths):
        index = current_bank.OverlapIndex()
        for path in paths:
            index.add(path)
        return index

    def test_same_prompt_same_state_and_shared_passages_are_hits(self):
        same = write_jsonl(
            self.root / "exam.jsonl",
            [{k: self.case[k] for k in ("id", "state", "question", "label")}],
        )
        self.assertEqual(self.index(same).hit(self.case), "prompt")
        other = {"type": "noul", "instructions": "Is this edit about geography?"}
        state = self.root / "bank.json"
        state.write_text(json.dumps([{"state": self.case["state"], "question": other}]))
        self.assertEqual(self.index(state).hit(self.case), "state")
        passage_record = resolved_record(
            "wikipedia-revert", 2, "true", wiki("Glacier", added=[PASSAGE])
        )
        passage_case = current_items.build(passage_record)
        teacher = write_jsonl(
            self.root / "pool.jsonl",
            [
                {
                    "state": f"Some intro. {PASSAGE.upper()} And an ending.",
                    "questions": {"decision": other},
                }
            ],
        )
        self.assertEqual(self.index(teacher).hit(passage_case), "passage")

    def test_short_shared_phrases_and_unrelated_rows_are_not_hits(self):
        rows = [
            {
                "state": {"text": "The glacier retreated four kilometres between surveys."},
                "question": {"type": "noul"},
            },
            {"state": "Totally unrelated text about trains.", "question": {"type": "noul"}},
        ]
        index = self.index(write_jsonl(self.root / "x.jsonl", rows))
        record = resolved_record(
            "wikipedia-revert",
            3,
            "true",
            wiki("Glacier", added=["The glacier retreated four kilometres."]),
        )
        self.assertIsNone(index.hit(current_items.build(record)))

    def test_items_sharing_only_the_template_do_not_match(self):
        earlier = current_items.build(
            resolved_record("wikipedia-revert", 8, "true", wiki("Other page"))
        )
        index = self.index(write_jsonl(self.root / "earlier.jsonl", [earlier]))
        self.assertIsNone(index.hit(self.case))

    def test_sources_are_summarised_by_name_and_hash_and_directories_are_read(self):
        folder = self.root / "refs"
        folder.mkdir()
        write_jsonl(folder / "a.jsonl", [{"state": "x", "question": {"type": "noul"}}])
        write_jsonl(
            folder / "relay.jsonl", [{"content_sha256": "0" * 64, "probabilities": [0.5, 0.5]}]
        )
        index = self.index(folder)
        names = {s["name"]: s for s in index.sources}
        self.assertEqual(set(names), {"a.jsonl", "relay.jsonl"})
        self.assertEqual(names["relay.jsonl"]["hash_only_rows"], 1)
        self.assertEqual(names["a.jsonl"]["text_rows"], 1)
        digest = hashlib.sha256((folder / "a.jsonl").read_bytes()).hexdigest()
        self.assertEqual(names["a.jsonl"]["sha256"], digest)
        self.assertNotIn(str(self.root), json.dumps(index.sources))


class BankExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = store_with(self.tmp.name, window_records())
        hit = current_items.build(window_records()[0])
        self.reference = write_jsonl(
            self.root / "exam.jsonl", [{"state": hit["state"], "question": hit["question"]}]
        )
        self.bank = self.root / "bank"

    def tearDown(self):
        self.tmp.cleanup()

    def export(self, **options):
        options.setdefault("references", [self.reference])
        return current_bank.export_bank(self.store, self.bank, "2026-10-30", **options)

    def test_dated_folder_holds_items_and_a_manifest(self):
        manifest = self.export()
        folder = self.bank / "2026-10-30"
        items = [json.loads(line) for line in (folder / "items.jsonl").read_text().splitlines()]
        self.assertEqual(json.loads((folder / "manifest.json").read_text()), manifest)
        self.assertEqual(manifest["version"], current_bank.VERSION)
        self.assertEqual(manifest["collector"], current.VERSION)
        self.assertEqual(manifest["mode"], "local")
        self.assertEqual(manifest["window"]["resolved_from"], "2026-10-30T00:00:00Z")
        self.assertEqual(manifest["window"]["resolved_until"], "2026-10-31T00:00:00Z")
        self.assertEqual(
            manifest["files"]["items.jsonl"]["sha256"],
            hashlib.sha256((folder / "items.jsonl").read_bytes()).hexdigest(),
        )
        self.assertEqual(manifest["files"]["items.jsonl"]["rows"], len(items))
        self.assertEqual(manifest["settings"]["leak_filters"], True)
        self.assertEqual(manifest["settings"]["seed"], 0)
        wiki_stats = manifest["streams"]["wikipedia-revert"]
        self.assertEqual(wiki_stats["resolved"], 6, "the late page belongs to the next day")
        self.assertEqual(wiki_stats["overlap"], 1)
        self.assertEqual(wiki_stats["exported_labels"], {"false": 2, "true": 2})
        self.assertTrue(wiki_stats["publishable"])
        self.assertIn("CC-BY-SA", wiki_stats["licence"])
        self.assertFalse(manifest["streams"]["github-pr-merge"]["publishable"])
        self.assertEqual(manifest["streams"]["github-pr-merge"]["filtered"], {"late-text": 1})
        self.assertEqual(manifest["overlap"]["checked"], True)
        self.assertEqual(
            manifest["overlap"]["hits"], [{"id": window_records()[0]["id"], "kind": "prompt"}]
        )
        self.assertEqual([s["name"] for s in manifest["overlap"]["references"]], ["exam.jsonl"])
        self.assertNotIn(window_records()[0]["id"], {c["id"] for c in items})
        self.assertEqual(sum(s["exported"] for s in manifest["streams"].values()), len(items))
        self.assertEqual(
            manifest["counts"],
            {
                stream: s["exported_labels"]
                for stream, s in manifest["streams"].items()
                if s["exported"]
            },
        )

    def test_an_existing_day_is_kept_unless_forced(self):
        first = self.export()
        path = self.bank / "2026-10-30" / "manifest.json"
        path.write_text(json.dumps({**first, "marker": True}))
        self.assertTrue(self.export()["marker"])
        self.assertNotIn("marker", self.export(force=True))

    def test_overlap_check_needs_references_or_an_explicit_skip(self):
        with self.assertRaisesRegex(ValueError, "reference"):
            self.export(references=[])
        manifest = self.export(references=[], check_overlap=False)
        self.assertEqual(manifest["overlap"], {"checked": False, "references": [], "hits": []})

    def test_filters_can_be_turned_off_and_are_recorded(self):
        manifest = self.export(leak_filters=False, balance=False)
        self.assertEqual(manifest["settings"]["leak_filters"], False)
        self.assertEqual(manifest["streams"]["github-pr-merge"]["exported"], 1)

    def test_publishable_exports_hold_only_attributed_cc_by_sa_streams(self):
        manifest = self.export(publishable=True)
        self.assertEqual(manifest["mode"], "publishable")
        items = (self.bank / "2026-10-30" / "items.jsonl").read_text().splitlines()
        self.assertEqual({json.loads(line)["family"] for line in items}, {"wikipedia-revert"})
        self.assertEqual(set(manifest["streams"]), set(current_bank.publishable_streams()))
        self.assertIn("wikipedia-revert", manifest["streams"])
        for stream in ("github-pr-merge", "github-issue-type", "github-issue-close"):
            with self.subTest(stream=stream), self.assertRaisesRegex(ValueError, "local-only"):
                self.export(publishable=True, streams=[stream], force=True)

    def test_publishable_cases_need_an_attribution_url(self):
        record = resolved_record(
            "wikipedia-revert", 90, "true", wiki("No link"), resolved="2026-10-30T05:00:00Z"
        )
        record["source"].pop("url")
        store = store_with(Path(self.tmp.name) / "other", [record])
        with self.assertRaisesRegex(ValueError, "attribution"):
            current_bank.export_bank(
                store, self.bank, "2026-10-30", publishable=True, check_overlap=False
            )

    def test_command_line_bank_export(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = current.main(
                [
                    "--dir",
                    str(self.root / "state"),
                    "export",
                    "--bank",
                    str(self.bank),
                    "--date",
                    "2026-10-30",
                    "--reference",
                    str(self.reference),
                ]
            )
        self.assertEqual(code, 0)
        summary = json.loads(out.getvalue())
        self.assertEqual(summary["date"], "2026-10-30")
        self.assertTrue((self.bank / "2026-10-30" / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
