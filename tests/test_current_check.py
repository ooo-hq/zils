"""Zils Current proof and cross-checks: every item's outcome re-derived by a second method."""

import json
import tempfile
import unittest
from pathlib import Path

from tests.test_current import FakeHttp, at
from zils import current, current_check

NOW = at("2026-10-31T12:00:00Z")
API = "https://en.wikipedia.org/w/api.php"


def wiki_case(rev, label, *, title="Frog", wiki="enwiki", stream="wikipedia-revert"):
    return {
        "id": f"{stream}-{rev}",
        "family": stream,
        "label": label,
        "state": {"text": "Wiki: English Wikipedia\nArticle: Frog\n"},
        "question": {"type": "noul", "instructions": "Will it be reverted?"},
        "source": {
            "name": "wikimedia",
            "wiki": wiki,
            "rev_id": rev,
            "old_rev_id": rev - 1,
            "title": title,
            "url": f"https://en.wikipedia.org/w/index.php?diff={rev}&oldid={rev - 1}",
            "api": API,
        },
        "asked_at": "2026-10-25T10:00:00Z",
        "resolved_at": "2026-10-28T10:00:00Z",
        "outcome_at": "2026-10-28T10:00:00Z",
    }


def rev(revid, sha1, ts, tags=(), comment=""):
    return {"revid": revid, "sha1": sha1, "timestamp": ts, "tags": list(tags), "comment": comment}


HISTORIES = {
    # reverted by identity: 103 restores 99's content
    100: [
        rev(98, "a", "2026-10-24T00:00:00Z"),
        rev(99, "b", "2026-10-25T09:00:00Z"),
        rev(100, "c", "2026-10-25T10:00:00Z"),
        rev(101, "d", "2026-10-25T11:00:00Z"),
        rev(102, "b", "2026-10-25T12:00:00Z", ["mw-rollback"]),
    ],
    # undone (content differs) with a summary naming the revision
    200: [
        rev(199, "x", "2026-10-25T09:00:00Z"),
        rev(200, "y", "2026-10-25T10:00:00Z"),
        rev(201, "z", "2026-10-25T11:00:00Z"),
        rev(202, "w", "2026-10-26T11:00:00Z", ["mw-undo"], "Undid revision 200 by [user]"),
    ],
    # never reverted
    300: [
        rev(299, "p", "2026-10-25T09:00:00Z"),
        rev(300, "q", "2026-10-25T10:00:00Z"),
        rev(301, "r", "2026-10-26T10:00:00Z"),
    ],
    # reverted only after the check: still "not reverted" at the time of the label
    400: [
        rev(399, "m", "2026-10-25T09:00:00Z"),
        rev(400, "n", "2026-10-25T10:00:00Z"),
        rev(401, "m", "2026-10-30T10:00:00Z", ["mw-manual-revert"]),
    ],
}


def history_handler(url, params):
    if params.get("prop") != "revisions":
        return 404, {}, {}
    start, direction = int(params["rvstartid"]), params["rvdir"]
    for key, revisions in HISTORIES.items():
        ids = [r["revid"] for r in revisions]
        if start in ids:
            if direction == "newer":
                rows = [r for r in revisions if r["revid"] >= start]
            else:
                rows = [r for r in reversed(revisions) if r["revid"] <= start]
            return 200, {"query": {"pages": [{"revisions": rows}]}}, {}
    return 200, {"query": {"pages": [{"missing": True}]}}, {}


class ProofTest(unittest.TestCase):
    def test_items_need_a_link_ids_and_ordered_times(self):
        good = wiki_case(100, "true")
        self.assertIsNone(current_check.proof_reason(good))
        self.assertEqual(
            current_check.proof_reason({**good, "source": {**good["source"], "url": ""}}),
            "no-url",
        )
        no_id = {**good, "source": {k: v for k, v in good["source"].items() if k != "rev_id"}}
        self.assertEqual(current_check.proof_reason(no_id), "no-source-id")
        self.assertEqual(
            current_check.proof_reason({**good, "outcome_at": "2026-10-24T00:00:00Z"}),
            "outcome-before-ask",
        )
        self.assertEqual(
            current_check.proof_reason({k: v for k, v in good.items() if k != "outcome_at"}),
            "no-outcome-time",
        )

    def test_built_cases_carry_an_outcome_time(self):
        from zils import current_items

        record = {
            "id": "wikipedia-revert-1",
            "stream": "wikipedia-revert",
            "asked_at": "2026-10-25T10:00:00Z",
            "resolved_at": "2026-10-28T10:00:00Z",
            "label": "false",
            "source": {"url": "u", "rev_id": 1},
            "data": {
                "wiki": "enwiki",
                "title": "T",
                "comment": "",
                "length_old": 1,
                "length_new": 2,
                "minor": False,
                "editor": "registered account",
                "removed": [],
                "added": ["x"],
            },
        }
        case = current_items.build(record)
        self.assertEqual(case["outcome_at"], "2026-10-28T10:00:00Z")
        self.assertEqual(case["outcome_time"], "checked")


class WikiCheckTest(unittest.TestCase):
    def test_page_history_rederives_the_revert_label(self):
        cases = [
            wiki_case(100, "true"),
            wiki_case(200, "true"),
            wiki_case(300, "false"),
            wiki_case(400, "false"),
            wiki_case(300, "true"),
        ]
        results = [current_check.check_wiki(c, FakeHttp(history_handler)) for c in cases]
        self.assertEqual(
            [r["derived"] for r in results], ["true", "true", "false", "false", "false"]
        )
        self.assertEqual([r["agree"] for r in results], [True, True, True, True, False])
        self.assertIn("102", results[0]["evidence"])


class GateTest(unittest.TestCase):
    def test_streams_under_the_threshold_are_removed_and_reported(self):
        good = [wiki_case(100, "true"), wiki_case(300, "false"), wiki_case(400, "false")]
        bad = [
            wiki_case(300, "true", stream="wikipedia-revert-de", wiki="dewiki"),
            wiki_case(100, "true", stream="wikipedia-revert-de", wiki="dewiki"),
        ]
        no_proof = {**wiki_case(200, "true"), "outcome_at": None}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "2026-10-31"
            folder.mkdir()
            rows = good + bad + [no_proof]
            (folder / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (folder / "manifest.json").write_text(
                json.dumps({"files": {"items.jsonl": {"rows": len(rows)}}})
            )
            report = current_check.run(folder, FakeHttp(history_handler), sample=100, seed=1)
            kept = [json.loads(x) for x in (folder / "items.jsonl").read_text().splitlines()]
            saved = json.loads((folder / "crosscheck.json").read_text())
            sheet = (folder / "spotcheck.md").read_text()
            manifest = json.loads((folder / "manifest.json").read_text())
            original = (folder / "items.unchecked.jsonl").exists()
        self.assertEqual(report["streams"]["wikipedia-revert"]["agreement"], 1.0)
        self.assertEqual(report["streams"]["wikipedia-revert-de"]["agreement"], 0.5)
        self.assertEqual(report["excluded"], ["wikipedia-revert-de"])
        self.assertEqual(report["proof_rejected"], {"no-outcome-time": 1})
        self.assertEqual(sorted(c["id"] for c in kept), sorted(c["id"] for c in good))
        self.assertEqual(
            saved["streams"]["wikipedia-revert-de"]["disagreements"][0]["url"][:8], "https://"
        )
        self.assertIn("wikipedia-revert", sheet)
        self.assertIn("https://en.wikipedia.org/w/index.php?diff=100", sheet)
        self.assertEqual(manifest["files"]["items.jsonl"]["rows"], 3)
        self.assertEqual(manifest["crosscheck"]["excluded"], ["wikipedia-revert-de"])
        self.assertTrue(original)


class AfdCheckTest(unittest.TestCase):
    def test_article_existence_agrees_with_the_result(self):
        def handler(url, params):
            if params.get("list") == "logevents":
                events = (
                    [{"type": "delete", "action": "delete", "timestamp": "2026-10-12T18:00:00Z"}]
                    if params["letitle"] == "Recreated"
                    else []
                )
                return 200, {"query": {"logevents": events}}, {}
            pages = [
                {"title": "Gone", "missing": True},
                {"title": "Here"},
                {"title": "Merged", "redirect": True},
                {"title": "Recreated"},
            ]
            return 200, {"query": {"pages": pages}}, {}

        def afd(article, label):
            return {
                "id": article,
                "family": "wikipedia-afd",
                "label": label,
                "state": {"text": f"Article: {article}\n"},
                "source": {"title": f"Wikipedia:Articles for deletion/{article}", "url": "u"},
            }

        cases = [
            afd("Gone", "delete"),
            afd("Here", "keep"),
            afd("Merged", "merge-redirect"),
            afd("Here", "delete"),
            {
                **afd("Gone", "B"),
                "question": {"type": "choice", "criteria": {"A": "Kept", "B": "Deleted"}},
            },
            {**afd("Recreated", "delete"), "asked_at": "2026-10-04T09:25:00Z"},
        ]
        results = current_check.check_afd(cases, FakeHttp(handler))
        self.assertEqual([r["agree"] for r in results], [True, True, True, False, True, True])
        self.assertIn("deletion log", results[-1]["evidence"])


class StackExchangeCheckTest(unittest.TestCase):
    def test_a_second_fetch_confirms_closes_scores_and_accepts(self):
        def handler(url, params):
            if url.endswith("/answers"):
                items = [
                    {
                        "answer_id": 21,
                        "question_id": 2,
                        "is_accepted": True,
                        "body": "<p>Use dateutil.</p>",
                    },
                    {
                        "answer_id": 22,
                        "question_id": 2,
                        "is_accepted": False,
                        "body": "<p>Regex.</p>",
                    },
                ]
                return 200, {"items": items, "quota_remaining": 200}, {}
            items = [
                {"question_id": 1, "closed_date": 5, "score": -3},
                {"question_id": 2, "score": 5, "accepted_answer_id": 21},
                {"question_id": 3, "score": 3},
            ]
            return 200, {"items": items, "quota_remaining": 200}, {}

        def se(stream, number, label, text="Site: Stack Overflow\n"):
            return {
                "id": f"{stream}-{number}",
                "family": stream,
                "label": label,
                "state": {"text": text},
                "source": {"site": "stackoverflow", "question_id": number, "url": "u"},
            }

        accepted_text = "Question:\nQ\n\nAnswer A:\nRegex.\n\nAnswer B:\nUse dateutil."
        cases = [
            se("stackexchange-closed-tools", 1, "true"),
            se("stackexchange-closed-tools", 2, "false"),
            se("stackexchange-score-tools", 2, "3"),
            se("stackexchange-accepted-tools", 2, "B", accepted_text),
            se("stackexchange-accepted-tools", 2, "A", accepted_text),
            se("stackexchange-score-tools", 3, "2"),  # 1 or 2 then, 3 now: within one vote
            se("stackexchange-score-tools", 2, "1"),  # 0 then, 5 now: a real difference
        ]
        from unittest.mock import patch

        from zils import current_stackexchange

        with patch.object(current_stackexchange, "sleep", lambda s: None):
            results = current_check.check_stackexchange(cases, FakeHttp(handler))
        self.assertEqual(
            [r["agree"] for r in results], [True, True, True, True, False, True, False]
        )


class NightlyPauseTest(unittest.TestCase):
    def test_a_stream_under_the_threshold_is_paused_for_later_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            current_check.update_paused(
                state,
                {
                    "streams": {
                        "a": {"agreement": 0.5, "sample": 20},
                        "b": {"agreement": 1.0, "sample": 20},
                    }
                },
            )
            self.assertEqual(current_check.paused(state), ["a"])
            current_check.update_paused(state, {"streams": {"a": {"agreement": 1.0, "sample": 20}}})
            self.assertEqual(
                current_check.paused(state), ["a"], "a pause stays until cleared by hand"
            )


class NightlyGateTest(unittest.TestCase):
    def test_nightly_cross_checks_each_export_and_skips_paused_streams(self):
        from tests.test_current_nightly import NightlyHttp
        from zils import current_nightly

        with tempfile.TemporaryDirectory() as tmp:
            state, bank = Path(tmp) / "s", Path(tmp) / "b"
            state.mkdir()
            (state / "paused.json").write_text(json.dumps(["github-issue-close"]))
            calls = []
            original = current_check.run

            def spy(folder, http, **options):
                calls.append((Path(folder).name, options["sample"]))
                return {"streams": {}, "excluded": []}

            current_check.run = spy
            try:
                result = current_nightly.run(
                    state,
                    bank,
                    NightlyHttp(["2026-10-31-9", "2026-10-31-10"]),
                    at("2026-10-31T12:30:00Z"),
                    check_overlap=False,
                    gh_first_hours=3,
                    wiki_first_hours=2,
                    se_pages=0,
                    afd_first_days=0,
                    crosscheck_sample=20,
                )
            finally:
                current_check.run = original
            manifest = json.loads((bank / "2026-10-30" / "manifest.json").read_text())
        self.assertEqual(calls, [("2026-10-30", 20)])
        self.assertNotIn("github-issue-close", manifest["settings"]["streams"])
        self.assertEqual(result["paused"], ["github-issue-close"])

    def test_the_crosscheck_command(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "2026-10-31"
            folder.mkdir()
            (folder / "items.jsonl").write_text(json.dumps(wiki_case(100, "true")) + "\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = current.main(
                    ["crosscheck", "--folder", str(folder), "--sample", "5"],
                    http=FakeHttp(history_handler),
                    now=NOW,
                )
            report = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(report["agreement"], {"wikipedia-revert": 1.0})


class ReexportTest(unittest.TestCase):
    def test_a_forced_export_removes_the_previous_check(self):
        from tests.test_current_bank import store_with, window_records
        from zils import current_bank

        with tempfile.TemporaryDirectory() as tmp:
            store = store_with(tmp, window_records())
            bank = Path(tmp) / "bank"
            folder = bank / "2026-10-30"
            folder.mkdir(parents=True)
            for name in ("items.unchecked.jsonl", "crosscheck.json", "spotcheck.md"):
                (folder / name).write_text("old")
            current_bank.export_bank(store, bank, "2026-10-30", check_overlap=False, force=True)
            left = sorted(p.name for p in folder.iterdir())
        self.assertEqual(left, ["items.jsonl", "manifest.json"])


if __name__ == "__main__":
    unittest.main()
