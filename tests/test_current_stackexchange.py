"""Zils Current Stack Exchange streams: closed, score and accepted answer, without an API key."""

import contextlib
import io
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import zils
from tests.test_current import FakeHttp, at
from zils import base_metrics, current, current_bank, current_items, current_stackexchange as se

NOW = at("2026-10-31T12:00:00Z")
ASKED = int(at("2026-10-20T10:00:00Z").timestamp())


def question(number, **extra):
    row = {
        "question_id": number,
        "creation_date": ASKED + number,
        "title": f"How do I parse dates &amp; times? #{number}",
        "body": "<p>I tried <code>strptime</code> and asked Sam Asker.</p><pre><code>x = 1\ny = 2</code></pre>",
        "tags": ["python", "datetime"],
        "score": 0,
        "answer_count": 0,
        "link": f"https://stackoverflow.com/questions/{number}/x",
        "content_license": "CC BY-SA 4.0",
        "owner": {"display_name": "Sam Asker", "user_type": "registered"},
    }
    row.update(extra)
    return row


def answer(number, question_id, body):
    return {
        "answer_id": number,
        "question_id": question_id,
        "body": f"<p>{body}</p>",
        "owner": {"display_name": f"Helper {number}"},
    }


class SEHttp(FakeHttp):
    def __init__(self, pages, answers=(), quota=250):
        self.pages, self.answers, self.quota = list(pages), list(answers), quota

        def handler(url, params):
            self.quota -= 1
            if url.endswith("/answers"):
                ids = {int(i) for i in url.split("/questions/")[1].split("/")[0].split(";")}
                items = [a for a in self.answers if a["question_id"] in ids]
                return 200, {"items": items, "has_more": False, "quota_remaining": self.quota}, {}
            items = self.pages.pop(0) if self.pages else []
            more = bool(self.pages)
            return 200, {"items": items, "has_more": more, "quota_remaining": self.quota}, {}

        super().__init__(json_handler=handler)


def collect(tmp, http, **options):
    store = current.Store(tmp)
    with patch.object(se, "sleep", lambda s: None):
        result = se.collect(store, http, NOW, sites=["stackoverflow"], **options)
    return store, result


class StackExchangeTest(unittest.TestCase):
    def test_every_group_gives_all_three_question_types_in_its_category(self):
        for group in se.SITES:
            self.assertIn(group, base_metrics.CATEGORY_WEIGHTS)
            kinds = {
                current_items.STREAMS[f"stackexchange-{name}-{group}"]["question"]["type"]
                for name in ("closed", "score", "accepted")
            }
            self.assertEqual(kinds, {"noul", "score", "choice"})
            for name in ("closed", "score", "accepted"):
                spec = current_items.STREAMS[f"stackexchange-{name}-{group}"]
                self.assertEqual(spec["category"], group)
                self.assertTrue(spec["publishable"])
        score = current_items.STREAMS["stackexchange-score-tools"]
        self.assertTrue(score["ordered_equally_spaced"])
        self.assertEqual(len(score["question"]["criteria"]), len(se.SCORE_LEVELS))

    def test_one_page_becomes_closed_score_and_accepted_items(self):
        pages = [
            [
                question(1, closed_date=ASKED + 3600, closed_reason="Duplicate", score=-2),
                question(2, score=5, accepted_answer_id=21, answer_count=2),
                question(3, score=0, last_edit_date=ASKED + 3 + 2 * 3600),
                question(4, score=1, accepted_answer_id=41, answer_count=7),
            ]
        ]
        answers = [answer(21, 2, "Use dateutil."), answer(22, 2, "Use Helper 21's idea, regex.")]
        http = SEHttp(pages, answers)
        with tempfile.TemporaryDirectory() as tmp:
            store, result = collect(tmp, http)
            counts = current.resolve(store, http, NOW)
            resolved = {r["stream"]: r for r in store.resolved() if r["source"]["question_id"] == 2}
            closed = {
                r["source"]["question_id"]: r["label"]
                for r in store.resolved()
                if r["stream"] == "stackexchange-closed-tools"
            }
            scores = {
                r["source"]["question_id"]: r["label"]
                for r in store.resolved()
                if r["stream"] == "stackexchange-score-tools"
            }
            dropped = {r["source"]["question_id"]: r["reason"] for r in store.dropped()}
        self.assertEqual(closed, {1: "true", 2: "false", 4: "false"})
        self.assertEqual(scores, {2: "3", 4: "2"}, "closed questions get no score item")
        self.assertEqual(dropped, {3: "edited-after-ask"})
        self.assertEqual(counts["resolved"], 6)
        self.assertEqual(result["calls"], 2, "one question page and one answer request")
        accepted = resolved["stackexchange-accepted-tools"]
        case = current_items.build(accepted)
        zils.validate_cases([case])
        self.assertEqual(case["question"]["type"], "choice")
        self.assertEqual(sorted(case["question"]["criteria"]), ["A", "B"])
        text = case["state"]["text"]
        right = "Use dateutil."
        self.assertIn(f"Answer {case['label']}:\n{right}", text)
        self.assertIn("Site: Stack Overflow", text)
        self.assertIn("Title: How do I parse dates & times? #2", text)
        self.assertIn("x = 1\ny = 2", text)
        for name in ("Sam Asker", "Helper 21", "Helper 22"):
            self.assertNotIn(name, json.dumps(case))
        self.assertNotIn("score", text.lower().replace("site:", ""))
        self.assertEqual(case["category"], "tools")
        self.assertEqual(case["source"]["url"], "https://stackoverflow.com/questions/2/x")
        self.assertEqual(
            current_items.build(resolved["stackexchange-closed-tools"])["question"]["type"], "noul"
        )
        score_case = current_items.build(resolved["stackexchange-score-tools"])
        zils.validate_cases([score_case])
        self.assertEqual(score_case["label"], "3")

    def test_cursor_advances_and_quota_floor_stops_the_run(self):
        cursors = {}
        http = SEHttp([[question(1)], [question(2)]])
        with tempfile.TemporaryDirectory() as tmp:
            collect(tmp, http, cursors=cursors, pages=1)
            self.assertEqual(cursors["stackoverflow"], ASKED + 1)
            store, _ = collect(tmp, http, cursors=cursors, pages=1)
            self.assertEqual(http.calls[-1][1]["fromdate"], ASKED + 2)
            ids = sorted({r["source"]["question_id"] for r in store.pending()})
        self.assertEqual(ids, [1, 2])
        low = SEHttp([[question(5)], [question(6)]], quota=se.QUOTA_RESERVE + 1)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(current_items.RateLimited):
            collect(tmp, low, pages=2)

    def test_throttle_errors_stop_the_run(self):
        http = FakeHttp(lambda url, params: (400, {"error_id": 502, "error_message": "slow"}, {}))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(current_items.RateLimited):
            collect(tmp, http)

    def test_answer_wording_that_gives_the_outcome_away_is_filtered(self):
        record = {"stream": "stackexchange-accepted-tools", "data": {}}
        for leak in ("Thanks for accepting!", "This is the accepted answer.", "EDIT: OP confirmed"):
            self.assertEqual(current_bank.leak_reason(record, leak), "states-outcome", leak)
        self.assertIsNone(current_bank.leak_reason(record, "Accept-Encoding headers matter."))
        closed = {"stream": "stackexchange-closed-tools", "data": {}}
        self.assertEqual(
            current_bank.leak_reason(closed, "Possible duplicate of: How to X"), "states-outcome"
        )


class BlockTest(unittest.TestCase):
    def test_blocks_follow_the_stream_and_can_be_overridden(self):
        record = {
            "id": "x",
            "stream": "stackexchange-closed-arts",
            "source": {"url": "u"},
            "asked_at": "2026-10-20T10:00:00Z",
            "resolved_at": "2026-10-31T10:00:00Z",
            "label": "true",
            "data": {"site": "movies", "title": "T", "tags": [], "body": "B"},
        }
        self.assertEqual(current_items.build(record)["block"], "known")
        accepted_spec = current_items.STREAMS["stackexchange-accepted-arts"]
        self.assertEqual(accepted_spec["block"], "new")
        with patch.dict("os.environ", {"ZILS_CURRENT_BLOCKS": "stackexchange-closed-=new"}):
            self.assertEqual(current_items.build(record)["block"], "new")


class WiringTest(unittest.TestCase):
    def test_collect_command_keeps_site_cursors_in_the_store(self):
        http = SEHttp([[question(1)]])
        with tempfile.TemporaryDirectory() as tmp, patch.object(se, "sleep", lambda s: None):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                current.main(
                    [
                        "--dir",
                        tmp,
                        "collect",
                        "--source",
                        "stackexchange",
                        "--site",
                        "stackoverflow",
                    ],
                    http=http,
                    now=NOW,
                )
            cursors = json.loads((Path(tmp) / "stackexchange-cursor.json").read_text())
        self.assertEqual(json.loads(out.getvalue())["added"], 2)
        self.assertEqual(cursors, {"stackoverflow": int((NOW - timedelta(days=8)).timestamp())})

    def test_nightly_reads_stack_exchange_and_survives_its_rate_limit(self):
        from zils import current_nightly

        http = FakeHttp(lambda url, params: (429, {}, {}))
        with tempfile.TemporaryDirectory() as tmp, patch.object(se, "sleep", lambda s: None):
            result = current_nightly.run(
                Path(tmp) / "s",
                Path(tmp) / "b",
                http,
                NOW,
                check_overlap=False,
                gh_max_hours=0,
                wiki_max_hours=0,
                afd_first_days=0,
                afc_first_days=0,
                arxiv_first_days=0,
                se_sites=["stackoverflow"],
            )
        self.assertEqual(result["collected"]["stackexchange"]["status"], "rate-limited")
        self.assertEqual(result["errors"], [])


class AcceptedBalanceTest(unittest.TestCase):
    def test_shuffled_answer_letters_are_not_downsampled_to_the_rarest_letter(self):
        rows = []
        for number in range(1, 13):
            count = 4 if number == 1 else 2
            label = "D" if number == 1 else "AB"[number % 2]
            rows.append(
                {
                    "id": f"stackexchange-accepted-tools-{number}",
                    "stream": "stackexchange-accepted-tools",
                    "asked_at": "2026-10-20T10:00:00Z",
                    "resolved_at": "2026-10-31T10:00:00Z",
                    "label": label,
                    "source": {"url": f"u{number}", "question_id": number, "site": "stackoverflow"},
                    "data": {
                        "site": "stackoverflow",
                        "title": f"T{number}",
                        "tags": [],
                        "body": f"Body {number}",
                        "answers": [f"answer {number}-{i}" for i in range(count)],
                    },
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            store = current.Store(tmp)
            store.save([], rows, [])
            cases, stats, _ = current_bank.prepare(store)
        self.assertEqual(stats["stackexchange-accepted-tools"]["exported"], 12)


if __name__ == "__main__":
    unittest.main()
