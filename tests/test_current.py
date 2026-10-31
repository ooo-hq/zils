"""Zils Current: collectors, waiting room, resolvers, items and export."""

import contextlib
import gzip
import io
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import zils
from zils import current, current_github, current_items, current_wikimedia

FIXTURES = Path(__file__).with_name("current_fixtures")
LOGINS = ("alice-dev", "carol-x", "dave-k", "erin", "bob-owner")


def at(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


class FakeHttp:
    """Recorded responses only; any request without one fails the test."""

    def __init__(self, json_handler=None, files=None, sse=None):
        self.json_handler = json_handler or (lambda url, params: (404, {}, {}))
        self.files = files or {}
        self.sse = list(sse or [])
        self.calls = []

    def get_json(self, url, params=None, headers=None):
        self.calls.append((url, dict(params or {})))
        return self.json_handler(url, dict(params or {}))

    def get_bytes(self, url):
        self.calls.append((url, {}))
        return (200, self.files[url]) if url in self.files else (404, b"")

    def lines(self, url, params=None, headers=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return iter(self.sse.pop(0) if self.sse else [])


def archive_http():
    raw = gzip.compress((FIXTURES / "gharchive.jsonl").read_bytes())
    return FakeHttp(files={current_github.hour_url("2026-10-26-15"): raw})


def github_responses(overrides=None):
    issues = "https://api.github.com/repos/acme/widgets/issues/"
    responses = {
        issues + "7": {
            "state": "closed",
            "state_reason": "completed",
            "closed_at": "2026-10-27T10:00:00Z",
            "labels": [{"name": "type: bug"}, {"name": "P1"}],
            "closed_by": {"login": "maint", "type": "User"},
            "user": {"login": "alice-dev", "type": "User"},
        },
        issues + "9": {
            "state": "closed",
            "state_reason": "not_planned",
            "closed_at": "2026-10-29T10:00:00Z",
            "labels": [{"name": "enhancement"}],
            "closed_by": {"login": "stale[bot]", "type": "Bot"},
            "user": {"login": "erin", "type": "User"},
        },
        "https://api.github.com/repos/acme/widgets/pulls/8": {
            "state": "closed",
            "merged": True,
            "merged_at": "2026-10-26T18:00:00Z",
            "closed_at": "2026-10-26T18:00:00Z",
            "title": "Fix emoji filenames",
            "body": "Fixes #7. Thanks @alice-dev! Tested by dave-k on his laptop.",
            "user": {"login": "dave-k", "type": "User"},
        },
    }
    responses.update(overrides or {})

    def handler(url, params):
        body = responses.get(url)
        if isinstance(body, tuple):
            return body
        return (200, body, {}) if body is not None else (404, {"message": "Not Found"}, {})

    return handler


def collected(root):
    store = current.Store(root)
    current_github.collect(store, archive_http(), ["2026-10-26-15"], now=at("2026-10-31T12:00:00Z"))
    return store


class Snapshot:
    """The store's files, read before the temporary directory goes away."""

    def __init__(self, store):
        rows = store.pending(), store.resolved(), store.dropped()
        self.pending, self.resolved, self.dropped = (lambda r=r: r for r in rows)


def no_logins(test, value):
    text = json.dumps(value, ensure_ascii=False).lower()
    for login in LOGINS:
        test.assertNotIn(login, text)


class ScrubTest(unittest.TestCase):
    def test_removes_contacts_addresses_mentions_and_known_names(self):
        text = (
            "cc @carol-x, mail alice@example.com from 192.168.1.20 or 2001:db8:0:0:0:0:2:1 "
            "at 12:30:45. Thanks @bob. See https://github.com/alice-dev/notes, ALICE-DEV wrote it. "
            "Ask [[User:Some Person|them]] or User talk:Other_Person. Uses @pytest.fixture, @types/node."
        )
        clean = current_items.scrub(text, names=["alice-dev"])
        for leaked in (
            "carol-x",
            "alice@example.com",
            "192.168.1.20",
            "2001:db8",
            "@bob",
            "alice-dev",
            "ALICE-DEV",
            "Some Person",
            "Other_Person",
        ):
            self.assertNotIn(leaked, clean)
        for kept in ("12:30:45", "@pytest.fixture", "@types/node", "[email]", "[ip]", "@[user]"):
            self.assertIn(kept, clean)

    def test_mentions_before_markdown_punctuation_are_removed(self):
        clean = current_items.scrub("_Reviewed and posted by @someone-else._ Thanks @dev_ and @x.")
        self.assertEqual(clean, "_Reviewed and posted by @[user]._ Thanks @[user]_ and @[user].")

    def test_short_names_are_not_replaced_inside_words(self):
        self.assertEqual(
            current_items.scrub("ann wrote this, x too", names=["ann", "x"]),
            "[user] wrote this, x too",
        )
        self.assertEqual(current_items.scrub("planned", names=["ann"]), "planned")


class GHArchiveTest(unittest.TestCase):
    def test_hour_urls_follow_the_archive_naming(self):
        self.assertEqual(
            current_github.hour_url("2026-10-26-5"),
            "https://data.gharchive.org/2026-10-26-5.json.gz",
        )
        self.assertEqual(
            current_github.hour_range("2026-10-26-23", 2), ["2026-10-26-23", "2026-10-27-0"]
        )
        with self.assertRaises(ValueError):
            current_github.hour_range("2026-10-26-05", 1)

    def test_collects_human_issues_and_outside_prs_without_personal_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = collected(tmp)
            pending = store.pending()
        streams = sorted((r["stream"], r["source"]["number"]) for r in pending)
        self.assertEqual(
            streams,
            [
                ("github-issue-close", 7),
                ("github-issue-close", 9),
                ("github-issue-type", 7),
                ("github-issue-type", 9),
                ("github-pr-merge", 8),
            ],
        )
        issue = next(
            r for r in pending if r["stream"] == "github-issue-type" and r["source"]["number"] == 7
        )
        self.assertEqual(issue["asked_at"], "2026-10-26T15:00:02Z")
        self.assertEqual(issue["source"]["url"], "https://github.com/acme/widgets/issues/7")
        self.assertEqual(issue["source"]["event_id"], "1")
        self.assertEqual(issue["data"]["repo"], "widgets")
        self.assertIn("@[user]", issue["data"]["body"])
        self.assertIn("[email]", issue["data"]["body"])
        self.assertEqual(
            next(r for r in pending if r["source"]["number"] == 9)["data"]["labels_at_open"],
            ["enhancement"],
        )
        no_logins(self, [r["data"] for r in pending])

    def test_collecting_the_same_hour_twice_adds_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = collected(tmp)
            added = current_github.collect(
                store, archive_http(), ["2026-10-26-15"], now=at("2026-10-31T12:00:00Z")
            )
            self.assertEqual(added, 0)
            self.assertEqual(len(store.pending()), 5)

    def test_missing_hour_is_an_error(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "not published"),
        ):
            current_github.collect(
                current.Store(tmp), FakeHttp(), ["2026-10-26-16"], now=at("2026-10-31T12:00:00Z")
            )


class GitHubResolveTest(unittest.TestCase):
    def resolve(self, overrides=None, now="2026-10-31T12:00:00Z"):
        with tempfile.TemporaryDirectory() as tmp:
            store = collected(tmp)
            http = FakeHttp(github_responses(overrides))
            counts = current.resolve(store, http, now=at(now), max_age_days=14)
            return Snapshot(store), http, counts

    def test_labels_come_from_maintainer_actions(self):
        store, http, counts = self.resolve()
        resolved = {(r["stream"], r["source"]["number"]): r for r in store.resolved()}
        self.assertEqual(resolved[("github-issue-type", 7)]["label"], "bug")
        self.assertEqual(resolved[("github-issue-type", 9)]["label"], "feature")
        self.assertEqual(resolved[("github-issue-close", 7)]["label"], "true")
        self.assertEqual(resolved[("github-issue-close", 7)]["outcome_at"], "2026-10-27T10:00:00Z")
        pr = resolved[("github-pr-merge", 8)]
        self.assertEqual(pr["label"], "true")
        self.assertEqual(pr["resolved_at"], "2026-10-31T12:00:00Z")
        self.assertEqual(pr["data"]["title"], "Fix emoji filenames")
        self.assertEqual(pr["data"]["text_at"], "2026-10-31T12:00:00Z")
        no_logins(self, pr["data"])
        dropped = {(r["stream"], r["source"]["number"]): r["reason"] for r in store.dropped()}
        self.assertEqual(dropped, {("github-issue-close", 9): "closed-by-bot"})
        self.assertEqual(store.pending(), [])
        self.assertEqual(counts, {"resolved": 4, "dropped": 1, "pending": 0})
        self.assertEqual(len(http.calls), 3, "one API call per issue or PR")

    def test_open_items_wait_and_unmerged_or_deleted_items_resolve_or_drop(self):
        issues = "https://api.github.com/repos/acme/widgets/issues/"
        store, _, _ = self.resolve(
            {
                issues + "7": {
                    "state": "open",
                    "labels": [],
                    "user": {"login": "alice-dev", "type": "User"},
                },
                issues + "9": (410, {"message": "Gone"}, {}),
                "https://api.github.com/repos/acme/widgets/pulls/8": {
                    "state": "closed",
                    "merged": False,
                    "closed_at": "2026-10-28T00:00:00Z",
                    "title": "Fix emoji filenames",
                    "body": None,
                    "user": {"login": "dave-k", "type": "User"},
                },
            }
        )
        self.assertEqual(
            sorted((r["stream"], r["source"]["number"]) for r in store.pending()),
            [("github-issue-close", 7), ("github-issue-type", 7)],
        )
        self.assertEqual(store.resolved()[0]["label"], "false")
        self.assertEqual({r["reason"] for r in store.dropped()}, {"deleted"})

    def test_ambiguous_labels_bots_and_old_items_are_dropped(self):
        issues = "https://api.github.com/repos/acme/widgets/issues/"
        store, _, _ = self.resolve(
            {
                issues + "7": {
                    "state": "open",
                    "labels": [{"name": "bug"}, {"name": "Feature Request"}],
                    "user": {"login": "alice-dev", "type": "User"},
                },
                issues + "9": {
                    "state": "open",
                    "labels": [],
                    "user": {"login": "erin", "type": "User"},
                },
                "https://api.github.com/repos/acme/widgets/pulls/8": {
                    "state": "open",
                    "merged": False,
                    "title": "t",
                    "body": "b",
                    "user": {"login": "dave-k", "type": "Bot"},
                },
            },
            now="2026-11-10T12:00:00Z",
        )
        reasons = {(r["stream"], r["source"]["number"]): r["reason"] for r in store.dropped()}
        self.assertEqual(
            reasons,
            {
                ("github-issue-type", 7): "ambiguous-label",
                ("github-issue-close", 7): "expired",
                ("github-issue-type", 9): "expired",
                ("github-issue-close", 9): "expired",
                ("github-pr-merge", 8): "bot",
            },
        )

    def test_rate_limit_keeps_the_rest_pending(self):
        limited = (403, {"message": "API rate limit exceeded"}, {"x-ratelimit-remaining": "0"})
        issues = "https://api.github.com/repos/acme/widgets/issues/"
        store, _, counts = self.resolve({issues + "9": limited})
        self.assertTrue(counts["rate_limited"])
        self.assertIn(
            ("github-issue-type", 9),
            [(r["stream"], r["source"]["number"]) for r in store.pending()],
        )
        self.assertTrue(store.resolved())


class WikimediaTest(unittest.TestCase):
    def sse(self):
        return (FIXTURES / "recentchange.sse").read_text().splitlines()

    def test_sse_parser_reads_ids_and_data(self):
        events = list(current_wikimedia.sse_events(self.sse()))
        self.assertEqual(len(events), 9)
        last_id, first = events[0]
        self.assertIn("eqiad.mediawiki.recentchange", last_id)
        self.assertEqual(first["title"], "Green tree frog")

    def test_replay_keeps_human_article_edits_inside_the_window(self):
        http = FakeHttp(sse=[self.sse()])
        with tempfile.TemporaryDirectory() as tmp:
            store = current.Store(tmp)
            added = current_wikimedia.collect(
                store,
                http,
                minutes=5,
                since="2026-10-28T12:00:00Z",
                now=at("2026-10-31T12:00:00Z"),
            )
            pending = store.pending()
        self.assertEqual(added, 3)
        self.assertEqual(sorted(r["source"]["rev_id"] for r in pending), [101, 201, 601])
        self.assertEqual(http.calls[0][1], {"since": "2026-10-28T12:00:00Z"})
        frog = next(r for r in pending if r["source"]["rev_id"] == 101)
        self.assertEqual(frog["stream"], "wikipedia-revert")
        self.assertEqual(frog["asked_at"], "2026-10-28T12:00:01Z")
        self.assertEqual(
            frog["source"]["url"], "https://en.wikipedia.org/w/index.php?diff=101&oldid=100"
        )
        self.assertEqual(frog["data"]["editor"], "registered account")
        self.assertNotIn("Other Person", frog["data"]["comment"])
        self.assertNotIn("user", frog["data"])
        kinds = {r["source"]["rev_id"]: r["data"]["editor"] for r in pending}
        self.assertEqual(kinds[201], "unregistered (IP address)")
        self.assertEqual(kinds[601], "temporary account")
        self.assertNotIn("203.0.113.5", json.dumps(pending))

    def test_live_collection_stops_at_the_wall_clock_and_reconnects_from_the_last_id(self):
        ticks = iter([0, 0, 0, 0, 1e9])
        http = FakeHttp(sse=[self.sse()[:12], self.sse()[12:]])
        with tempfile.TemporaryDirectory() as tmp:
            current_wikimedia.collect(
                current.Store(tmp),
                http,
                minutes=5,
                now=at("2026-10-31T12:00:00Z"),
                clock=lambda: next(ticks, 1e9),
            )
        self.assertEqual(len(http.calls), 2)
        self.assertIn("Last-Event-ID", http.calls[1][2])

    def test_diff_text_reads_removed_and_added_lines(self):
        body = json.loads((FIXTURES / "mediawiki_compare.json").read_text())["compare"]["body"]
        removed, added = current_wikimedia.diff_text(body)
        self.assertEqual(removed, ["The frog lives in ponds."])
        self.assertEqual(
            added[0],
            "The frog lives in ponds & swamps. Ask [[User:Example Editor]] or mail ed@example.org",
        )
        self.assertEqual(added[1], "{{cite web|url=https://example.org/frogs}}")

    def wiki_handler(self, revisions, missing=()):
        compare = json.loads((FIXTURES / "mediawiki_compare.json").read_text())

        def handler(url, params):
            if params.get("action") == "query":
                ids = [int(i) for i in params["revids"].split("|")]
                pages = [{"title": "T", "revisions": [revisions[i]]} for i in ids if i in revisions]
                body = {"query": {"pages": pages}}
                if missing:
                    body["query"]["badrevids"] = {
                        str(i): {"revid": i, "missing": True} for i in missing
                    }
                return 200, body, {}
            if params.get("action") == "compare":
                return 200, compare, {}
            return 404, {}, {}

        return handler

    def collected(self, tmp):
        store = current.Store(tmp)
        current_wikimedia.collect(
            store,
            FakeHttp(sse=[self.sse()]),
            minutes=5,
            since="2026-10-28T12:00:00Z",
            now=at("2026-10-28T12:10:00Z"),
        )
        return store

    def test_edits_wait_then_resolve_from_the_reverted_tag(self):
        revisions = {
            101: {"revid": 101, "parentid": 100, "tags": ["mw-reverted", "visualeditor"]},
            201: {"revid": 201, "parentid": 200, "tags": []},
            601: {"revid": 601, "parentid": 600, "tags": ["mw-undo"]},
        }
        http = FakeHttp(self.wiki_handler(revisions))
        with tempfile.TemporaryDirectory() as tmp:
            store = self.collected(tmp)
            current.resolve(store, http, now=at("2026-10-29T12:00:00Z"), max_age_days=14)
            self.assertEqual(len(store.pending()), 3)
            self.assertEqual(http.calls, [], "nothing is checked before the waiting period")
            current.resolve(store, http, now=at("2026-10-30T13:00:00Z"), max_age_days=14)
            labels = {r["source"]["rev_id"]: r["label"] for r in store.resolved()}
            dropped = {r["source"]["rev_id"]: r["reason"] for r in store.dropped()}
            frog = next(r for r in store.resolved() if r["source"]["rev_id"] == 101)
        self.assertEqual(labels, {101: "true", 201: "false"})
        self.assertEqual(dropped, {601: "is-revert"})
        self.assertEqual(frog["data"]["removed"], ["The frog lives in ponds."])
        self.assertNotIn("Example Editor", json.dumps(frog["data"]))
        query = [c for c in http.calls if c[1].get("action") == "query"]
        self.assertEqual(len(query), 1, "revisions are checked in one batch")
        self.assertEqual(query[0][1]["maxlag"], "5")

    def test_deleted_or_hidden_revisions_are_dropped(self):
        revisions = {
            101: {"revid": 101, "parentid": 100, "tags": [], "texthidden": True},
            201: {"revid": 201, "parentid": 200, "tags": []},
        }
        with tempfile.TemporaryDirectory() as tmp:
            store = self.collected(tmp)
            current.resolve(
                store,
                FakeHttp(self.wiki_handler(revisions, missing=[601])),
                now=at("2026-10-31T12:00:00Z"),
                max_age_days=14,
            )
            dropped = {r["source"]["rev_id"]: r["reason"] for r in store.dropped()}
        self.assertEqual(dropped, {101: "deleted", 601: "deleted"})


def resolved_record(
    stream_name, number, label, data, asked="2026-10-26T15:00:00Z", resolved="2026-10-27T15:00:00Z"
):
    source = {"name": "test", "number": number, "url": f"https://example.org/{number}"}
    return {
        "id": f"{stream_name}-{number:016x}",
        "stream": stream_name,
        "source": source,
        "asked_at": asked,
        "resolved_at": resolved,
        "label": label,
        "data": data,
    }


def issue_data(title, body="Body text."):
    return {"repo": "widgets", "title": title, "body": body, "labels_at_open": []}


def wiki_data(title):
    return {
        "wiki": "enwiki",
        "title": title,
        "comment": "copyedit",
        "length_old": 100,
        "length_new": 90,
        "minor": False,
        "editor": "registered account",
        "removed": ["Old sentence."],
        "added": ["New sentence."],
    }


class ItemTest(unittest.TestCase):
    def test_items_use_the_bank_case_format(self):
        records = [
            resolved_record("github-issue-type", 1, "bug", issue_data("Crash on save")),
            resolved_record("github-issue-type", 2, "feature", issue_data("Add dark mode")),
            resolved_record("github-issue-close", 3, "true", issue_data("Crash on save")),
            resolved_record(
                "github-pr-merge",
                4,
                "false",
                {"repo": "widgets", "head": "fix", "base": "main", "title": "Fix", "body": "Body"},
            ),
            resolved_record("wikipedia-revert", 5, "true", wiki_data("Green tree frog")),
        ]
        cases = [current_items.build(r) for r in records]
        zils.validate_cases(cases)
        for case, record in zip(cases, records, strict=True):
            self.assertEqual(case["family"], record["stream"])
            self.assertEqual(case["source_id"], record["id"])
            self.assertEqual(case["source"], record["source"])
            self.assertEqual(set(case["state"]), {"text"})
            self.assertRegex(case["content_sha256"], "^[0-9a-f]{64}$")
            self.assertEqual(
                (case["asked_at"], case["resolved_at"]), (record["asked_at"], record["resolved_at"])
            )
            self.assertIn(case["category"], {"language", "reasoning"})
            self.assertTrue(case["licence"])
        bug, feature = cases[0], cases[1]
        self.assertEqual(bug["question"]["type"], "choice")
        self.assertEqual(sorted(bug["question"]["criteria"]), ["A", "B"])
        self.assertTrue(bug["question"]["criteria"][bug["label"]].startswith("Bug report"))
        self.assertTrue(
            feature["question"]["criteria"][feature["label"]].startswith("Feature request")
        )
        self.assertEqual(cases[3]["question"]["type"], "noul")
        self.assertEqual(cases[3]["label"], "false")
        self.assertIn("Removed:", cases[4]["state"]["text"])
        self.assertIn("Green tree frog", cases[4]["state"]["text"])

    def test_long_states_are_truncated_to_the_model_budget(self):
        body = ("A paragraph about the crash. " * 40 + "\n\n") * 60
        record = resolved_record("github-issue-type", 1, "bug", issue_data("Crash", body))
        lite = current_items.build(record, max_tokens=4096)["state"]["text"]
        mini = current_items.build(record, max_tokens=8192)["state"]["text"]
        self.assertLessEqual(len(lite), current_items.state_chars(4096))
        self.assertLessEqual(len(mini), current_items.state_chars(8192))
        self.assertGreater(len(mini), len(lite))
        self.assertTrue(lite.startswith("Repository: widgets\nIssue title: Crash\n"))
        self.assertTrue(lite.endswith("[truncated]"))
        cjk = resolved_record(
            "github-issue-type", 3, "bug", issue_data("Crash", "保存时崩溃。" * 2000)
        )
        text = current_items.build(cjk, max_tokens=4096)["state"]["text"]
        self.assertLessEqual(
            current_items.estimated_tokens(text), 4096 - current_items.QUESTION_TOKENS
        )
        self.assertGreater(current_items.estimated_tokens(text), 3000)
        self.assertGreaterEqual(
            current_items.estimated_tokens("保存"), 2, "one token per non-ASCII char"
        )
        self.assertGreaterEqual(current_items.estimated_tokens("|1||0||"), 4, "markup is dense")
        short = current_items.build(
            resolved_record("github-issue-type", 2, "bug", issue_data("Crash"))
        )
        self.assertNotIn("[truncated]", short["state"]["text"])


class ExportTest(unittest.TestCase):
    def store(self, tmp, records):
        store = current.Store(tmp)
        store.add(records)
        store.save([], records, [])
        return store

    def test_export_dedupes_balances_and_reports(self):
        records = [
            resolved_record(
                "wikipedia-revert",
                i,
                "true" if i < 3 else "false",
                wiki_data(f"Page {i}"),
                resolved=f"2026-10-{27 + i % 2}T15:00:00Z",
            )
            for i in range(8)
        ]
        records.append(
            resolved_record("wikipedia-revert", 99, "false", wiki_data("Page 5"))
        )  # duplicate text
        records.append(
            resolved_record(
                "github-pr-merge",
                50,
                "true",
                {"repo": "w", "head": "h", "base": "m", "title": "T", "body": ""},
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store(tmp, records)
            store.save([], [], [{**records[0], "id": "gone", "reason": "expired"}])
            out = Path(tmp) / "items.jsonl"
            stats = current.export(store, out, seed=7)
            items = [json.loads(line) for line in out.read_text().splitlines()]
            again = current.export(store, Path(tmp) / "again.jsonl", seed=7)
            same = [
                json.loads(line) for line in (Path(tmp) / "again.jsonl").read_text().splitlines()
            ]
        zils.validate_cases(items)
        self.assertEqual(items, same)
        self.assertEqual(stats, again)
        wiki = [c for c in items if c["family"] == "wikipedia-revert"]
        self.assertEqual(sorted(c["label"] for c in wiki), ["false"] * 3 + ["true"] * 3)
        self.assertEqual(len({c["content_sha256"] for c in items}), len(items))
        w = stats["streams"]["wikipedia-revert"]
        self.assertEqual(w["resolved"], 9)
        self.assertEqual(w["duplicates"], 1)
        self.assertEqual(w["labels"], {"false": 5, "true": 3})
        self.assertEqual(w["exported_labels"], {"false": 3, "true": 3})
        self.assertEqual(w["dropped"], {"expired": 1})
        self.assertEqual(w["median_hours_to_resolution"], 24.0)
        pr = stats["streams"]["github-pr-merge"]
        self.assertEqual(pr["exported"], 0, "a stream with one outcome exports nothing")

    def test_stats_separate_the_real_outcome_time_from_our_check_time(self):
        pr = {"repo": "w", "head": "h", "base": "m", "title": "T", "body": ""}
        records = [
            {
                **resolved_record("github-pr-merge", i, label, {**pr, "title": f"T{i}"}),
                "outcome_at": outcome,
            }
            for i, (label, outcome) in enumerate(
                [("true", "2026-10-26T16:00:00Z"), ("false", "2026-10-26T18:00:00Z")]
            )
        ]
        records.append(resolved_record("wikipedia-revert", 9, "true", wiki_data("Page")))
        with tempfile.TemporaryDirectory() as tmp:
            stats = current.export(self.store(tmp, records), Path(tmp) / "x.jsonl")["streams"]
        self.assertEqual(stats["github-pr-merge"]["median_hours_to_resolution"], 24.0)
        self.assertEqual(stats["github-pr-merge"]["median_hours_to_outcome"], 2.0)
        self.assertIsNone(stats["wikipedia-revert"]["median_hours_to_outcome"])

    def test_unbalanced_export_keeps_everything(self):
        records = [
            resolved_record("wikipedia-revert", i, "false", wiki_data(f"Page {i}"))
            for i in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            stats = current.export(self.store(tmp, records), Path(tmp) / "x.jsonl", balance=False)
        self.assertEqual(stats["streams"]["wikipedia-revert"]["exported"], 3)


class CommandLineTest(unittest.TestCase):
    def test_collect_resolve_export(self):
        http = archive_http()
        http.json_handler = github_responses()
        now = at("2026-10-31T12:00:00Z")
        with tempfile.TemporaryDirectory() as tmp:
            for args in (
                ["collect", "--source", "gharchive", "--start", "2026-10-26-15", "--hours", "1"],
                ["resolve"],
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(current.main(["--dir", tmp, *args], http=http, now=now), 0)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = current.main(
                    [
                        "--dir",
                        tmp,
                        "export",
                        "--out",
                        str(Path(tmp) / "items.jsonl"),
                        "--stats",
                        "--no-balance",
                    ],
                    http=http,
                    now=now,
                )
            self.assertEqual(code, 0)
            stats = json.loads(out.getvalue())
            self.assertEqual(stats["streams"]["github-issue-type"]["exported"], 2)
            self.assertEqual(stats["version"], current.VERSION)

    def test_module_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "zils.current", "--help"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("collect", result.stdout)

    def test_user_agent_names_a_contact(self):
        agent = current.user_agent()
        self.assertRegex(agent, r"^zils-current/\S+ \(.+\)")


if __name__ == "__main__":
    unittest.main()
