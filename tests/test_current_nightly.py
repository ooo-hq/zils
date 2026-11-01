"""Zils Current nightly runner: cursors, idempotent reruns, resumption, lock and run log."""

import contextlib
import fcntl
import gzip
import io
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from tests.test_current import (  # noqa: F401 (setUpModule)
    FIXTURES,
    FakeHttp,
    at,
    github_responses,
    setUpModule,
)
from zils import current, current_github, current_nightly

NOW = at("2026-10-31T12:30:00Z")


class NightlyHttp(FakeHttp):
    """Archive hours, a replayable recentchange stream and the two APIs, all recorded."""

    def __init__(self, hours):
        archive = gzip.compress((FIXTURES / "gharchive.jsonl").read_bytes())
        empty = gzip.compress(b"")
        files = {
            current_github.hour_url(h): archive if i == 0 else empty for i, h in enumerate(hours)
        }
        compare = json.loads((FIXTURES / "mediawiki_compare.json").read_text())
        github = github_responses()

        def handler(url, params):
            if "api.github.com" in url:
                return github(url, params)
            if params.get("action") == "query":
                ids = [int(i) for i in params["revids"].split("|")]
                revisions = [{"revid": i, "parentid": i - 1, "tags": []} for i in ids]
                return 200, {"query": {"pages": [{"revisions": revisions}]}}, {}
            return 200, compare, {}

        super().__init__(json_handler=handler, files=files)
        self.sse_lines = (FIXTURES / "recentchange.sse").read_text().splitlines()

    def lines(self, url, params=None, headers=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return iter(self.sse_lines if "since" in (params or {}) else [])

    def downloads(self):
        return [c[0] for c in self.calls if c[0].startswith("https://data.gharchive.org")]

    def replays(self):
        return [c[1]["since"] for c in self.calls if len(c) == 3 and "since" in c[1]]


class NightlyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "state"
        self.bank = Path(self.tmp.name) / "bank"
        self.options = {
            "gh_first_hours": 3,
            "wiki_first_hours": 2,
            "wiki_minutes": 2,
            "check_overlap": False,
            "se_pages": 0,
            "afd_first_days": 0,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def run_once(self, http, now=NOW, **options):
        return current_nightly.run(self.state, self.bank, http, now, **{**self.options, **options})

    def test_first_run_collects_resolves_exports_and_logs(self):
        http = NightlyHttp(["2026-10-31-9", "2026-10-31-10"])
        result = self.run_once(http)
        self.assertEqual(result["errors"], [])
        self.assertEqual(
            http.downloads(),
            [current_github.hour_url(h) for h in ("2026-10-31-9", "2026-10-31-10")],
        )
        self.assertEqual(http.replays(), ["2026-10-31T10:00:00Z", "2026-10-31T11:00:00Z"])
        cursor = json.loads((self.state / "cursor.json").read_text())
        self.assertEqual(cursor["gharchive"], "2026-10-31-10")
        self.assertEqual(cursor["wikimedia"], "2026-10-31T11:00:00Z")
        self.assertEqual(cursor["export"], "2026-10-30")
        self.assertTrue((self.bank / "2026-10-30" / "manifest.json").exists())
        self.assertEqual(result["collected"]["gharchive"]["added"], 5)
        # Four English edits and one German edit: the default wiki list reads both.
        self.assertEqual(result["collected"]["wikimedia"]["added"], 5)
        self.assertGreater(result["resolve"]["resolved"], 0)
        log = [json.loads(line) for line in (self.state / "runs.jsonl").read_text().splitlines()]
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["started_at"], "2026-10-31T12:30:00Z")
        stats = json.loads((self.state / "stats.json").read_text())
        self.assertIn("wikipedia-revert", stats["streams"])

    def test_a_rerun_downloads_and_exports_nothing_new(self):
        self.run_once(NightlyHttp(["2026-10-31-9", "2026-10-31-10"]))
        manifest = (self.bank / "2026-10-30" / "manifest.json").read_text()
        again = NightlyHttp(["2026-10-31-9", "2026-10-31-10"])
        result = self.run_once(again)
        self.assertEqual(again.downloads(), [])
        self.assertEqual(again.replays(), [])
        self.assertEqual(result["exports"], [])
        self.assertEqual((self.bank / "2026-10-30" / "manifest.json").read_text(), manifest)
        self.assertEqual(len((self.state / "runs.jsonl").read_text().splitlines()), 2)

    def test_an_unpublished_hour_is_retried_next_time(self):
        http = NightlyHttp(["2026-10-31-9"])
        self.run_once(http)
        self.assertEqual(
            json.loads((self.state / "cursor.json").read_text())["gharchive"], "2026-10-31-9"
        )
        later = NightlyHttp(["2026-10-31-10", "2026-10-31-11"])
        self.run_once(later, now=NOW + timedelta(hours=1))
        self.assertEqual(
            later.downloads(),
            [current_github.hour_url(h) for h in ("2026-10-31-10", "2026-10-31-11")],
        )

    def test_missed_days_are_exported_in_order(self):
        self.run_once(NightlyHttp(["2026-10-31-9", "2026-10-31-10"]))
        result = self.run_once(
            NightlyHttp([]), now=NOW + timedelta(days=2), gh_max_hours=0, wiki_max_hours=0
        )
        self.assertEqual([e["date"] for e in result["exports"]], ["2026-10-31", "2026-11-01"])

    def test_a_running_job_makes_the_next_one_wait(self):
        self.state.mkdir(parents=True)
        with (self.state / "nightly.lock").open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            http = NightlyHttp(["2026-10-31-9"])
            result = self.run_once(http)
        self.assertEqual(result, {"status": "busy"})
        self.assertEqual(http.calls, [])

    def test_failures_are_logged_and_later_steps_still_run(self):
        class Broken(NightlyHttp):
            def get_bytes(self, url):
                raise OSError("network down")

        result = self.run_once(Broken(["2026-10-31-9"]))
        self.assertEqual(result["errors"][0]["step"], "gharchive")
        self.assertIn("resolve", result)
        self.assertTrue((self.bank / "2026-10-30" / "manifest.json").exists())

    def test_command_line(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = current.main(
                [
                    "nightly",
                    "--state",
                    str(self.state),
                    "--bank",
                    str(self.bank),
                    "--no-overlap-check",
                    "--gh-first-hours",
                    "3",
                    "--wiki-first-hours",
                    "2",
                    "--wiki-minutes",
                    "2",
                    "--se-pages",
                    "0",
                    "--afd-first-days",
                    "0",
                    "--wiki-source",
                    "stream",
                ],
                http=NightlyHttp(["2026-10-31-9", "2026-10-31-10"]),
                now=NOW,
            )
        self.assertEqual(code, 0)
        self.assertTrue((self.state / "runs.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
