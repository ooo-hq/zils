"""Zils Current sources beyond the pilot: more Wikipedias, Wikidata, sampled resolution."""

import json
import tempfile
import unittest
from unittest.mock import patch

import zils
from tests.test_current import FIXTURES, FakeHttp, at
from zils import current, current_bank, current_items, current_wikimedia


def event(wiki, rev, *, title="Page", namespace=0, comment="copyedit", bot=False, user="Ed Itor"):
    server = {
        "enwiki": "https://en.wikipedia.org",
        "dewiki": "https://de.wikipedia.org",
        "wikidatawiki": "https://www.wikidata.org",
    }.get(wiki, f"https://{wiki[:2]}.wikipedia.org")
    return {
        "meta": {"domain": server.removeprefix("https://"), "dt": "2026-10-28T12:00:01Z"},
        "type": "edit",
        "namespace": namespace,
        "title": title,
        "comment": comment,
        "timestamp": 1793188801,
        "user": user,
        "bot": bot,
        "minor": False,
        "length": {"old": 100, "new": 120},
        "revision": {"old": rev - 1, "new": rev},
        "server_url": server,
        "server_script_path": "/w",
        "wiki": wiki,
    }


WIKIS = set(current_wikimedia.DEFAULT_WIKIS)


class WikiListTest(unittest.TestCase):
    def test_default_list_covers_nine_wikipedias_and_wikidata(self):
        self.assertEqual(
            set(current_wikimedia.DEFAULT_WIKIS),
            {
                "enwiki",
                "dewiki",
                "frwiki",
                "eswiki",
                "jawiki",
                "itwiki",
                "ptwiki",
                "ruwiki",
                "zhwiki",
                "wikidatawiki",
            },
        )

    def test_each_wikipedia_is_its_own_stream_english_keeps_the_pilot_name(self):
        en = current_wikimedia.record(event("enwiki", 11), WIKIS)
        de = current_wikimedia.record(event("dewiki", 12), WIKIS)
        ja = current_wikimedia.record(event("jawiki", 13), WIKIS)
        self.assertEqual(en["stream"], "wikipedia-revert")
        self.assertEqual(de["stream"], "wikipedia-revert-de")
        self.assertEqual(ja["stream"], "wikipedia-revert-ja")
        self.assertIsNone(current_wikimedia.record(event("nlwiki", 14), WIKIS))
        for stream in ("wikipedia-revert-de", "wikipedia-revert-ja", "wikidata-revert"):
            spec = current_items.STREAMS[stream]
            self.assertTrue(spec["publishable"])
            self.assertIn("CC", spec["licence"])

    def test_pooled_records_from_the_pilot_release_are_split_by_wiki_on_read(self):
        legacy = current_wikimedia.record(event("frwiki", 21), WIKIS)
        legacy["stream"] = "wikipedia-revert"
        english = current_wikimedia.record(event("enwiki", 22), WIKIS)
        with tempfile.TemporaryDirectory() as tmp:
            store = current.Store(tmp)
            store.add([legacy, english])
            streams = sorted(r["stream"] for r in store.pending())
        self.assertEqual(streams, ["wikipedia-revert", "wikipedia-revert-fr"])

    def test_wiki_name_and_language_appear_in_the_state(self):
        record = current_wikimedia.record(event("dewiki", 31, title="Frosch"), WIKIS)
        record.update(
            label="false",
            resolved_at="2026-10-30T12:00:00Z",
            data={**record["data"], "removed": ["Alt."], "added": ["Neu."]},
        )
        case = current_items.build(record)
        zils.validate_cases([case])
        self.assertIn("Wiki: German Wikipedia", case["state"]["text"])
        self.assertEqual(case["category"], "language")

    def test_revert_wording_in_other_languages_is_a_leak(self):
        for stream, comment in (
            ("wikipedia-revert-de", "Änderungen zurückgesetzt"),
            ("wikipedia-revert-fr", "Révocation des modifications"),
            ("wikipedia-revert-ru", "откат правок"),
            ("wikipedia-revert-ja", "差し戻し"),
            ("wikipedia-revert-zh", "回退到之前的版本"),
            ("wikipedia-revert-es", "Revertidos los cambios"),
        ):
            record = {"stream": stream, "data": {}}
            self.assertEqual(current_bank.leak_reason(record, comment), "states-outcome", stream)


def wiki_handler(revisions, compare=None, labels=None):
    compare = compare or json.loads((FIXTURES / "mediawiki_compare.json").read_text())
    calls = {"compare": 0}

    def handler(url, params):
        if params.get("action") == "query":
            ids = [int(i) for i in params["revids"].split("|")]
            pages = [{"title": "T", "revisions": [revisions[i]]} for i in ids if i in revisions]
            return 200, {"query": {"pages": pages}}, {}
        if params.get("action") == "compare":
            calls["compare"] += 1
            return 200, compare, {}
        if params.get("action") == "wbgetentities":
            ids = params["ids"].split("|")
            return (
                200,
                {
                    "entities": {
                        i: {"labels": {"en": {"value": (labels or {}).get(i, i)}}} for i in ids
                    }
                },
                {},
            )
        return 404, {}, {}

    return handler, calls


class SampledResolutionTest(unittest.TestCase):
    def test_kept_edits_per_revert_bound_the_diff_requests(self):
        records = [current_wikimedia.record(event("enwiki", 100 + i), WIKIS) for i in range(20)]
        revisions = {
            100 + i: {"revid": 100 + i, "tags": ["mw-reverted"] if i < 2 else []} for i in range(20)
        }
        handler, calls = wiki_handler(revisions)
        outcomes = list(
            current_wikimedia.resolve(
                records, FakeHttp(handler), at("2026-10-31T12:00:00Z"), negatives_per_positive=2
            )
        )
        resolved = [r for s, r in outcomes if s == "resolved"]
        sampled = [r for s, r in outcomes if s == "dropped" and r["reason"] == "sampled-out"]
        self.assertEqual(sorted(r["label"] for r in resolved), ["false"] * 4 + ["true"] * 2)
        self.assertEqual(len(sampled), 14)
        self.assertEqual(calls["compare"], 6, "only kept edits fetch their diff")
        again = list(
            current_wikimedia.resolve(
                records,
                FakeHttp(wiki_handler(revisions)[0]),
                at("2026-10-31T12:00:00Z"),
                negatives_per_positive=2,
            )
        )
        self.assertEqual(
            sorted(r["id"] for s, r in again if s == "resolved"),
            sorted(r["id"] for r in resolved),
            "the sample is deterministic",
        )

    def test_without_a_ratio_every_edit_is_kept(self):
        records = [current_wikimedia.record(event("enwiki", 200 + i), WIKIS) for i in range(5)]
        revisions = {200 + i: {"revid": 200 + i, "tags": []} for i in range(5)}
        handler, calls = wiki_handler(revisions)
        outcomes = list(
            current_wikimedia.resolve(records, FakeHttp(handler), at("2026-10-31T12:00:00Z"))
        )
        self.assertEqual([s for s, _ in outcomes], ["resolved"] * 5)
        self.assertEqual(calls["compare"], 5)

    def test_the_resolve_command_passes_the_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = current.Store(tmp)
            store.add([current_wikimedia.record(event("enwiki", 300 + i), WIKIS) for i in range(6)])
            revisions = {
                300 + i: {"revid": 300 + i, "tags": ["mw-reverted"] if i == 0 else []}
                for i in range(6)
            }
            handler, _ = wiki_handler(revisions)
            counts = current.resolve(
                store, FakeHttp(handler), at("2026-10-31T12:00:00Z"), negatives_per_positive=1
            )
        self.assertEqual(counts["resolved"], 2)
        self.assertEqual(counts["dropped"], 4)


WIKIDATA_DIFF = (
    '<tr><td colspan="2" class="diff-lineno"></td><td colspan="2" class="diff-lineno">'
    'Property / <a title="Property:P569" href="/wiki/Property:P569">date of birth</a></td></tr>'
    '<tr><td colspan="2"> </td><td class="diff-marker" data-marker="+"></td>'
    '<td class="diff-addedline"><div><ins class="diffchange">'
    "<span>3 March 1961</span></ins></div></td></tr>"
    '<tr><td colspan="2" class="diff-lineno">description / en</td>'
    '<td colspan="2" class="diff-lineno">description / en</td></tr>'
    '<tr><td class="diff-marker" data-marker="-"></td><td class="diff-deletedline"><div>'
    "<del>American actor</del></div></td>"
    '<td class="diff-marker" data-marker="+"></td><td class="diff-addedline"><div>'
    "<ins>American actor and singer, see User:Ed Itor</ins></div></td></tr>"
)


class WikidataTest(unittest.TestCase):
    def test_manual_item_edits_are_kept_tool_batches_are_skipped(self):
        manual = current_wikimedia.record(
            event(
                "wikidatawiki",
                41,
                title="Q42",
                comment="/* wbsetclaim-create:2||1 */ [[Property:P569]]: 3 March 1961",
            ),
            WIKIS,
        )
        self.assertEqual(manual["stream"], "wikidata-revert")
        self.assertEqual(manual["source"]["item"], "Q42")
        for comment in (
            "/* wbeditentity-update:0| */ QuickStatements 3.0 [[:toollabs:qs-dev/batch/1|batch #1]]",
            "/* wbcreateclaim-create:1| */ [[Property:P50]]: [[Q1]], #quickstatements",
            "/* wbeditentity-update:0| */ #openrefine",
        ):
            self.assertIsNone(
                current_wikimedia.record(event("wikidatawiki", 42, comment=comment), WIKIS)
            )
        self.assertIsNone(
            current_wikimedia.record(event("wikidatawiki", 43, namespace=120), WIKIS),
            "items only",
        )

    def test_resolution_reads_the_item_label_and_labelled_diff_lines(self):
        record = current_wikimedia.record(
            event(
                "wikidatawiki",
                51,
                title="Q42",
                comment="/* wbsetclaim-create:2||1 */ [[Property:P569]]: 3 March 1961",
            ),
            WIKIS,
        )
        handler, _ = wiki_handler(
            {51: {"revid": 51, "tags": ["mw-reverted"], "user": "Ed Itor"}},
            compare={"compare": {"body": WIKIDATA_DIFF}},
            labels={"Q42": "Douglas Adams"},
        )
        http = FakeHttp(handler)
        [(status, resolved)] = list(
            current_wikimedia.resolve([record], http, at("2026-10-31T12:00:00Z"))
        )
        self.assertEqual(status, "resolved")
        self.assertEqual(resolved["label"], "true")
        self.assertEqual(resolved["data"]["label"], "Douglas Adams")
        self.assertEqual(resolved["data"]["added"][0], "Property / date of birth: 3 March 1961")
        self.assertEqual(resolved["data"]["removed"], ["description / en: American actor"])
        compare = next(c for c in http.calls if c[1].get("action") == "compare")
        self.assertEqual(compare[1]["uselang"], "en")
        case = current_items.build(resolved)
        zils.validate_cases([case])
        text = case["state"]["text"]
        self.assertIn("Item: Douglas Adams (Q42)", text)
        self.assertIn("Wiki: Wikidata", text)
        self.assertIn("set claim (create)", text)
        self.assertNotIn("Ed Itor", text)
        self.assertEqual(case["category"], "retrieval")
        self.assertEqual(
            case["question"]["instructions"],
            "Will this Wikidata item edit be reverted by another edit?",
        )


class ReplayRetryTest(unittest.TestCase):
    def test_a_failed_connection_is_retried_before_giving_up(self):
        lines = (FIXTURES / "recentchange.sse").read_text().splitlines()

        class Flaky(FakeHttp):
            def lines(self, url, params=None, headers=None):
                self.calls.append((url, dict(params or {}), dict(headers or {})))
                if len(self.calls) == 1:
                    raise OSError("read timed out")
                return iter(lines if len(self.calls) == 2 else [])

        waits = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(current_wikimedia, "sleep", waits.append),
        ):
            added = current_wikimedia.collect(
                current.Store(tmp),
                Flaky(),
                5,
                since="2026-10-28T12:00:00Z",
                now=at("2026-10-31T12:00:00Z"),
                wikis=["enwiki"],
            )
        self.assertEqual(added, 3)
        self.assertEqual(len(waits), 1)

    def test_a_burst_of_later_events_from_the_other_datacentre_does_not_end_the_replay(self):
        lines = (FIXTURES / "recentchange.sse").read_text().splitlines()
        later = []
        for i in range(current_wikimedia.LATE_RUN + 100):
            row = event("enwiki", 5000 + i)
            row["meta"]["dt"] = "2026-10-30T09:00:00Z"
            later += ["event: message", "id: x", "data: " + json.dumps(row), ""]
        with tempfile.TemporaryDirectory() as tmp:
            added = current_wikimedia.collect(
                current.Store(tmp),
                FakeHttp(sse=[later + lines]),
                5,
                since="2026-10-28T12:00:00Z",
                now=at("2026-10-31T12:00:00Z"),
                wikis=["enwiki"],
            )
        self.assertEqual(added, 3)


class PoliteApiTest(unittest.TestCase):
    def test_too_many_requests_waits_for_retry_after_then_continues(self):
        answers = [(429, {}, {"retry-after": "7"}), (200, {"ok": 1}, {})]
        http = FakeHttp(lambda url, params: answers.pop(0))
        waits = []
        with patch.object(current_wikimedia, "sleep", waits.append):
            status, body = current_wikimedia._api(http, "https://x/api.php", {})
        self.assertEqual((status, body), (200, {"ok": 1}))
        self.assertIn(7.0, waits)

    def test_requests_are_spaced_and_a_lasting_limit_stops_the_pass(self):
        http = FakeHttp(lambda url, params: (429, {}, {}))
        waits = []
        with (
            patch.object(current_wikimedia, "sleep", waits.append),
            self.assertRaises(current_items.RateLimited),
        ):
            current_wikimedia._api(http, "https://x/api.php", {})
        self.assertEqual(len(http.calls), current_wikimedia.LIMIT_RETRIES + 1)

    def test_the_http_client_spaces_requests_to_one_host(self):
        class Response:
            status_code, headers = 200, {}

            def json(self):
                return {}

        class Session:
            headers = {}

            def get(self, url, **kwargs):
                return Response()

        waits = []
        times = iter([0.0, 0.0, 0.1, 0.1, 0.1, 0.1])
        with (
            patch.object(current.time, "sleep", waits.append),
            patch.object(current.time, "monotonic", lambda: next(times)),
        ):
            http = current.Http(session=Session(), gap=0.5)
            http.get_json("https://en.wikipedia.org/w/api.php")
            http.get_json("https://en.wikipedia.org/w/api.php")
            http.get_json("https://www.wikidata.org/w/api.php")
        self.assertEqual(len(waits), 1)
        self.assertAlmostEqual(waits[0], 0.4)
        self.assertIn("zils-current/", Session.headers["User-Agent"])


class CheckpointTest(unittest.TestCase):
    def test_a_long_pass_saves_what_it_has_resolved_as_it_goes(self):
        records = [current_wikimedia.record(event("enwiki", 700 + i), WIKIS) for i in range(5)]

        seen_saved = []

        def resolver(rows, http, now, **options):
            for i, row in enumerate(rows[:3]):
                if i == 2:
                    seen_saved.append(len(store.resolved()))
                yield "resolved", {**row, "label": "false", "resolved_at": "2026-10-31T00:00:00Z"}
            raise RuntimeError("killed")

        with tempfile.TemporaryDirectory() as tmp:
            store = current.Store(tmp)
            store.add(records)
            with (
                patch.dict(current.RESOLVERS, {"wikipedia-revert": resolver}),
                patch.object(current, "CHECKPOINT", 2),
                self.assertRaises(RuntimeError),
            ):
                current.resolve(store, FakeHttp(), at("2026-10-31T12:00:00Z"))
            self.assertEqual(seen_saved, [2], "saved at the checkpoint, during the pass")
            self.assertEqual(len(store.resolved()), 3, "and again when the pass stops")
            self.assertEqual(len(store.pending()), 2)


def change(rev, tags=(), *, wiki_bot=False, comment="copyedit", ns=0, user="Ed Itor"):
    row = {
        "type": "edit",
        "ns": ns,
        "title": f"Page {rev}",
        "revid": rev,
        "old_revid": rev - 1,
        "oldlen": 100,
        "newlen": 90,
        "user": user,
        "comment": comment,
        "timestamp": "2026-10-25T10:00:00Z",
        "tags": list(tags),
        # formatversion=2: flags are always present, as booleans
        "bot": wiki_bot,
        "minor": False,
        "new": False,
    }
    return row


class RecentChangesTest(unittest.TestCase):
    def test_bulk_pull_keeps_reverts_and_a_sample_of_kept_edits(self):
        changes = [change(1000 + i, ["mw-reverted"] if i < 3 else []) for i in range(30)]
        changes += [change(2000, ["mw-undo"]), change(2001, ["mw-reverted"], wiki_bot=True)]
        calls = []

        def handler(url, params):
            calls.append((url, params))
            return 200, {"query": {"recentchanges": changes}}, {}

        with tempfile.TemporaryDirectory() as tmp:
            store = current.Store(tmp)
            result = current_wikimedia.collect_recent(
                store,
                FakeHttp(handler),
                at("2026-10-31T12:00:00Z"),
                wikis=["enwiki"],
                samples=1,
                negatives_per_positive=1,
            )
            pending = store.pending()
        self.assertEqual(result["added"], 6)
        self.assertEqual(result["positives"], {"enwiki": 3})
        self.assertEqual(len(pending), 6)
        frog = pending[0]
        self.assertEqual(frog["stream"], "wikipedia-revert")
        self.assertEqual(frog["source"]["api"], "https://en.wikipedia.org/w/api.php")
        self.assertEqual(frog["asked_at"], "2026-10-25T10:00:00Z")
        self.assertEqual(frog["data"]["editor"], "registered account")
        self.assertNotIn("Ed Itor", json.dumps(pending))
        params = calls[0][1]
        self.assertEqual(params["list"], "recentchanges")
        self.assertEqual(params["rcshow"], "!bot")
        self.assertEqual(params["rclimit"], "500")
        self.assertIn("tags", params["rcprop"])

    def test_windows_spread_over_the_age_range_and_stop_at_the_target(self):
        seen = []

        def handler(url, params):
            seen.append(params["rcstart"])
            rows = [change(3000 + len(seen) * 10 + i, ["mw-reverted"]) for i in range(5)]
            return 200, {"query": {"recentchanges": rows}}, {}

        with tempfile.TemporaryDirectory() as tmp:
            result = current_wikimedia.collect_recent(
                current.Store(tmp),
                FakeHttp(handler),
                at("2026-10-31T12:00:00Z"),
                wikis=["dewiki"],
                samples=10,
                min_days=2,
                max_days=12,
                target_positives=12,
            )
        self.assertEqual(len(seen), 3, "stops once the target is reached")
        self.assertTrue(all("2026-10-19" <= s <= "2026-10-29T12:00:00Z" for s in seen))
        self.assertEqual(result["positives"], {"dewiki": 15})

    def test_the_collect_command_and_the_nightly_mode_use_the_bulk_pull(self):
        import contextlib
        import io
        from pathlib import Path

        from zils import current_nightly

        def handler(url, params):
            if params.get("list") == "recentchanges":
                rows = [change(4000 + i, ["mw-reverted"] if i < 2 else []) for i in range(6)]
                return 200, {"query": {"recentchanges": rows}}, {}
            return 404, {}, {}

        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                current.main(
                    ["--dir", tmp, "collect", "--source", "recentchanges", "--wiki", "enwiki"]
                    + ["--samples", "1", "--negatives-per-positive", "1"],
                    http=FakeHttp(handler),
                    now=at("2026-10-31T12:00:00Z"),
                )
            self.assertEqual(json.loads(out.getvalue())["added"], 4)
            result = current_nightly.run(
                Path(tmp) / "s",
                Path(tmp) / "b",
                FakeHttp(handler),
                at("2026-10-31T12:00:00Z"),
                check_overlap=False,
                gh_max_hours=0,
                se_pages=0,
                afd_first_days=0,
                wikis=["enwiki"],
                wiki_source="recentchanges",
                wiki_samples=1,
            )
        self.assertEqual(result["collected"]["wikimedia"]["added"], 4)


class NetworkFailureTest(unittest.TestCase):
    def test_a_network_failure_ends_one_source_and_the_others_still_resolve(self):
        def broken(rows, http, now, **options):
            raise OSError("Temporary failure in name resolution")
            yield

        ready = {
            "id": "wikipedia-afd-1",
            "stream": "wikipedia-afd",
            "asked_at": "2026-10-04T09:25:00Z",
            "source": {"url": "u", "title": "t"},
            "data": {"article": "A", "text": "x"},
            "outcome": {"label": "keep", "outcome_at": "2026-10-12T00:00:00Z"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            store = current.Store(tmp)
            store.add([current_wikimedia.record(event("enwiki", 900), WIKIS), ready])
            with patch.dict(current.RESOLVERS, {"wikipedia-revert": broken}):
                counts = current.resolve(store, FakeHttp(), at("2026-10-31T12:00:00Z"))
        self.assertEqual(counts["resolved"], 1)
        self.assertTrue(counts["rate_limited"])
        self.assertEqual(list(current.RESOLVERS)[:2], ["stackexchange-", "wikipedia-afd"])

    def test_a_bulk_pull_skips_a_wiki_that_stays_lagged(self):
        def handler(url, params):
            if "wikidata" in url:
                return 200, {"error": {"code": "maxlag"}}, {"retry-after": "1"}
            return (
                200,
                {"query": {"recentchanges": [change(5000, ["mw-reverted"]), change(5001)]}},
                {},
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(current_wikimedia, "sleep", lambda s: None),
        ):
            result = current_wikimedia.collect_recent(
                current.Store(tmp),
                FakeHttp(handler),
                at("2026-10-31T12:00:00Z"),
                wikis=["wikidatawiki", "enwiki"],
                samples=1,
            )
        self.assertEqual(result["positives"], {"wikidatawiki": 0, "enwiki": 1})
        self.assertIn("wikidatawiki", result["skipped"])


class SettingsTest(unittest.TestCase):
    def test_nightly_defaults_sample_more_and_read_every_wiki(self):
        args = current._parser().parse_args(["nightly", "--state", "s", "--bank", "b"])
        self.assertEqual(args.wiki_minutes, 10)
        self.assertEqual(args.negatives_per_positive, 3)
        self.assertEqual(current._wikis(args.wiki), list(current_wikimedia.DEFAULT_WIKIS))
        with patch.dict("os.environ", {"ZILS_CURRENT_WIKIS": "enwiki, dewiki"}):
            self.assertEqual(current._wikis(None), ["enwiki", "dewiki"])
        resolve = current._parser().parse_args(["resolve"])
        self.assertIsNone(resolve.negatives_per_positive, "a plain resolve keeps every edit")
        self.assertEqual(args.wiki_source, "recentchanges")


if __name__ == "__main__":
    unittest.main()
