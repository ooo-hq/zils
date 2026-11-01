"""Zils Current Articles for Creation stream: will a reviewer accept this draft?"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_current import FakeHttp, at
from zils import current, current_afc, current_bank, current_check, current_items, current_nightly

NOW = at("2026-10-31T12:00:00Z")
DAY = "2026-10-29"
PENDING = "{{AfC submission|||u=Some Writer|ns=118|ts=20261020093000}}"
OLD_DECLINE = "{{AfC submission|d|nn|u=Some Writer|ns=118|decliner=Rev Iewer|ts=20261001000000}}"
COMMENT = "{{AfC comment|1=Needs better sources. [[User:Rev Iewer|Rev Iewer]] 10:00, 2 October 2026 (UTC)}}"
DRAFT = (
    f"{PENDING}\n{OLD_DECLINE}\n{COMMENT}\n<!-- Important, do not remove this line -->\n"
    "{{Short description|Pond in Ohio}}\n{{Infobox lake|name=Frog Pond|area=2 ha}}\n"
    "'''Frog Pond''' is a [[pond]] in [[Ohio|the state of Ohio]], written by "
    "[[User:Some Writer|Some Writer]].<ref>{{cite web |title=Ponds of Ohio |website=Ohio News "
    "|url=https://ohionews.example/ponds}}</ref> It hosts frogs.<ref name=a/>\n\n"
    "== History ==\nDug in 1901 by farmers who needed water for cattle during a dry summer. "
    "It was enlarged in 1950.<ref>Local history book, p. 4</ref>\n\n"
    "[[File:Pond.jpg|thumb|The pond]]\n[[:Category:Ponds]]\n"
)

RC = [
    {
        "title": "Draft:Frog Pond",
        "revid": 102,
        "old_revid": 101,
        "timestamp": f"{DAY}T10:00:00Z",
        "comment": "Declining submission: nn - Submission is about a topic not yet shown to "
        "meet general notability guidelines (AFCH)",
    },
    {
        "title": "Draft:Spam Co",
        "revid": 202,
        "old_revid": 201,
        "timestamp": f"{DAY}T11:00:00Z",
        "comment": "Rejecting submission: adv - Submission is advertising (AFCH)",
    },
    {
        "title": "Draft:Talky",
        "revid": 402,
        "old_revid": 401,
        "timestamp": f"{DAY}T12:00:00Z",
        "comment": "Commenting on submission (AFCH)",
    },
]
MOVES = [
    {
        "title": "Draft:Lake Inn",
        "revid": 302,
        "timestamp": f"{DAY}T13:00:00Z",
        "comment": "Publishing accepted [[Wikipedia:Articles for creation|Articles for creation]] "
        "submission",
        "params": {"target_ns": 0, "target_title": "Lake Inn"},
    },
    {
        "title": "Draft:Elsewhere",
        "revid": 502,
        "timestamp": f"{DAY}T14:00:00Z",
        "comment": "Publishing accepted [[Wikipedia:Articles for creation|Articles for creation]] "
        "submission",
        "params": {"target_ns": 2, "target_title": "User:Someone/Elsewhere"},
    },
    {
        "title": "Draft:Renamed",
        "revid": 602,
        "timestamp": f"{DAY}T15:00:00Z",
        "comment": "Fix title",
        "params": {"target_ns": 118, "target_title": "Draft:Renamed better"},
    },
]
REVISIONS = {
    101: {"timestamp": "2026-10-20T09:30:00Z", "content": DRAFT},
    201: {"timestamp": "2026-10-28T08:00:00Z", "content": "Spam Co sells the best widgets. " * 10},
    301: {
        "timestamp": "2026-10-25T07:00:00Z",
        "content": "{{AfC submission|||ts=20261025070000|u=X}}\n'''Lake Inn''' is a hotel on a "
        "lake, covered by three national newspapers.<ref>{{cite news|title=Lake Inn opens"
        "|newspaper=The Times}}</ref> " * 3,
    },
    # The review edits themselves, read by the cross-check.
    102: {"timestamp": f"{DAY}T10:00:00Z", "content": "{{AfC submission|d|nn|ts=1}}\nFrog"},
    202: {"timestamp": f"{DAY}T11:00:00Z", "content": "Spam Co, no templates left"},
}
PARENTS = {302: 301}


def handler(pages=None, rc_pages=None):
    """Answer the API calls the collector and the cross-check make."""
    rc_pages = list(rc_pages or [RC])
    pages = pages or {}

    def answer(url, params):
        if params.get("list") == "recentchanges":
            assert params["rctag"] == "AFCH" and params["rcnamespace"] == "118"
            batch = rc_pages.pop(0) if rc_pages else []
            body = {"query": {"recentchanges": batch}}
            if rc_pages:
                body["continue"] = {"rccontinue": "next", "continue": "-||"}
            return 200, body, {}
        if params.get("list") == "logevents" and params.get("letype") == "move":
            assert params["letag"] == "AFCH"
            return 200, {"query": {"logevents": MOVES}}, {}
        if params.get("list") == "logevents":
            return (
                200,
                {"query": {"logevents": pages.get(("deletions", params["letitle"]), [])}},
                {},
            )
        if params.get("prop") == "revisions":
            revs = []
            for revid in map(int, params["revids"].split("|")):
                if params["rvprop"] == "ids" and revid in PARENTS:
                    revs.append({"revid": revid, "parentid": PARENTS[revid]})
                elif revid in REVISIONS:
                    rev = REVISIONS[revid]
                    revs.append(
                        {
                            "revid": revid,
                            "timestamp": rev["timestamp"],
                            "slots": {"main": {"content": rev["content"]}},
                        }
                    )
            return 200, {"query": {"pages": [{"revisions": revs}]}}, {}
        if params.get("prop") == "info":
            found = []
            for title in params["titles"].split("|"):
                found.append(pages.get(title, {"title": title, "missing": True}))
            return 200, {"query": {"pages": found}}, {}
        return 404, {}, {}

    return answer


def collect(tmp, http=None):
    store = current.Store(tmp)
    http = http or FakeHttp(handler())
    result = current_afc.collect(store, http, NOW, days=[DAY])
    current.resolve(store, http, now=NOW)
    return store, http, result


def by_title(store):
    return {r["source"]["title"]: r for r in store.resolved()}


class CollectTest(unittest.TestCase):
    def test_declines_rejects_and_accepts_resolve_with_the_review_as_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _, result = collect(tmp)
            records = by_title(store)
        self.assertEqual(set(records), {"Draft:Frog Pond", "Draft:Spam Co", "Draft:Lake Inn"})
        self.assertEqual(result["added"], 3)
        self.assertEqual(records["Draft:Frog Pond"]["label"], "false")
        self.assertEqual(records["Draft:Spam Co"]["label"], "false")
        self.assertEqual(records["Draft:Lake Inn"]["label"], "true")
        frog = records["Draft:Frog Pond"]
        self.assertEqual(frog["stream"], "wikipedia-afc")
        self.assertEqual(
            frog["asked_at"], "2026-10-20T09:30:00Z"
        )  # the pending ts, not the decline
        self.assertEqual(frog["outcome_at"], f"{DAY}T10:00:00Z")
        self.assertEqual(frog["source"]["rev_id"], 101)
        self.assertEqual(frog["source"]["review"], {"rev_id": 102})
        self.assertEqual(frog["source"]["url"], "https://en.wikipedia.org/w/index.php?oldid=101")
        lake = records["Draft:Lake Inn"]
        self.assertEqual(lake["source"]["rev_id"], 301)  # the draft as reviewed, before the move
        self.assertEqual(lake["source"]["review"], {"target": "Lake Inn"})
        # Without a pending ts, the reviewed revision's time is the ask time.
        self.assertEqual(records["Draft:Spam Co"]["asked_at"], "2026-10-28T08:00:00Z")

    def test_text_is_the_reviewed_draft_without_templates_or_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = by_title(collect(tmp)[0])["Draft:Frog Pond"]["data"]["text"]
        self.assertTrue(
            text.startswith("Frog Pond is a pond in the state of Ohio, written by [user]."), text
        )
        self.assertIn("[ref: Ponds of Ohio, Ohio News]", text)
        self.assertIn("[ref: Local history book, p. 4]", text)
        self.assertIn("== History ==", text)
        for gone in (
            "AfC",
            "Needs better sources",
            "Some Writer",
            "Rev Iewer",
            "Infobox",
            "Short description",
            "do not remove",
            "Category",
            "Pond.jpg",
            "'''",
            "[[",
        ):
            self.assertNotIn(gone, text)

    def test_case_is_a_yes_no_reasoning_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = by_title(collect(tmp)[0])["Draft:Lake Inn"]
        case = current_items.build(record)
        self.assertEqual(case["question"]["type"], "noul")
        self.assertEqual(case["label"], "true")
        self.assertEqual(case["category"], "reasoning")
        self.assertEqual(case["block"], "new")
        self.assertTrue(
            case["state"]["text"].startswith("Wiki: English Wikipedia\nDraft: Lake Inn")
        )
        self.assertIsNone(current_check.proof_reason(case))

    def test_drafts_declined_as_unsafe_are_never_read(self):
        # Copyright violations, unsourced claims about living people, vandalism and attack pages:
        # their text is not ours to keep or publish, so the draft is not even fetched.
        unsafe = {
            "Draft:Copied": "Declining submission: cv - Submission is a copyright violation (AFCH)",
            "Draft:Gossip": "Declining submission: blp - Submission is in violation of the "
            "biographies of living persons policy (AFCH)",
            "Draft:Junk": "Rejecting submission: van - Submission is vandalism (AFCH)",
            "Draft:Mixed": "Declining submission: v, cv - Submission is improperly sourced, "
            "Submission is a copyright violation (AFCH)",
            "Draft:Mean": "Declining submission: reason - This draft is an attack page (AFCH)",
        }
        edits = [
            {"title": title, "revid": 900 + 2 * i, "old_revid": 901 + 2 * i,
             "timestamp": f"{DAY}T16:00:00Z", "comment": comment}
            for i, (title, comment) in enumerate(unsafe.items())
        ]  # fmt: skip
        extra = {
            e["old_revid"]: {"timestamp": "2026-10-20T00:00:00Z", "content": DRAFT} for e in edits
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(REVISIONS, extra):
            http = FakeHttp(handler(rc_pages=[RC + edits]))
            store, _, result = collect(tmp, http)
            titles = {r["source"]["title"] for r in store.resolved()}
            fetched = {
                int(revid)
                for _, params in http.calls
                if params.get("prop") == "revisions"
                for revid in params["revids"].split("|")
            }
        self.assertEqual(titles, {"Draft:Frog Pond", "Draft:Spam Co", "Draft:Lake Inn"})
        self.assertEqual(result["added"], 3)
        self.assertFalse(fetched & set(extra))
        # An ordinary decline that merely mentions sources is kept.
        self.assertFalse(current_afc.unsafe_decline(RC[0]["comment"]))

    def test_recent_changes_are_paged(self):
        with tempfile.TemporaryDirectory() as tmp:
            http = FakeHttp(handler(rc_pages=[RC[:1], RC[1:]]))
            titles = set(by_title(collect(tmp, http)[0]))
            calls = [p for _, p in http.calls if p.get("list") == "recentchanges"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["rccontinue"], "next")
        self.assertIn("Draft:Spam Co", titles)

    def test_a_draft_that_states_its_outcome_is_filtered(self):
        record = {
            "id": "x",
            "stream": "wikipedia-afc",
            "data": {"title": "T", "text": "This submission was declined twice before."},
        }
        self.assertEqual(current_bank.leak_reason(record, record["data"]["text"]), "states-outcome")

    def test_review_days_are_finished_utc_days(self):
        self.assertEqual(current_afc.review_days(NOW, "2026-10-28"), ["2026-10-29", "2026-10-30"])
        days = current_afc.review_days(NOW, None)
        self.assertEqual(len(days), current_afc.FIRST_DAYS)
        self.assertEqual(days[-1], "2026-10-30")


class AfcCheckTest(unittest.TestCase):
    def cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            return [current_items.build(r) for r in collect(tmp)[0].resolved()]

    def test_declines_need_a_declined_template_and_accepts_an_article(self):
        cases = sorted(self.cases(), key=lambda c: c["source"]["title"])
        pages = {"Lake Inn": {"title": "Lake Inn", "ns": 0, "pageid": 9}}
        results = current_check.check_afc(cases, FakeHttp(handler(pages)))
        agree = {c["source"]["title"]: r["agree"] for c, r in zip(cases, results, strict=True)}
        # Spam Co's rejection edit kept no AfC template: the second derivation disagrees.
        self.assertEqual(
            agree, {"Draft:Frog Pond": True, "Draft:Lake Inn": True, "Draft:Spam Co": False}
        )

    def test_an_accepted_article_deleted_later_still_agrees(self):
        cases = [c for c in self.cases() if c["label"] == "true"]
        deletions = {
            ("deletions", "Lake Inn"): [{"action": "delete", "timestamp": "2026-10-30T00:00:00Z"}]
        }
        results = current_check.check_afc(cases, FakeHttp(handler(deletions)))
        self.assertTrue(results[0]["agree"])

    def test_crosscheck_dispatches_the_stream(self):
        pages = {"Lake Inn": {"title": "Lake Inn", "ns": 0, "pageid": 9}}
        streams = current_check.crosscheck(self.cases(), FakeHttp(handler(pages)), sample=5)
        self.assertEqual(streams["wikipedia-afc"]["agreement"], round(2 / 3, 4))
        self.assertNotEqual(streams["wikipedia-afc"]["method"], "none")


class AfcNightlyTest(unittest.TestCase):
    def test_nightly_reads_finished_days_from_the_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, cursor = current.Store(tmp), current_nightly.Cursor(Path(tmp) / "cursor.json")
            cursor.set("afc", "2026-10-28")
            http = FakeHttp(handler())
            with patch.object(current_afc, "collect", wraps=current_afc.collect) as spy:
                result = current_nightly._afc(store, http, cursor, NOW, None)
            self.assertEqual(result["days"], ["2026-10-29", "2026-10-30"])
            self.assertEqual(cursor.get("afc"), "2026-10-30")
            self.assertEqual(spy.call_count, 2)


if __name__ == "__main__":
    unittest.main()
