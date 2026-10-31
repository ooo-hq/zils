"""Zils Current Articles for deletion stream: closed English Wikipedia deletion discussions."""

import json
import tempfile
import unittest

import zils
from tests.test_current import FakeHttp, at
from zils import current, current_afd, current_bank, current_items

NOW = at("2026-10-31T12:00:00Z")
SIG = "[[User:Nom Inator|Nom Inator]] ([[User talk:Nom Inator|talk]]) 09:25, 4 October 2026 (UTC)"
VOTER = (
    "[[User:Lib Star|<span style='color:black'>Lib Star</span>]] "
    "([[User talk:Lib Star|talk]]) 00:57, 9 October 2026 (UTC)"
)


def discussion(title, result, body=None, closer="[[User:Clo Ser|Clo Ser]] 17:52, 12 October 2026 (UTC)"):
    body = body or (
        f"Fails [[WP:GNG]], see [[Special:Contributions/Some One|Some One]]. {SIG}\n"
        f"*'''Delete''' all the sources are primary, as Nom Inator says. {VOTER}\n"
    )
    return (
        '<div class="boilerplate afd vfd xfd-closed archived mw-archivedtalk">\n'
        ":''The following discussion is an archived debate of the proposed deletion of the "
        "article below. '''Please do not modify it.'''''\n"
        "<!--Template:Afd top-->\n\n"
        f"The result was '''{result}'''. A clear consensus. {closer}\n"
        f"===[[:{title}]]===\n"
        "<noinclude>{{AFD help}}</noinclude>\n"
        f":{{{{la|1={title}}}}} – (<includeonly>[[Wikipedia:Articles for deletion/{title}|View AfD]]"
        "</includeonly>)\n"
        f":({{{{Find sources AFD|title={title}}}}})\n"
        f"{body}"
        ":''The above discussion is preserved as an archive of the debate. "
        "'''Please do not modify it.'''''<!--Template:Afd bottom--></div>"
    )


PAGES = {
    "Wikipedia:Articles for deletion/Frog Pond": discussion("Frog Pond", "delete"),
    "Wikipedia:Articles for deletion/Lake Tahoe Inn": discussion("Lake Tahoe Inn", "keep"),
    "Wikipedia:Articles for deletion/Old Mill (2nd nomination)": discussion(
        "Old Mill", "merge to [[Mill]]"
    ),
    "Wikipedia:Articles for deletion/Some Band": discussion("Some Band", "no consensus"),
    "Wikipedia:Articles for deletion/Draft Thing": discussion("Draft Thing", "draftify"),
    "Wikipedia:Articles for deletion/Gone Fast": discussion(
        "Gone Fast",
        "speedy delete",
        body=f"Spam. {SIG}\n*Article was speedily deleted under G11. {VOTER}\n",
    ),
    "Wikipedia:Articles for deletion/Still Open": "===[[:Still Open]]===\nNominated. " + SIG,
}


def handler(url, params):
    if params.get("prop") == "templates":
        templates = [{"ns": 4, "title": t} for t in PAGES] + [
            {"ns": 4, "title": "Wikipedia:Articles for deletion/Log/2026 October 3"}
        ]
        return 200, {"query": {"pages": [{"title": params["titles"], "templates": templates}]}}, {}
    if params.get("prop") == "revisions":
        pages = [
            {
                "title": t,
                "pageid": i + 1,
                "revisions": [{"revid": 900 + i, "slots": {"main": {"content": PAGES[t]}}}],
            }
            for i, t in enumerate(params["titles"].split("|"))
        ]
        return 200, {"query": {"pages": pages}}, {}
    return 404, {}, {}


class AfdTest(unittest.TestCase):
    def collect(self, tmp, **options):
        store = current.Store(tmp)
        http = FakeHttp(handler)
        result = current_afd.collect(store, http, NOW, days=["2026-10-04"], **options)
        return store, http, result

    def test_results_map_to_a_small_choice_set(self):
        self.assertEqual(current_afd.result_class("speedy keep"), "keep")
        self.assertEqual(current_afd.result_class("soft delete"), "delete")
        self.assertEqual(current_afd.result_class("redirect to [[Foo]]"), "merge-redirect")
        self.assertEqual(current_afd.result_class("merge selectively"), "merge-redirect")
        self.assertEqual(current_afd.result_class("no consensus"), "no-consensus")
        self.assertIsNone(current_afd.result_class("draftify"))
        self.assertIsNone(current_afd.result_class("withdrawn"))

    def test_closed_discussions_become_items_with_the_text_before_the_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, http, result = self.collect(tmp)
            current.resolve(store, http, NOW)
            resolved = {r["data"]["article"]: r for r in store.resolved()}
            dropped = {r["data"]["article"]: r["reason"] for r in store.dropped()}
        self.assertEqual(
            {a: r["label"] for a, r in resolved.items()},
            {
                "Frog Pond": "delete",
                "Lake Tahoe Inn": "keep",
                "Old Mill": "merge-redirect",
                "Some Band": "no-consensus",
                "Gone Fast": "delete",
            },
        )
        self.assertEqual(dropped, {"Draft Thing": "ambiguous-label"})
        self.assertEqual(result["calls"], 2, "one log page, one batch of discussions")
        frog = resolved["Frog Pond"]
        self.assertEqual(frog["asked_at"], "2026-10-04T09:25:00Z")
        self.assertEqual(frog["outcome_at"], "2026-10-12T17:52:00Z")
        self.assertEqual(frog["stream"], "wikipedia-afd")
        self.assertIn("oldid=900", frog["source"]["url"])
        case = current_items.build(frog)
        zils.validate_cases([case])
        text = case["state"]["text"]
        self.assertIn("Article: Frog Pond", text)
        self.assertIn("'''Delete''' all the sources are primary", text)
        for gone in ("The result was", "Clo Ser", "Nom Inator", "Lib Star", "Some One", "archived"):
            self.assertNotIn(gone, json.dumps(case))
        self.assertEqual(case["category"], "reasoning")
        self.assertEqual(case["block"], "known")
        self.assertEqual(case["question"]["type"], "choice")
        self.assertEqual(len(case["question"]["criteria"]), 4)
        self.assertTrue(current_items.STREAMS["wikipedia-afd"]["publishable"])

    def test_a_note_that_states_the_outcome_is_a_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, http, _ = self.collect(tmp)
            current.resolve(store, http, NOW)
            gone = next(r for r in store.resolved() if r["data"]["article"] == "Gone Fast")
        text = current_items.build(gone)["state"]["text"]
        self.assertEqual(current_bank.leak_reason(gone, text), "states-outcome")

    def test_a_second_run_over_the_same_day_adds_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _, first = self.collect(tmp)
            second = current_afd.collect(store, FakeHttp(handler), NOW, days=["2026-10-04"])
        self.assertEqual(first["added"], 5)
        self.assertEqual(second["added"], 0)

    def test_log_days_follow_the_cursor_and_stay_old_enough_to_be_closed(self):
        self.assertEqual(
            current_afd.log_days(NOW, cursor="2026-10-18", first_days=30),
            ["2026-10-19", "2026-10-20", "2026-10-21"],
        )
        self.assertEqual(len(current_afd.log_days(NOW, cursor=None, first_days=5)), 5)
        self.assertEqual(
            current_afd.log_title("2026-10-04"), "Wikipedia:Articles for deletion/Log/2026 October 4"
        )


class AfdWiringTest(unittest.TestCase):
    def test_collect_command_reads_the_named_days(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                current.main(
                    ["--dir", tmp, "collect", "--source", "afd", "--day", "2026-10-04"],
                    http=FakeHttp(handler),
                    now=NOW,
                )
        self.assertEqual(json.loads(out.getvalue())["added"], 5)

    def test_nightly_reads_new_log_days_once(self):
        from pathlib import Path

        from zils import current_nightly

        with tempfile.TemporaryDirectory() as tmp:
            options = {"check_overlap": False, "gh_max_hours": 0, "wiki_max_hours": 0}
            options |= {"se_pages": 0, "afd_first_days": 2}
            first = current_nightly.run(
                Path(tmp) / "s", Path(tmp) / "b", FakeHttp(handler), NOW, **options
            )
            again = current_nightly.run(
                Path(tmp) / "s", Path(tmp) / "b", FakeHttp(handler), NOW, **options
            )
            cursor = json.loads((Path(tmp) / "s" / "cursor.json").read_text())
        self.assertEqual(first["collected"]["afd"]["days"], ["2026-10-20", "2026-10-21"])
        self.assertEqual(again["collected"]["afd"]["days"], [])
        self.assertEqual(cursor["afd"], "2026-10-21")


if __name__ == "__main__":
    unittest.main()
