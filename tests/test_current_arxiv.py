"""Zils Current arXiv streams: which category was a new paper filed under?"""

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from tests.test_current import FakeHttp, at
from zils import current, current_arxiv, current_check, current_items, current_nightly

NOW = at("2026-10-31T12:00:00Z")
DAY = "2026-10-26"


def entry(number, primary, others=(), title=None, abstract=None, version=1, at_="10:00:00"):
    cats = [primary, *others]
    return (
        "<entry>"
        f"<id>http://arxiv.org/abs/2610.{number:05d}v{version}</id>"
        f"<title>{title or f'Paper {number} on\n  things'}</title>"
        f"<summary>{abstract or f'We study thing {number}.\n  It works well.'}</summary>"
        f"<published>{DAY}T{at_}Z</published>"
        f"<updated>{DAY}T{at_}Z</updated>"
        "<author><name>Ada Lovelace</name></author>"
        "<arxiv:comment>12 pages, accepted at NeurIPS</arxiv:comment>"
        f'<arxiv:primary_category term="{primary}" scheme="http://arxiv.org/schemas/atom"/>'
        + "".join(f'<category term="{c}" scheme="http://arxiv.org/schemas/atom"/>' for c in cats)
        + "</entry>"
    )


def feed(entries, total=None):
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" '
        'xmlns:arxiv="http://arxiv.org/schemas/atom">'
        f"<opensearch:totalResults>{len(entries) if total is None else total}"
        "</opensearch:totalResults>" + "".join(entries) + "</feed>"
    ).encode()


PAPERS = [
    entry(1, "cs.LG", ("stat.ML",)),
    entry(2, "cs.CV"),
    entry(3, "cs.CL", ("cs.AI",)),
    entry(4, "cs.CV"),
    entry(5, "cs.RO"),
    entry(6, "math.AP"),
    entry(7, "hep-ph"),
    entry(8, "stat.ME"),
    entry(9, "cs.CR", abstract="This paper has been withdrawn by the author."),
]


def oai(categories):
    return (
        '<?xml version="1.0" encoding="UTF-8"?><OAI-PMH><GetRecord><record><metadata>'
        '<arXiv xmlns="http://arxiv.org/OAI/arXiv/">'
        f"<categories>{categories}</categories></arXiv></metadata></record></GetRecord></OAI-PMH>"
    ).encode()


class ArxivHttp(FakeHttp):
    """Answers arXiv API pages (by `start`) and OAI-PMH records; any other request fails."""

    def __init__(self, pages=None, records=None):
        super().__init__()
        self.pages = list(pages or [feed(PAPERS)])
        self.records = records or {}

    def get_bytes(self, url):
        self.calls.append((url, {}))
        parts = urlsplit(url)
        query = {k: v[0] for k, v in parse_qs(parts.query).items()}
        if url.startswith(current_arxiv.API):
            return (200, self.pages.pop(0)) if self.pages else (200, feed([]))
        if url.startswith(current_arxiv.OAI):
            number = query["identifier"].rsplit(":", 1)[1]
            return (200, oai(self.records[number])) if number in self.records else (404, b"")
        return 404, b""


def collect(tmp, http=None, days=(DAY,)):
    store = current.Store(tmp)
    http = http or ArxivHttp()
    with patch.object(current_arxiv, "sleep", lambda s: None):
        result = current_arxiv.collect(store, http, NOW, days=list(days))
    current.resolve(store, http, now=NOW)
    return store, http, result


def by_number(store):
    return {r["source"]["arxiv_id"]: r for r in store.resolved()}


def code(option):
    return option.rsplit("(", 1)[1].rstrip(")")


class CollectTest(unittest.TestCase):
    def test_papers_resolve_with_four_options_and_the_primary_as_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _, result = collect(tmp)
            papers = by_number(store)
        self.assertEqual(result["added"], 8)
        paper = papers["2610.00001"]
        self.assertEqual(paper["stream"], "arxiv-category-cs")
        options = paper["data"]["options"]
        self.assertEqual(len(options), current_arxiv.OPTIONS)
        self.assertEqual(code(options[ord(paper["label"]) - 65]), "cs.LG")
        self.assertEqual(options[ord(paper["label"]) - 65], "Machine Learning (cs.LG)")
        self.assertEqual(len({code(o) for o in options}), current_arxiv.OPTIONS)
        self.assertNotIn(
            "stat.ML", {code(o) for o in options}
        )  # a cross-list is never a distractor
        self.assertEqual(paper["asked_at"], f"{DAY}T10:00:00Z")
        self.assertEqual(paper["source"]["url"], "https://arxiv.org/abs/2610.00001v1")

    def test_cross_lists_never_appear_among_the_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper = by_number(collect(tmp)[0])["2610.00003"]
        self.assertNotIn("cs.AI", {code(o) for o in paper["data"]["options"]})

    def test_distractors_come_from_the_same_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper = by_number(collect(tmp)[0])["2610.00002"]
        self.assertTrue(all(code(o).startswith("cs.") for o in paper["data"]["options"]))

    def test_streams_follow_the_primary_archive_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            papers = by_number(collect(tmp)[0])
        streams = {n: r["stream"] for n, r in papers.items()}
        self.assertEqual(streams["2610.00006"], "arxiv-category-math")
        self.assertEqual(streams["2610.00007"], "arxiv-category-physics")
        self.assertEqual(streams["2610.00008"], "arxiv-category-other")

    def test_withdrawn_papers_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = collect(tmp)[0]
            self.assertNotIn("2610.00009", by_number(store))
            self.assertEqual({r["reason"] for r in store.dropped()}, {"withdrawn"})

    def test_options_are_deterministic(self):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            first, second = by_number(collect(one)[0]), by_number(collect(two)[0])
        self.assertEqual(
            {n: r["data"]["options"] for n, r in first.items()},
            {n: r["data"]["options"] for n, r in second.items()},
        )

    def test_text_is_title_and_abstract_without_authors_or_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = by_number(collect(tmp)[0])["2610.00001"]
        case = current_items.build(record)
        self.assertEqual(
            case["state"]["text"],
            "Title: Paper 1 on things\n\nAbstract:\nWe study thing 1. It works well.",
        )
        self.assertNotIn("Ada", case["state"]["text"])
        self.assertNotIn("NeurIPS", case["state"]["text"])
        self.assertEqual(case["question"]["type"], "choice")
        self.assertEqual(case["question"]["criteria"][case["label"]], "Machine Learning (cs.LG)")
        self.assertEqual(case["category"], "tools")
        self.assertEqual(case["block"], "known")
        self.assertTrue(case["licence"].startswith("CC0-1.0"))

    def test_requests_are_three_seconds_apart_and_pages_follow_the_total(self):
        pages = [feed(PAPERS[:2], total=3), feed(PAPERS[2:3], total=3)]
        waits = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(current_arxiv, "PAGE_SIZE", 2),
            patch.object(current_arxiv, "sleep", waits.append),
        ):
            http = ArxivHttp(pages=pages)
            result = current_arxiv.collect(current.Store(tmp), http, NOW, days=[DAY])
        starts = [parse_qs(urlsplit(u).query)["start"][0] for u, _ in http.calls]
        self.assertEqual(starts, ["0", "2"])
        self.assertEqual(waits, [current_arxiv.GAP_SECONDS])
        self.assertEqual(result["papers"], 3)
        query = parse_qs(urlsplit(http.calls[0][0]).query)["search_query"][0]
        self.assertEqual(query, "submittedDate:[202610260000 TO 202610262359]")

    def test_an_empty_page_with_results_is_retried(self):
        pages = [feed([], total=9), feed(PAPERS)]
        with tempfile.TemporaryDirectory() as tmp:
            _, http, result = collect(tmp, ArxivHttp(pages=pages))
        self.assertEqual(len(http.calls), 2)
        self.assertEqual(result["added"], 8)

    def test_a_day_without_papers_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            collect(tmp, ArxivHttp(pages=[feed([])] * (current_arxiv.RETRIES + 1)))

    def test_days_wait_for_the_announcement(self):
        days = current_arxiv.days(NOW, None, 14)
        self.assertEqual(len(days), 14)
        self.assertEqual(
            days[-1], (NOW - timedelta(days=current_arxiv.MIN_AGE_DAYS)).strftime("%Y-%m-%d")
        )
        self.assertEqual(current_arxiv.days(NOW, "2026-10-26"), ["2026-10-27", "2026-10-28"])


class ArxivCheckTest(unittest.TestCase):
    def test_crosscheck_reads_the_primary_from_oai_pmh(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = [current_items.build(r) for r in collect(tmp)[0].resolved()]
        cs = [c for c in cases if c["source"]["arxiv_id"] in ("2610.00001", "2610.00002")]
        records = {"2610.00001": "cs.LG stat.ML", "2610.00002": "cs.RO cs.CV"}
        waits = []
        with patch.object(current_arxiv, "sleep", waits.append):
            results = current_check.check_arxiv(cs, ArxivHttp(records=records))
        agree = {c["source"]["arxiv_id"]: r["agree"] for c, r in zip(cs, results, strict=True)}
        self.assertEqual(agree, {"2610.00001": True, "2610.00002": False})
        self.assertEqual(waits, [current_arxiv.GAP_SECONDS])

    def test_crosscheck_dispatches_arxiv_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = [current_items.build(r) for r in collect(tmp)[0].resolved()]
        records = {c["source"]["arxiv_id"]: "x.Y" for c in cases}
        with patch.object(current_arxiv, "sleep", lambda s: None):
            streams = current_check.crosscheck(cases, ArxivHttp(records=records), sample=5)
        self.assertEqual(streams["arxiv-category-cs"]["method"], "OAI-PMH record categories")
        self.assertEqual(streams["arxiv-category-cs"]["agreement"], 0.0)


class ArxivNightlyTest(unittest.TestCase):
    def test_nightly_reads_new_days_from_the_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, cursor = current.Store(tmp), current_nightly.Cursor(Path(tmp) / "cursor.json")
            cursor.set("arxiv", "2026-10-26")
            http = ArxivHttp(pages=[feed(PAPERS), feed(PAPERS)])
            with patch.object(current_arxiv, "sleep", lambda s: None):
                result = current_nightly._arxiv(store, http, cursor, NOW, None)
            self.assertEqual(result["days"], ["2026-10-27", "2026-10-28"])
            self.assertEqual(cursor.get("arxiv"), "2026-10-28")


if __name__ == "__main__":
    unittest.main()
