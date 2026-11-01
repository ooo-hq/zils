"""Zils Current arXiv streams: which category was a new paper filed under?

One request to the arXiv API (`search_query=submittedDate:[day]`) returns up to 2,000 papers
submitted on one UTC day, with their title, abstract, primary category and cross-lists. Each paper
becomes one choice item: the title and abstract, and four categories to choose from. The answer
is the primary category the paper was announced under (authors propose it, moderators may move
it). The three other options are categories other papers of the same day were filed under, in
the same group (computer science, mathematics, physics, or the rest), never one of the paper's
own cross-lists, so exactly one option is a category the paper is listed in.

Days are read once they are MIN_AGE_DAYS old, when their papers have been announced. Authors,
comments, journal references and dates are never shown. arXiv metadata is CC0; the API terms ask
for one request every three seconds on a single connection.
"""

import hashlib
import random
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

from .current_items import RateLimited, iso

API = "https://export.arxiv.org/api/query"
OAI = "https://oaipmh.arxiv.org/oai"
MIN_AGE_DAYS = 3  # a day's papers are announced within three days (weekends included)
FIRST_DAYS = 14
PAGE_SIZE = 2000
GAP_SECONDS = 3.0  # arXiv API terms: one request every three seconds
RETRIES, RETRY_WAIT = 3, 30.0
OPTIONS = 4
sleep = time.sleep
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "x": "http://arxiv.org/schemas/atom",
    "o": "http://a9.com/-/spec/opensearch/1.1/",
}
PHYSICS = (
    "astro-ph cond-mat gr-qc hep-ex hep-lat hep-ph hep-th math-ph nlin nucl-ex nucl-th physics "
    "quant-ph"
).split()
# Categories that are one subject under two names; listing one lists both.
ALIASES = {
    "math.IT": "cs.IT",
    "math.MP": "math-ph",
    "math.ST": "stat.TH",
    "q-fin.EC": "econ.GN",
    "eess.SY": "cs.SY",
    "math.NA": "cs.NA",
}
ALIASES |= {v: k for k, v in ALIASES.items()}
WITHDRAWN = re.compile(
    r"\b(?:paper|article|submission|manuscript) (?:has been|is) withdrawn\b", re.I
)
# From https://arxiv.org/category_taxonomy (1 Nov 2026).
CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering, Finance, and Science",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation and Language",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CY": "Computers and Society",
    "cs.DB": "Databases",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structures and Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.GT": "Computer Science and Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in Computer Science",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.MS": "Mathematical Software",
    "cs.NA": "Numerical Analysis",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.NI": "Networking and Internet Architecture",
    "cs.OH": "Other Computer Science",
    "cs.OS": "Operating Systems",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SC": "Symbolic Computation",
    "cs.SD": "Sound",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social and Information Networks",
    "cs.SY": "Systems and Control",
    "econ.EM": "Econometrics",
    "econ.GN": "General Economics",
    "econ.TH": "Theoretical Economics",
    "eess.AS": "Audio and Speech Processing",
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems and Control",
    "math.AC": "Commutative Algebra",
    "math.AG": "Algebraic Geometry",
    "math.AP": "Analysis of PDEs",
    "math.AT": "Algebraic Topology",
    "math.CA": "Classical Analysis and ODEs",
    "math.CO": "Combinatorics",
    "math.CT": "Category Theory",
    "math.CV": "Complex Variables",
    "math.DG": "Differential Geometry",
    "math.DS": "Dynamical Systems",
    "math.FA": "Functional Analysis",
    "math.GM": "General Mathematics",
    "math.GN": "General Topology",
    "math.GR": "Group Theory",
    "math.GT": "Geometric Topology",
    "math.HO": "History and Overview",
    "math.IT": "Information Theory",
    "math.KT": "K-Theory and Homology",
    "math.LO": "Logic",
    "math.MG": "Metric Geometry",
    "math.MP": "Mathematical Physics",
    "math.NA": "Numerical Analysis",
    "math.NT": "Number Theory",
    "math.OA": "Operator Algebras",
    "math.OC": "Optimization and Control",
    "math.PR": "Probability",
    "math.QA": "Quantum Algebra",
    "math.RA": "Rings and Algebras",
    "math.RT": "Representation Theory",
    "math.SG": "Symplectic Geometry",
    "math.SP": "Spectral Theory",
    "math.ST": "Statistics Theory",
    "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
    "astro-ph.EP": "Earth and Planetary Astrophysics",
    "astro-ph.GA": "Astrophysics of Galaxies",
    "astro-ph.HE": "High Energy Astrophysical Phenomena",
    "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
    "astro-ph.SR": "Solar and Stellar Astrophysics",
    "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
    "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
    "cond-mat.mtrl-sci": "Materials Science",
    "cond-mat.other": "Other Condensed Matter",
    "cond-mat.quant-gas": "Quantum Gases",
    "cond-mat.soft": "Soft Condensed Matter",
    "cond-mat.stat-mech": "Statistical Mechanics",
    "cond-mat.str-el": "Strongly Correlated Electrons",
    "cond-mat.supr-con": "Superconductivity",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math-ph": "Mathematical Physics",
    "nlin.AO": "Adaptation and Self-Organizing Systems",
    "nlin.CD": "Chaotic Dynamics",
    "nlin.CG": "Cellular Automata and Lattice Gases",
    "nlin.PS": "Pattern Formation and Solitons",
    "nlin.SI": "Exactly Solvable and Integrable Systems",
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",
    "physics.acc-ph": "Accelerator Physics",
    "physics.ao-ph": "Atmospheric and Oceanic Physics",
    "physics.app-ph": "Applied Physics",
    "physics.atm-clus": "Atomic and Molecular Clusters",
    "physics.atom-ph": "Atomic Physics",
    "physics.bio-ph": "Biological Physics",
    "physics.chem-ph": "Chemical Physics",
    "physics.class-ph": "Classical Physics",
    "physics.comp-ph": "Computational Physics",
    "physics.data-an": "Data Analysis, Statistics and Probability",
    "physics.ed-ph": "Physics Education",
    "physics.flu-dyn": "Fluid Dynamics",
    "physics.gen-ph": "General Physics",
    "physics.geo-ph": "Geophysics",
    "physics.hist-ph": "History and Philosophy of Physics",
    "physics.ins-det": "Instrumentation and Detectors",
    "physics.med-ph": "Medical Physics",
    "physics.optics": "Optics",
    "physics.plasm-ph": "Plasma Physics",
    "physics.pop-ph": "Popular Physics",
    "physics.soc-ph": "Physics and Society",
    "physics.space-ph": "Space Physics",
    "quant-ph": "Quantum Physics",
    "q-bio.BM": "Biomolecules",
    "q-bio.CB": "Cell Behavior",
    "q-bio.GN": "Genomics",
    "q-bio.MN": "Molecular Networks",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.OT": "Other Quantitative Biology",
    "q-bio.PE": "Populations and Evolution",
    "q-bio.QM": "Quantitative Methods",
    "q-bio.SC": "Subcellular Processes",
    "q-bio.TO": "Tissues and Organs",
    "q-fin.CP": "Computational Finance",
    "q-fin.EC": "Economics",
    "q-fin.GN": "General Finance",
    "q-fin.MF": "Mathematical Finance",
    "q-fin.PM": "Portfolio Management",
    "q-fin.PR": "Pricing of Securities",
    "q-fin.RM": "Risk Management",
    "q-fin.ST": "Statistical Finance",
    "q-fin.TR": "Trading and Market Microstructure",
    "stat.AP": "Applications",
    "stat.CO": "Computation",
    "stat.ME": "Methodology",
    "stat.ML": "Machine Learning",
    "stat.OT": "Other Statistics",
    "stat.TH": "Statistics Theory",
}


def group_of(category):
    archive = category.split(".")[0]
    if archive in ("cs", "math"):
        return archive
    return "physics" if archive in PHYSICS else "other"


def stream_of(category):
    return f"arxiv-category-{group_of(category)}"


def option(category):
    return f"{CATEGORIES.get(category, category)} ({category})"


def days(now, cursor=None, first_days=FIRST_DAYS):
    """Submission days after `cursor` (or the last `first_days`) that are announced by now."""
    last = (now.astimezone(UTC) - timedelta(days=MIN_AGE_DAYS)).date()
    day = (
        datetime.strptime(cursor, "%Y-%m-%d").date() + timedelta(days=1)
        if cursor
        else last - timedelta(days=first_days - 1)
    )
    result = []
    while day <= last:
        result.append(day.strftime("%Y-%m-%d"))
        day += timedelta(days=1)
    return result


class Api:
    """Serial requests GAP_SECONDS apart; HTTP 429 and 503 wait and retry, then stop the run."""

    def __init__(self, http):
        self.http, self.calls = http, 0

    def get(self, url, params):
        for attempt in range(RETRIES + 1):
            if self.calls:
                sleep(GAP_SECONDS if attempt == 0 else RETRY_WAIT)
            status, raw = self.http.get_bytes(f"{url}?{urlencode(params)}")
            self.calls += 1
            if status not in (429, 503):
                return status, raw
        raise RateLimited(f"arXiv HTTP {status}")


def _text(element):
    return " ".join((element.text or "").split()) if element is not None else ""


def parse(raw):
    """(total results, papers) from one API page."""
    root = ET.fromstring(raw)
    total = int(_text(root.find("o:totalResults", NS)) or 0)
    papers = []
    for entry in root.findall("a:entry", NS):
        number, _, version = _text(entry.find("a:id", NS)).rsplit("/", 1)[-1].rpartition("v")
        primary = entry.find("x:primary_category", NS)
        papers.append(
            {
                "arxiv_id": number,
                "version": int(version or 1),
                "title": _text(entry.find("a:title", NS)),
                "abstract": _text(entry.find("a:summary", NS)),
                "published": _text(entry.find("a:published", NS)),
                "primary": primary.get("term") if primary is not None else "",
                "categories": [c.get("term") for c in entry.findall("a:category", NS)],
            }
        )
    return total, papers


def _day(api, day):
    """Every paper submitted on `day` (UTC), a page at a time."""
    stamp = day.replace("-", "")
    query = f"submittedDate:[{stamp}0000 TO {stamp}2359]"
    papers, start = [], 0
    while True:
        params = {
            "search_query": query,
            "start": str(start),
            "max_results": str(PAGE_SIZE),
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
        for _ in range(RETRIES + 1):
            status, raw = api.get(API, params)
            if status != 200:
                raise ValueError(f"arXiv API HTTP {status} for {day}")
            total, page = parse(raw)
            if page or start >= total > 0:
                break  # the API sometimes answers a page with results with no entries: retry
        if not page and not papers:
            raise ValueError(f"arXiv API returned no papers for {day}")
        papers += page
        start += len(page)
        if not page or start >= total:
            return papers


def options(paper, pool):
    """The primary and OPTIONS - 1 distractors, shuffled; the same paper and pool, the same list."""
    own = set(paper["categories"]) | {paper["primary"]}
    own |= {ALIASES[c] for c in own if c in ALIASES}
    group = group_of(paper["primary"])
    rng = random.Random(f"arxiv:{paper['arxiv_id']}")
    # Other papers' primaries, as often as they occur: frequent neighbours are harder to rule out.
    candidates = [c for c in pool if group_of(c) == group and c not in own]
    chosen = []
    while candidates and len(chosen) < OPTIONS - 1:
        pick = rng.choice(candidates)
        chosen.append(pick)
        candidates = [c for c in candidates if c != pick]
    for fallback in (
        [c for c in CATEGORIES if group_of(c) == group],
        list(CATEGORIES),
    ):
        spare = [c for c in fallback if c not in own and c not in chosen]
        rng.shuffle(spare)
        chosen += spare[: OPTIONS - 1 - len(chosen)]
    result = [paper["primary"], *chosen]
    rng.shuffle(result)
    return result


def record(paper, pool, now):
    """("ready", record) for a usable paper, ("dropped", record) otherwise."""
    stream = stream_of(paper["primary"])
    digest = hashlib.sha256(f"arxiv:{paper['arxiv_id']}".encode()).hexdigest()[:16]
    row = {
        "id": f"{stream}-{digest}",
        "stream": stream,
        "asked_at": paper["published"],
        "collected_at": iso(now),
        "source": {
            "name": "arxiv",
            "arxiv_id": paper["arxiv_id"],
            "version": paper["version"],
            "url": f"https://arxiv.org/abs/{paper['arxiv_id']}v{paper['version']}",
        },
        "data": {"title": paper["title"], "abstract": paper["abstract"]},
    }
    if not paper["title"] or not paper["abstract"] or not paper["primary"]:
        return "dropped", {**row, "reason": "no-text"}
    if WITHDRAWN.search(paper["abstract"]):
        return "dropped", {**row, "reason": "withdrawn"}
    codes = options(paper, pool)
    row["data"]["options"] = [option(c) for c in codes]
    return "ready", {**row, "outcome": {"label": chr(65 + codes.index(paper["primary"]))}}


def collect(store, http, now, *, days):
    """Read each submission day; add its papers, ready to resolve."""
    api, added, count = Api(http), 0, 0
    for day in days:
        papers = _day(api, day)
        count += len(papers)
        pool = [p["primary"] for p in papers if p["primary"]]
        ready, dropped = [], []
        for paper in papers:
            status, row = record(paper, pool, now)
            (ready if status == "ready" else dropped).append(row)
        added += store.add(ready)
        store.drop(dropped)
    return {"added": added, "papers": count, "calls": api.calls}


def primary_from_oai(api, arxiv_id):
    """The first category of the paper's OAI-PMH record (arXiv lists the primary first), or None."""
    params = {
        "verb": "GetRecord",
        "identifier": f"oai:arXiv.org:{arxiv_id}",
        "metadataPrefix": "arXiv",
    }
    status, raw = api.get(OAI, params)
    match = re.search(rb"<categories>([^<]*)</categories>", raw or b"") if status == 200 else None
    return (match.group(1).decode().split() or [None])[0] if match else None
