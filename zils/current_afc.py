"""Zils Current Articles for Creation stream: will a reviewer accept this English Wikipedia draft?

New editors submit drafts (`Draft:` pages) for review at Articles for Creation. A reviewer, nearly
always with the AFCH helper script, either declines or rejects the submission (an edit to the
draft, summary "Declining submission: ..." or "Rejecting submission: ...") or accepts it by moving
the draft into the article space (move log, summary "Publishing accepted ... submission"). Both
carry the `AFCH` change tag, so two list requests per day find every review.

The state is the draft as the reviewer saw it: the revision just before the decline edit, or
the last revision before the move (the parent of the move's own revision). Templates (the
submission box, earlier declines and reviewer comments, infoboxes), comments, files and
categories are removed, citations are reduced to their title and publication, and editor names
are scrubbed. The label is whether the submission was accepted. The ask time is the submission
time recorded in the pending submission box, or the reviewed revision's time.
"""

import hashlib
import re
from datetime import UTC, datetime, timedelta

from .current_afd import clean
from .current_items import iso

API = "https://en.wikipedia.org/w/api.php"
NAMESPACE = "118"  # Draft
FIRST_DAYS = 28  # recent changes keep 30 days
BATCH = 50
MIN_CHARS = 200
DECLINE = re.compile(r"^(?:Declining|Rejecting) submission\b")
# Declines whose draft text is not ours to keep or publish: copyright violations, breaches of the
# biographies of living persons policy, vandalism and attack pages. AFCH writes the reason codes
# after the colon ("Declining submission: v, cv - ..."); the words catch custom reasons.
UNSAFE_CODES = {"cv", "blp", "van"}
REASON_CODES = re.compile(r"[:,]\s*([a-z0-9-]+)(?=\s*(?:,|-\s))")
UNSAFE_WORDS = re.compile(r"copyright|copyvio|living person|\bblp\b|attack page|vandal", re.I)
ACCEPT = "Publishing accepted"
PENDING_TS = re.compile(r"\{\{\s*AfC submission\s*\|\s*\|[^{}]*?\bts\s*=\s*(\d{14})", re.I)
CITE = re.compile(r"\{\{\s*cite[^{}]*\}\}", re.I)
CITE_FIELD = re.compile(
    r"\|\s*(title|website|work|newspaper|journal|magazine|publisher)\s*=\s*([^|{}]*)", re.I
)
EMPTY_REF = re.compile(r"<ref\b[^>]*/>", re.I)
REF = re.compile(r"<ref\b[^>]*>(.*?)</ref>", re.I | re.S)
FILE_OR_CATEGORY = re.compile(r"\[\[\s*:?\s*(?:File|Image|Category)\s*:[^\[\]]*\]\]", re.I)
PIPED = re.compile(r"\[\[[^\[\]|]*\|([^\[\]]*)\]\]")
LINK = re.compile(r"\[\[([^\[\]|]*)\]\]")


def review_days(now, cursor=None, first_days=FIRST_DAYS):
    """Finished UTC days after `cursor` (or the last `first_days`)."""
    last = (now.astimezone(UTC) - timedelta(days=1)).date()
    day = (
        datetime.strptime(cursor, "%Y-%m-%d").date() + timedelta(days=1)
        if cursor
        else last - timedelta(days=first_days - 1)
    )
    days = []
    while day <= last:
        days.append(day.strftime("%Y-%m-%d"))
        day += timedelta(days=1)
    return days


def _get(http, params):
    from .current_afd import _get as get

    return get(http, params)


def _listing(http, params, key):
    """Every row of a list query, following `continue`."""
    rows, more = [], {}
    while True:
        status, body = _get(http, {**params, **more})
        if status != 200:
            raise ValueError(f"Wikipedia API HTTP {status} for {key}")
        rows += (body.get("query") or {}).get(key) or []
        more = body.get("continue") or {}
        if not more:
            return rows


def _revisions(http, revids, prop):
    """{revid: revision} for the given ids, BATCH per request."""
    found = {}
    revids = sorted(set(revids))
    for start in range(0, len(revids), BATCH):
        params = {
            "action": "query",
            "prop": "revisions",
            "revids": "|".join(map(str, revids[start : start + BATCH])),
            "rvprop": prop,
        }
        if "content" in prop:
            params["rvslots"] = "main"
        status, body = _get(http, params)
        for page in (body.get("query") or {}).get("pages") or [] if status == 200 else []:
            for revision in page.get("revisions") or []:
                found[revision["revid"]] = revision
    return found


def _content(revision):
    return ((revision.get("slots") or {}).get("main") or {}).get("content") or ""


def _citation(match):
    fields = {}
    for name, value in CITE_FIELD.findall(match.group(0)):
        fields.setdefault(name.lower(), " ".join(value.split()))
    title = fields.get("title", "")
    where = next(
        (
            fields[k]
            for k in ("website", "work", "newspaper", "journal", "magazine", "publisher")
            if fields.get(k)
        ),
        "",
    )
    return ", ".join(p for p in (title, where) if p)


def draft_text(content):
    """Readable draft text: citations as [ref: title, publication], no templates or markup."""
    text = re.sub(r"<!--.*?-->", "", content, flags=re.S)
    text = CITE.sub(_citation, text)
    text = EMPTY_REF.sub("", text)
    text = REF.sub(
        lambda m: f" [ref: {' '.join(m.group(1).split())}]" if m.group(1).strip() else "", text
    )
    text = FILE_OR_CATEGORY.sub("", text)
    text, _ = clean(text)
    text = LINK.sub(r"\1", PIPED.sub(r"\1", text))
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"[ \t]+\[ref:", " [ref:", text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(line.strip() for line in text.split("\n"))).strip()


def _asked_at(content, revision, outcome_at):
    """The latest pending submission time, if it is before the review; else the revision time."""
    stamps = sorted(PENDING_TS.findall(content))
    if stamps:
        moment = iso(datetime.strptime(stamps[-1], "%Y%m%d%H%M%S").replace(tzinfo=UTC))
        if moment < outcome_at:
            return moment
    return revision.get("timestamp")


def record(review, revision, now):
    """("ready", record) for a usable review, ("dropped", record) otherwise."""
    content = _content(revision)
    digest = hashlib.sha256(f"enwiki:{review['reviewed']}".encode()).hexdigest()[:16]
    row = {
        "id": f"wikipedia-afc-{digest}",
        "stream": "wikipedia-afc",
        "asked_at": _asked_at(content, revision, review["at"]),
        "collected_at": iso(now),
        "source": {
            "name": "wikipedia-afc",
            "wiki": "enwiki",
            "title": review["title"],
            "rev_id": review["reviewed"],
            "review": review["review"],
            "url": f"https://en.wikipedia.org/w/index.php?oldid={review['reviewed']}",
        },
        "data": {"title": review["title"].removeprefix("Draft:"), "text": draft_text(content)},
    }
    if len(row["data"]["text"]) < MIN_CHARS:
        return "dropped", {**row, "reason": "no-text"}
    if not row["asked_at"] or row["asked_at"] >= review["at"]:
        return "dropped", {**row, "reason": "no-ask-time"}
    return "ready", {**row, "outcome": {"label": review["label"], "outcome_at": review["at"]}}


def unsafe_decline(comment):
    """True when a decline or rejection summary gives an unsafe reason (see ``UNSAFE_CODES``)."""
    head = comment.split(" - ", 1)[0] + " - "
    return bool(UNSAFE_CODES & set(REASON_CODES.findall(head)) or UNSAFE_WORDS.search(comment))


def _reviews(http, day):
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC)
    end = start + timedelta(days=1) - timedelta(seconds=1)
    edits = _listing(
        http,
        {
            "action": "query",
            "list": "recentchanges",
            "rcnamespace": NAMESPACE,
            "rctag": "AFCH",
            "rctype": "edit",
            "rcprop": "title|ids|comment|timestamp",
            "rcdir": "newer",
            "rcstart": iso(start),
            "rcend": iso(end),
            "rclimit": "max",
        },
        "recentchanges",
    )
    reviews = [
        {
            "title": e["title"],
            "reviewed": e["old_revid"],
            "review": {"rev_id": e["revid"]},
            "at": e["timestamp"],
            "label": "false",
        }
        for e in edits
        if DECLINE.match(e.get("comment") or "")
        and not unsafe_decline(e["comment"])
        and e.get("old_revid")
    ]
    moves = _listing(
        http,
        {
            "action": "query",
            "list": "logevents",
            "letype": "move",
            "letag": "AFCH",
            "lenamespace": NAMESPACE,
            "leprop": "title|ids|comment|timestamp|details",
            "ledir": "newer",
            "lestart": iso(start),
            "leend": iso(end),
            "lelimit": "max",
        },
        "logevents",
    )
    accepts = [
        m
        for m in moves
        if ACCEPT in (m.get("comment") or "") and (m.get("params") or {}).get("target_ns") == 0
    ]
    # A move adds a revision to the moved page; its parent is the draft as the reviewer saw it.
    parents = _revisions(http, [m["revid"] for m in accepts], "ids")
    for move in accepts:
        parent = (parents.get(move["revid"]) or {}).get("parentid")
        if parent:
            reviews.append(
                {
                    "title": move["title"],
                    "reviewed": parent,
                    "review": {"target": move["params"]["target_title"]},
                    "at": move["timestamp"],
                    "label": "true",
                }
            )
    return reviews


def collect(store, http, now, *, days):
    """Read each day's reviews; add the reviewed drafts, ready to resolve."""
    added = reviewed = 0
    for day in days:
        reviews = _reviews(http, day)
        reviewed += len(reviews)
        revisions = _revisions(http, [r["reviewed"] for r in reviews], "ids|timestamp|content")
        ready, dropped = [], []
        for review in reviews:
            revision = revisions.get(review["reviewed"])
            if revision is None:
                continue  # deleted or hidden revision
            status, row = record(review, revision, now)
            (ready if status == "ready" else dropped).append(row)
        added += store.add(ready)
        store.drop(dropped)
    return {"added": added, "reviews": reviewed}
