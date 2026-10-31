"""Zils Current Articles for deletion stream: will English Wikipedia keep or delete this article?

Each day's log page (`Wikipedia:Articles for deletion/Log/2026 October 4`) transcludes that
day's deletion discussions. Discussions close after about seven days (longer when relisted). A
closed discussion is wrapped in an archive box that starts with "The result was '''keep'''" and
the closer's signature, above the discussion heading. The state is everything from the heading to
the archive footer: the nomination and the discussion as they stood at the close, which by
convention is never edited afterwards. The label is the closing result, mapped to four choices.

Editor names are removed from signatures and from the text. One request lists a day's
discussions; one more request reads up to 50 of them.
"""

import hashlib
import html
import re
from datetime import UTC, datetime, timedelta

from .current_items import iso, scrub

API = "https://en.wikipedia.org/w/api.php"
PREFIX = "Wikipedia:Articles for deletion/"
MIN_AGE_DAYS = 10  # most discussions on a log day are closed ten days later
FIRST_DAYS = 45
BATCH = 50
MONTHS = (
    "January February March April May June July August September October November December".split()
)
RESULT = re.compile(r"The result was\s*'''(.*?)'''", re.S | re.I)
STAMP = re.compile(r"(\d{1,2}):(\d{2}), (\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4}) \(UTC\)")
HEADING = re.compile(r"^===\s*\[\[:?(.+?)\]\]\s*===\s*$", re.M)
FOOTER = re.compile(r":?''The above discussion is preserved as an archive.*", re.S)
USER_LINK = re.compile(
    r"\[\[\s*(?:User(?: talk)?|Special:Contributions)\s*[:/]\s*([^\]|]+)(?:\|[^\]]*)?\]\]", re.I
)
TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
CHOICES = ("keep", "delete", "merge-redirect", "no-consensus")


def log_title(day):
    moment = datetime.strptime(day, "%Y-%m-%d")
    return f"{PREFIX}Log/{moment.year} {MONTHS[moment.month - 1]} {moment.day}"


def log_days(now, cursor=None, first_days=FIRST_DAYS):
    """Log days after `cursor` (or the last `first_days`) that are at least MIN_AGE_DAYS old."""
    last = (now.astimezone(UTC) - timedelta(days=MIN_AGE_DAYS)).date()
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


def result_class(result):
    """keep / delete / merge-redirect / no-consensus, or None for other results."""
    words = " ".join(re.sub(r"\[\[.*?\]\]", " ", result.lower()).split())
    if "no consensus" in words:
        return "no-consensus"
    if re.search(r"\b(?:withdrawn|draftif|userf|transwiki|moot|procedural)", words):
        return None
    if re.search(r"\b(?:merge|redirect)", words):
        return "merge-redirect"
    if re.search(r"\bdelete", words):
        return "delete"
    if re.search(r"\bkeep\b", words):
        return "keep"
    return None


def stamp(text):
    match = STAMP.search(text or "")
    if not match:
        return None
    hour, minute, day, month, year = match.groups()
    moment = datetime(
        int(year), MONTHS.index(month) + 1, int(day), int(hour), int(minute), tzinfo=UTC
    )
    return iso(moment)


def clean(text):
    """Readable discussion text: no user links, templates, comments or markup tags."""
    names = {n.strip() for n in USER_LINK.findall(text)}
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<(includeonly|noinclude)>.*?</\1>", "", text, flags=re.S)
    text = USER_LINK.sub("[user]", text)
    for _ in range(3):
        text = TEMPLATE.sub("", text)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    text = re.sub(r"(?:\[user\]\s*(?:\(\s*\[user\]\s*\))?\s*)+", "[user] ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n(?:\s*:?\(\s*\)\s*\n)+", "\n", text)
    return scrub(re.sub(r"\n{3,}", "\n\n", text).strip(), sorted(names)), names


def record(title, page, now):
    """A ready record for one closed discussion, ("dropped", record) for an unusable one, or None."""
    revision = (page.get("revisions") or [{}])[0]
    content = ((revision.get("slots") or {}).get("main") or {}).get("content") or ""
    result, heading = RESULT.search(content), HEADING.search(content)
    if not result or not heading or result.start() > heading.start():
        return None  # still open
    body = FOOTER.sub("", content[heading.start() :])
    text, _ = clean(body)
    article = heading.group(1).split("|")[0].strip()
    digest = hashlib.sha256(f"enwiki:{page.get('pageid') or title}".encode()).hexdigest()[:16]
    label = result_class(result.group(1))
    row = {
        "id": f"wikipedia-afd-{digest}",
        "stream": "wikipedia-afd",
        "asked_at": stamp(body) or stamp(content),
        "collected_at": iso(now),
        "source": {
            "name": "wikipedia-afd",
            "wiki": "enwiki",
            "title": title,
            "rev_id": revision.get("revid"),
            "url": f"https://en.wikipedia.org/w/index.php?title={title.replace(' ', '_')}"
            f"&oldid={revision.get('revid')}",
        },
        "data": {
            "article": article,
            "text": text.split("\n", 1)[1].strip() if "\n" in text else "",
        },
    }
    if not label or not row["asked_at"]:
        return "dropped", {**row, "reason": "ambiguous-label"}
    outcome_at = stamp(content[result.start() : heading.start()])
    return "ready", {**row, "outcome": {"label": label, "outcome_at": outcome_at}}


def _get(http, params):
    params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
    from .current_wikimedia import _api

    return _api(http, API, params)


def collect(store, http, now, *, days):
    """Read the discussions listed on each log day; add closed ones, ready to resolve."""
    added, calls, seen = 0, 0, set()
    for day in days:
        status, body = _get(
            http,
            {
                "action": "query",
                "prop": "templates",
                "titles": log_title(day),
                "tlnamespace": "4",
                "tllimit": "max",
            },
        )
        calls += 1
        pages = (body.get("query") or {}).get("pages") or [] if status == 200 else []
        titles = [
            t["title"]
            for p in pages
            for t in p.get("templates") or []
            if t["title"].startswith(PREFIX) and not t["title"].startswith(PREFIX + "Log/")
        ]
        titles = [t for t in titles if t not in seen]
        seen.update(titles)
        for start in range(0, len(titles), BATCH):
            chunk = titles[start : start + BATCH]
            status, body = _get(
                http,
                {
                    "action": "query",
                    "prop": "revisions",
                    "rvprop": "ids|content",
                    "rvslots": "main",
                    "titles": "|".join(chunk),
                },
            )
            calls += 1
            ready, dropped = [], []
            for page in (body.get("query") or {}).get("pages") or [] if status == 200 else []:
                outcome = record(page["title"], page, now)
                if outcome:
                    (ready if outcome[0] == "ready" else dropped).append(outcome[1])
            added += store.add(ready)
            store.drop(dropped)
    return {"added": added, "calls": calls}
