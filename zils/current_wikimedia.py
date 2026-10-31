"""Zils Current Wikipedia stream: article edits from Wikimedia EventStreams, resolved by the revert tag.

Edits come from the public `recentchange` server-sent event stream (live, or replayed with
`since` within the stream's retention of several days). After a waiting period the MediaWiki
Action API is asked for each revision's change tags: MediaWiki adds `mw-reverted` to every edit
that a later undo, rollback or manual revert (an edit restoring an earlier version) reverses.
Edits that are themselves reverts (`mw-undo`, `mw-rollback`, `mw-manual-revert`) are dropped.
"""

import hashlib
import html
import json
import re
import time
from datetime import UTC, datetime, timedelta

from .current_items import IPV4, IPV6, RateLimited, iso, parse_time, scrub

STREAM = "https://stream.wikimedia.org/v2/stream/recentchange"
WAIT_HOURS = 48
BATCH = 50  # revids per Action API request
REVERT_TAGS = {"mw-undo", "mw-rollback", "mw-manual-revert"}
HIDDEN = ("texthidden", "commenthidden", "userhidden", "suppressed")
MAX_LINES, MAX_LINE = 40, 2000
LATE_RUN = 500  # consecutive replayed events past the window before a replay stops
CELL = re.compile(r'<td class="diff-(deleted|added)line[^"]*">\s*<div>(.*?)</div>\s*</td>', re.S)
TAG = re.compile(r"<[^>]+>")
sleep = time.sleep


def sse_events(lines):
    """(last event id, event) pairs from text/event-stream lines."""
    last_id, data = None, []
    for line in lines:
        if line == "":
            if data:
                try:
                    yield last_id, json.loads("\n".join(data))
                except ValueError:
                    pass
            data = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "data":
            data.append(value)
        elif field == "id":
            last_id = value


def editor(user):
    if IPV4.fullmatch(user) or IPV6.fullmatch(user):
        return "unregistered (IP address)"
    return "temporary account" if user.startswith("~") else "registered account"


def record(event, wikis, collected_at=None):
    """A waiting-room record for a human edit to an existing article, or None."""
    meta, revision = event.get("meta") or {}, event.get("revision") or {}
    if (
        event.get("type") != "edit"
        or event.get("wiki") not in wikis
        or event.get("namespace") != 0
        or event.get("bot")
        or meta.get("domain") == "canary"
        or not revision.get("old")
        or not revision.get("new")
    ):
        return None
    wiki, new, old, server = event["wiki"], revision["new"], revision["old"], event["server_url"]
    length = event.get("length") or {}
    return {
        "id": f"wikipedia-revert-{hashlib.sha256(f'{wiki}:{new}'.encode()).hexdigest()[:16]}",
        "stream": "wikipedia-revert",
        "asked_at": iso(datetime.fromtimestamp(event["timestamp"], UTC)),
        "collected_at": collected_at,
        "source": {
            "name": "wikimedia",
            "wiki": wiki,
            "rev_id": new,
            "old_rev_id": old,
            "title": event["title"],
            "url": f"{server}/w/index.php?diff={new}&oldid={old}",
            "api": f"{server}{event.get('server_script_path', '/w')}/api.php",
        },
        "data": {
            "wiki": wiki,
            "title": event["title"],
            "comment": scrub(event.get("comment"), [event.get("user", "")]),
            "length_old": length.get("old") or 0,
            "length_new": length.get("new") or 0,
            "minor": bool(event.get("minor")),
            "editor": editor(event.get("user", "")),
        },
    }


def collect(store, http, minutes, *, now, wikis=None, since=None, limit=None, clock=time.monotonic):
    """Read the stream for `minutes` of wall time, or of stream time from `since`."""
    wikis, batch, added, last_id, done = set(wikis or ["enwiki"]), [], 0, None, False
    end = parse_time(since) + timedelta(minutes=minutes) if since else None
    deadline = clock() + minutes * 60
    while not done:
        headers = {"Last-Event-ID": last_id} if last_id else {}
        params = {"since": since} if since and not last_id else {}
        seen = late = 0
        try:
            for last_id_, event in sse_events(http.lines(STREAM, params=params, headers=headers)):
                seen, last_id = seen + 1, last_id_ or last_id
                meta = event.get("meta") or {}
                if end and meta.get("domain") != "canary":
                    # Two datacentres' topics interleave out of order: stop after a run of late events.
                    late = late + 1 if parse_time(meta.get("dt", since)) > end else 0
                    if late >= LATE_RUN:
                        done = True
                        break
                    if late:
                        continue
                row = record(event, wikis, iso(now))
                if row:
                    batch.append(row)
                if len(batch) >= 200 or (limit and added + len(batch) >= limit):
                    added, batch = added + store.add(batch), []
                    done = bool(limit and added >= limit)
                if done or (not end and clock() > deadline):
                    done = True
                    break
        except OSError:  # the server closes connections after 15 minutes; resume from last_id
            pass
        if not seen or (not end and clock() > deadline):
            break
    return added + store.add(batch)


def diff_text(body):
    """Removed and added lines from an Action API compare table."""
    removed, added = [], []
    for side, cell in CELL.findall(body or ""):
        text = html.unescape(TAG.sub("", cell)).strip()
        if text:
            (removed if side == "deleted" else added).append(text[:MAX_LINE])
    return removed[:MAX_LINES], added[:MAX_LINES]


def _api(http, url, params):
    params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
    for attempt in range(4):
        status, body, headers = http.get_json(url, params=params)
        if status == 429:
            raise RateLimited("HTTP 429")
        error = (body or {}).get("error") or {}
        if error.get("code") != "maxlag":
            return status, body
        if attempt < 3:
            sleep(float(headers.get("retry-after", 5)))
    raise RateLimited("replication lag stayed high")


def _resolve_one(record, revision, http, now):
    if any(revision.get(flag) for flag in HIDDEN):
        return "dropped", {**record, "reason": "deleted"}
    tags = set(revision.get("tags") or [])
    if tags & REVERT_TAGS:
        return "dropped", {**record, "reason": "is-revert"}
    source = record["source"]
    status, body = _api(
        http,
        source["api"],
        {
            "action": "compare",
            "fromrev": str(source["old_rev_id"]),
            "torev": str(source["rev_id"]),
            "prop": "diff",
            "difftype": "table",
        },
    )
    if status != 200 or "error" in body or "compare" not in body:
        return "dropped", {**record, "reason": "deleted"}
    names = [revision.get("user", "")]
    removed, added = diff_text(body["compare"].get("body"))
    data = {
        **record["data"],
        "removed": [scrub(s, names) for s in removed],
        "added": [scrub(s, names) for s in added],
    }
    label = "true" if "mw-reverted" in tags else "false"
    return "resolved", {**record, "data": data, "resolved_at": iso(now), "label": label}


def resolve(records, http, now, wait_hours=WAIT_HOURS):
    """Yield (status, record) for edits older than the waiting period; younger ones stay pending."""
    due = [r for r in records if now - parse_time(r["asked_at"]) >= timedelta(hours=wait_hours)]
    groups = {}
    for row in due:
        groups.setdefault(row["source"]["api"], []).append(row)
    for url, rows in groups.items():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start : start + BATCH]
            status, body = _api(
                http,
                url,
                {
                    "action": "query",
                    "prop": "revisions",
                    "revids": "|".join(str(r["source"]["rev_id"]) for r in chunk),
                    "rvprop": "ids|tags|flags|timestamp|user",
                },
            )
            if status != 200 or "query" not in body:
                continue
            query = body["query"]
            revisions = {
                rev["revid"]: rev
                for page in query.get("pages") or []
                for rev in page.get("revisions") or []
            }
            missing = {int(k) for k in query.get("badrevids") or {}}
            for row in chunk:
                rev_id = row["source"]["rev_id"]
                if rev_id in missing:
                    yield "dropped", {**row, "reason": "deleted"}
                elif rev_id in revisions:
                    yield _resolve_one(row, revisions[rev_id], http, now)
