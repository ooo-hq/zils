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

from .current_items import IPV4, IPV6, WIKIDATA, WIKIS, RateLimited, iso, parse_time, scrub, wiki_stream

STREAM = "https://stream.wikimedia.org/v2/stream/recentchange"
WAIT_HOURS = 48
BATCH = 50  # revids per Action API request
REVERT_TAGS = {"mw-undo", "mw-rollback", "mw-manual-revert"}
HIDDEN = ("texthidden", "commenthidden", "userhidden", "suppressed")
MAX_LINES, MAX_LINE = 40, 2000
RETRIES, RETRY_SECONDS = 3, 10  # connections that deliver nothing, e.g. a timed-out replay
LATE_RUN = 500  # consecutive replayed events past the window before a replay stops
# ...once the window has delivered something. Before that, a run of late events is the other
# datacentre's topic arriving first (after a switchover it holds days of later events), so keep
# reading, up to this many late events in all.
LATE_LIMIT = 200_000
CELL = re.compile(r'<td class="diff-(deleted|added)line[^"]*">\s*<div>(.*?)</div>\s*</td>', re.S)
TAG = re.compile(r"<[^>]+>")
ROW = re.compile(r"<tr\b.*?</tr>", re.S)
LINENO = re.compile(r'<td[^>]*class="diff-lineno"[^>]*>(.*?)</td>', re.S)
# Wikidata edits made in bulk by tools; their outcome says little about the edit itself.
TOOL_EDIT = re.compile(r"toollabs:|quickstatements|#openrefine|#mix'n'match|\[\[:toollabs", re.I)
DEFAULT_WIKIS = (*WIKIS, WIKIDATA)
LIMIT_RETRIES, LIMIT_WAIT = 4, 60  # HTTP 429: retries, and the wait without Retry-After
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
        or (event["wiki"] == WIKIDATA and TOOL_EDIT.search(event.get("comment") or ""))
    ):
        return None
    wiki, new, old, server = event["wiki"], revision["new"], revision["old"], event["server_url"]
    length = event.get("length") or {}
    stream = wiki_stream(wiki)
    digest = hashlib.sha256(f"{wiki}:{new}".encode()).hexdigest()[:16]
    prefix = "wikidata-revert" if wiki == WIKIDATA else "wikipedia-revert"
    source = {"item": event["title"]} if wiki == WIKIDATA else {}
    return {
        "id": f"{prefix}-{digest}",
        "stream": stream,
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
            **source,
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
    wikis, batch, added, last_id, done = set(wikis or DEFAULT_WIKIS), [], 0, None, False
    end = parse_time(since) + timedelta(minutes=minutes) if since else None
    failures = late_total = 0
    inside = False
    deadline = clock() + minutes * 60
    while not done:
        headers = {"Last-Event-ID": last_id} if last_id else {}
        params = {"since": since} if since and not last_id else {}
        seen = late = 0
        errored = False
        try:
            for last_id_, event in sse_events(http.lines(STREAM, params=params, headers=headers)):
                seen, last_id = seen + 1, last_id_ or last_id
                meta = event.get("meta") or {}
                if end and meta.get("domain") != "canary":
                    # Two datacentres' topics interleave out of order: stop after a run of late events.
                    is_late = parse_time(meta.get("dt", since)) > end
                    late, late_total = (late + 1, late_total + 1) if is_late else (0, late_total)
                    inside = inside or not is_late
                    if (late >= LATE_RUN and inside) or late_total >= LATE_LIMIT:
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
            errored = True
        if not seen and errored and failures < RETRIES:
            failures += 1  # a refused or timed-out connection: wait, then ask again
            sleep(RETRY_SECONDS * failures)
            continue
        if not seen or (not end and clock() > deadline):
            break
        failures = 0
    return added + store.add(batch)


def _cell_text(cell):
    return " ".join(html.unescape(TAG.sub("", cell)).split())


def diff_text(body, context=False):
    """Removed and added lines from an Action API compare table.

    With `context`, each line is prefixed by its section heading, as Wikibase diffs label their
    rows (`Property / date of birth: 3 March 1961`)."""
    removed, added, heading = [], [], ""
    for row in ROW.findall(body or "") if context else [body or ""]:
        headings = [_cell_text(c) for c in LINENO.findall(row)]
        if context and headings:
            heading = next((h for h in reversed(headings) if h), "")
            continue
        for side, cell in CELL.findall(row):
            text = _cell_text(cell) if context else html.unescape(TAG.sub("", cell)).strip()
            if text:
                line = f"{heading}: {text}" if context and heading else text
                (removed if side == "deleted" else added).append(line[:MAX_LINE])
    return removed[:MAX_LINES], added[:MAX_LINES]


def _api(http, url, params):
    """One Action API request: paced, retried after `maxlag` and after HTTP 429 (Retry-After)."""
    params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
    lagged = limited = 0
    while True:
        status, body, headers = http.get_json(url, params=params)
        if status == 429:
            if limited >= LIMIT_RETRIES:
                raise RateLimited("HTTP 429")
            limited += 1
            sleep(float(headers.get("retry-after") or LIMIT_WAIT * limited))
            continue
        error = (body or {}).get("error") or {}
        if error.get("code") != "maxlag":
            return status, body
        lagged += 1
        if lagged >= 4:
            raise RateLimited("replication lag stayed high")
        sleep(float(headers.get("retry-after", 5)))


def _precheck(record, revision):
    """("dropped", record) for hidden or revert edits, else None."""
    if any(revision.get(flag) for flag in HIDDEN):
        return "dropped", {**record, "reason": "deleted"}
    if set(revision.get("tags") or []) & REVERT_TAGS:
        return "dropped", {**record, "reason": "is-revert"}
    return None


def _resolve_one(record, revision, http, now):
    source = record["source"]
    wikidata = record["data"]["wiki"] == WIKIDATA
    params = {
        "action": "compare",
        "fromrev": str(source["old_rev_id"]),
        "torev": str(source["rev_id"]),
        "prop": "diff",
        "difftype": "table",
    }
    status, body = _api(http, source["api"], {**params, "uselang": "en"} if wikidata else params)
    if status != 200 or "error" in body or "compare" not in body:
        return "dropped", {**record, "reason": "deleted"}
    names = [revision.get("user", "")]
    removed, added = diff_text(body["compare"].get("body"), context=wikidata)
    data = {
        **record["data"],
        "removed": [scrub(s, names) for s in removed],
        "added": [scrub(s, names) for s in added],
    }
    label = "true" if "mw-reverted" in set(revision.get("tags") or []) else "false"
    return "resolved", {**record, "data": data, "resolved_at": iso(now), "label": label}


def _sample(rows, revisions, ratio):
    """Ids of the not-reverted edits to keep: `ratio` per reverted edit, chosen by hashed id."""
    if ratio is None:
        return None
    positive = sum("mw-reverted" in (revisions[r]["tags"] or []) for r in rows)
    negatives = sorted(
        (r for r in rows if "mw-reverted" not in (revisions[r]["tags"] or [])),
        key=lambda r: hashlib.sha256(r.encode()).hexdigest(),
    )
    return set(negatives[: int(positive * ratio)])


def _labels(http, url, records):
    """English (or fallback) labels of the Wikidata items behind resolved records."""
    items = sorted({r["source"]["item"] for r in records})
    if not items:
        return {}
    status, body = _api(
        http,
        url,
        {
            "action": "wbgetentities",
            "ids": "|".join(items),
            "props": "labels",
            "languages": "en",
            "languagefallback": "1",
        },
    )
    if status != 200:
        return {}
    labels = {}
    for item, entity in ((body or {}).get("entities") or {}).items():
        values = (entity or {}).get("labels") or {}
        value = (values.get("en") or next(iter(values.values()), {})).get("value")
        if value:
            labels[item] = value
    return labels


def resolve(records, http, now, wait_hours=WAIT_HOURS, negatives_per_positive=None):
    """Yield (status, record) for edits older than the waiting period; younger ones stay pending.

    With `negatives_per_positive`, each batch keeps that many not-reverted edits per reverted
    one and drops the rest as `sampled-out` before any diff is fetched. Reverts are rare, export
    balances labels anyway, and this saves most of the per-edit diff requests."""
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
            candidates = {}
            for row in chunk:
                rev_id = row["source"]["rev_id"]
                if rev_id in missing:
                    yield "dropped", {**row, "reason": "deleted"}
                elif rev_id in revisions:
                    early = _precheck(row, revisions[rev_id])
                    if early:
                        yield early
                    else:
                        candidates[row["id"]] = row
            by_id = {i: revisions[r["source"]["rev_id"]] for i, r in candidates.items()}
            keep = _sample(list(candidates), by_id, negatives_per_positive)
            done = []
            for row_id, row in candidates.items():
                if keep is not None and row_id not in keep and "mw-reverted" not in (
                    by_id[row_id].get("tags") or []
                ):
                    yield "dropped", {**row, "reason": "sampled-out"}
                    continue
                status_, resolved = _resolve_one(row, by_id[row_id], http, now)
                if status_ == "resolved" and row["data"]["wiki"] == WIKIDATA:
                    done.append(resolved)
                else:
                    yield status_, resolved
            labels = _labels(http, url, done)
            for resolved in done:
                item = resolved["source"]["item"]
                yield "resolved", {**resolved, "data": {**resolved["data"], "label": labels.get(item)}}
