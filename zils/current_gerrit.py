"""Zils Current Gerrit stream: will this proposed code change be merged?

Public Gerrit servers (OpenStack's OpenDev, Wikimedia, Android's AOSP) review every change before
it lands. A change either merges or is abandoned. One query per server, status and day
(`status:merged after:"D" before:"D+1"`, 200 changes a page) returns the changes with every
patch set's commit message, the owner and the message history. The state is the change as first
proposed: the server, project, branch and patch set 1's commit message, without trailers
(Signed-off-by, Change-Id, Reviewed-by, ...), cherry-pick notes or names. The label is whether
it merged; the outcome time is the submit time or the "Abandoned" message.

Not every outcome is a judgement of the change, so some are dropped:

- changes owned by bots and service accounts;
- titles that say not to merge (WIP, DNM, "do not merge", [RFC], ...);
- abandons whose reason names a replacement (superseded, duplicate, squashed, in favour of, ...);
- changes decided within an hour of upload (a mistaken upload or an automated flow);
- more than OWNER_CAP changes per owner, outcome and day (bulk clean-ups).

Days are read once they are MIN_AGE_DAYS old, so most abandon-then-restore cycles are over.
"""

import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from urllib.parse import quote, urlencode

from .current_items import RateLimited, iso, scrub

HOSTS = {
    "opendev": ("https://review.opendev.org", "OpenStack (OpenDev)"),
    "wikimedia": ("https://gerrit.wikimedia.org/r", "Wikimedia"),
    "android": ("https://android-review.googlesource.com", "Android (AOSP)"),
}
MIN_AGE_DAYS = 3
FIRST_DAYS = 14
PAGE_SIZE = 200
GAP_SECONDS = 1.0
RETRIES, RETRY_WAIT = 3, 30.0
MIN_LIFE = timedelta(hours=1)
OWNER_CAP = 5
MIN_CHARS = 15
OPTIONS = ("ALL_REVISIONS", "ALL_COMMITS", "DETAILED_ACCOUNTS", "MESSAGES")
sleep = time.sleep
BOT = re.compile(
    r"bot\b|bot$|zuul|proposal|l10n|translat|autoroll|automerger|coastguard|service|\bci\b", re.I
)
NOT_FOR_MERGE = re.compile(
    r"\b(?:WIP|DNM|DO[\s_-]*NOT[\s_-]*MERGE|NOT[\s_-]+FOR[\s_-]+MERGE)\b"
    r"|\[(?:RFC|POC|DRAFT|TEST)\]|^(?:RFC|POC|DRAFT|TEST)\s*:",
    re.I,
)
SUPERSEDED = re.compile(
    r"supersed|duplicate|squash|replaced (?:by|with)|in favou?r of|moved to|cherry[- ]?pick"
    r"|automated revert|release submitter|rebased|done elsewhere|elsewise|another change"
    r"|already merged|merged (?:as|in|into|via)",
    re.I,
)
FOOTER_KEYS = re.compile(
    r"^(?:[A-Za-z-]+-by|Change-Id|Bug|Closes-Bug|Partial-Bug|Related-Bug|Depends-On|Needed-By"
    r"|Implements|Story|Task|Fixes|Test|Tests|Flag|Merged-In|Ignore-AOSP-First|Cherry-picked-from"
    r"|Reviewed-on|Signed-off-by|Change-Id|Co-Authored-By|Generated-By|Assisted-By|Bug-Id)\s*:",
    re.I,
)
CHERRY = re.compile(r"^\(cherry picked from commit [0-9a-f]+\)\s*$", re.I)


def days(now, cursor=None, first_days=FIRST_DAYS):
    """Outcome days after `cursor` (or the last `first_days`) that are MIN_AGE_DAYS old."""
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


def _time(text):
    """Gerrit's `2026-10-26 15:00:00.000000000` (UTC) as an ISO time."""
    return iso(datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC))


class Api:
    """Serial requests GAP_SECONDS apart; HTTP 429 and 503 wait and retry, then stop the run."""

    def __init__(self, http):
        self.http, self.calls = http, 0

    def get(self, url):
        for attempt in range(RETRIES + 1):
            if self.calls:
                sleep(GAP_SECONDS if attempt == 0 else RETRY_WAIT)
            status, raw = self.http.get_bytes(url)
            self.calls += 1
            if status not in (429, 503):
                break
        else:
            raise RateLimited(f"Gerrit HTTP {status}")
        if status != 200:
            return status, None
        # Gerrit prefixes JSON with )]}' against cross-site script inclusion.
        return status, json.loads(raw.decode("utf-8").split("\n", 1)[1])


def _changes(api, base, status, day):
    following = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    query = f'status:{status} after:"{day}" before:"{following}"'
    rows, start = [], 0
    while True:
        params = [("q", query), ("n", str(PAGE_SIZE)), ("S", str(start))]
        params += [("o", option) for option in OPTIONS]
        code, page = api.get(f"{base}/changes/?{urlencode(params)}")
        if code != 200:
            raise ValueError(f"Gerrit {base} HTTP {code} for {status} on {day}")
        rows += page
        start += len(page)
        if not page or not page[-1].get("_more_changes"):
            return rows


def _outcome(row):
    """(label, outcome time, abandon reason) of a merged or abandoned change."""
    if row.get("status") == "MERGED":
        return "true", row.get("submitted"), ""
    abandons = [m for m in row.get("messages") or [] if m["message"].startswith("Abandoned")]
    if not abandons:
        return "false", None, ""
    return "false", abandons[-1]["date"], abandons[-1]["message"].removeprefix("Abandoned").strip()


def _names(row):
    names = set()
    for revision in (row.get("revisions") or {}).values():
        for person in (
            revision.get("uploader") or {},
            (revision.get("commit") or {}).get("author") or {},
        ):
            names |= {person.get("name"), person.get("username")}
    owner = row.get("owner") or {}
    names |= {owner.get("name"), owner.get("username"), owner.get("email")}
    return sorted(n for n in names if n)


def message_text(message, names=()):
    """The commit message without trailers, cherry-pick notes or names."""
    lines = [
        line
        for line in (message or "").strip().split("\n")
        if not FOOTER_KEYS.match(line.strip()) and not CHERRY.match(line.strip())
    ]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return scrub(text, names)


def _bot(owner):
    if "SERVICE_USER" in (owner.get("tags") or []):
        return True
    return bool(BOT.search(f"{owner.get('username', '')} {owner.get('name', '')}"))


def record(host, row, now):
    """(kind, record, outcome day): kind "ready" or "dropped", or None for an undecided change."""
    base, name = HOSTS[host]
    label, outcome_at, reason = _outcome(row)
    revisions = sorted((row.get("revisions") or {}).items(), key=lambda kv: kv[1]["_number"])
    if not outcome_at or not revisions:
        return None, None, None
    sha, first = revisions[0]
    number, project = row["_number"], row["project"]
    digest = hashlib.sha256(f"gerrit:{host}:{number}".encode()).hexdigest()[:16]
    out = {
        "id": f"gerrit-merge-{digest}",
        "stream": "gerrit-merge",
        "asked_at": _time(row["created"]),
        "collected_at": iso(now),
        "source": {
            "name": "gerrit",
            "host": host,
            "number": number,
            "project": project,
            "revision": sha,
            "url": f"{base}/c/{project}/+/{number}",
        },
        "data": {
            "host": name,
            "project": project,
            "branch": row.get("branch", ""),
            "message": message_text((first.get("commit") or {}).get("message"), _names(row)),
        },
        "owner": (row.get("owner") or {}).get("_account_id"),
    }
    outcome = {"label": label, "outcome_at": _time(outcome_at)}
    day = outcome["outcome_at"][:10]
    subject = (first.get("commit") or {}).get("subject") or row.get("subject") or ""
    if _bot(row.get("owner") or {}):
        return "dropped", {**out, "reason": "bot-owner"}, day
    if NOT_FOR_MERGE.search(subject):
        return "dropped", {**out, "reason": "not-for-merge"}, day
    if label == "false" and SUPERSEDED.search(reason):
        return "dropped", {**out, "reason": "superseded"}, day
    if (
        datetime.fromisoformat(outcome["outcome_at"][:-1])
        - datetime.fromisoformat(out["asked_at"][:-1])
        < MIN_LIFE
    ):
        return "dropped", {**out, "reason": "too-quick"}, day
    if len(out["data"]["message"]) < MIN_CHARS:
        return "dropped", {**out, "reason": "no-text"}, day
    return "ready", {**out, "outcome": outcome}, day


def collect(store, http, now, *, days, hosts=None):
    """Read each day's merged and abandoned changes on each server; add them, ready to resolve."""
    api, added, seen = Api(http), 0, 0
    for day in days:
        for host in hosts or list(HOSTS):
            ready, dropped = [], []
            for status in ("merged", "abandoned"):
                for row in _changes(api, HOSTS[host][0], status, day):
                    seen += 1
                    kind, out, decided = record(host, row, now)
                    if kind is None or decided != day:
                        continue  # undecided, or decided on another day (that day's query has it)
                    (ready if kind == "ready" else dropped).append(out)
            ready = _owner_capped(ready, dropped)
            added += store.add([{k: v for k, v in r.items() if k != "owner"} for r in ready])
            store.drop([{k: v for k, v in r.items() if k != "owner"} for r in dropped])
    return {"added": added, "changes": seen, "calls": api.calls}


def _owner_capped(ready, dropped):
    """At most OWNER_CAP changes per owner and outcome, lowest change numbers first."""
    groups, kept = defaultdict(list), []
    for row in sorted(ready, key=lambda r: r["source"]["number"]):
        groups[(row["owner"], row["outcome"]["label"])].append(row)
    for rows in groups.values():
        kept += rows[:OWNER_CAP]
        dropped += [{**r, "reason": "owner-cap"} for r in rows[OWNER_CAP:]]
    return sorted(kept, key=lambda r: r["source"]["number"])


def recheck(api, host, project, number):
    """The outcome read again from the change's history: "true", "false", or None if open."""
    base = HOSTS[host][0]
    code, row = api.get(f"{base}/changes/{quote(project, safe='')}~{number}/detail")
    if code != 200 or not isinstance(row, dict):
        return None, f"HTTP {code}"
    if row.get("submitted"):
        return "true", f"submitted {row['submitted'][:19]}"
    if any(m["message"].startswith("Abandoned") for m in row.get("messages") or []):
        return "false", "Abandoned message, never submitted"
    return None, "neither submitted nor abandoned"
