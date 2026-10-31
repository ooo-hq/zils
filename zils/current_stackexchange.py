"""Zils Current Stack Exchange streams: questions from a week or two ago and what happened to them.

One request to the public API (`/2.3/questions`, `withbody` filter) returns up to 100 questions
of one site with their title, body, tags, close state, score and accepted answer. Each question
becomes up to three items, all with the question as asked:

- `stackexchange-closed-<group>`: was it closed? (yes/no)
- `stackexchange-score-<group>`: its score, in four ordered levels (score)
- `stackexchange-accepted-<group>`: which of its answers did the asker accept? (choice; one more
  request per 100 questions reads the answers)

Sites are grouped by subject so each group is one exam category (see SITES). Without an API key
the API allows 300 requests a day per IP address, so a run reads a few pages per site, waits
between requests, honours `backoff`, and stops when the quota runs low. Questions edited more
than an hour after they were asked are dropped, so the text is the text the outcome followed.
"""

import hashlib
import html
import random
import re
import time
from datetime import UTC, datetime, timedelta

from .current_items import SCORE_LEVELS, SE_SITES, RateLimited, iso, scrub

API = "https://api.stackexchange.com/2.3"
SITES = SE_SITES
GROUP = {site: group for group, sites in SITES.items() for site in sites}
MIN_AGE_DAYS = 8  # closes and accepts mostly happen in the first days; scores settle within a week
WINDOW_DAYS = 14  # a first run reads questions asked up to this long before MIN_AGE_DAYS
PAGES_PER_SITE = 2
PAGE_SIZE = 100
EDIT_GRACE = timedelta(hours=1)
MIN_ANSWERS, MAX_ANSWERS = 2, 4
QUOTA_RESERVE = 20
GAP_SECONDS = 2.0
TAG = re.compile(r"<[^>]+>")
BLOCK_TAGS = re.compile(r"</?(?:p|pre|li|ul|ol|h\d|blockquote|br|hr|div|table|tr)\b[^>]*>", re.I)
sleep = time.sleep


def text(body):
    """Readable text from post HTML: block tags become line breaks, code keeps its lines."""
    body = BLOCK_TAGS.sub("\n", body or "")
    lines = [line.rstrip() for line in html.unescape(TAG.sub("", body)).split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def score_level(score):
    for index, (bound, _) in enumerate(SCORE_LEVELS):
        if bound is None or score <= bound:
            return index
    raise AssertionError("unreachable")


def _id(stream, site, number):
    digest = hashlib.sha256(f"{site}:{number}".encode()).hexdigest()[:16]
    return f"{stream}-{digest}"


class Api:
    """Serial API calls with a gap between them, `backoff` honoured and a quota floor."""

    def __init__(self, http, key=None, gap=GAP_SECONDS, reserve=QUOTA_RESERVE):
        self.http, self.key, self.gap, self.reserve = http, key, gap, reserve
        self.calls, self.quota, self.wait = 0, None, 0.0

    def get(self, path, params):
        if self.quota is not None and self.quota <= self.reserve:
            raise RateLimited(f"Stack Exchange quota at {self.quota}")
        if self.calls:
            sleep(max(self.gap, self.wait))
        params = {**params, **({"key": self.key} if self.key else {})}
        status, body, _ = self.http.get_json(f"{API}{path}", params=params)
        self.calls += 1
        body = body if isinstance(body, dict) else {}
        if status == 429 or body.get("error_id") == 502:
            raise RateLimited(body.get("error_message") or f"HTTP {status}")
        if status != 200 or "items" not in body:
            raise ValueError(f"Stack Exchange {path}: HTTP {status} {body.get('error_message', '')}")
        self.quota = body.get("quota_remaining", self.quota)
        self.wait = float(body.get("backoff") or 0)
        return body


def _base(site, question, collected_at):
    owner = (question.get("owner") or {}).get("display_name") or ""
    names = [html.unescape(owner)]
    return {
        "asked_at": iso(datetime.fromtimestamp(question["creation_date"], UTC)),
        "collected_at": collected_at,
        "source": {
            "name": "stackexchange",
            "site": site,
            "question_id": question["question_id"],
            "url": question["link"],
            "content_license": question.get("content_license"),
        },
        "data": {
            "site": site,
            "group": GROUP.get(site, "tools"),
            "title": scrub(html.unescape(question["title"]), names),
            "tags": list(question.get("tags") or []),
            "body": scrub(text(question.get("body")), names),
        },
    }


def _edited_late(question):
    edited = question.get("last_edit_date")
    return bool(edited) and edited - question["creation_date"] > EDIT_GRACE.total_seconds()


def records(site, questions, now, answers=None):
    """Ready (already resolved) records for one site's questions; dropped ones with a reason."""
    group, collected = GROUP.get(site, "tools"), iso(now)
    ready, dropped = [], []
    for question in questions:
        base = _base(site, question, collected)
        number = question["question_id"]
        if _edited_late(question):
            dropped.append(
                {**base, "id": _id("stackexchange", site, number), "reason": "edited-after-ask"}
            )
            dropped[-1]["stream"] = f"stackexchange-closed-{group}"
            continue
        closed = "closed_date" in question
        outcome_at = (
            iso(datetime.fromtimestamp(question["closed_date"], UTC)) if closed else None
        )
        ready.append(
            {
                **base,
                "id": _id(f"stackexchange-closed-{group}", site, number),
                "stream": f"stackexchange-closed-{group}",
                "outcome": {"label": "true" if closed else "false", "outcome_at": outcome_at},
                "data": {**base["data"], "closed_reason": question.get("closed_reason")},
            }
        )
        if not closed:  # a closed question's score reflects the closing, not the question
            ready.append(
                {
                    **base,
                    "id": _id(f"stackexchange-score-{group}", site, number),
                    "stream": f"stackexchange-score-{group}",
                    "outcome": {"label": str(score_level(question["score"]))},
                }
            )
        accepted = question.get("accepted_answer_id")
        options = (answers or {}).get(number) or []
        if accepted and MIN_ANSWERS <= len(options) <= MAX_ANSWERS:
            if not any(a["answer_id"] == accepted for a in options):
                continue
            rng = random.Random(f"{site}:{number}")
            options = sorted(options, key=lambda a: a["answer_id"])
            rng.shuffle(options)
            letters = [chr(65 + i) for i in range(len(options))]
            names = [base["data"]["title"]] + [
                html.unescape((a.get("owner") or {}).get("display_name") or "") for a in options
            ]
            names += [html.unescape((question.get("owner") or {}).get("display_name") or "")]
            ready.append(
                {
                    **base,
                    "id": _id(f"stackexchange-accepted-{group}", site, number),
                    "stream": f"stackexchange-accepted-{group}",
                    "outcome": {
                        "label": letters[[a["answer_id"] for a in options].index(accepted)]
                    },
                    "data": {
                        **base["data"],
                        "answers": [scrub(text(a.get("body")), names[1:]) for a in options],
                    },
                }
            )
    return ready, dropped


def collect(store, http, now, *, sites=None, cursors=None, pages=PAGES_PER_SITE, key=None):
    """Read questions asked since each site's cursor and at least MIN_AGE_DAYS ago.

    `cursors` maps site -> last creation time read (epoch seconds) and is updated in place,
    so a stopped run resumes. Returns the number of records added and the API calls made."""
    api, cursors = Api(http, key=key), cursors if cursors is not None else {}
    until = int((now - timedelta(days=MIN_AGE_DAYS)).timestamp())
    added = 0
    for site in sites or [s for group in SITES.values() for s in group]:
        since = cursors.get(site) or int((now - timedelta(days=MIN_AGE_DAYS + WINDOW_DAYS)).timestamp())
        for _ in range(pages):
            if since >= until:
                break
            body = api.get(
                "/questions",
                {
                    "site": site,
                    "fromdate": since + 1,
                    "todate": until,
                    "sort": "creation",
                    "order": "asc",
                    "pagesize": PAGE_SIZE,
                    "filter": "withbody",
                },
            )
            questions = [q for q in body["items"] if not _is_bot(q)]
            if not body["items"]:
                cursors[site] = until
                break
            wanted = [
                q["question_id"]
                for q in questions
                if q.get("accepted_answer_id")
                and MIN_ANSWERS <= q.get("answer_count", 0) <= MAX_ANSWERS
                and not _edited_late(q)
            ]
            answers = _answers(api, site, wanted) if wanted else {}
            ready, dropped = records(site, questions, now, answers)
            added += store.add(ready)
            store.drop(dropped)
            since = max(q["creation_date"] for q in body["items"])
            cursors[site] = since
            if not body.get("has_more"):
                cursors[site] = until
                break
    return {"added": added, "calls": api.calls, "quota": api.quota}


def _is_bot(question):
    return (question.get("owner") or {}).get("user_type") == "does_not_exist"


def _answers(api, site, question_ids):
    """Answers of up to 100 questions per request, by question id."""
    found = {}
    for start in range(0, len(question_ids), PAGE_SIZE):
        chunk = question_ids[start : start + PAGE_SIZE]
        page = 1
        while True:
            body = api.get(
                f"/questions/{';'.join(map(str, chunk))}/answers",
                {
                    "site": site,
                    "pagesize": PAGE_SIZE,
                    "page": page,
                    "filter": "withbody",
                    "sort": "creation",
                    "order": "asc",
                },
            )
            for answer in body["items"]:
                found.setdefault(answer["question_id"], []).append(answer)
            if not body.get("has_more") or page >= 3:
                break
            page += 1
    return found


def resolve(records_, http, now):
    """Stack Exchange records are collected with their outcome: label them straight away."""
    for record in records_:
        outcome = record.get("outcome")
        if not outcome:
            yield "dropped", {**record, "reason": "no-outcome"}
            continue
        row = {k: v for k, v in record.items() if k != "outcome"}
        row.update(label=outcome["label"], resolved_at=iso(now))
        if outcome.get("outcome_at"):
            row["outcome_at"] = outcome["outcome_at"]
        yield "resolved", row
