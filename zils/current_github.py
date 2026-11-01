"""Zils Current GitHub streams: issues and pull requests opened in GH Archive hours, resolved from the REST API.

GH Archive (https://www.gharchive.org) publishes every public GitHub event in hourly files. Since
GitHub trimmed Events API payloads in 2025, pull request events carry ids and branch names but no
title or body, so pull request text is read from the REST API at the first resolution pass.
Resolution reads the issue or pull request's current state: maintainer labels, close reason, merge.
"""

import gzip
import hashlib
import itertools
import json
import re
import time
from datetime import datetime, timedelta

from . import settings
from .current_items import RateLimited, iso, scrub

ARCHIVE = "https://data.gharchive.org/{}.json.gz"
API = "https://api.github.com/repos/{}/{}/{}"
HOUR = re.compile(r"\d{4}-\d{2}-\d{2}-(?:[1-9]|1\d|2[0-3]|0)")
ISSUE_STREAMS = ("github-issue-type", "github-issue-close")
BUG = re.compile(r"\bbugs?\b")
FEATURE = re.compile(r"\b(?:enhancements?|features?|feature requests?)\b")
# GitHub's guidance against secondary rate limits: serial requests with a gap between them, and
# after a limit, wait Retry-After (or a minute, doubling) before retrying.
GAP_SECONDS = 1.0
LIMIT_RETRIES, LIMIT_WAIT, MAX_LIMIT_WAIT = 4, 60.0, 900.0
sleep = time.sleep


def hour_url(hour):
    return ARCHIVE.format(hour)


def hour_range(start, count):
    """Consecutive archive hours; the archive writes hours without a leading zero (2026-10-26-5)."""
    if not HOUR.fullmatch(start):
        raise ValueError("GH Archive hours look like 2026-10-26-5 (UTC, hour without leading zero)")
    first = datetime.strptime(start.rsplit("-", 1)[0], "%Y-%m-%d") + timedelta(
        hours=int(start.rsplit("-", 1)[1])
    )
    return [f"{t:%Y-%m-%d}-{t.hour}" for t in (first + timedelta(hours=i) for i in range(count))]


def _bot(user):
    user = user or {}
    return user.get("type") == "Bot" or str(user.get("login", "")).endswith("[bot]")


def _id(stream, repo, number):
    return f"{stream}-{hashlib.sha256(f'{repo}#{number}'.encode()).hexdigest()[:16]}"


def _names(event, *users):
    owner = event["repo"]["name"].split("/")[0]
    names = {event["actor"]["login"], *(u.get("login", "") for u in users if u)}
    if "org" not in event:  # a personal repository's owner is a person
        names.add(owner)
    return [n.removesuffix("[bot]") for n in names if n]


def _record(stream, event, kind, number, data, collected_at):
    repo = event["repo"]["name"]
    path = "issues" if kind == "issue" else "pulls"
    return {
        "id": _id(stream, repo, number),
        "stream": stream,
        "asked_at": event["created_at"],
        "collected_at": collected_at,
        "source": {
            "name": "gharchive",
            "event_id": event["id"],
            "repo": repo,
            "number": number,
            "kind": kind,
            "url": f"https://github.com/{repo}/{'issues' if kind == 'issue' else 'pull'}/{number}",
            "api": API.format(*repo.split("/", 1), f"{path}/{number}"),
        },
        "data": data,
    }


def records(events, collected_at):
    """Waiting-room records for human-opened issues and pull requests from outside contributors."""
    result = []
    for event in events:
        payload, actor = event.get("payload") or {}, event.get("actor") or {}
        if payload.get("action") != "opened" or _bot(actor):
            continue
        repo = event["repo"]["name"]
        if event["type"] == "IssuesEvent":
            issue = payload["issue"]
            if "pull_request" in issue or _bot(issue.get("user")):
                continue
            names = _names(event, issue.get("user"), *(issue.get("assignees") or []))
            data = {
                "repo": repo.split("/", 1)[1],
                "title": scrub(issue.get("title"), names),
                "body": scrub(issue.get("body"), names),
                "labels_at_open": [label["name"] for label in issue.get("labels") or []],
            }
            for stream in ISSUE_STREAMS:
                result.append(_record(stream, event, "issue", issue["number"], data, collected_at))
        elif event["type"] == "PullRequestEvent":
            # Owners merge almost every pull request in their own personal repositories.
            if repo.split("/")[0].lower() == actor.get("login", "").lower():
                continue
            pull, names = payload["pull_request"], _names(event)
            data = {
                "repo": repo.split("/", 1)[1],
                "head": scrub(pull["head"]["ref"], names),
                "base": scrub(pull["base"]["ref"], names),
            }
            result.append(
                _record("github-pr-merge", event, "pull", payload["number"], data, collected_at)
            )
    return result


def collect(store, http, hours, now):
    added = 0
    for hour in hours:
        status, raw = http.get_bytes(hour_url(hour))
        if status == 404:
            raise ValueError(f"GH Archive hour {hour} is not published yet")
        if status != 200:
            raise ValueError(f"GH Archive hour {hour} returned HTTP {status}")
        lines = gzip.decompress(raw).decode("utf-8").split("\n")
        added += store.add(records((json.loads(line) for line in lines if line), iso(now)))
    return added


def _headers():
    token = _token()
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    return {**headers, "Authorization": f"Bearer {token}"} if token else headers


def _limited(status, body, headers):
    message = str((body or {}).get("message", "")).lower()
    return status == 429 or (
        status == 403
        and (
            headers.get("x-ratelimit-remaining") == "0"
            or "retry-after" in headers
            or "rate limit" in message
        )
    )


def _exhausted(headers):
    """The hourly quota is spent: waiting could take up to an hour, so the pass stops instead."""
    return headers.get("x-ratelimit-remaining") == "0" and "retry-after" not in headers


class _Paced:
    """Serial GitHub requests GAP_SECONDS apart; secondary rate limits are waited out and retried."""

    def __init__(self):
        self.last = None

    def __call__(self, send, label):
        for attempt in range(LIMIT_RETRIES + 1):
            if self.last is not None:
                gap = GAP_SECONDS - (time.monotonic() - self.last)
                if gap > 0:
                    sleep(gap)
            status, body, headers = send()
            self.last = time.monotonic()
            if not _limited(status, body, headers):
                return status, body, headers
            if _exhausted(headers) or attempt == LIMIT_RETRIES:
                break
            wait = float(headers.get("retry-after") or LIMIT_WAIT * 2**attempt)
            sleep(min(wait, MAX_LIMIT_WAIT))
        raise RateLimited(f"{label} HTTP {status}")


def issue_kind(labels):
    """'bug', 'feature', 'ambiguous' or None from label names such as 'type: bug' or 'enhancement'."""
    kinds = set()
    for name in labels:
        name = re.sub(r"[^a-z0-9]+", " ", name.lower())
        if "not a bug" in name:
            continue
        kinds |= {"bug"} if BUG.search(name) else set()
        kinds |= {"feature"} if FEATURE.search(name) else set()
    return ("ambiguous" if len(kinds) > 1 else kinds.pop()) if kinds else None


def _outcome(record, body, now):
    stream = record["stream"]
    if stream == "github-issue-type":
        kind = issue_kind(label["name"] for label in body.get("labels") or [])
        if kind == "ambiguous":
            return "dropped", {**record, "reason": "ambiguous-label"}
        return (
            ("resolved", {**record, "resolved_at": iso(now), "label": kind})
            if kind
            else ("pending", record)
        )
    if stream == "github-issue-close":
        if body.get("state") != "closed":
            return "pending", record
        if _bot(body.get("closed_by")):
            return "dropped", {**record, "reason": "closed-by-bot"}
        reason = body.get("state_reason")
        if reason not in ("completed", "not_planned", "duplicate"):
            return "dropped", {**record, "reason": "no-close-reason"}
        label = "true" if reason == "completed" else "false"
        return "resolved", {
            **record,
            "resolved_at": iso(now),
            "label": label,
            "outcome_at": body["closed_at"],
        }
    if _bot(body.get("user")):
        return "dropped", {**record, "reason": "bot"}
    data = record["data"]
    if "title" not in data:
        owner = record["source"]["repo"].split("/")[0]
        names = [body["user"].get("login", ""), owner]
        data = {
            **data,
            "title": scrub(body.get("title"), names),
            "body": scrub(body.get("body"), names),
            "text_at": iso(now),
        }
        record = {**record, "data": data}
    if body.get("merged"):
        return "resolved", {
            **record,
            "resolved_at": iso(now),
            "label": "true",
            "outcome_at": body["merged_at"],
        }
    if body.get("state") == "closed":
        return "resolved", {
            **record,
            "resolved_at": iso(now),
            "label": "false",
            "outcome_at": body["closed_at"],
        }
    return "pending", record


GRAPHQL = "https://api.github.com/graphql"
GRAPHQL_BATCH = 50  # issues and pull requests per GraphQL query
REPO_NAME = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
ACTOR = "{ __typename login }"
FRAGMENT = f"""fragment item on IssueOrPullRequest {{
  __typename
  ... on Issue {{ state stateReason closedAt labels(first: 50) {{ nodes {{ name }} }} author {ACTOR}
    timelineItems(last: 1, itemTypes: [CLOSED_EVENT]) {{ nodes {{ ... on ClosedEvent {{ actor {ACTOR} }} }} }} }}
  ... on PullRequest {{ state merged mergedAt closedAt title body author {ACTOR} }}
}}"""


def _token():
    return settings.get("ZILS_GITHUB_TOKEN") or settings.get("GITHUB_TOKEN")


def _user(actor):
    if not actor:
        return None
    return {
        "login": actor.get("login", ""),
        "type": "Bot" if actor.get("__typename") == "Bot" else "User",
    }


def _rest_shape(node):
    """A GraphQL issue or pull request in the REST fields that _outcome reads."""
    state = "open" if node.get("state") == "OPEN" else "closed"
    if node.get("__typename") == "PullRequest":
        return {
            "state": state,
            "merged": bool(node.get("merged")),
            "merged_at": node.get("mergedAt"),
            "closed_at": node.get("closedAt"),
            "title": node.get("title"),
            "body": node.get("body"),
            "user": _user(node.get("author")) or {},
        }
    closes = (node.get("timelineItems") or {}).get("nodes") or []
    return {
        "state": state,
        "state_reason": (node.get("stateReason") or "").lower() or None,
        "closed_at": node.get("closedAt"),
        "labels": [{"name": n["name"]} for n in (node.get("labels") or {}).get("nodes") or []],
        "closed_by": _user(closes[-1].get("actor")) if closes else None,
        "user": _user(node.get("author")) or {},
    }


def _query(keys):
    repos = {}
    for repo, number in keys:
        repos.setdefault(repo, []).append(number)
    aliases, parts = {}, []
    for r, (repo, numbers) in enumerate(repos.items()):
        owner, name = repo.split("/", 1)
        items = " ".join(f"n{n}: issueOrPullRequest(number: {n}) {{ ...item }}" for n in numbers)
        parts.append(f'r{r}: repository(owner: "{owner}", name: "{name}") {{ {items} }}')
        aliases.update({(repo, n): (f"r{r}", f"n{n}") for n in numbers})
    return FRAGMENT + "\nquery { " + " ".join(parts) + " }", aliases


def _graphql(groups, http, token):
    """Yield (key, status, body) per issue or pull request, GRAPHQL_BATCH keys per request."""
    keys = list(groups)
    headers, paced = {"Authorization": f"Bearer {token}"}, _Paced()
    for start in range(0, len(keys), GRAPHQL_BATCH):
        chunk = keys[start : start + GRAPHQL_BATCH]
        query, aliases = _query(chunk)
        status, body, _ = paced(
            lambda: http.post_json(GRAPHQL, {"query": query}, headers=headers), "GraphQL"
        )
        errors = (body or {}).get("errors") or []
        if any(e.get("type") == "RATE_LIMITED" for e in errors):  # the hourly GraphQL quota
            raise RateLimited(f"GraphQL HTTP {status}")
        if status != 200 or not isinstance((body or {}).get("data"), dict):
            continue  # a timeout or server error: these stay pending
        missing = {tuple(e.get("path") or ())[:2] for e in errors if e.get("type") == "NOT_FOUND"}
        for key in chunk:
            repo_alias, item_alias = aliases[key]
            repository = body["data"].get(repo_alias)
            node = (repository or {}).get(item_alias)
            if node:
                yield key, 200, _rest_shape(node)
            elif (repo_alias,) in missing or (repo_alias, item_alias) in missing:
                yield key, 404, {}


def _rest(groups, http):
    headers, paced = _headers(), _Paced()
    for key, group in groups.items():
        url = group[0]["source"]["api"]
        try:
            status, body, _ = paced(lambda: http.get_json(url, headers=headers), "REST")
        except RateLimited as error:
            raise RateLimited(f"{error}; set GITHUB_TOKEN for 5,000 requests an hour") from None
        yield key, status, body


def resolve(records, http, now):
    """Yield (status, record); one lookup serves every stream built from the same issue.

    With a token, lookups go GRAPHQL_BATCH at a time through GraphQL (one request instead of
    fifty); without one, through the REST API, which allows 60 unauthenticated requests an hour.
    """
    groups = {}
    for record in records:
        source = record["source"]
        groups.setdefault((source["repo"], source["number"]), []).append(record)
    token = _token()
    batched = token and hasattr(http, "post_json")
    graphql = {k: v for k, v in groups.items() if batched and REPO_NAME.fullmatch(k[0])}
    rest = {k: v for k, v in groups.items() if k not in graphql}
    for key, status, body in itertools.chain(_graphql(graphql, http, token), _rest(rest, http)):
        for record in groups[key]:
            if status in (404, 410, 451):
                yield "dropped", {**record, "reason": "deleted"}
            elif status == 200:
                yield _outcome(record, body, now)
