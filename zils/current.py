"""Zils Current: real-world decisions, labelled by what actually happened.

    python -m zils.current collect --source gharchive --start 2026-10-26-15 --hours 1
    python -m zils.current collect --source wikimedia --minutes 5 [--since 2026-10-28T12:00:00Z]
    python -m zils.current resolve
    python -m zils.current export --out items.jsonl --stats

`collect` puts each new event in a waiting room with `asked_at`; `resolve` labels the ones whose
outcome is known (`resolved_at`, `label`) and drops items that never resolve within the maximum
age; `export` writes bank-format cases, deduplicated by content hash and balanced per stream.
Storage is a local directory of JSON lines (default `runs/current`). See docs/current.md.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import ROOT, current_github, current_items, current_wikimedia, settings

VERSION = "zils-current/v1"
DEFAULT_DIR = ROOT / "runs" / "current"
MAX_AGE_DAYS = 14
CONTACT = "https://github.com/ooo-hq/zils"
FILES = ("pending.jsonl", "resolved.jsonl", "dropped.jsonl")


RateLimited, iso, parse_time = (
    current_items.RateLimited,
    current_items.iso,
    current_items.parse_time,
)


def user_agent():
    contact = settings.get("ZILS_CURRENT_CONTACT") or CONTACT
    return f"zils-current/{VERSION.rsplit('/', 1)[1]} ({contact})"


class Http:
    """requests with an identifying User-Agent; every call returns (status, body[, headers])."""

    def __init__(self, timeout=30):
        import requests

        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent()
        self.timeout = timeout

    def get_json(self, url, params=None, headers=None):
        response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
        try:
            body = response.json()
        except ValueError:
            body = {}
        return response.status_code, body, {k.lower(): v for k, v in response.headers.items()}

    def get_bytes(self, url):
        response = self.session.get(url, timeout=max(self.timeout, 120))
        return response.status_code, response.content

    def lines(self, url, params=None, headers=None):
        headers = {"Accept": "text/event-stream", **(headers or {})}
        with self.session.get(
            url, params=params, headers=headers, stream=True, timeout=self.timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                yield line


class Store:
    """Waiting room (pending), labelled items (resolved) and dropped items with a reason."""

    def __init__(self, root):
        self.root = Path(root)

    def _read(self, name):
        path = self.root / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def pending(self):
        return self._read("pending.jsonl")

    def resolved(self):
        return self._read("resolved.jsonl")

    def dropped(self):
        return self._read("dropped.jsonl")

    def known_ids(self):
        return {r["id"] for name in FILES for r in self._read(name)}

    def _append(self, name, rows):
        if rows:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / name).open("a") as handle:
                handle.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    def add(self, records):
        """Append records whose id the store has never seen; return how many were added."""
        known, fresh = self.known_ids(), []
        for record in records:
            if record["id"] not in known:
                known.add(record["id"])
                fresh.append(record)
        self._append("pending.jsonl", fresh)
        return len(fresh)

    def save(self, pending, resolved, dropped):
        """Replace the waiting room and append newly resolved and dropped items."""
        self._append("resolved.jsonl", resolved)
        self._append("dropped.jsonl", dropped)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "pending.jsonl"
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pending))
        os.replace(temporary, path)


RESOLVERS = {"github-": current_github.resolve, "wikipedia-": current_wikimedia.resolve}


def resolve(store, http, now, max_age_days=MAX_AGE_DAYS, wiki_wait_hours=None):
    """One resolution pass over the waiting room; returns counts of what happened."""
    pending = store.pending()
    outcomes, limited = {}, False
    for prefix, resolver in RESOLVERS.items():
        records = [r for r in pending if r["stream"].startswith(prefix)]
        options = (
            {"wait_hours": wiki_wait_hours} if prefix == "wikipedia-" and wiki_wait_hours else {}
        )
        try:
            for status, record in resolver(records, http, now, **options):
                outcomes[record["id"]] = (status, record)
        except RateLimited as error:
            print(
                f"{prefix.rstrip('-')}: rate limited, will continue later ({error})",
                file=sys.stderr,
            )
            limited = True
    keep, resolved, dropped = [], [], []
    oldest = now - timedelta(days=max_age_days)
    for record in pending:
        status, record = outcomes.get(record["id"], ("pending", record))
        if status == "pending" and parse_time(record["asked_at"]) < oldest:
            status, record = "dropped", {**record, "reason": "expired"}
        if status == "dropped":
            dropped.append({**record, "dropped_at": iso(now)})
        else:
            (resolved if status == "resolved" else keep).append(record)
    store.save(keep, resolved, dropped)
    counts = {"resolved": len(resolved), "dropped": len(dropped), "pending": len(keep)}
    return {**counts, "rate_limited": True} if limited else counts


def _balance(cases, outcomes, seed, stream):
    """Downsample every outcome to the rarest one's count; choice keys are shuffled per case."""
    by_label = defaultdict(list)
    for case in sorted(cases, key=lambda c: c["id"]):
        by_label[outcomes[case["id"]]].append(case)
    if len(by_label) < 2:
        return []
    size = min(map(len, by_label.values()))
    rng = random.Random(f"{seed}:{stream}")
    return [c for label in sorted(by_label) for c in rng.sample(by_label[label], size)]


def _median_hours(records, field):
    hours = [
        (parse_time(r[field]) - parse_time(r["asked_at"])).total_seconds() / 3600 for r in records
    ]
    return round(statistics.median(hours), 2) if hours else None


def export(store, out, *, seed=0, balance=True, max_tokens=4096):
    """Write deduplicated (and by default label-balanced) cases; return per-stream statistics."""
    resolved, streams = store.resolved(), {}
    groups = defaultdict(list)
    for record in resolved:
        groups[record["stream"]].append(record)
    dropped = defaultdict(Counter)
    for record in store.dropped():
        dropped[record["stream"]][record["reason"]] += 1
    pending = Counter(r["stream"] for r in store.pending())
    exported = []
    for stream in sorted(set(groups) | set(dropped) | set(pending)):
        records = sorted(groups.get(stream, []), key=lambda r: (r["asked_at"], r["id"]))
        cases, seen = [], set()
        for record in records:
            case = current_items.build(record, max_tokens)
            if case["content_sha256"] not in seen:
                seen.add(case["content_sha256"])
                cases.append(case)
        outcomes = {r["id"]: r["label"] for r in records}
        chosen = _balance(cases, outcomes, seed, stream) if balance else cases
        # resolved_at is when this tool saw the outcome; outcome_at is when it happened, if known.
        checked = _median_hours(records, "resolved_at")
        happened = _median_hours([r for r in records if r.get("outcome_at")], "outcome_at")
        streams[stream] = {
            "pending": pending.get(stream, 0),
            "resolved": len(records),
            "dropped": dict(sorted(dropped[stream].items())),
            "duplicates": len(records) - len(cases),
            "labels": dict(sorted(Counter(outcomes[c["id"]] for c in cases).items())),
            "exported": len(chosen),
            "exported_labels": dict(sorted(Counter(outcomes[c["id"]] for c in chosen).items())),
            "median_hours_to_resolution": checked,
            "median_hours_to_outcome": happened,
        }
        exported += chosen
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in exported))
    return {
        "version": VERSION,
        "seed": seed,
        "balanced": balance,
        "max_tokens": max_tokens,
        "exported": len(exported),
        "streams": streams,
    }


def _parser():
    parser = argparse.ArgumentParser(
        prog="python -m zils.current", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--dir", default=settings.get("ZILS_CURRENT_DIR") or str(DEFAULT_DIR))
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="add new events to the waiting room")
    collect.add_argument("--source", choices=("gharchive", "wikimedia"), required=True)
    collect.add_argument("--hour", action="append", default=[], help="GH Archive hour YYYY-MM-DD-H")
    collect.add_argument("--start", help="first GH Archive hour YYYY-MM-DD-H")
    collect.add_argument("--hours", type=int, default=1, help="consecutive GH Archive hours")
    collect.add_argument("--minutes", type=float, default=5, help="Wikimedia stream minutes")
    collect.add_argument("--since", help="replay Wikimedia from this UTC time (retention is days)")
    collect.add_argument("--wiki", action="append", help="wiki database name (default enwiki)")
    collect.add_argument("--limit", type=int, help="stop after this many Wikimedia edits")
    resolve_ = commands.add_parser("resolve", help="label waiting items whose outcome is known")
    resolve_.add_argument(
        "--max-age-days",
        type=float,
        default=float(settings.get("ZILS_CURRENT_MAX_AGE_DAYS") or MAX_AGE_DAYS),
    )
    resolve_.add_argument(
        "--wiki-wait-hours", type=float, help=f"default {current_wikimedia.WAIT_HOURS}"
    )
    export_ = commands.add_parser("export", help="write bank-format cases")
    export_.add_argument("--out", required=True)
    export_.add_argument("--stats", action="store_true", help="print per-stream statistics as JSON")
    export_.add_argument("--seed", type=int, default=0)
    export_.add_argument("--no-balance", action="store_true", help="keep every label")
    export_.add_argument(
        "--max-tokens", type=int, default=4096, help="model input cap (mini: 8192)"
    )
    return parser


def main(argv=None, http=None, now=None):
    args = _parser().parse_args(argv)
    store = Store(args.dir)
    now = now or datetime.now(UTC)
    if args.command == "export":
        stats = export(
            store, args.out, seed=args.seed, balance=not args.no_balance, max_tokens=args.max_tokens
        )
        print(json.dumps(stats, indent=1) if args.stats else f"exported {stats['exported']} cases")
        return 0
    http = http or Http()
    if args.command == "resolve":
        print(json.dumps(resolve(store, http, now, args.max_age_days, args.wiki_wait_hours)))
        return 0
    if args.source == "gharchive":
        hours = list(args.hour)
        if args.start:
            hours += current_github.hour_range(args.start, args.hours)
        if not hours:
            raise SystemExit("give --hour or --start")
        added = current_github.collect(store, http, hours, now=now)
    else:
        added = current_wikimedia.collect(
            store,
            http,
            args.minutes,
            wikis=args.wiki,
            since=args.since,
            now=now,
            limit=args.limit,
            clock=time.monotonic,
        )
    print(json.dumps({"source": args.source, "added": added}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
