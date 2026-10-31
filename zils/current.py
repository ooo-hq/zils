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
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import ROOT, current_bank, current_github, current_items, current_wikimedia, settings

VERSION = current_items.COLLECTOR
DEFAULT_DIR = ROOT / "runs" / "current"
MAX_AGE_DAYS = 14
WIKI_MINUTES = 10  # stream minutes per hour: about 3,000 balanced edits a day over the wiki list
NEGATIVES_PER_POSITIVE = 3
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

    def post_json(self, url, payload, headers=None):
        response = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
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
            # Bytes, not decode_unicode: str.splitlines would also split on U+2028 inside JSON.
            for line in response.iter_lines():
                yield line.decode("utf-8", "replace")


def _upgrade(record):
    """Records the pilot release pooled into `wikipedia-revert` get their own wiki's stream."""
    if record.get("stream") == "wikipedia-revert":
        wiki = (record.get("data") or {}).get("wiki", "enwiki")
        return {**record, "stream": current_items.wiki_stream(wiki)}
    return record


class Store:
    """Waiting room (pending), labelled items (resolved) and dropped items with a reason."""

    def __init__(self, root):
        self.root = Path(root)

    def _read(self, name):
        path = self.root / name
        if not path.exists():
            return []
        return [_upgrade(json.loads(line)) for line in path.read_text().split("\n") if line]

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


RESOLVERS = {
    "github-": current_github.resolve,
    "wikipedia-revert": current_wikimedia.resolve,
    "wikidata-": current_wikimedia.resolve,
}
WIKIMEDIA = ("wikipedia-revert", "wikidata-")


def resolve(
    store,
    http,
    now,
    max_age_days=MAX_AGE_DAYS,
    wiki_wait_hours=None,
    negatives_per_positive=None,
):
    """One resolution pass over the waiting room; returns counts of what happened."""
    pending = store.pending()
    outcomes, limited = {}, False
    for prefix, resolver in RESOLVERS.items():
        records = [r for r in pending if r["stream"].startswith(prefix)]
        options = {}
        if prefix in WIKIMEDIA:
            options["negatives_per_positive"] = negatives_per_positive
            if wiki_wait_hours:
                options["wait_hours"] = wiki_wait_hours
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


def export(store, out, *, seed=0, balance=True, max_tokens=4096, leak_filters=True):
    """Write every resolved item as cases (leak-filtered, deduplicated, by default balanced)."""
    cases, streams, _ = current_bank.prepare(
        store, leak_filters=leak_filters, seed=seed, balance=balance, max_tokens=max_tokens
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    return {
        "version": VERSION,
        "seed": seed,
        "balanced": balance,
        "leak_filters": leak_filters,
        "max_tokens": max_tokens,
        "exported": len(cases),
        "streams": streams,
    }


def _references(values):
    if values:
        return values
    joined = settings.get("ZILS_CURRENT_REFERENCES") or ""
    return [p for p in joined.split(os.pathsep) if p]


def _export_options(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-balance", action="store_true", help="keep every label")
    parser.add_argument("--max-tokens", type=int, default=4096, help="model input cap (mini: 8192)")
    parser.add_argument(
        "--no-leak-filters", action="store_true", help="keep items whose text gives the answer away"
    )


def _sampling_option(parser, default=None):
    parser.add_argument(
        "--negatives-per-positive",
        type=float,
        default=default,
        help="not-reverted edits kept per reverted edit before diffs are fetched "
        f"(default {default or 'all'})",
    )


def _wikis(values=None):
    """Wikis to read: --wiki, else ZILS_CURRENT_WIKIS (comma-separated), else the default list."""
    if values:
        return values
    joined = settings.get("ZILS_CURRENT_WIKIS") or ""
    return [w.strip() for w in joined.split(",") if w.strip()] or list(
        current_wikimedia.DEFAULT_WIKIS
    )


def _overlap_options(parser):
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help="exam/bank/benchmark/training set to keep out (file or folder; repeatable; "
        "or ZILS_CURRENT_REFERENCES)",
    )
    parser.add_argument(
        "--no-overlap-check", action="store_true", help="export without reference sets"
    )


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
    collect.add_argument("--wiki", action="append", help="wiki database name (default: _wikis())")
    collect.add_argument("--limit", type=int, help="stop after this many Wikimedia edits")
    resolve_ = commands.add_parser("resolve", help="label waiting items whose outcome is known")
    max_age = float(settings.get("ZILS_CURRENT_MAX_AGE_DAYS") or MAX_AGE_DAYS)
    resolve_.add_argument("--max-age-days", type=float, default=max_age)
    resolve_.add_argument(
        "--wiki-wait-hours", type=float, help=f"default {current_wikimedia.WAIT_HOURS}"
    )
    _sampling_option(resolve_)
    export_ = commands.add_parser("export", help="write bank-format cases")
    target = export_.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", help="one file with every resolved item")
    target.add_argument(
        "--bank", help="dated folder DIR/<date>/ with items.jsonl and manifest.json"
    )
    export_.add_argument("--date", help="UTC day of resolution for --bank (default yesterday)")
    export_.add_argument("--stats", action="store_true", help="print per-stream statistics as JSON")
    export_.add_argument(
        "--publishable", action="store_true", help="only streams whose licence allows republishing"
    )
    export_.add_argument("--stream", action="append", help="limit --bank to these streams")
    export_.add_argument("--force", action="store_true", help="rewrite an existing day")
    _export_options(export_)
    _overlap_options(export_)
    nightly = commands.add_parser("nightly", help="collect, resolve and export, from cursors")
    nightly.add_argument("--state", required=True, help="waiting room, cursors and run log")
    nightly.add_argument("--bank", required=True, help="dated exports")
    nightly.add_argument("--gh-first-hours", type=int, default=24)
    nightly.add_argument("--gh-max-hours", type=int, default=48)
    nightly.add_argument("--wiki-first-hours", type=int, default=24)
    nightly.add_argument("--wiki-max-hours", type=int, default=48)
    nightly.add_argument(
        "--wiki-minutes",
        type=float,
        default=float(settings.get("ZILS_CURRENT_WIKI_MINUTES") or WIKI_MINUTES),
        help=f"stream minutes sampled per hour (default {WIKI_MINUTES:g})",
    )
    nightly.add_argument("--wiki", action="append", help="wiki database name (default: all)")
    _sampling_option(nightly, NEGATIVES_PER_POSITIVE)
    nightly.add_argument("--max-age-days", type=float, default=max_age)
    nightly.add_argument("--publishable", action="store_true")
    _export_options(nightly)
    _overlap_options(nightly)
    grade_ = commands.add_parser("grade", help="score a dated export with a checkpoint")
    grade_.add_argument("--items", required=True, help="DIR/<date>/items.jsonl")
    grade_.add_argument("--model", required=True, help="d3-family checkpoint folder")
    grade_.add_argument("--key", help="prediction cache key (default: the folder name)")
    grade_.add_argument("--cache", help="prediction cache (default: next to the items)")
    grade_.add_argument("--out", help="results JSON (default: grades-<key>.json next to the items)")
    grade_.add_argument("--lock", default=None, help="shared GPU lock (ZILS_COMPUTE_LOCK)")
    grade_.add_argument("--lock-wait", type=float, default=600, help="seconds to wait for the GPU")
    grade_.add_argument("--device", default="cuda")
    return parser


def _grade(args):
    from . import current_grade, soup_grade

    items = Path(args.items)
    key = args.key or Path(args.model).name
    result = current_grade.grade(
        items,
        soup_grade.production_runner(args.model, args.device),
        key=key,
        cache=args.cache or items.parent / "predictions",
        lock=args.lock or current_grade.lock_path(),
        wait_seconds=args.lock_wait,
    )
    out = Path(args.out) if args.out else items.parent / f"grades-{key}.json"
    if result["status"] == "busy":
        print(json.dumps(result))
        return current_grade.BUSY
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"status": "graded", "out": str(out), "all": result["all"]}))
    return 0


def _export_settings(args):
    return {
        "seed": args.seed,
        "balance": not args.no_balance,
        "max_tokens": args.max_tokens,
        "leak_filters": not args.no_leak_filters,
    }


def main(argv=None, http=None, now=None):
    args = _parser().parse_args(argv)
    now = now or datetime.now(UTC)
    if args.command == "grade":
        return _grade(args)
    if args.command == "nightly":
        from . import current_nightly

        result = current_nightly.run(
            args.state,
            args.bank,
            http or Http(),
            now,
            references=_references(args.reference),
            check_overlap=not args.no_overlap_check,
            gh_first_hours=args.gh_first_hours,
            gh_max_hours=args.gh_max_hours,
            wiki_first_hours=args.wiki_first_hours,
            wiki_max_hours=args.wiki_max_hours,
            wiki_minutes=args.wiki_minutes,
            wikis=_wikis(args.wiki),
            negatives_per_positive=args.negatives_per_positive,
            max_age_days=args.max_age_days,
            export_options={**_export_settings(args), "publishable": args.publishable},
        )
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "errors"}
                | {"errors": [e["step"] + ": " + e["error"] for e in result.get("errors", [])]}
            )
        )
        return 1 if result.get("errors") else 0
    store = Store(args.dir)
    if args.command == "export":
        options = _export_settings(args)
        if args.out:
            stats = export(store, args.out, **options)
            print(
                json.dumps(stats, indent=1) if args.stats else f"exported {stats['exported']} cases"
            )
            return 0
        date = args.date or (now - timedelta(days=1)).strftime("%Y-%m-%d")
        manifest = current_bank.export_bank(
            store,
            args.bank,
            date,
            references=_references(args.reference),
            check_overlap=not args.no_overlap_check,
            publishable=args.publishable,
            streams=args.stream,
            force=args.force,
            **options,
        )
        summary = {
            "date": date,
            "mode": manifest["mode"],
            "rows": manifest["files"]["items.jsonl"]["rows"],
            "counts": manifest["counts"],
            "overlap_hits": len(manifest["overlap"]["hits"]),
        }
        print(json.dumps(manifest if args.stats else summary, indent=1 if args.stats else None))
        return 0
    http = http or Http()
    if args.command == "resolve":
        counts = resolve(
            store,
            http,
            now,
            args.max_age_days,
            args.wiki_wait_hours,
            negatives_per_positive=args.negatives_per_positive,
        )
        print(json.dumps(counts))
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
            wikis=_wikis(args.wiki),
            since=args.since,
            now=now,
            limit=args.limit,
            clock=time.monotonic,
        )
    print(json.dumps({"source": args.source, "added": added}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
