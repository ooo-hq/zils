"""Zils Current exports: leak filters, overlap checks, the licence gate and dated bank folders.

    python -m zils.current export --bank DIR [--date 2026-10-30] --reference EXAM.jsonl ...

A dated export holds the cases resolved on one UTC day, in `DIR/<date>/items.jsonl`, with a
`manifest.json` that records counts, file hashes, licences, the window and every setting used.
"""

import hashlib
import json
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import base_exam, current_github, current_items
from .current_items import iso, parse_time

VERSION = "zils-current-bank/v1"
# Phrases that state a stream's outcome in its own state text; kept short and specific so that
# ordinary text ("depends on #4 being merged first", "merge sort") stays in.
OUTCOME_PHRASES = {
    "github-pr-merge": (
        r"closed without merg",
        r"\b(?:was|been|got|is now) merged\b",
        r"\bmerged (?:in|into|via|as)\b",
        r"\bsuperseded by\b",
        r"\bclos(?:ed|ing) in favou?r of\b",
        r"\bclosing this\b",
        r"\bwo(?:n'?t| not) (?:be )?merg",
        r"\bnot (?:be )?merged\b",
    ),
    "github-issue-close": (
        r"\bduplicate of\b",
        r"\bclos(?:ed|ing) as\b",
        r"\bclosing this\b",
        r"\bwo(?:n'?t| not) fix\b",
        r"\bwontfix\b",
        r"\b(?:fixed|resolved|addressed) (?:in|by) #\d+",
    ),
}
# Revert wording in the default wikis' languages; the edits themselves are not reverts (those are
# dropped), but a summary that mentions one gives the outcome away.
REVERT_PHRASES = (
    r"\brevert(?:ed|ing|s)?\b",
    r"\bundid\b",
    r"\bundo\b",
    r"\brvv?\b",
    r"\brollback\b",
    r"zurückgesetzt|rückgängig|revertiert",
    r"révoqu|révocation|\bannul(?:é|ation)",
    r"\brevertid|\bdeshech|\bdeshac|\brevertir",
    r"\bannullat|\bripristinat",
    r"\bdesfeit|\bdesfaz|\brevers(?:ão|ao)\b",
    r"откат|отмен",
    r"差し戻|取り消|リバート",
    r"回退|撤销|撤銷|还原|還原",
)
SE_CLOSED_PHRASES = (
    r"\bpossible duplicate\b",
    r"\bmarked as (?:a )?duplicate\b",
    r"\balready has answers? here\b",
    r"\[(?:closed|duplicate|on hold)\]",
    r"\bclosed as\b",
    r"\bvot(?:ed|ing) to close\b",
)
SE_ACCEPTED_PHRASES = (
    r"\baccepted answer\b",
    r"\bthanks? for accepting\b",
    r"\b(?:OP|asker) (?:has )?confirmed\b",
    r"\byou accepted\b",
)
for _stream in current_items.STREAMS:
    if _stream.startswith(("wikipedia-revert", "wikidata-revert")):
        OUTCOME_PHRASES[_stream] = REVERT_PHRASES
    elif _stream.startswith("stackexchange-closed-"):
        OUTCOME_PHRASES[_stream] = SE_CLOSED_PHRASES
    elif _stream.startswith("stackexchange-accepted-"):
        OUTCOME_PHRASES[_stream] = SE_ACCEPTED_PHRASES
OUTCOME = {s: re.compile("|".join(p), re.I) for s, p in OUTCOME_PHRASES.items()}
SHINGLE_WORDS = 16
SHINGLE_KEEP = 4  # keep one shingle in four, chosen by hash, so both sides keep the same ones
WORD = re.compile(r"\w+")
TEMPLATE_LABELS = {
    "Repository",
    "Branch",
    "Issue title",
    "Pull request title",
    "Wiki",
    "Article",
    "Item",
    "Edit summary",
    "Site",
    "Title",
    "Tags",
}
TEMPLATE_FIELDS = {"Size change", "Minor edit", "Editor"}
TEMPLATE_LINES = {"Removed:", "Added:", "(nothing)", "[truncated]", "Question:", "(empty)"}
TEMPLATE_LINES |= {f"Answer {chr(65 + i)}:" for i in range(26)}


def leak_reason(record, text):
    """Why a resolved item gives its answer away, or None."""
    data, stream = record["data"], record["stream"]
    if data.get("text_at") and record.get("outcome_at") and data["text_at"] > record["outcome_at"]:
        return "late-text"
    if stream == "github-issue-type" and current_github.issue_kind(
        data.get("labels_at_open") or []
    ):
        return "label-at-open"
    if stream in OUTCOME and OUTCOME[stream].search(text):
        return "states-outcome"
    return None


def _normal(text):
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def _state_text(state):
    if isinstance(state, dict) and isinstance(state.get("text"), str):
        return state["text"]
    return state if isinstance(state, str) else json.dumps(state, sort_keys=True)


def shingles(text):
    words = WORD.findall(_normal(text))
    kept = set()
    for i in range(len(words) - SHINGLE_WORDS + 1):
        digest = hashlib.blake2b(" ".join(words[i : i + SHINGLE_WORDS]).encode(), digest_size=8)
        value = int.from_bytes(digest.digest(), "big")
        if value % SHINGLE_KEEP == 0:
            kept.add(value)
    return kept


def passage_text(text):
    """State text without the fixed lines of Zils Current's own templates (labels, sizes, flags),
    so items never match each other, or an earlier export, on template wording alone."""
    kept = []
    for line in text.split("\n"):
        label, _, value = line.partition(": ")
        if line in TEMPLATE_LINES or label in TEMPLATE_FIELDS:
            continue
        kept.append(value if label in TEMPLATE_LABELS else line)
    return "\n".join(kept)


def _rows(path):
    if path.suffix == ".jsonl":
        for line in path.read_text().split("\n"):
            if line.strip():
                yield json.loads(line)
        return
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = next(
            (value[k] for k in ("cases", "items", "rows") if isinstance(value.get(k), list)), []
        )
    yield from value if isinstance(value, list) else []


class OverlapIndex:
    """Exam, bank, benchmark and training rows to keep out of exports.

    A case overlaps when its prompt fingerprint (base_exam.fingerprints), its normalized state
    text, or a shared passage of about fifty words or more matches a reference row. Rows that
    carry only `content_sha256` (rebuild-only datasets) are compared on that hash.
    """

    def __init__(self):
        self.prompts, self.hashes, self.states, self.passages = set(), set(), set(), set()
        self.sources = []

    def add(self, path):
        path = Path(path)
        files = (
            sorted(p for p in path.rglob("*") if p.suffix in (".jsonl", ".json"))
            if path.is_dir()
            else [path]
        )
        for file in files:
            summary = {
                "name": file.name,
                "sha256": _sha256(file),
                "rows": 0,
                "text_rows": 0,
                "hash_only_rows": 0,
            }
            for row in _rows(file):
                if not isinstance(row, dict):
                    continue
                summary["rows"] += 1
                if isinstance(row.get("content_sha256"), str):
                    self.hashes.add(row["content_sha256"])
                if "state" not in row:
                    summary["hash_only_rows"] += "content_sha256" in row
                    continue
                summary["text_rows"] += 1
                text = _state_text(row["state"])
                self.states.add(_normal(text))
                self.passages |= shingles(passage_text(text))
                questions = [row.get("question"), *((row.get("questions") or {}).values())]
                for question in questions:
                    if isinstance(question, dict) and "type" in question:
                        try:
                            self.prompts |= set(
                                base_exam.fingerprints(
                                    {"state": row["state"], "question": question}
                                )
                            )
                        except (KeyError, TypeError, AttributeError):
                            pass
            self.sources.append(summary)

    def hit(self, case):
        """The first kind of overlap found ("prompt", "hash", "state", "passage"), or None."""
        if set(base_exam.fingerprints(case)) & self.prompts:
            return "prompt"
        if case["content_sha256"] in self.hashes:
            return "hash"
        text = case["state"]["text"]
        if _normal(text) in self.states:
            return "state"
        if shingles(passage_text(text)) & self.passages:
            return "passage"
        return None


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


def prepare(
    store,
    *,
    records=None,
    streams=None,
    overlap=None,
    leak_filters=True,
    seed=0,
    balance=True,
    max_tokens=4096,
):
    """Cases and per-stream statistics: build, filter leaks, dedupe, drop overlaps, balance."""
    records = store.resolved() if records is None else records
    groups = defaultdict(list)
    for record in records:
        groups[record["stream"]].append(record)
    dropped = defaultdict(Counter)
    for record in store.dropped():
        dropped[record["stream"]][record["reason"]] += 1
    pending = Counter(r["stream"] for r in store.pending())
    names = (
        sorted(set(groups) | set(dropped) | set(pending)) if streams is None else sorted(streams)
    )
    exported, stats, hits = [], {}, []
    for stream in names:
        rows = sorted(groups.get(stream, []), key=lambda r: (r["asked_at"], r["id"]))
        cases, seen, filtered, overlaps = [], set(), Counter(), 0
        for record in rows:
            case = current_items.build(record, max_tokens)
            reason = leak_reason(record, case["state"]["text"]) if leak_filters else None
            if reason:
                filtered[reason] += 1
                continue
            if case["content_sha256"] in seen:
                filtered["duplicate"] += 1
                continue
            seen.add(case["content_sha256"])
            kind = overlap.hit(case) if overlap else None
            if kind:
                overlaps += 1
                hits.append({"id": case["id"], "kind": kind})
                continue
            cases.append(case)
        outcomes = {r["id"]: r["label"] for r in rows}
        chosen = _balance(cases, outcomes, seed, stream) if balance else cases
        spec = current_items.STREAMS[stream]
        stats[stream] = {
            "licence": spec["licence"],
            "publishable": spec["publishable"],
            "pending": pending.get(stream, 0),
            "resolved": len(rows),
            "dropped": dict(sorted(dropped[stream].items())),
            "filtered": dict(sorted((k, v) for k, v in filtered.items() if k != "duplicate")),
            "duplicates": filtered["duplicate"],
            "overlap": overlaps,
            "labels": dict(sorted(Counter(outcomes[c["id"]] for c in cases).items())),
            "exported": len(chosen),
            "exported_labels": dict(sorted(Counter(outcomes[c["id"]] for c in chosen).items())),
            # resolved_at is when this tool saw the outcome; outcome_at is when it happened.
            "median_hours_to_resolution": _median_hours(rows, "resolved_at"),
            "median_hours_to_outcome": _median_hours(
                [r for r in rows if r.get("outcome_at")], "outcome_at"
            ),
        }
        exported += chosen
    return exported, stats, hits


def publishable_streams(streams=None):
    """The requested streams, refusing any whose licence does not allow republishing."""
    allowed = {s for s, spec in current_items.STREAMS.items() if spec["publishable"]}
    if streams is None:
        return sorted(allowed)
    refused = sorted(set(streams) - allowed)
    if refused:
        raise ValueError(f"{', '.join(refused)}: local-only stream, never in a publishable export")
    return sorted(streams)


def _write(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def export_bank(
    store,
    bank,
    date,
    *,
    references=(),
    check_overlap=True,
    publishable=False,
    streams=None,
    leak_filters=True,
    seed=0,
    balance=True,
    max_tokens=4096,
    force=False,
):
    """Write DIR/<date>/items.jsonl and manifest.json for items resolved on that UTC day.

    An existing day is returned unchanged unless `force` is set, so reruns are idempotent.
    """
    folder = Path(bank) / date
    manifest_path = folder / "manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())
    start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
    end = start + timedelta(days=1)
    if publishable:
        streams = publishable_streams(streams)
    if check_overlap and not references:
        raise ValueError(
            "give exam/bank/benchmark reference sets, or skip the overlap check explicitly"
        )
    overlap = OverlapIndex()
    for path in references if check_overlap else ():
        overlap.add(path)
    records = [r for r in store.resolved() if start <= parse_time(r["resolved_at"]) < end]
    if publishable and any(
        not r["source"].get("url") for r in records if r["stream"] in set(streams)
    ):
        raise ValueError("publishable cases need an attribution URL in source.url")
    cases, stats, hits = prepare(
        store,
        records=records,
        streams=streams,
        overlap=overlap if check_overlap else None,
        leak_filters=leak_filters,
        seed=seed,
        balance=balance,
        max_tokens=max_tokens,
    )
    if publishable:
        local = {c["family"] for c in cases} - set(publishable_streams())
        if local:
            raise ValueError(f"{sorted(local)}: local-only stream, never in a publishable export")
        if any(not c["source"].get("url") for c in cases):
            raise ValueError("publishable cases need an attribution URL in source.url")
    folder.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in cases)
    _write(folder / "items.jsonl", text)
    asked = [c["asked_at"] for c in cases]
    manifest = {
        "version": VERSION,
        "collector": current_items.COLLECTOR,
        "date": date,
        "mode": "publishable" if publishable else "local",
        "window": {
            "resolved_from": iso(start),
            "resolved_until": iso(end),
            "asked_from": min(asked) if asked else None,
            "asked_until": max(asked) if asked else None,
        },
        "settings": {
            "streams": sorted(stats),
            "leak_filters": leak_filters,
            "outcome_phrases": {s: list(p) for s, p in OUTCOME_PHRASES.items()}
            if leak_filters
            else {},
            "seed": seed,
            "balance": balance,
            "max_tokens": max_tokens,
        },
        "counts": {s: v["exported_labels"] for s, v in stats.items() if v["exported"]},
        "streams": stats,
        "overlap": {"checked": check_overlap, "references": overlap.sources, "hits": hits},
        "files": {
            "items.jsonl": {
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "bytes": len(text.encode()),
                "rows": len(cases),
            }
        },
    }
    _write(manifest_path, json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest
