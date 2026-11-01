"""Zils Current proof and cross-checks: is every item a real event with a real outcome?

    python -m zils.current crosscheck --folder BANK/2026-10-31 [--sample 100] [--threshold 0.97]

Two gates run on a dated export, after it is written:

1. **Proof.** Every item needs its source link (`source.url`), its source ids (revision ids,
   question id, discussion page), `asked_at`, and `outcome_at` after `asked_at`. Items without
   them are removed.
2. **Cross-check.** For a seeded random sample per stream, the label is derived again by a
   different method than the one that produced it:

   - reverts (`mw-reverted` tag): the page history after the edit. A later revision, up to the
     time of the label, whose content hash equals the parent or an earlier revision (an identity
     revert), or a later undo, rollback or manual revert whose summary names the revision;
   - Articles for deletion (closing statement): whether the article exists now (delete: missing
     or a redirect; keep or no consensus: an article; merge or redirect: a redirect or an
     article);
   - Stack Exchange (first fetch): a second fetch of the questions, and for accepted answers the
     answers' own `is_accepted` flag, matched to our options by their text;
   - Articles for Creation (review edit summary or move log): a decline or rejection needs the
     review edit's own text to hold a declined or rejected submission box; an acceptance needs
     the article to exist now (or a deletion log entry after the review);
   - arXiv (API primary category): the paper's OAI-PMH record, whose first category is the
     primary;
   - Gerrit (the change's `status`): the change's detail read again, merged if it has a submit
     time, abandoned if its history holds an "Abandoned" message and no submit.

   A stream whose agreement is under the threshold is removed from the export.

The folder gets `crosscheck.json` (sample sizes, agreement, disagreements with links) and
`spotcheck.md` (20 random items per stream for a person to check), the unchecked items are kept
as `items.unchecked.jsonl`, and the manifest records the result.
"""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import current_stackexchange
from .current_items import iso, parse_time

THRESHOLD = 0.97
SAMPLE = 100
SPOT = 20
WINDOW = 60  # revisions read after the edit
REVERT_TAGS = {"mw-undo", "mw-rollback", "mw-manual-revert"}
ID_FIELDS = ("rev_id", "question_id", "number", "title", "arxiv_id")
REVIEWED_BOX = re.compile(r"\{\{\s*AfC submission\s*\|\s*[dr]\s*\|", re.I)


def proof_reason(case):
    """Why an item lacks proof of its event and outcome, or None."""
    source = case.get("source") or {}
    if not source.get("url"):
        return "no-url"
    if not any(source.get(k) for k in ID_FIELDS):
        return "no-source-id"
    if case.get("family", "").startswith(("wikipedia-revert", "wikidata-revert")) and not (
        source.get("rev_id") and source.get("old_rev_id")
    ):
        return "no-source-id"
    if not case.get("asked_at"):
        return "no-ask-time"
    if not case.get("outcome_at"):
        return "no-outcome-time"
    if parse_time(case["outcome_at"]) <= parse_time(case["asked_at"]):
        return "outcome-before-ask"
    return None


def _revisions(http, api, title, start, direction, limit):
    from .current_wikimedia import _api

    status, body = _api(
        http,
        api,
        {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvstartid": str(start),
            "rvdir": direction,
            "rvlimit": str(limit),
            "rvprop": "ids|sha1|tags|timestamp|comment",
        },
    )
    pages = (body.get("query") or {}).get("pages") or [] if status == 200 else []
    return [r for p in pages for r in p.get("revisions") or []]


def check_wiki(case, http):
    """Derive the revert label from the page history, without the `mw-reverted` tag."""
    source = case["source"]
    rev_id, cutoff = source["rev_id"], parse_time(case.get("resolved_at") or case["outcome_at"])
    title = source.get("item") or source["title"]
    before = _revisions(http, source["api"], title, source["old_rev_id"], "older", 16)
    after = _revisions(http, source["api"], title, rev_id, "newer", WINDOW)
    if not after or after[0].get("revid") != rev_id:
        return {"derived": None, "agree": False, "evidence": "revision not found in history"}
    earlier = {r.get("sha1") for r in before if r.get("sha1")}
    derived, evidence = "false", f"no revert of {rev_id} in {len(after) - 1} later revisions"
    for later in after[1:]:
        if parse_time(later["timestamp"]) > cutoff:
            break
        named = set(later.get("tags") or []) & REVERT_TAGS and re.search(
            rf"\b{rev_id}\b", later.get("comment") or ""
        )
        if later.get("sha1") in earlier or named:
            how = "restores an earlier revision" if later.get("sha1") in earlier else "names it"
            derived = "true"
            evidence = f"revision {later['revid']} at {later['timestamp']} {how}"
            break
    return {"derived": derived, "agree": derived == case["label"], "evidence": evidence}


def _article(case):
    match = re.search(r"^Article: (.+)$", case["state"]["text"], re.M)
    return match.group(1).strip() if match else None


def _afd_class(case):
    """The result class of an AfD case, whose label may be a shuffled letter."""
    from .current_items import STREAMS

    label, criteria = case["label"], (case.get("question") or {}).get("criteria") or {}
    if label in STREAMS["wikipedia-afd"]["question"]["criteria"]:
        return label
    names = {text: key for key, text in STREAMS["wikipedia-afd"]["question"]["criteria"].items()}
    return names[criteria[label]]


def _deleted_since(http, title, since):
    """Timestamp of a deletion of `title` after `since`, from the deletion log, or None."""
    from .current_afd import _get

    status, body = _get(
        http,
        {
            "action": "query",
            "list": "logevents",
            "letype": "delete",
            "letitle": title,
            "lelimit": "20",
        },
    )
    events = (body.get("query") or {}).get("logevents") or [] if status == 200 else []
    for event in events:
        if event.get("action") == "delete" and event.get("timestamp", "") >= since:
            return event["timestamp"]
    return None


def check_afd(cases, http):
    """Compare each closing result with whether the article exists today."""
    from .current_afd import API, _get

    titles = sorted({_article(c) for c in cases if _article(c)})
    pages = {}
    for start in range(0, len(titles), 50):
        status, body = _get(
            http,
            {"action": "query", "prop": "info", "titles": "|".join(titles[start : start + 50])},
        )
        for page in (body.get("query") or {}).get("pages") or [] if status == 200 else []:
            pages[page["title"]] = page
        for pair in (body.get("query") or {}).get("normalized") or []:
            if pair.get("to") in pages:
                pages[pair["from"]] = pages[pair["to"]]
    del API
    results = []
    for case in cases:
        page = pages.get(_article(case)) or {}
        state = (
            "missing"
            if page.get("missing") or not page
            else "redirect"
            if page.get("redirect")
            else "article"
        )
        allowed = {
            "delete": {"missing", "redirect"},
            "keep": {"article"},
            "no-consensus": {"article"},
            "merge-redirect": {"redirect", "article"},
        }[_afd_class(case)]
        evidence_class = _afd_class(case)
        note = ""
        if evidence_class == "delete" and state == "article" and case.get("asked_at"):
            # Deleted, then recreated: the deletion log still shows the deletion.
            deleted = _deleted_since(http, _article(case), case["asked_at"])
            if deleted:
                state, note = "missing", f" (recreated; deletion log {deleted})"
        derived = case["label"] if state in allowed else f"page is {state}"
        results.append(
            {
                "derived": derived,
                "agree": state in allowed,
                "evidence": f"result {evidence_class}; article is now: {state}{note}",
            }
        )
    return results


def check_afc(cases, http):
    """Declines: the review edit holds a declined box. Accepts: the article exists (or was deleted)."""
    from .current_afc import _content, _revisions

    declines = [c["source"]["review"]["rev_id"] for c in cases if c["label"] == "false"]
    edits = _revisions(http, declines, "ids|content")
    targets = sorted({c["source"]["review"]["target"] for c in cases if c["label"] == "true"})
    pages = {}
    from .current_afd import _get

    for start in range(0, len(targets), 50):
        status, body = _get(
            http,
            {"action": "query", "prop": "info", "titles": "|".join(targets[start : start + 50])},
        )
        for page in (body.get("query") or {}).get("pages") or [] if status == 200 else []:
            pages[page["title"]] = page
    results = []
    for case in cases:
        review = case["source"]["review"]
        if case["label"] == "false":
            edit = edits.get(review["rev_id"])
            boxed = bool(edit and REVIEWED_BOX.search(_content(edit)))
            derived = "false" if boxed else "no declined box"
            evidence = f"review edit {review['rev_id']}: " + (
                "declined/rejected box" if boxed else "no declined or rejected box"
            )
            results.append({"derived": derived, "agree": boxed, "evidence": evidence})
            continue
        page = pages.get(review["target"]) or {}
        exists = bool(page) and not page.get("missing") and page.get("ns", 0) == 0
        note = "article exists"
        if not exists:
            deleted = _deleted_since(http, review["target"], case["outcome_at"])
            exists, note = bool(deleted), f"deleted {deleted}" if deleted else "no article"
        results.append(
            {
                "derived": "true" if exists else "no article",
                "agree": exists,
                "evidence": f"{review['target']}: {note}",
            }
        )
    return results


def check_gerrit(cases, http):
    """Read each change again: a submit time means merged, an Abandoned message abandoned."""
    from .current_gerrit import Api, recheck

    api, results = Api(http), []
    for case in cases:
        source = case["source"]
        derived, evidence = recheck(api, source["host"], source["project"], source["number"])
        results.append(
            {"derived": derived, "agree": derived == case["label"], "evidence": evidence}
        )
    return results


def check_arxiv(cases, http):
    """Compare each label's category with the first category of the paper's OAI-PMH record."""
    from .current_arxiv import Api, primary_from_oai

    api, results = Api(http), []
    for case in cases:
        chosen = case["question"]["criteria"][case["label"]]
        labelled = re.search(r"\(([^()]+)\)$", chosen).group(1)
        primary = primary_from_oai(api, case["source"]["arxiv_id"])
        results.append(
            {
                "derived": primary,
                "agree": primary == labelled,
                "evidence": f"OAI-PMH primary {primary}; label {labelled}",
            }
        )
    return results


def _answers(text):
    parts = re.split(r"\n\nAnswer ([A-Z]):\n", text)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def _same(ours, theirs):
    ours = re.sub(r"\s+", " ", ours.replace("[truncated]", "")).strip()
    theirs = re.sub(r"\s+", " ", theirs).strip()
    return bool(ours) and (ours == theirs or theirs.startswith(ours[: max(40, len(ours) - 20)]))


def check_stackexchange(cases, http, key=None):
    """Second fetch of the questions (and of accepted items' answers), one site at a time."""
    api = current_stackexchange.Api(http, key=key)
    by_site = defaultdict(list)
    for case in cases:
        by_site[case["source"]["site"]].append(case)
    fresh, answers = {}, {}
    for site, rows in by_site.items():
        ids = sorted({c["source"]["question_id"] for c in rows})
        for start in range(0, len(ids), 100):
            chunk = ids[start : start + 100]
            body = api.get(
                f"/questions/{';'.join(map(str, chunk))}", {"site": site, "pagesize": 100}
            )
            for question in body["items"]:
                fresh[(site, question["question_id"])] = question
        wanted = sorted(
            {
                c["source"]["question_id"]
                for c in rows
                if c["family"].startswith("stackexchange-accepted-")
            }
        )
        if wanted:
            for answer in current_stackexchange._answers(api, site, wanted).values():
                for row in answer:
                    answers.setdefault((site, row["question_id"]), []).append(row)
    results = []
    for case in cases:
        key_ = (case["source"]["site"], case["source"]["question_id"])
        question = fresh.get(key_)
        if question is None:
            results.append({"derived": None, "agree": False, "evidence": "question not returned"})
            continue
        family = case["family"]
        if family.startswith("stackexchange-closed-"):
            derived = "true" if "closed_date" in question else "false"
            evidence = f"closed_date={question.get('closed_date')}"
        elif family.startswith("stackexchange-score-"):
            level = current_stackexchange.score_level(question["score"])
            # Votes keep arriving: a score within one vote of the label's range agrees.
            near = {current_stackexchange.score_level(question["score"] + d) for d in (-1, 0, 1)}
            derived = case["label"] if int(case["label"]) in near else str(level)
            evidence = f"score now {question['score']} (level {level})"
        else:
            options = _answers(case["state"]["text"])
            accepted = [a for a in answers.get(key_, []) if a.get("is_accepted")]
            derived = None
            for letter, text in options.items():
                if accepted and _same(text, current_stackexchange.text(accepted[0].get("body"))):
                    derived = letter
            evidence = (
                f"is_accepted answer {accepted[0]['answer_id']}"
                if accepted
                else "no accepted answer now"
            )
        results.append(
            {"derived": derived, "agree": derived == case["label"], "evidence": evidence}
        )
    return results


def _sample(cases, size, seed, stream):
    rows = sorted(cases, key=lambda c: c["id"])
    return random.Random(f"{seed}:{stream}").sample(rows, min(size, len(rows)))


def crosscheck(cases, http, *, sample=SAMPLE, seed=0, threshold=THRESHOLD):
    """Per-stream agreement of a sample with a second derivation of the label."""
    groups = defaultdict(list)
    for case in cases:
        groups[case["family"]].append(case)
    streams = {}
    # Stack Exchange samples of every stream are fetched together, a site at a time, to spend
    # as few of the day's 300 keyless requests as possible.
    se_chosen = {
        s: _sample(rows, sample, seed, s)
        for s, rows in groups.items()
        if s.startswith("stackexchange-")
    }
    flat = [c for rows in se_chosen.values() for c in rows]
    se_results = (
        dict(zip(map(id, flat), check_stackexchange(flat, http), strict=True)) if flat else {}
    )
    for stream, rows in sorted(groups.items()):
        chosen = se_chosen.get(stream) or _sample(rows, sample, seed, stream)
        if stream.startswith(("wikipedia-revert", "wikidata-revert")):
            results = [check_wiki(c, http) for c in chosen]
            method = "page history (identity revert or a revert naming the revision)"
        elif stream == "wikipedia-afd":
            results = check_afd(chosen, http)
            method = "article exists now"
        elif stream == "wikipedia-afc":
            results = check_afc(chosen, http)
            method = "review edit text or article exists now"
        elif stream == "gerrit-merge":
            results = check_gerrit(chosen, http)
            method = "change history (submitted, Abandoned)"
        elif stream.startswith("arxiv-"):
            results = check_arxiv(chosen, http)
            method = "OAI-PMH record categories"
        elif stream.startswith("stackexchange-"):
            results = [se_results[id(c)] for c in chosen]
            method = "second fetch of the question and its answers"
        else:
            streams[stream] = {"sample": 0, "agreement": None, "method": "none", "passed": False}
            continue
        agree = sum(r["agree"] for r in results)
        streams[stream] = {
            "items": len(rows),
            "sample": len(results),
            "agreement": round(agree / len(results), 4) if results else None,
            "method": method,
            "passed": bool(results) and agree / len(results) >= threshold,
            "disagreements": [
                {
                    "id": c["id"],
                    "url": c["source"].get("url"),
                    "label": c["label"],
                    "derived": r["derived"],
                    "evidence": r["evidence"],
                }
                for c, r in zip(chosen, results, strict=True)
                if not r["agree"]
            ][:25],
            "evidence": {c["id"]: r["evidence"] for c, r in zip(chosen, results, strict=True)},
        }
    return streams


def _excerpt(text, limit=600):
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def spotcheck(cases, streams, seed=0):
    """A Markdown sheet: 20 random items per stream with link, excerpt, label and evidence."""
    lines = ["# Zils Current spot check", ""]
    lines.append("Each item: the source link, the start of the input, our label, and the evidence.")
    groups = defaultdict(list)
    for case in cases:
        groups[case["family"]].append(case)
    for stream, rows in sorted(groups.items()):
        info = streams.get(stream, {})
        agreement = info.get("agreement")
        lines += ["", f"## {stream}", ""]
        lines.append(
            f"{len(rows)} items; cross-check {info.get('sample', 0)} sampled, agreement "
            f"{'n/a' if agreement is None else f'{agreement:.1%}'} ({info.get('method', 'none')})."
        )
        evidence = info.get("evidence") or {}
        for i, case in enumerate(_sample(rows, SPOT, f"spot:{seed}", stream), 1):
            question = case["question"]
            criteria = question.get("criteria")
            label = case["label"]
            if isinstance(criteria, dict):
                meaning = criteria.get(label, label)
            elif isinstance(criteria, list):
                meaning = criteria[int(label)]
            else:
                meaning = label
            lines += [
                "",
                f"### {i}. {case['source'].get('url')}",
                "",
                f"- Asked: {case.get('asked_at')}; outcome: {case.get('outcome_at')}",
                f"- Question: {question.get('instructions')}",
                f"- Our label: `{label}` ({meaning})",
                f"- Evidence: {evidence.get(case['id'], 'label source: ' + _label_source(stream))}",
                "",
                "```text",
                _excerpt(case["state"]["text"]).replace("```", "'''"),
                "```",
            ]
    return "\n".join(lines) + "\n"


def _label_source(stream):
    if stream.startswith(("wikipedia-revert", "wikidata-revert")):
        return "MediaWiki mw-reverted change tag"
    if stream == "wikipedia-afd":
        return "closing statement (The result was ...)"
    if stream.startswith("stackexchange-"):
        return "Stack Exchange API fields"
    if stream == "wikipedia-afc":
        return "AFCH review edit summary or move log"
    if stream.startswith("arxiv-"):
        return "arXiv API primary category"
    if stream == "gerrit-merge":
        return "Gerrit change status"
    return "source API"


def _sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def run(folder, http, *, sample=SAMPLE, seed=0, threshold=THRESHOLD, now=None):
    """Proof gate and cross-check gate on one export folder; rewrites items.jsonl in place."""
    folder = Path(folder)
    items = folder / "items.jsonl"
    original = folder / "items.unchecked.jsonl"
    if not original.exists():
        original.write_text(items.read_text())
    cases = [json.loads(line) for line in original.read_text().splitlines() if line.strip()]
    rejected, proven = Counter(), []
    for case in cases:
        reason = proof_reason(case)
        if reason:
            rejected[reason] += 1
        else:
            proven.append(case)
    streams = crosscheck(proven, http, sample=sample, seed=seed, threshold=threshold)
    excluded = sorted(s for s, v in streams.items() if not v["passed"])
    kept = [c for c in proven if c["family"] not in set(excluded)]
    text = "".join(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in kept)
    items.write_text(text)
    report = {
        "checked_at": iso(now) if now else None,
        "threshold": threshold,
        "sample_per_stream": sample,
        "seed": seed,
        "proof_rejected": dict(rejected),
        "excluded": excluded,
        "kept": dict(Counter(c["family"] for c in kept)),
        "streams": streams,
    }
    (folder / "crosscheck.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    (folder / "spotcheck.md").write_text(spotcheck(proven, streams, seed))
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("files", {})["items.jsonl"] = {
            "sha256": _sha256(text),
            "bytes": len(text.encode()),
            "rows": len(kept),
        }
        manifest["crosscheck"] = {
            "threshold": threshold,
            "excluded": excluded,
            "proof_rejected": dict(rejected),
            "agreement": {s: v["agreement"] for s, v in streams.items()},
        }
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return report


def paused(state):
    path = Path(state) / "paused.json"
    return sorted(json.loads(path.read_text())) if path.exists() else []


def update_paused(state, report, threshold=THRESHOLD):
    """Pause every stream whose cross-check fell under the threshold; pauses stay until a person
    removes them from `paused.json`."""
    current = set(paused(state))
    for stream, info in report["streams"].items():
        if (
            info.get("sample")
            and info.get("agreement") is not None
            and info["agreement"] < threshold
        ):
            current.add(stream)
    path = Path(state) / "paused.json"
    path.write_text(json.dumps(sorted(current), indent=1) + "\n")
    return sorted(current)
