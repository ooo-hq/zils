"""Zils Current cases from resolved events: scrubbed state text, bank-format questions."""

import hashlib
import json
import random
import re
import unicodedata
from datetime import UTC, datetime

# d3 counts state and question together against the profile cap (4,096 for lite and nano, 8,192
# for mini). Without a tokenizer dependency, estimate tokens conservatively: wiki tables and code
# measured 1.6 ASCII characters per token on the d3 tokenizer (31 Oct 2026), CJK text about one
# per character. Reserve room for the question and the runtime's prompt template.
CHARS_PER_TOKEN = 1.5
QUESTION_TOKENS = 384
TRUNCATED = "\n[truncated]"


class RateLimited(Exception):
    """A source asked us to stop; unfinished items stay pending for the next pass."""


def iso(moment):
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


GITHUB_TERMS = "github-user-content: source ids only; text is rebuilt from GitHub, not republished"
WIKIMEDIA_TERMS = "CC-BY-SA-4.0: attribute via source.url; share-alike applies to published text"
# Categories follow base_metrics.CATEGORY_WEIGHTS; provisional until the bank integration reviews them.
STREAMS = {
    "github-issue-type": {
        "category": "language",
        "licence": GITHUB_TERMS,
        "question": {
            "type": "choice",
            "instructions": "Maintainers later labelled this newly opened GitHub issue. Which kind is it?",
            "criteria": {
                "bug": "Bug report: something is broken or behaves incorrectly",
                "feature": "Feature request or enhancement: new or improved behaviour",
            },
        },
    },
    "github-issue-close": {
        "category": "reasoning",
        "licence": GITHUB_TERMS,
        "question": {
            "type": "noul",
            "instructions": "Will the maintainers close this newly opened GitHub issue as completed, "
            "rather than as not planned or a duplicate?",
            "criteria": {
                "true": "Closed as completed",
                "false": "Closed as not planned or as a duplicate",
            },
        },
    },
    "github-pr-merge": {
        "category": "reasoning",
        "licence": GITHUB_TERMS,
        "question": {
            "type": "noul",
            "instructions": "Will this newly opened GitHub pull request be merged?",
            "criteria": {"true": "Merged", "false": "Closed without merging"},
        },
    },
    "wikipedia-revert": {
        "category": "language",
        "licence": WIKIMEDIA_TERMS,
        "question": {
            "type": "noul",
            "instructions": "Will this Wikipedia article edit be reverted by another edit?",
            "criteria": {"true": "Reverted", "false": "Not reverted"},
        },
    },
}
WIKIS = {"enwiki": "English Wikipedia"}

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.]*\d)")
IPV6 = re.compile(
    r"(?<![\w:])(?:[0-9a-f]{1,4}:){3,7}[0-9a-f]{1,4}(?![\w:])|\b[0-9a-f]{1,4}::[0-9a-f:]+", re.I
)
# @name, but not decorators (@pytest.fixture), scoped packages (@types/node) or e-mail remnants.
MENTION = re.compile(r"(?<![\w@])@[A-Za-z0-9][A-Za-z0-9-]{0,38}(?![A-Za-z0-9/-])(?!\.[A-Za-z0-9])")
WIKI_USER_LINK = re.compile(
    r"\[\[\s*(User(?: talk)?|Special:Contributions)\s*[:/]\s*[^\]|\[]+", re.I
)
WIKI_USER = re.compile(
    r"\b(User(?: talk)?|Special:Contributions)([:/])(?!\[user\])[^\s\]|\[{}<>,;]+", re.I
)


def scrub(text, names=()):
    """Replace e-mail addresses, IP addresses, @mentions, wiki user links and known account names."""
    text = EMAIL.sub("[email]", text or "")
    text = IPV6.sub("[ip]", IPV4.sub("[ip]", text))
    text = MENTION.sub("@[user]", text)
    text = WIKI_USER_LINK.sub(lambda m: f"[[{m.group(1)}:[user]", text)
    text = WIKI_USER.sub(lambda m: f"{m.group(1)}{m.group(2)}[user]", text)
    for name in sorted({n for n in names if n and len(n) >= 3}, key=len, reverse=True):
        variants = {name, name.replace(" ", "_"), name.replace("_", " ")}
        for variant in variants:
            pattern = r"(?<![\w-])" + re.escape(variant) + r"(?![\w-])"
            text = re.sub(pattern, "[user]", text, flags=re.I)
    return text


def state_chars(max_tokens):
    return int((max_tokens - QUESTION_TOKENS) * CHARS_PER_TOKEN)


def estimated_tokens(text):
    ascii_chars = sum(ch < "\x80" for ch in text)
    return ascii_chars / CHARS_PER_TOKEN + (len(text) - ascii_chars)


def fit(text, max_tokens):
    """Truncate text so its estimated tokens leave room for the question within max_tokens."""
    budget, used = max_tokens - QUESTION_TOKENS, 0.0
    if estimated_tokens(text) <= budget:
        return text
    room = budget - estimated_tokens(TRUNCATED)
    for index, ch in enumerate(text):
        used += 1 / CHARS_PER_TOKEN if ch < "\x80" else 1
        if used > room:
            return truncate(text, index + len(TRUNCATED))
    return text


def truncate(text, limit):
    """Cut at a paragraph, line or word boundary near the limit and mark the cut."""
    if len(text) <= limit:
        return text
    room = max(0, limit - len(TRUNCATED))
    cut = text[:room]
    for boundary in ("\n\n", "\n", " "):
        at = cut.rfind(boundary)
        if at >= room * 0.8:
            cut = cut[:at]
            break
    return cut.rstrip() + TRUNCATED


def _normal(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def _github_state(record):
    data = record["data"]
    if record["stream"] == "github-pr-merge":
        head = f"Repository: {data['repo']}\nBranch: {data['head']} -> {data['base']}\n"
        return head + f"Pull request title: {data['title']}\n", data.get("body") or ""
    return f"Repository: {data['repo']}\nIssue title: {data['title']}\n", data.get("body") or ""


def _wiki_state(record, limit):
    data = record["data"]
    change = data["length_new"] - data["length_old"]
    head = (
        f"Wiki: {WIKIS.get(data['wiki'], data['wiki'])}\nArticle: {data['title']}\n"
        f"Edit summary: {data['comment'] or '(none)'}\n"
        f"Size change: {change:+d} bytes ({data['length_old']} -> {data['length_new']})\n"
        f"Minor edit: {'yes' if data['minor'] else 'no'}\nEditor: {data['editor']}\n"
    )
    removed = "\n".join("- " + line for line in data["removed"]) or "(nothing)"
    added = "\n".join("+ " + line for line in data["added"]) or "(nothing)"
    # Removed and added text share the room left after the header; unused room passes to the other.
    room = max(0, limit - len(head) - 24)
    removed = truncate(removed, max(room // 2, room - len(added)))
    added = truncate(added, room - len(removed))
    return truncate(f"{head}\nRemoved:\n{removed}\n\nAdded:\n{added}", limit)


def state_text(record, max_tokens=4096):
    limit = state_chars(max_tokens)
    if record["stream"] == "wikipedia-revert":
        return fit(_wiki_state(record, limit), max_tokens)
    head, body = _github_state(record)
    return fit(truncate(head + ("\n" + body if body.strip() else ""), limit), max_tokens)


def content_sha256(stream, text):
    payload = json.dumps({"stream": stream, "state": _normal(text)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _question(record):
    spec = STREAMS[record["stream"]]
    question = json.loads(json.dumps(spec["question"]))
    label = record["label"]
    if question["type"] == "choice":
        # As in authored exams: shuffle (order carries no signal), then rename keys A, B, ...
        pairs = list(question["criteria"].items())
        random.Random(record["id"]).shuffle(pairs)
        letters = [chr(65 + i) for i in range(len(pairs))]
        label = letters[[k for k, _ in pairs].index(label)]
        question["criteria"] = {
            letter: text for letter, (_, text) in zip(letters, pairs, strict=True)
        }
    return question, label


def build(record, max_tokens=4096):
    """One bank-format case (id, family, category, block, source_id, state, question, label)."""
    spec = STREAMS[record["stream"]]
    text = state_text(record, max_tokens)
    question, label = _question(record)
    case = {
        "id": record["id"],
        "family": record["stream"],
        "category": spec["category"],
        "block": "new",
        "source_id": record["id"],
        "state": {"text": text},
        "question": question,
        "label": label,
        "source": record["source"],
        "asked_at": record["asked_at"],
        "resolved_at": record["resolved_at"],
        "licence": spec["licence"],
        "content_sha256": content_sha256(record["stream"], text),
    }
    if "outcome_at" in record:
        case["outcome_at"] = record["outcome_at"]
    return case
