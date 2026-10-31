"""Zils Current cases from resolved events: scrubbed state text, bank-format questions."""

import hashlib
import json
import random
import re
import unicodedata
from datetime import UTC, datetime

from . import settings

# d3 counts state and question together against the profile cap (4,096 for lite and nano, 8,192
# for mini). Without a tokenizer dependency, estimate tokens conservatively: wiki tables and code
# measured 1.6 ASCII characters per token on the d3 tokenizer (31 Oct 2026), CJK text about one
# per character. Reserve room for the question and the runtime's prompt template.
CHARS_PER_TOKEN = 1.5
QUESTION_TOKENS = 384
COLLECTOR = "zils-current/v1"
TRUNCATED = "\n[truncated]"


class RateLimited(Exception):
    """A source asked us to stop; unfinished items stay pending for the next pass."""


def iso(moment):
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


GITHUB_TERMS = "github-user-content: source ids only; text is rebuilt from GitHub, not republished"
WIKIMEDIA_TERMS = "CC-BY-SA-4.0: attribute via source.url; share-alike applies to published text"
WIKIDATA_TERMS = "CC0-1.0: Wikidata structured data; link source.url as a courtesy"
# Categories follow base_metrics.CATEGORY_WEIGHTS; provisional until the bank integration reviews them.
STREAMS = {
    "github-issue-type": {
        "category": "language",
        "licence": GITHUB_TERMS,
        "publishable": False,
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
        "publishable": False,
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
        "publishable": False,
        "question": {
            "type": "noul",
            "instructions": "Will this newly opened GitHub pull request be merged?",
            "criteria": {"true": "Merged", "false": "Closed without merging"},
        },
    },
    "wikipedia-revert": {
        "category": "language",
        "licence": WIKIMEDIA_TERMS,
        "publishable": True,
        "question": {
            "type": "noul",
            "instructions": "Will this Wikipedia article edit be reverted by another edit?",
            "criteria": {"true": "Reverted", "false": "Not reverted"},
        },
    },
}
# Wikipedias and Wikidata read by default: the largest projects by human article edits on
# EventStreams (31 Oct 2026). Each Wikipedia is its own stream so labels balance per language;
# English keeps the pilot's stream name.
WIKIS = {
    "enwiki": ("en", "English Wikipedia"),
    "dewiki": ("de", "German Wikipedia"),
    "frwiki": ("fr", "French Wikipedia"),
    "eswiki": ("es", "Spanish Wikipedia"),
    "jawiki": ("ja", "Japanese Wikipedia"),
    "itwiki": ("it", "Italian Wikipedia"),
    "ptwiki": ("pt", "Portuguese Wikipedia"),
    "ruwiki": ("ru", "Russian Wikipedia"),
    "zhwiki": ("zh", "Chinese Wikipedia"),
}
WIKIDATA = "wikidatawiki"
WIKI_NAMES = {db: name for db, (_, name) in WIKIS.items()} | {WIKIDATA: "Wikidata"}


def wiki_stream(wiki):
    """The stream of an edit on `wiki` (a database name such as dewiki)."""
    if wiki == WIKIDATA:
        return "wikidata-revert"
    if wiki == "enwiki":
        return "wikipedia-revert"
    lang = WIKIS[wiki][0] if wiki in WIKIS else wiki.removesuffix("wiki")
    return f"wikipedia-revert-{lang}"


def _revert_spec(what):
    return {
        "category": "retrieval" if what == "Wikidata item" else "language",
        "licence": WIKIDATA_TERMS if what == "Wikidata item" else WIKIMEDIA_TERMS,
        "publishable": True,
        "question": {
            "type": "noul",
            "instructions": f"Will this {what} edit be reverted by another edit?",
            "criteria": {"true": "Reverted", "false": "Not reverted"},
        },
    }


for _db in WIKIS:
    if _db != "enwiki":
        STREAMS[wiki_stream(_db)] = _revert_spec("Wikipedia article")
STREAMS["wikidata-revert"] = _revert_spec("Wikidata item")

# Stack Exchange: sites grouped by subject, one exam category per group. Each group has three
# streams (closed: yes/no, score: ordered levels, accepted: choice), so every category gets every
# question type.
SE_SITES = {
    "tools": ("stackoverflow", "superuser", "serverfault", "askubuntu", "unix", "dba"),
    "reasoning": ("math", "physics", "stats", "cs", "chemistry", "electronics"),
    "language": ("english", "ell", "german", "french", "spanish", "japanese", "chinese"),
    "retrieval": ("history", "travel", "law", "biology", "money", "academia", "skeptics"),
    "arts": ("movies", "scifi", "music", "literature", "photo", "graphicdesign", "writing"),
}
SE_NAMES = {
    "stackoverflow": "Stack Overflow",
    "superuser": "Super User",
    "serverfault": "Server Fault",
    "askubuntu": "Ask Ubuntu",
    "ell": "English Language Learners Stack Exchange",
}
# Question score a week or more after asking: (inclusive upper bound, description), ordered.
SCORE_LEVELS = (
    (-1, "Negative score (-1 or lower)"),
    (0, "Score of 0"),
    (2, "Score of 1 or 2"),
    (None, "Score of 3 or higher"),
)
SE_TERMS = "CC-BY-SA-4.0: attribute via source.url (the post and its authors); share-alike applies"
for _group in SE_SITES:
    _se = {"category": _group, "licence": SE_TERMS, "publishable": True}
    STREAMS[f"stackexchange-closed-{_group}"] = {
        **_se,
        "block": "known",
        "question": {
            "type": "noul",
            "instructions": "Will this newly asked Stack Exchange question be closed "
            "(as a duplicate, off-topic, unclear, too broad or opinion-based)?",
            "criteria": {"true": "Closed", "false": "Still open at least a week later"},
        },
    }
    STREAMS[f"stackexchange-score-{_group}"] = {
        **_se,
        "block": "new",
        "ordered_equally_spaced": True,
        "question": {
            "type": "score",
            "instructions": "What score (upvotes minus downvotes) will this newly asked Stack "
            "Exchange question have a week or more later?",
            "criteria": [f"Level {i}: {text}" for i, (_, text) in enumerate(SCORE_LEVELS)],
        },
    }
    STREAMS[f"stackexchange-accepted-{_group}"] = {
        **_se,
        "block": "new",
        "per_record_options": True,
        "question": {
            "type": "choice",
            "instructions": "The asker of this Stack Exchange question accepted one of these "
            "answers. Which one?",
            "criteria": {},
        },
    }
STREAMS["wikipedia-afd"] = {
    "category": "reasoning",
    "licence": WIKIMEDIA_TERMS,
    "publishable": True,
    "block": "known",
    "question": {
        "type": "choice",
        "instructions": "This English Wikipedia article was nominated for deletion. "
        "How did the deletion discussion close?",
        "criteria": {
            "keep": "Kept",
            "delete": "Deleted",
            "merge-redirect": "Merged into, or redirected to, another article",
            "no-consensus": "No consensus (the article stays by default)",
        },
    },
}
# Blocks: `known` for task types with an established public benchmark (Wikipedia vandalism,
# PAN-WVC; Wikidata vandalism, WSDM Cup 2017; Stack Overflow closed questions, Kaggle 2012;
# Articles for deletion outcomes), `new` for the rest. ZILS_CURRENT_BLOCKS overrides them:
# comma-separated `stream-prefix=block` pairs, longest prefix first.
for _stream, _spec in STREAMS.items():
    if _stream.startswith(("wikipedia-revert", "wikidata-revert")):
        _spec.setdefault("block", "known")
    _spec.setdefault("block", "new")
BLOCKS = ("known", "new")


def block_of(stream):
    overrides = {}
    for pair in (settings.get("ZILS_CURRENT_BLOCKS") or "").split(","):
        prefix, _, block = pair.partition("=")
        if prefix.strip() and block.strip() in BLOCKS:
            overrides[prefix.strip()] = block.strip()
    for prefix in sorted(overrides, key=len, reverse=True):
        if stream.startswith(prefix):
            return overrides[prefix]
    return STREAMS[stream]["block"]


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
    text = text or ""
    # Known names first, so a multi-word name inside a User: link is removed whole.
    for name in sorted({n for n in names if n and len(n) >= 3}, key=len, reverse=True):
        variants = {name, name.replace(" ", "_"), name.replace("_", " ")}
        for variant in variants:
            pattern = r"(?<![\w-])" + re.escape(variant) + r"(?![\w-])"
            text = re.sub(pattern, "[user]", text, flags=re.I)
    text = EMAIL.sub("[email]", text)
    text = IPV6.sub("[ip]", IPV4.sub("[ip]", text))
    text = MENTION.sub("@[user]", text)
    text = WIKI_USER_LINK.sub(lambda m: f"[[{m.group(1)}:[user]", text)
    return WIKI_USER.sub(lambda m: f"{m.group(1)}{m.group(2)}[user]", text)


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


WIKIBASE_SUMMARY = re.compile(r"/\*\s*wb([a-z]+)-([a-z-]+)(?::[^*]*)?\*/\s*")
WIKIBASE_ACTIONS = {
    "createclaim": "create claim",
    "setclaim": "set claim",
    "setclaimvalue": "set claim value",
    "removeclaims": "remove claims",
    "setlabel": "set label",
    "setdescription": "set description",
    "setaliases": "set aliases",
    "setlabeldescriptionaliases": "set label, description and aliases",
    "setsitelink": "set sitelink",
    "linktitles": "link titles",
    "editentity": "edit entity",
    "setreference": "set reference",
    "removereferences": "remove references",
    "setqualifier": "set qualifier",
    "removequalifiers": "remove qualifiers",
    "mergeitems": "merge items",
}


def wikibase_summary(comment):
    """A Wikibase autosummary in words: `/* wbsetclaim-create:2||1 */ x` -> `set claim (create): x`."""

    def words(match):
        action = WIKIBASE_ACTIONS.get(match.group(1), match.group(1))
        return f"{action} ({match.group(2).replace('-', ' ')}): "

    return WIKIBASE_SUMMARY.sub(words, comment or "").strip()


def _wiki_state(record, limit):
    data = record["data"]
    change = data["length_new"] - data["length_old"]
    if data["wiki"] == WIKIDATA:
        label = data.get("label")
        page = f"Item: {label} ({data['title']})" if label else f"Item: {data['title']}"
        summary = wikibase_summary(data["comment"])
    else:
        page, summary = f"Article: {data['title']}", data["comment"]
    head = (
        f"Wiki: {WIKI_NAMES.get(data['wiki'], data['wiki'])}\n{page}\n"
        f"Edit summary: {summary or '(none)'}\n"
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


def _se_state(record, limit):
    data = record["data"]
    site = data["site"]
    name = SE_NAMES.get(site, site.replace("-", " ").title() + " Stack Exchange")
    head = (
        f"Site: {name}\nTitle: {data['title']}\nTags: {', '.join(data['tags']) or '(none)'}\n"
        f"\nQuestion:\n"
    )
    answers = data.get("answers") or []
    if not answers:
        return truncate(head + (data["body"] or "(empty)"), limit)
    # The question takes up to a third of the room; the answers share the rest equally.
    room = max(0, limit - len(head))
    labels = [f"\n\nAnswer {chr(65 + i)}:\n" for i in range(len(answers))]
    question = truncate(data["body"] or "(empty)", max(room // 3, 200))
    each = max(0, (room - len(question) - sum(map(len, labels))) // len(answers))
    parts = [
        label + truncate(a or "(empty)", each) for label, a in zip(labels, answers, strict=True)
    ]
    return head + question + "".join(parts)


def state_text(record, max_tokens=4096):
    limit = state_chars(max_tokens)
    if record["stream"] == "wikipedia-afd":
        data = record["data"]
        head = f"Wiki: English Wikipedia\nArticle: {data['article']}\n\nDeletion discussion:\n"
        return fit(truncate(head + (data["text"] or "(empty)"), limit), max_tokens)
    if record["stream"].startswith("stackexchange-"):
        return fit(_se_state(record, limit), max_tokens)
    if record["stream"].startswith(("wikipedia-revert", "wikidata-revert")):
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
    if spec.get("per_record_options"):
        # Options were shuffled at collection and are named in the state text (Answer A, ...).
        count = len(record["data"]["answers"])
        question["criteria"] = {chr(65 + i): f"Answer {chr(65 + i)}" for i in range(count)}
        return question, label
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
        "block": block_of(record["stream"]),
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
