"""Author private base-election exam draws with Claude, ready for `zils.base_exam seal`.

Each item is written new by Claude Opus 5.5 from one public row used only as a style example, then
kept only if an independent blind pass (same model, no label shown) gives the same answer. Seeds come
from held-out validation/test splits of datasets outside the Decision Index, S1MB and JevBench test
sets, and from BIG-bench training tasks for new domains. Known families match the public seeds
published at zils-ai/base-exam-seed-v1 and -v2; their receipts live in the family history.

Qualified for d3-mini on 22 Oct 2026: two consecutive sealed draws ranked five Decision Index
references at Spearman 0.9 on Choice and yes/no families (see docs/base-election.md).

    python -m scripts.author_exam author --work exams --draw r1 --per-family 72
    python -m scripts.author_exam verify --work exams --draw r1
    python -m scripts.author_exam finish --work exams --draw r1 --calibration 8
    python -m zils.base_exam seal ...   (see docs/base-election.md)

Needs `ANTHROPIC_API_KEY` and requirements/exam.txt. Rows used by any earlier draw under --work are
never reused. Keep the work directory private: it holds labels.
"""

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "claude-opus-5-5"
SEEDS_PER_REQUEST = 10
MAX_SEED_CHARS = 3000
SCORE_KEYS = ["0", "1", "2", "3", "4"]


def hf_parquet(repo, path):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo, path, repo_type="dataset")
    return [(f"{repo}:{path}:{i}", r) for i, r in enumerate(pq.read_table(p).to_pylist())]


def hf_jsonl(repo, path):
    from huggingface_hub import hf_hub_download

    p = Path(hf_hub_download(repo, path, repo_type="dataset"))
    lines = [line for line in p.read_text().splitlines() if line.strip()]
    return [(f"{repo}:{path}:{i}", json.loads(line)) for i, line in enumerate(lines)]


def glaive():
    from huggingface_hub import hf_hub_download

    path = "glaive-function-calling-v2.json"
    p = Path(hf_hub_download("glaiveai/glaive-function-calling-v2", path, repo_type="dataset"))
    rows = json.loads(p.read_text())
    return [(f"glaive:{i}", r) for i, r in enumerate(rows) if "functioncall" in r.get("chat", "")]


GLUE = "nyu-mll/glue"
SOURCES = {
    "csqa-val": lambda: hf_parquet("tau/commonsense_qa", "data/validation-00000-of-00001.parquet"),
    "svamp-test": lambda: hf_parquet("ChilleD/SVAMP", "data/test-00000-of-00001.parquet"),
    "mnli-val": lambda: hf_parquet(GLUE, "mnli/validation_mismatched-00000-of-00001.parquet"),
    "rte-val": lambda: hf_parquet(GLUE, "rte/validation-00000-of-00001.parquet"),
    "stsb-val": lambda: hf_parquet(GLUE, "stsb/validation-00000-of-00001.parquet"),
    "qnli-val": lambda: hf_parquet(GLUE, "qnli/validation-00000-of-00001.parquet"),
    "newsgroups-test": lambda: hf_jsonl("SetFit/20_newsgroups", "test.jsonl"),
    "tweet-emotion-test": lambda: hf_parquet(
        "cardiffnlp/tweet_eval", "emotion/test-00000-of-00001.parquet"
    ),
    "tweet-irony-test": lambda: hf_parquet(
        "cardiffnlp/tweet_eval", "irony/test-00000-of-00001.parquet"
    ),
    "glaive": glaive,
}
# New-domain pools: BIG-bench training tasks (Glaive rows for tools). A task is used by one draw only.
BIGBENCH = {
    "reasoning": [
        "logical_deduction", "date_understanding", "causal_judgment", "navigate",
        "temporal_sequences", "tracking_shuffled_objects", "penguins_in_a_table",
        "reasoning_about_colored_objects", "elementary_math_qa", "unit_conversion",
        "physical_intuition", "strategyqa", "geometric_shapes", "checkmate_in_one",
    ],
    "language": [
        "implicatures", "disambiguation_qa", "hyperbaton", "analytic_entailment",
        "epistemic_reasoning", "discourse_marker_prediction", "salient_translation_error_detection",
        "misconceptions", "logical_fallacy_detection", "presuppositions_as_nli",
        "entailed_polarity", "conceptual_combinations",
    ],
    "retrieval": [
        "fact_checker", "vitaminc_fact_verification", "auto_categorization", "question_selection",
        "evaluating_information_essentiality", "human_organs_senses", "intent_recognition",
        "movie_recommendation", "fantasy_reasoning", "hindu_knowledge", "cause_and_effect",
        "timedial", "social_iqa",
    ],
    "tools": ["code_line_description"],
    "arts": [
        "figure_of_speech_detection", "metaphor_boolean", "understanding_fables", "riddle_sense",
        "irony_identification", "snarks", "emoji_movie", "ruin_names",
        "emojis_emotion_prediction", "phrase_relatedness", "color", "kanji_ascii",
    ],
}  # fmt: skip
for _tasks in BIGBENCH.values():
    for _task in _tasks:
        SOURCES[f"bb-{_task}"] = lambda t=_task: hf_parquet(
            "tasksource/bigbench", f"{t}/train-00000-of-00001.parquet"
        )

# Known families: names match the published seed (zils-ai/base-exam-seed-v2) and the history receipts.
KNOWN = {
    "reasoning-known-choice-k2": ("reasoning", "choice", "csqa-val", "Commonsense multiple-choice reasoning with 5 plausible options (CommonsenseQA style)."),
    "reasoning-known-noul-k2": ("reasoning", "noul", "csqa-val", "Is a stated answer to a commonsense question correct? Requires ruling out close alternatives."),
    "reasoning-known-score-k2": ("reasoning", "score", "svamp-test", "Arithmetic word problems with distracting numbers; the answer is a small count."),
    "language-known-choice-k2": ("language", "choice", "mnli-val", "Premise/hypothesis inference across genres: entailment, neutral or contradiction."),
    "language-known-noul-k2": ("language", "noul", "rte-val", "Does the passage entail the hypothesis?"),
    "language-known-score-k2": ("language", "score", "stsb-val", "Rate the semantic similarity of two sentences."),
    "retrieval-known-choice-k2": ("retrieval", "choice", "newsgroups-test", "Which of several close topic categories does a post belong to (20 Newsgroups style)?"),
    "retrieval-known-noul-k2": ("retrieval", "noul", "qnli-val", "Does the sentence contain the answer to the question?"),
    "retrieval-known-score-k2": ("retrieval", "score", "qnli-val", "How completely does a passage answer a question (none to fully)?"),
    "tools-known-choice-k2": ("tools", "choice", "glaive", "Which function call (name and arguments) correctly serves the request, among close candidates?"),
    "tools-known-noul-k2": ("tools", "noul", "glaive", "Should the assistant call a function now, given the conversation and schemas?"),
    "tools-known-score-k2": ("tools", "score", "glaive", "How many required arguments can be filled from the conversation so far?"),
    "arts-known-choice-k2": ("arts", "choice", "tweet-emotion-test", "Which emotion does a short post express: anger, joy, optimism or sadness?"),
    "arts-known-noul-k2": ("arts", "noul", "tweet-irony-test", "Is this short post ironic?"),
    "arts-known-score-k2": ("arts", "score", "tweet-emotion-test", "Rate the intensity of the emotion a short post expresses."),
}  # fmt: skip

TYPE_RULES = {
    "choice": "Question type choice: 3-8 options with short snake_case keys; exactly one correct.",
    "noul": (
        "Question type noul (yes/no): leave options empty; answer true or false. "
        "Balance true and false across items."
    ),
    "score": (
        'Question type score: exactly 5 ordered, equally spaced levels with keys "0","1","2","3","4" '
        "in that order (for example counts 0-4, or ratings from none to very high), each with a "
        "description; the answer is one key. The level must be objectively determinable from the "
        "state."
    ),
}
REWRITE_SYSTEM = """You write new private evaluation items for small (2-4B parameter) decision models, using real dataset rows only as inspiration for the task style.

For each source row, write one new item in the stated task family and question type:
- Do not copy the row. Change entities, numbers, facts, setting and wording so it is not a paraphrase.
- Make the item harder than the row: add plausible distractors and require combining two or three pieces of information. Aim for items a strong 4B model gets right about half the time, without ambiguity.
- The state is plain text (up to about 500 words) holding everything needed. The instructions state the question.
- Exactly one defensible answer; explain it in `explanation`.
Return the new items in the same order as the sources."""  # noqa: E501

OPTION = {
    "type": "object",
    "properties": {"key": {"type": "string"}, "description": {"type": "string"}},
    "required": ["key", "description"],
    "additionalProperties": False,
}
ITEM = {
    "type": "object",
    "properties": {
        "state": {"type": "string"},
        "type": {"type": "string", "enum": ["choice", "noul", "score"]},
        "instructions": {"type": "string"},
        "options": {"type": "array", "items": OPTION},
        "answer": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["state", "type", "instructions", "options", "answer", "explanation"],
    "additionalProperties": False,
}
ITEMS_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": ITEM}},
    "required": ["items"],
    "additionalProperties": False,
}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "confidence": {"type": "number"}},
    "required": ["answer", "confidence"],
    "additionalProperties": False,
}


def run(requests, workers=25):
    """Send (custom_id, params) pairs concurrently; returns custom_id -> text or None."""
    import anthropic

    client = anthropic.Anthropic()

    def one(item):
        custom_id, params = item
        for attempt in range(3):
            try:
                with client.messages.stream(**params) as stream:
                    message = stream.get_final_message()
                if message.stop_reason != "end_turn":
                    return custom_id, None, message.usage
                text = next(b.text for b in message.content if b.type == "text")
                return custom_id, text, message.usage
            except Exception as exc:  # transport resets are not APIError subclasses
                print(json.dumps({"retry": custom_id, "error": type(exc).__name__}), flush=True)
                time.sleep(10 * (attempt + 1))
        return custom_id, None, None

    out, usage = {}, {"input_tokens": 0, "output_tokens": 0}
    with ThreadPoolExecutor(workers) as pool:
        for custom_id, text, used in pool.map(one, requests):
            out[custom_id] = text
            if used is not None:
                usage["input_tokens"] += used.input_tokens
                usage["output_tokens"] += used.output_tokens
    return out, usage


def used_rows(work, draw):
    used = set()
    for plan in Path(work).glob("*/plan.json"):
        if plan.parent.name != draw:
            used |= {sid for ids in json.loads(plan.read_text()).values() for sid in ids}
    return used


def new_families(work, draw, need):
    """Three unused BIG-bench tasks per category (one per type) with at least `need` rows."""
    used = set()
    for path in Path(work).glob("*/families.json"):
        if path.parent.name != draw:
            used |= {v[2] for k, v in json.loads(path.read_text()).items() if "-new-" in k} - {
                "glaive"
            }
    families = {}
    for category, tasks in BIGBENCH.items():
        pool = [f"bb-{t}" for t in tasks if f"bb-{t}" not in used]
        pool = [s for s in pool if len(SOURCES[s]()) >= need]
        if category == "tools":
            pool += ["glaive"] * 3
        pool.sort(key=lambda s: hashlib.sha256(f"{draw}:{s}".encode()).hexdigest())
        if len(pool) < 3:
            raise RuntimeError(f"{category}: fewer than three unused new-domain tasks remain")
        for kind, source in zip(("choice", "noul", "score"), pool):
            brief = f"Task in the style of {source} (training split)."
            families[f"{category}-new-{kind}-{draw}"] = (category, kind, source, brief)
    return families


def author(work, draw, per_family, known_only=False, workers=25):
    root = Path(work) / draw
    root.mkdir(parents=True, exist_ok=True)
    families = dict(KNOWN)
    if not known_only:
        families.update(new_families(work, draw, per_family))
    used, taken, cache, requests, plan = used_rows(work, draw), set(), {}, [], {}
    for family, (_, kind, source, brief) in families.items():
        if source not in cache:
            cache[source] = SOURCES[source]()
        pool = [(sid, r) for sid, r in cache[source] if sid not in used | taken]
        pool.sort(key=lambda x: hashlib.sha256(f"{draw}:{family}:{x[0]}".encode()).hexdigest())
        if len(pool) < per_family:
            raise RuntimeError(f"{family}: only {len(pool)} unused rows in {source}")
        chosen = pool[:per_family]
        taken |= {sid for sid, _ in chosen}
        for k in range(0, per_family, SEEDS_PER_REQUEST):
            chunk = chosen[k : k + SEEDS_PER_REQUEST]
            custom_id = f"{family}--{k // SEEDS_PER_REQUEST}"
            plan[custom_id] = [sid for sid, _ in chunk]
            rows = "\n\n".join(
                f"Source row {i + 1}:\n"
                + json.dumps(r, ensure_ascii=False, default=str)[:MAX_SEED_CHARS]
                for i, (_, r) in enumerate(chunk)
            )
            prompt = (
                f"Task family: {brief}\n{TYPE_RULES[kind]}\n"
                f"Write {len(chunk)} new items, one per source row.\n\n{rows}"
            )
            requests.append((custom_id, {
                "model": MODEL, "max_tokens": 32000, "system": REWRITE_SYSTEM,
                "messages": [{"role": "user", "content": prompt}],
                "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": ITEMS_SCHEMA}},
            }))  # fmt: skip
    (root / "families.json").write_text(json.dumps(families, indent=1))
    (root / "plan.json").write_text(json.dumps(plan, indent=1))
    texts, usage = run(requests, workers)
    (root / "author-usage.json").write_text(json.dumps(usage))
    items = candidates(texts, families, plan, draw)
    (root / "candidates.json").write_text(json.dumps(items, indent=1))
    print(json.dumps({"draw": draw, "requests": len(requests), "candidates": len(items), **usage}))


def candidates(texts, families, plan, draw):
    """Structurally valid authored items; anything malformed is dropped, never repaired."""
    items = []
    for custom_id, text in sorted(texts.items()):
        if text is None:
            continue
        family = custom_id.split("--")[0]
        category, kind = families[family][0], families[family][1]
        for i, item in enumerate(json.loads(text)["items"][: len(plan[custom_id])]):
            options = item["options"]
            if kind == "noul" and item["answer"] in ("true", "false"):
                criteria = None
            elif kind == "score" and [o["key"] for o in options] == SCORE_KEYS:
                criteria = [o["description"] for o in options]
            elif kind == "choice" and item["type"] == "choice" and len(options) >= 2:
                criteria = {o["key"]: o["description"] for o in options}
            else:
                continue
            if kind != "noul" and item["answer"] not in (
                criteria if kind == "choice" else SCORE_KEYS
            ):
                continue
            items.append({
                "id": f"{draw}-{custom_id}-{i}", "family": family, "category": category,
                "block": "new" if "-new-" in family else "known",
                "source_id": plan[custom_id][i], "state": item["state"],
                "question": {"type": kind, "instructions": item["instructions"], "criteria": criteria},
                "label": item["answer"], "explanation": item["explanation"],
            })  # fmt: skip
    return items


def verify(work, draw, workers=25):
    root = Path(work) / draw
    items = json.loads((root / "candidates.json").read_text())
    requests = []
    for item in items:
        q = item["question"]
        if q["type"] == "noul":
            options = 'Answer "true" or "false".'
        else:
            criteria = q["criteria"]
            if isinstance(criteria, list):
                criteria = {str(i): v for i, v in enumerate(criteria)}
            lines = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
            options = f"Options:\n{lines}\nAnswer with one option key."
        content = f"State:\n{item['state']}\n\nQuestion: {q['instructions']}\n{options}"
        requests.append((item["id"], {
            "model": MODEL, "max_tokens": 8000, "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
        }))  # fmt: skip
    texts, usage = run(requests, workers)
    answers = {k: json.loads(v)["answer"].strip() for k, v in texts.items() if v is not None}
    (root / "answers.json").write_text(json.dumps(answers, indent=1))
    (root / "verify-usage.json").write_text(json.dumps(usage))
    print(json.dumps({"draw": draw, "checked": len(items), "answered": len(answers), **usage}))


def finish(work, draw, calibration):
    """Keep blind-pass agreements, shuffle Choice options, split calibration/test, write registry."""
    root = Path(work) / draw
    items = json.loads((root / "candidates.json").read_text())
    answers = json.loads((root / "answers.json").read_text())
    families = json.loads((root / "families.json").read_text())
    rng = random.Random(f"{draw}-options")
    kept, seen = {}, set()
    for item in items:
        if answers.get(item["id"]) != item["label"]:
            continue
        q = item["question"]
        key = (item["state"].strip() + json.dumps(q["instructions"])).lower()
        if key in seen:
            continue
        seen.add(key)
        kept.setdefault(item["family"], []).append(item)
    splits = {"calibration": [], "test": []}
    for family, rows in sorted(kept.items()):
        for j, item in enumerate(rows):
            q = dict(item["question"])
            if q["type"] == "choice":  # option order carries no signal
                pairs = list(q["criteria"].items())
                rng.shuffle(pairs)
                q["criteria"] = dict(pairs)
            splits["calibration" if j < calibration else "test"].append({
                "id": item["id"], "family": family, "category": item["category"],
                "block": item["block"], "source_id": item["source_id"],
                "state": {"text": item["state"]}, "question": q, "label": item["label"],
            })  # fmt: skip
    for name, rows in splits.items():
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (root / f"{name}.jsonl").write_text(text)
    registry = {
        family: {
            "type": kind,
            "category": category,
            **({"ordered_equally_spaced": True} if kind == "score" else {}),
        }
        for family, (category, kind, _, _) in families.items()
        if family in kept
    }
    (root / "registry.json").write_text(json.dumps(registry, indent=1, sort_keys=True))
    print(
        json.dumps({"draw": draw, **{k: len(v) for k, v in splits.items()}, "families": len(kept)})
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("author", "verify", "finish"):
        p = sub.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--draw", required=True)
        if name == "author":
            p.add_argument("--per-family", type=int, required=True)
            p.add_argument("--known-only", action="store_true", help="public seed samples")
        if name in ("author", "verify"):
            p.add_argument("--workers", type=int, default=25)
        if name == "finish":
            p.add_argument("--calibration", type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == "author":
        author(args.work, args.draw, args.per_family, args.known_only, args.workers)
    elif args.command == "verify":
        verify(args.work, args.draw, args.workers)
    else:
        finish(args.work, args.draw, args.calibration)


if __name__ == "__main__":
    main()
