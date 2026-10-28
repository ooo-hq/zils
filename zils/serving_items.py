"""Fresh, label-free decision requests for grading serving miners against the validator's own run."""

import hashlib
import random

from zils.benchmark import scenario

QIDS = ("decision", "answer", "q")
# Severity is left out: every model size answers it >= 0.99 sure (28 Oct 2026 calibration), so it never
# tells a model from a smaller substitute.
FAMILIES = ("policy", "routing", "evidence")


def round_seed(validator_hotkey, round_id, miner_hotkey, model):
    key = f"{validator_hotkey}|{round_id}|{miner_hotkey}|{model}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


def items(seed, count):
    """`count` request bodies (no model) from clean benchmark scenarios, deterministic per seed."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        family = rng.choice(FAMILIES)
        rows = list(scenario(family, rng.randrange(10), rng))
        record, question, _args = rng.choice(rows)
        if question["type"] == "choice":
            keys = rng.sample(list(question["criteria"]), len(question["criteria"]))
            question = {**question, "criteria": {k: question["criteria"][k] for k in keys}}
        out.append(
            {"state": {"record": record, "note": ""}, "questions": {rng.choice(QIDS): question}}
        )
    return out
