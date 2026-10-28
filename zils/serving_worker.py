"""Inference worker for `zils serve`, run under the pinned d3 interpreter (`miner.d3_python()`).

It loads each `--checkpoint`, writes one ready line, then answers JSON lines on stdin/stdout:
  out  {"ready": true, "served": [{"model", "weights_sha256", "max_input_tokens"}]}
  in   {"id", "model", "questions": {qid: question}, "state"}
  out  {"id", "predictions": {qid: {"probabilities", "input_tokens"}}}
   or  {"id", "error": {"kind": "too_long" | "invalid" | "runtime", "message"}}
Error messages are fixed and never echo the input. HTTP, keys and the chain stay in the coordinator
(zils/serving_server.py); this module must not import bittensor, bittensor_wallet or zils.serving."""

import argparse
import json
import os
import sys
from pathlib import Path

from . import checkpoint_hash, models, settings

MESSAGES = {
    "too_long": "Input exceeds the served token limit.",
    "invalid": "Request could not be encoded.",
    "runtime": "Inference failed.",
}
# d3.Encoder.inputs raises ValueError with this prefix when a decision is over the profile's token cap.
TOO_LONG = "Decision exceeds"


def d3_model(checkpoint):
    from . import d3

    return d3.DecisionModel(checkpoint, "cuda")


def describe(checkpoint):
    """The manifest entry for a public d3-family checkpoint; refuses adapters and unpinned base weights.
    `weights_sha256` is `checkpoint_hash(dir)`: over model.json for a base (its weights are the pinned
    release), over the full weights for a candidate."""
    checkpoint = Path(checkpoint)
    meta = models.metadata(checkpoint)
    if meta is None or meta["kind"] not in ("base", "candidate") or meta["model"] not in models.d3_family():
        raise ValueError(f"{checkpoint}: serving needs a public d3-family checkpoint, not an adapter")
    override = models.candidate_base_env(meta["model"])
    if settings.get(override):  # d3.base_path would load these unpinned weights under the base's hash
        raise ValueError(f"unset {override}: served weights must be the pinned release")
    return {
        "model": meta["model"],
        "weights_sha256": checkpoint_hash(checkpoint),
        "max_input_tokens": models.spec(meta["model"]).get("max_input_tokens", 2048),
    }


def load(checkpoints, factory=d3_model):
    """({model: decision model}, manifest entries); every check runs before any weights load."""
    served = [describe(path) for path in checkpoints]
    names = [item["model"] for item in served]
    if len(set(names)) != len(names):
        raise ValueError("two --checkpoint directories serve the same model")
    return {item["model"]: factory(Path(path)) for item, path in zip(served, checkpoints)}, served


def error(rid, kind):
    return {"id": rid, "error": {"kind": kind, "message": MESSAGES[kind]}}


def answer(loaded, line):
    rid = None
    try:
        request = json.loads(line)
        rid = request["id"]
        decision = loaded[request["model"]]
        state, questions = request["state"], request["questions"]
        encoded = {qid: decision.encoder.inputs(state, q) for qid, q in questions.items()}
        # Every question encodes before any runs, so an over-long request costs no GPU time.
        predictions = {}
        for qid, (ids, keys) in encoded.items():
            probs = decision.probabilities(ids, keys, questions[qid], decision.meta["temperature"])
            predictions[qid] = {"probabilities": probs, "input_tokens": len(ids)}
    except ValueError as failure:
        return error(rid, "too_long" if str(failure).startswith(TOO_LONG) else "invalid")
    except (TypeError, KeyError, AttributeError):
        return error(rid, "invalid")
    except Exception as failure:  # OSError, CUDA RuntimeError: the node, not the caller
        print(f"serving worker: inference failed ({type(failure).__name__})", file=sys.stderr, flush=True)
        return error(rid, "runtime")
    return {"id": rid, "predictions": predictions}


def run(checkpoints, factory=d3_model, stdin=sys.stdin, stdout=sys.stdout):
    loaded, served = load(checkpoints, factory)

    def write(value):
        stdout.write(json.dumps(value) + "\n")
        stdout.flush()

    write({"ready": True, "served": served})
    for line in stdin:
        write(answer(loaded, line))


def main(argv=None, factory=d3_model):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    args = parser.parse_args(argv)
    # The protocol keeps the real stdout; fd 1 goes to stderr so library prints cannot corrupt it.
    protocol = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    try:
        run(args.checkpoint, factory, stdout=protocol)
    except (ValueError, OSError) as failure:
        parser.exit(1, f"serving worker: {failure}\n")


if __name__ == "__main__":
    main()
