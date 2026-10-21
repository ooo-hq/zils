"""Validator-owned text adapter inference worker; accepts no labels or model code from miners."""

import argparse
import importlib.metadata
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

# Executing this file directly must import the zils package, not the zils/ directory itself.
script_dir = Path(__file__).resolve().parent
sys.path = [str(script_dir.parent), *[p for p in sys.path if Path(p).resolve() != script_dir]]


SHARED_BATCH = 5  # one box: four edges and presence


def shared_image_groups(requests, shared):
    """Consecutive rows asking about the same single image, at most SHARED_BATCH per group."""
    groups = []
    for row in requests:
        if set(row) - {"images", "videos"} != {"id", "state", "question"}:
            raise ValueError("inference must receive only id, state, question and media paths")
        last = groups[-1] if groups else None
        if (
            shared
            and last
            and len(last) < SHARED_BATCH
            and "videos" not in row
            and len(row.get("images", ())) == 1
            and set(last[0]) == set(row)
            and last[0]["images"] == row["images"]
        ):
            last.append(row)
        else:
            groups.append([row])
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "base", "base-revision", "device"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    requests = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        from zils import models

        model_id = models.checkpoint_model(args.checkpoint)
        engine = models.text_runtime(model_id)
        spec = models.spec(model_id)
        if (args.base, args.base_revision) != (spec["base"], spec["base_revision"]):
            raise ValueError("Text base differs from its pinned revision")
        started = time.perf_counter()
        model = engine.DecisionModel(args.checkpoint, args.device)
        load_ms = (time.perf_counter() - started) * 1000
        predictions = []
        for group in shared_image_groups(requests, hasattr(model, "predict_shared")):
            row = group[0]
            # Media travel as local paths; only profiles that declare inputs ever receive them.
            media = {}
            if "images" in row:
                media["images"] = [Path(path).read_bytes() for path in row["images"]]
            if "videos" in row:
                media["videos"] = list(row["videos"])
            started = time.perf_counter()
            if len(group) > 1:
                results = model.predict_shared(
                    [(r["state"], r["question"]) for r in group], media["images"][0]
                )
            else:
                results = [model.predict(row["state"], row["question"], **media)]
            elapsed = (time.perf_counter() - started) * 1000 / len(group)
            predictions += [
                {"id": r["id"], "probabilities": p, "elapsed_ms": elapsed}
                for r, p in zip(group, results, strict=True)
            ]
        runtime = {
            "model": spec,
            "temperature": model.meta["temperature"],
            "dtype": "bfloat16",
            "device": args.device,
            "model_load_ms": load_ms,
            "runtime_revision": spec["runtime_revision"],
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        }
    print(json.dumps({"predictions": predictions, "runtime": runtime}, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError) as error:
        print(f"runtime/setup error: {error}", file=sys.stderr)
        sys.exit(78)
    except Exception as error:
        print(f"checkpoint evaluation failed: {error}", file=sys.stderr)
        sys.exit(2)
