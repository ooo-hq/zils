"""Validator-owned JevK5 inference worker; accepts no labels or model code from miners."""

import argparse
import importlib.metadata
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

# Executing this file must not let zils/jevk5.py shadow the upstream jevk5 package.
script_dir = Path(__file__).resolve().parent
sys.path = [str(script_dir.parent), *[p for p in sys.path if Path(p).resolve() != script_dir]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "base", "base-revision", "device"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    requests = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        from zils import models
        from zils.jevk5 import DecisionModel

        spec = models.spec(models.JEVK5)
        if (args.base, args.base_revision) != (spec["base"], spec["base_revision"]):
            raise ValueError("JevK5 base differs from its pinned revision")
        started = time.perf_counter()
        model = DecisionModel(args.checkpoint, args.device)
        load_ms = (time.perf_counter() - started) * 1000
        predictions = []
        for row in requests:
            if set(row) != {"id", "state", "question"}:
                raise ValueError("inference must receive only id, state and question")
            started = time.perf_counter()
            probabilities = model.predict(row["state"], row["question"])
            predictions.append(
                {
                    "id": row["id"],
                    "probabilities": probabilities,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                }
            )
        runtime = {
            "model": spec,
            "temperature": model.meta["temperature"],
            "dtype": "bfloat16",
            "device": args.device,
            "model_load_ms": load_ms,
            "jevk5": importlib.metadata.version("jevk5"),
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
