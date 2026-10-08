"""Isolated, offline image training and validator-owned inference."""

import argparse
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    with redirect_stdout(sys.stderr):
        from zils import imajev, models, settings

        if args.train:
            if args.out is None:
                raise ValueError("A new training output path is required")
            report = imajev.train(
                args.checkpoint, args.cases, args.images, args.out, args.seed, args.device
            )
        else:
            started = time.perf_counter()
            engine = imajev.ImageEngine(
                Path(settings.required("ZILS_IMAGE_REFERENCE")), args.device
            )
            meta = imajev.load_checkpoint(engine, args.checkpoint)
            load_ms = 1000 * (time.perf_counter() - started)
            rows = json.loads(args.cases.read_text())
            predictions = []
            for row in rows:
                if set(row) != {"id", "state", "question", "image"}:
                    raise ValueError(
                        "Image inference receives only ID, state, question and frozen image binding"
                    )
                tick = time.perf_counter()
                prepared = engine.prepare(
                    imajev.image_path(args.images, row["image"]), row["state"], row["question"]
                )
                import torch

                from zils.image_metrics import probabilities_at_temperature

                with torch.inference_mode():
                    logits = engine.logits(prepared).cpu().tolist()
                result = {
                    "probabilities": dict(
                        zip(
                            prepared["keys"],
                            probabilities_at_temperature(logits, meta["temperature"]),
                            strict=True,
                        )
                    ),
                    "input_tokens": prepared["input_tokens"],
                    "logits": logits,
                }
                predictions.append(
                    {"id": row["id"], **result, "elapsed_ms": 1000 * (time.perf_counter() - tick)}
                )
            report = {
                "predictions": predictions,
                "runtime": {
                    "model": models.spec(models.IMAJEV),
                    "temperature": meta["temperature"],
                    "dtype": "bfloat16",
                    "device": args.device,
                    "model_load_ms": load_ms,
                    **models.profile_identity(models.IMAJEV),
                },
            }
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError) as error:
        print(f"runtime/setup error: {type(error).__name__}", file=sys.stderr)
        sys.exit(78)
    except Exception as error:
        print(f"image worker failed: {type(error).__name__}", file=sys.stderr)
        sys.exit(2)
