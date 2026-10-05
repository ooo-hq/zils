"""Validator-owned inference worker. Install Kev in this worker's Python environment."""

import argparse
import importlib.metadata
import json
import math
import sys
import time
from contextlib import redirect_stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    requests = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        import torch
        from kev.api import SystemOneRequest, to_record
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.device import sync

        if args.device == "cuda" and not torch.cuda.is_available():
            raise OSError("CUDA is unavailable")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise OSError("MPS is unavailable")
        checkpoint = Checkpoint(args.checkpoint)
        if checkpoint.meta.base != args.base:
            raise ValueError("checkpoint uses a different base model")
        if checkpoint.meta.base_revision not in (None, args.base_revision):
            raise ValueError("checkpoint declares a different base revision")
        if (checkpoint.meta.lora, checkpoint.meta.head_dim, checkpoint.meta.weights_dtype) != (
            16,
            256,
            "fp32",
        ):
            raise ValueError("v1 requires rank-16 LoRA, a 256-dimensional head, and FP32 weights")
        if not math.isfinite(checkpoint.meta.temperature) or checkpoint.meta.temperature <= 0:
            raise ValueError("checkpoint temperature must be positive and finite")
        # Every candidate uses the operator's pinned base, never a mutable Hub tag.
        checkpoint.meta.base_revision = args.base_revision
        loading_started = time.perf_counter()
        tokenizer, model = checkpoint.load(
            args.device, LoadOptions(dtype=torch.float32, backend="torch")
        )
        sync(args.device)
        model_load_ms = (time.perf_counter() - loading_started) * 1000
        predictions = []
        for request in requests:
            payload = SystemOneRequest(
                state=request["state"], questions={"decision": request["question"]}
            )
            record, metadata = to_record(payload)
            started = time.perf_counter()
            encoded = model.encode(tokenizer, record, max_state=8192, max_branch=8192, strict=True)
            probabilities = model.probs(encoded)[0].tolist()
            sync(args.device)
            predictions.append(
                {
                    "id": request["id"],
                    "probabilities": dict(zip(metadata[0]["keys"], probabilities)),
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                }
            )
        package = importlib.metadata.distribution("kev")
        origin = json.loads(package.read_text("direct_url.json") or "{}")
        runtime = {
            "torch": torch.__version__,
            "kev": package.version,
            "kev_commit": origin.get("vcs_info", {}).get("commit_id"),
            "python": sys.version.split()[0],
            "backend": "torch",
            "dtype": "fp32",
            "temperature": checkpoint.meta.temperature,
            "model_load_ms": model_load_ms,
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
