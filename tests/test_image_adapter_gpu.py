"""Explicit private GPU switching check; never runs during unittest discovery."""

import argparse
import json
import time
from pathlib import Path


def run(stock, adapter, images, out):
    import torch

    from zils import models, settings
    from zils.adapter_releases import read_release
    from zils.imajev import ImageEngine, verify_reference
    from zils.runtime import CapacityUnavailable, gpu_ready, locked

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    release = read_release(adapter)
    reference = verify_reference(stock)
    if release["model"] != models.spec(models.IMAJEV):
        raise ValueError("An image release is required")
    photos = sorted(Path(images).glob("*.png"))[:2]
    if not photos:
        raise ValueError("Provide one or two canonical PNG test images")
    lock = Path(settings.required("ZILS_COMPUTE_LOCK"))
    with locked(lock):
        if not gpu_ready("cuda", model=models.IMAJEV):
            raise CapacityUnavailable()
        torch.set_num_threads(6)
        torch.cuda.set_per_process_memory_fraction(0.5)
        tick = time.perf_counter()
        engine = ImageEngine(stock, "cuda")
        load_seconds = time.perf_counter() - tick
        records = []
        for name, path, identity in [
            ("stock-before", stock, reference),
            ("adapter", adapter, release),
            ("adapter-warm", adapter, release),
            ("stock-after", stock, reference),
        ]:
            tick = time.perf_counter()
            engine.activate_release(path, identity)
            switched = time.perf_counter()
            predictions = []
            for photo in photos:
                prepared = engine.prepare(photo, {}, release["task"]["question"])
                predictions.append(engine.predict(prepared, identity["temperature"]))
            torch.cuda.synchronize()
            records.append(
                {
                    "name": name,
                    "release_id": identity["release_id"],
                    "fingerprint": identity["fingerprint"],
                    "activation_seconds": switched - tick,
                    "prediction_seconds": time.perf_counter() - switched,
                    "predictions": predictions,
                }
            )
        delta = max(
            abs(p["probabilities"][key] - q["probabilities"][key])
            for p, q in zip(records[0]["predictions"], records[-1]["predictions"], strict=True)
            for key in p["probabilities"]
        )
        if delta > 1e-6:
            raise ValueError("Stock predictions were not restored within tolerance")
        if records[1]["predictions"] != records[2]["predictions"]:
            raise ValueError("Warm adapter predictions changed")
        if verify_reference(stock) != reference or read_release(adapter) != release:
            raise ValueError("Immutable inputs changed during the check")
        report = {
            "completed": True,
            "stock_restore_max_probability_difference": delta,
            "tolerance": 1e-6,
            "model_load_seconds": load_seconds,
            "stock_cpu_cache_bytes": 0,
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "records": records,
            "private_switching_check_only": True,
            "registered": False,
        }
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stock", "adapter", "images", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.stock, args.adapter, args.images, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
