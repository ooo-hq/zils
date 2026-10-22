"""Write a staged five-pool config from an existing mini base_tier; never install it."""

import argparse
import json
import tempfile
from pathlib import Path

import zils
from zils import base_round, model_pools, models


def migrate(config, directory):
    if "base_pools" in config or "base_tier" not in config:
        raise ValueError("migration requires an existing legacy base_tier")
    model = models.validate_spec(config["model"])
    if models.pool_id(model) != "d3-mini":
        raise ValueError("legacy incumbent must belong to mini")
    track = config["base_tier"]
    result = {
        k: v
        for k, v in config.items()
        if k not in ("base_tier", "base_round_dir", "base_reference")
    }
    pools = {
        "d3-mini": {
            **{k: v for k, v in track.items() if k != "burn_uid"},
            "model": models.spec(model),
            "share": 1.0,
            "enabled": False,
            "draws": [],
            "policy": dict(base_round.DEFAULT_POLICY),
        }
    }
    for pool, root in models.D3_POOL_MODELS.items():
        if pool == "d3-mini":
            continue
        with tempfile.TemporaryDirectory() as tmp:
            models.write_metadata(tmp, model=root, kind="base")
            digest = zils.checkpoint_hash(tmp)
        pools[pool] = {
            "model": models.spec(root),
            "enabled": False,
            "share": 0.0,
            "incumbent_sha256": digest,
            "kings": [],
            "draws": [],
        }
    result["base_pools"] = model_pools.validate(
        {"share": track["share"], "burn_uid": track.get("burn_uid", 0), "pools": pools}
    )
    result["base_pool_rounds"] = {}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--out", required=True, help="new file for review; existing files are refused"
    )
    args = parser.parse_args()
    path = Path(args.config).resolve()
    result = migrate(json.loads(path.read_text()), path.parent)
    with Path(args.out).open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
