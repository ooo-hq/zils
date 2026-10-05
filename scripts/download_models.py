"""Download the pinned Kev checkpoint and base model for local experiments."""

import argparse
import os
from pathlib import Path

from zils import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "models/reference")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    from zils import ARTIFACT_FILES, checkpoint_hash
    from zils.runtime import prepare_base

    snapshot_download(
        "jaredpalmer/kev-0.8b",
        revision="54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8",
        local_dir=str(args.out),
        allow_patterns=list(ARTIFACT_FILES),
        cache_dir=str(Path(os.environ.get("HF_HOME", str(ROOT / ".cache/huggingface"))) / "hub"),
    )
    print(f"Reference checkpoint: {args.out} ({checkpoint_hash(args.out)})")
    prepare_base()


if __name__ == "__main__":
    main()
