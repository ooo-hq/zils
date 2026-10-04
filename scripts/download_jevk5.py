"""Explicitly download pinned Zils model files and create their integrity manifest."""

import argparse
import json
from pathlib import Path

from fez.jev_server import MODEL_REVISION, build_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new empty model output directory")
    from huggingface_hub import snapshot_download

    snapshot_download(
        "alibiserikbay/JevK5",
        revision=MODEL_REVISION,
        local_dir=args.out,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.jinja",
            "tokenizer*",
            "README.md",
            "LICENSE*",
            "NOTICE*",
        ],
    )
    manifest = build_manifest(args.out)
    (args.out / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps({"release_id": manifest["release_id"], "fingerprint": manifest["fingerprint"]})
    )


if __name__ == "__main__":
    main()
