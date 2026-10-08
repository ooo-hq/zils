"""Download only the approved image base, published adapter/head and runtime sources."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from zils import models
from zils.imajev import PINS, build_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new empty image output directory")
    from huggingface_hub import snapshot_download

    profile = models.spec(models.IMAJEV)
    for folder in ("base", "adapter"):
        snapshot_download(
            profile[folder],
            revision=profile[folder + "_revision"],
            local_dir=args.out / folder,
            allow_patterns=list(PINS[folder]),
        )
    with tempfile.TemporaryDirectory(prefix="imajev-source-") as temp:
        subprocess.run(
            ["git", "clone", "--no-checkout", "https://github.com/mohit67890/imajev", temp],
            check=True,
            capture_output=True,
            timeout=180,
        )
        subprocess.run(
            ["git", "-C", temp, "checkout", "--detach", profile["runtime_revision"]],
            check=True,
            capture_output=True,
            timeout=60,
        )
        for name in PINS["runtime"]:
            target = args.out / "runtime" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(temp) / name, target)
    manifest = build_manifest(args.out)
    (args.out / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("release_id", "fingerprint")}))


if __name__ == "__main__":
    main()
