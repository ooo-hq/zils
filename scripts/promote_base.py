"""Prepare a confirmed base-tier winner for publication: verify, stage with licence files, print the profile entry.

Dry run by default. ``--upload`` publishes the staged directory to the Hugging Face organisation with the
``hf`` CLI and prints the resulting revision to pin. Nothing here edits ``zils/models.py`` or a validator
config; the printed entry and the ``base_tier`` crowning (new ``incumbent_sha256`` and ``incumbent_uid``) are
applied in a reviewed pull request.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zils import models  # noqa: E402

ORIGIN = {
    models.H2O: " by h2oai, Apache License 2.0",
    **{m: "; licence terms are in LICENSE" for m in models.d3_family()},
}
CANDIDATE_ID_PREFIX = {
    models.H2O: "h2o-lightning-4b",
    **{m: "d3-mini-4.5b" for m in models.d3_family()},
}

NOTICE = """Zils base model, round {round}.
Derived from {base} (revision {revision}){origin}.
Produced by the Zils base-tier competition; candidate checkpoint sha256 {sha}.
Confirmed on sealed holdout manifest {manifest}.
"""


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="confirmed candidate directory")
    parser.add_argument("--decision", required=True, help="decision.json with confirmed_winner")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--model", choices=list(CANDIDATE_ID_PREFIX), default=models.H2O)
    parser.add_argument("--staging", required=True, help="new directory to stage the release")
    parser.add_argument(
        "--repo",
        default=None,
        help="Hugging Face repo id, default zils-ai/<h2o-lightning-4b|d3-mini-4.5b>-zils-r<round>",
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hf", default=str(Path.home() / ".local/bin/hf"))
    args = parser.parse_args()
    candidate, staging = Path(args.candidate), Path(args.staging)
    candidate_id = f"{CANDIDATE_ID_PREFIX[args.model]}-zils-r{args.round}"
    repo = args.repo or f"zils-ai/{candidate_id}"
    decision = json.loads(Path(args.decision).read_text())
    winner = decision.get("confirmed_winner")
    if not winner:
        raise SystemExit("decision has no confirmed winner; promotion requires two eligible draws")
    if models.metadata(candidate)["kind"] != "candidate":
        raise SystemExit("not a base candidate checkpoint")
    if models.metadata(candidate)["model"] != args.model:
        raise SystemExit(f"not a {args.model} base candidate checkpoint")
    models.text_runtime(args.model).validate_candidate(candidate)
    import zils

    digest = zils.checkpoint_hash(candidate)
    if digest != winner["sha256"]:
        raise SystemExit(
            f"candidate hash {digest} differs from the confirmed winner {winner['sha256']}"
        )
    staging.mkdir(parents=True, exist_ok=False)
    for name in models.candidate_files(args.model, "candidate"):
        if name != "model.json":
            shutil.copyfile(candidate / name, staging / name)
    parent = models.spec(args.model)
    notice = NOTICE.format(
        round=args.round,
        base=parent["base"],
        revision=parent["base_revision"],
        origin=ORIGIN[args.model],
        sha=digest,
        manifest=decision.get("manifest_sha256"),
    )
    with (staging / "NOTICE").open(
        "a"
    ) as f:  # d3 candidates already carry the vendor NOTICE; keep it
        f.write(notice)
    (staging / "README.md").write_text(
        f"# Zils base, round {args.round}\n\nFull-weight {parent['name']} decision model promoted by the Zils base-tier competition. "
        f"Candidate checkpoint sha256 `{digest}`. Same architecture, tokenizer and decision contract as the parent; only weights changed. "
        f"See NOTICE for licence attribution.\n"
    )
    entry = {
        **parent,
        "id": candidate_id,
        "name": f"Zils Base r{args.round} ({parent['name']})",
        "base": repo,
        "base_revision": "<revision printed by --upload>",
        "parent": {"id": parent["id"], "base_revision": parent["base_revision"]},
    }
    pins = {name: sha(staging / name) for name in sorted(p.name for p in staging.iterdir())}
    # The new profile's pins file: the parent's runtime pins over the published files (README.md excluded).
    pins_file = {
        **models.pins(args.model),
        "files": {k: v for k, v in pins.items() if k != "README.md"},
    }
    pins_path = staging.parent / f"{candidate_id}-pins.json"
    pins_path.write_text(json.dumps(pins_file, indent=2, sort_keys=True) + "\n")
    registration = {
        "TEXT_ADAPTER_MODELS": candidate_id,
        "TEXT_RUNTIME_MODULES": models.TEXT_RUNTIME_MODULES[args.model],
        "RUNTIME_PYTHON_ENV": models.RUNTIME_PYTHON_ENV[args.model],
        "CANDIDATE_BASE_ENV": models.CANDIDATE_BASE_ENV[args.model].replace(
            "_CANDIDATE_BASE", f"_ZILS_R{args.round}_CANDIDATE_BASE"
        ),
        "PINS_FILES": pins_path.name,
    }
    result = {
        "staged": str(staging),
        "repo": repo,
        "proposed_spec_entry": entry,
        "proposed_pin_files": pins,
        "proposed_pins_file": str(pins_path),
        "proposed_registration": registration,
        # Crowning: the same promotion PR sets the validator track to the new king, which is then the
        # only base-tier miner paid (king of the hill). Its uid is the confirmed winner's roster uid.
        "proposed_base_tier": {"incumbent_sha256": digest, "incumbent_uid": winner["uid"]},
        "uploaded": False,
    }
    if args.upload:
        out = subprocess.run(
            [
                args.hf,
                "upload",
                repo,
                str(staging),
                ".",
                "--type",
                "model",
                "--commit-message",
                f"Zils base round {args.round}",
                "--format",
                "json",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        result.update(uploaded=True, upload=out.stdout.strip()[-2000:])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
