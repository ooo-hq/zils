"""Prepare a confirmed base-tier winner for publication: verify, stage with licence files, print the profile entry.

Dry run by default. ``--upload`` publishes the staged directory to the Hugging Face organisation with the
``hf`` CLI and prints the resulting revision to pin. Nothing here edits ``zils/models.py`` or a validator
config; the printed entry and the ``base_tier`` crowning (new ``incumbent_sha256``, the new king's hotkey
prepended to ``kings``) are applied in a reviewed pull request. A validator-built soup winner adds no
king; its members, round and hash are recorded under ``release``.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zils import base_release, models, regression_suite  # noqa: E402

ORIGIN = {
    **{m: "; licence terms are in LICENSE" for m in models.d3_family()},
}
CANDIDATE_ID_PREFIX = {
    **{m: models.D3_POOL_MODELS[models.pool_id(m)].removesuffix("-v1") for m in models.d3_family()},
}

NOTICE = """Zils base model, round {round}.
Derived from {base} (revision {revision}){origin}.
Produced by the Zils base-tier competition; candidate checkpoint sha256 {sha}.
Confirmed on sealed holdout manifest {manifest}.
"""


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def archival_verify_regression(report, candidate, incumbent, model):
    if (
        report.get("candidate_sha256") != candidate
        or report.get("incumbent_sha256") != incumbent
        or report.get("incumbent_model") != model
        or report.get("candidate_model") != model
    ):
        raise ValueError("regression report differs from candidate, incumbent or pool profile")
    jobs = report.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        raise ValueError("regression report requires scored jobs")
    comparison = {}
    for job, sides in jobs.items():
        if not isinstance(sides, dict) or set(sides) != {"incumbent", "candidate"}:
            raise ValueError("regression job requires incumbent and candidate")
        values = {
            side: scores.get("brier") if isinstance(scores, dict) else None
            for side, scores in sides.items()
        }
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 2
            for v in values.values()
        ):
            raise ValueError("regression Brier must be finite and in [0, 2]")
        comparison[job] = values
    if not regression_suite.verdict(comparison)["passed"]:
        raise ValueError("candidate failed the adapter regression gate")


def verify_regression(report, candidate, incumbent, model, *, expected_suite):
    return regression_suite.verify_election(
        report, candidate, incumbent, model, expected_suite=expected_suite
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="confirmed candidate directory")
    parser.add_argument("--decision", help="decision.json with confirmed_winner")
    parser.add_argument("--first-decision", help="first draw's timestamped decision.json")
    parser.add_argument(
        "--regression-report", help="bound suite-result.json from adapter regression"
    )
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--model", choices=list(CANDIDATE_ID_PREFIX))
    parser.add_argument("--config", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--window-report", required=True)
    parser.add_argument("--staging", required=True, help="new directory to stage the release")
    parser.add_argument(
        "--repo",
        default=None,
        help="Hugging Face repo id, default zils-ai/<d3-mini-4.5b>-zils-r<round>",
    )
    parser.add_argument(
        "--king-hotkey",
        help="hotkey of the confirmed winner's uid, as in the round's chain-snapshot.json",
    )
    parser.add_argument(
        "--previous-kings",
        help="the track's current base_tier.kings, comma-separated, newest first",
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hf", default=str(Path.home() / ".local/bin/hf"))
    args = parser.parse_args()
    candidate, staging = Path(args.candidate), Path(args.staging)
    if (models.metadata(candidate) or {}).get("kind") != "candidate":
        raise SystemExit("not a base candidate checkpoint")
    if args.model and models.metadata(candidate)["model"] != args.model:
        raise SystemExit(f"not a {args.model} base candidate checkpoint")
    config = Path(args.config).resolve()
    active = base_release.active_context(
        base_release.read(config), args.pool, directory=config.parent
    )
    model = active["model"]["id"]
    if args.model and args.model != model or models.metadata(candidate)["model"] != model:
        raise ValueError("candidate/model flag differs from active pool")
    args.model = model
    window = base_release.verify_window(base_release.read(args.window_report), active)
    winner = window["winner"]
    if window["status"] != "qualified" or winner is None:
        raise ValueError("release window has no qualified winner")
    soup = winner.get("ownership")
    if soup and args.king_hotkey is not None:
        raise ValueError("a soup release crowns no king; omit --king-hotkey")
    if args.king_hotkey is not None and args.king_hotkey != winner["hotkey"]:
        raise ValueError("king-hotkey differs from verified snapshot")
    if (
        args.previous_kings is not None
        and list(filter(None, args.previous_kings.split(","))) != active["kings"]
    ):
        raise ValueError("previous-kings differs from installed configuration")
    args.king_hotkey = winner["hotkey"]
    args.previous_kings = ",".join(active["kings"])
    candidate_id = f"{CANDIDATE_ID_PREFIX[model]}-zils-r{args.round}"
    repo = args.repo or f"zils-ai/{candidate_id}"
    decision = window["closing"]["decision"]
    runtime = models.text_runtime(args.model)
    if args.model in models.d3_family():
        runtime.validate_candidate(candidate, args.model)
    else:
        runtime.validate_candidate(candidate)
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
        # Crowning: the same promotion PR sets the validator track to the new incumbent and prepends
        # the new king to the history (king of the hill; the last five distinct kings share the base).
        # A validator-built soup is released without a king: its members are paid by the soup pool.
        "proposed_base_tier": {
            "incumbent_sha256": digest,
            "kings": [
                *([] if soup else [args.king_hotkey]),
                *filter(None, args.previous_kings.split(",")),
            ],
        },
        "crowned_uid": None if soup else winner["uid"],
        "pool_id": models.pool_id(args.model) if args.model in models.d3_family() else None,
        "uploaded": False,
    }
    if soup:
        result["release"] = soup
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
