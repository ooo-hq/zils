"""Frozen image datasets: verified content identity and training-only exports.

A job on a text profile that declares `inputs` (d3) freezes one canonical PNG per case, an asset
catalog bound to the owner and job, and a training export of asset references.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import zils

from . import benchmark, jobs, models, version_selection
from .decisions import DecisionError, validate_request
from .image_contract import capabilities

VERSION = "zils-image-job/v1"
PREPROCESSOR = "zils-image-rgb-png/v1"
SPLITS = {"train": 1024, "calibration": 256, "test": 512}
ASSET_FIELDS = (
    "id",
    "owner_id",
    "job_id",
    "purpose",
    "canonical_sha256",
    "pixel_sha256",
    "canonical_bytes",
    "width",
    "height",
    "preprocessor",
)
CASE_FIELDS = ("id", "group_id", "family", "state", "question", "label", "image")


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def manifest_bytes(manifest):
    return (canonical(manifest) + "\n").encode()


def takes_images(model):
    """Profiles whose jobs freeze image assets: text profiles that declare `inputs`."""
    return "inputs" in models.spec(model)


def validate_training_question(model, question, state=None):
    """One choice question per frozen image case, within the profile's option limit."""
    body = {
        "model": model,
        "state": {} if state is None else state,
        "questions": {"decision": question},
    }
    try:
        validate_request(body)
        if (
            question["type"] != "choice"
            or len(question["criteria"]) > capabilities(model)["max_options"]
            or any(not name.strip() for name in question["criteria"])
        ):
            raise ValueError("image training takes one choice question per case")
    except DecisionError as error:
        raise ValueError(str(error)) from None
    return question


def export_path(entry):
    """Local cache file of a frozen training-export image, relative to the export file."""
    if (
        not isinstance(entry, dict)
        or set(entry) != {"asset_id", "sha256"}
        or not isinstance(entry["sha256"], str)
        or not re.fullmatch("[a-f0-9]{64}", entry["sha256"])
    ):
        raise ValueError("invalid frozen training image reference")
    return "images/" + entry["sha256"] + ".png"


def image_path(images, binding):
    """Content-addressed local cache; model inputs never contain source filenames."""
    digest = binding.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
        raise ValueError("Invalid frozen image hash")
    path = Path(images) / (digest + ".png")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 10 * 1024**2:
        raise ValueError("Frozen image bytes changed or are unavailable")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError("Frozen image bytes changed or are unavailable")
    return path


def verify_binding(job):
    manifest = job.get("manifest") or {}
    if (
        not takes_images(models.job_model(job))
        or manifest.get("version") != VERSION
        or hashlib.sha256(manifest_bytes(manifest)).hexdigest() != job.get("job_sha256")
    ):
        raise ValueError("image manifest binding changed")
    return manifest


def case_fingerprint(case, canonical_sha256):
    return hashlib.sha256(
        canonical(
            {
                "state": case["state"],
                "question": case["question"],
                "outcome_order": list(case["question"]["criteria"]),
                "image_sha256": canonical_sha256,
            }
        ).encode()
    ).hexdigest()


def validate_intake(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "seed", "snapshot_sha256"}
        or value["version"] != "zils-image-intake/v1"
        or not isinstance(value["seed"], str)
        or not 1 <= len(value["seed"]) <= 100
        or not isinstance(value["snapshot_sha256"], str)
        or not re.fullmatch("[a-f0-9]{64}", value["snapshot_sha256"])
    ):
        raise ValueError("invalid frozen image intake")
    return value


def _validate(splits, assets, job, *, live=False):
    model = models.job_model(job)
    if not takes_images(model) or set(splits) != set(SPLITS):
        raise ValueError("image job requires an image-capable profile and all three splits")
    version_selection.job_id(job["id"])
    version_selection.job_id(job["owner_id"])
    ids, groups, digests, pixels, bindings, used = set(), {}, {}, {}, {}, {}
    question, order = None, None
    for split, limit in SPLITS.items():
        cases = splits[split]
        zils.validate_cases(cases)
        if len(cases) > limit:
            raise ValueError(f"{split} exceeds image example limit")
        labels = set()
        for case in cases:
            if not set(CASE_FIELDS) <= case.keys() or set(case["image"]) != {"asset_id"}:
                raise ValueError("image rows require an opaque asset reference")
            aid = version_selection.job_id(case["image"]["asset_id"])
            asset = assets.get(aid)
            if (
                not isinstance(asset, dict)
                or any(key not in asset for key in ASSET_FIELDS)
                or asset["id"] != aid
                or asset["owner_id"] != job["owner_id"]
                or asset["job_id"] != job["id"]
                or asset["purpose"] != "training"
                or asset["preprocessor"] != PREPROCESSOR
            ):
                raise ValueError("image asset owner, job or profile binding is invalid")
            if live:
                if asset.get("state") != "ready":
                    raise ValueError("image is not ready")
                if job.get("status") == "uploading" and datetime.fromisoformat(
                    asset["expires_at"].replace("Z", "+00:00")
                ) <= datetime.now(timezone.utc):
                    raise ValueError("image has expired")
            for field in ("canonical_sha256", "pixel_sha256"):
                if not isinstance(asset[field], str) or not re.fullmatch(
                    "[a-f0-9]{64}", asset[field]
                ):
                    raise ValueError("invalid verified image hash")
            if (
                any(
                    type(asset[k]) is not int or not 1 <= asset[k] <= 8192
                    for k in ("width", "height")
                )
                or asset["width"] * asset["height"] > 16_000_000
                or type(asset["canonical_bytes"]) is not int
                or not 1 <= asset["canonical_bytes"] <= 10 * 1024 * 1024
            ):
                raise ValueError("image exceeds canonical limits")
            validate_training_question(model, case["question"], case["state"])
            current_order = list(case["question"]["criteria"])
            if question is None:
                question, order = case["question"], current_order
            if case["question"] != question or current_order != order:
                raise ValueError("all images must answer the same declared question")
            group = case["group_id"]
            if not isinstance(group, str) or not group or case["id"] in ids:
                raise ValueError("unique case IDs and source groups are required")
            ids.add(case["id"])
            if groups.setdefault(group, split) != split:
                raise ValueError("source group overlaps splits")
            for value, seen in (
                (asset["canonical_sha256"], digests),
                (asset["pixel_sha256"], pixels),
            ):
                previous = seen.setdefault(value, (split, case["label"]))
                if previous[0] != split:
                    raise ValueError("duplicate image content overlaps splits")
                if previous[1] != case["label"]:
                    raise ValueError("identical image has inconsistent labels")
            used[aid] = {**{k: asset[k] for k in ASSET_FIELDS}, "split": split}
            bindings[case["id"]] = case_fingerprint(case, asset["canonical_sha256"])
            labels.add(case["label"])
        if labels != set(order):
            raise ValueError(f"{split} needs independent examples of every outcome")
    if sum(a["canonical_bytes"] for a in used.values()) > 1024**3:
        raise ValueError("image job exceeds one GiB")
    return used, bindings, question, order, groups


def training_rows(cases, assets=None):
    rows = []
    for case in cases:
        aid = case["image"]["asset_id"]
        if assets is None or aid not in assets:
            raise ValueError("image export requires a verified asset catalog")
        rows.append(
            {
                "state": case["state"],
                "questions": {"decision": {**case["question"], "label": case["label"]}},
                "images": [{"asset_id": aid, "sha256": assets[aid]["canonical_sha256"]}],
            }
        )
    return rows


def build(root, job, splits, assets, policy):
    catalog, bindings, question, order, groups = _validate(splits, assets, job, live=True)
    model = models.job_model(job)
    jobs.validate_policy(policy)
    if policy != job["acceptance"]:
        raise ValueError("image acceptance policy changed")
    selection = job.get("selection")
    if selection is None:
        if "previous_job_id" in policy:
            raise ValueError("upgrade requires frozen version selection")
        selection = {
            "version": version_selection.VERSION,
            "root_job_id": job["id"],
            "previous": None,
        }
    version_selection.validate(selection, policy.get("previous_job_id"), current_job_id=job["id"])
    root = Path(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    # Preserve declared outcome order in row/export files. Filenames never enter model evidence.
    clean = {s: [{k: c[k] for k in CASE_FIELDS} for c in rows] for s, rows in splits.items()}
    for split, cases in clean.items():
        benchmark.write_private(
            root / f"{split}.jsonl",
            "".join(json.dumps(c, ensure_ascii=False, allow_nan=False) + "\n" for c in cases),
        )
    benchmark.write_private(
        root / "miner-training.jsonl",
        "".join(
            json.dumps(c, ensure_ascii=False, allow_nan=False) + "\n"
            for c in training_rows(clean["train"], catalog)
        ),
    )
    manifest = {
        "version": VERSION,
        "job_id": job["id"],
        "owner_id": job["owner_id"],
        "data_access": "approved-workers-training-export",
        "model": models.spec(model),
        "acceptance": policy,
        "selection": selection,
        "files": {name: benchmark.file_hash(root / name) for name in benchmark.FILES},
        "counts": {s: len(c) for s, c in clean.items()},
        "assets": catalog,
        "case_fingerprints": bindings,
        "question": question,
        "outcome_order": order,
        "split_plan": {
            "method": "customer-reviewed-groups/v1",
            "seed": validate_intake(job["image_intake"])["seed"]
            if job.get("image_intake")
            else None,
            "proportions": {s: len(c) / sum(map(len, clean.values())) for s, c in clean.items()},
            "groups": groups,
            "counts": {s: len(c) for s, c in clean.items()},
        },
    }
    if job.get("image_intake"):
        manifest["image_intake"] = validate_intake(job["image_intake"])
    benchmark.write_private(root / "manifest.json", manifest_bytes(manifest).decode())
    return manifest


def audit(root, manifest):
    if (
        manifest.get("version") != VERSION
        or manifest.get("data_access") != "approved-workers-training-export"
        or set(manifest.get("files", {})) != set(benchmark.FILES)
    ):
        raise ValueError("invalid image job manifest")
    job = {
        "id": manifest["job_id"],
        "owner_id": manifest["owner_id"],
        "model_profile": manifest["model"],
    }
    root = Path(root)
    for name in benchmark.FILES:
        if benchmark.file_hash(root / name) != manifest["files"][name]:
            raise ValueError("image dataset changed: " + name)
    splits = {s: benchmark.read_jsonl(root / f"{s}.jsonl") for s in SPLITS}
    catalog, bindings, question, order, groups = _validate(splits, manifest["assets"], job)
    jobs.validate_policy(manifest["acceptance"])
    version_selection.validate(
        manifest["selection"],
        manifest["acceptance"].get("previous_job_id"),
        current_job_id=job["id"],
    )
    if (
        catalog != manifest["assets"]
        or bindings != manifest["case_fingerprints"]
        or question != manifest["question"]
        or order != manifest["outcome_order"]
        or groups != manifest["split_plan"]["groups"]
        or manifest["split_plan"]["counts"] != {s: len(c) for s, c in splits.items()}
        or manifest["split_plan"]["proportions"]
        != {s: len(c) / sum(map(len, splits.values())) for s, c in splits.items()}
        or manifest["split_plan"]["seed"]
        != (
            validate_intake(manifest["image_intake"])["seed"]
            if manifest.get("image_intake")
            else None
        )
        or manifest["counts"] != {s: len(c) for s, c in splits.items()}
        or benchmark.read_jsonl(root / "miner-training.jsonl")
        != training_rows(splits["train"], catalog)
    ):
        raise ValueError("image manifest or export binding changed")
    return splits


def validate_predecessor(manifest, previous):
    if previous.get("model") != manifest["model"]:
        raise ValueError("previous model belongs to a different profile family")
    if task_contract(manifest) != task_contract(previous):
        raise ValueError("An image upgrade must retain its incumbent's callable task question")
    known = previous.get("assets", {})
    for field in ("canonical_sha256", "pixel_sha256"):
        prior = {a[field] for a in known.values() if a["split"] in ("train", "calibration")}
        holdout = {
            a[field] for a in manifest["assets"].values() if a["split"] in ("calibration", "test")
        }
        if prior & holdout:
            raise ValueError("new holdout images overlap predecessor training or calibration")


def task_contract(manifest):
    """Recover declared order after PostgreSQL jsonb has reordered object keys."""
    order = manifest["outcome_order"]
    question = manifest["question"]
    if (
        not isinstance(order, list)
        or len(set(order)) != len(order)
        or set(order) != set(question["criteria"])
    ):
        raise ValueError("Image task outcome order changed")
    question = {**question, "criteria": {key: question["criteria"][key] for key in order}}
    model = models.validate_spec(manifest["model"])
    validate_training_question(model, question)
    return {"question": question, "outcome_order": list(order)}
