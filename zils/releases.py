"""Validate immutable evaluated model artifacts without a hosted service."""

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

import zils

from . import jobs, models, version_selection
from .queue_protocol import identifier

VERSION = "zils-customer-adapter/v1"
PROMPT_VERSION = "zils-training-decision-v1"
RUNTIME_REVISION = "f26426d16f59e8bbe1470e5b162cc89329e29b29"
FILES = (*models.JEVK5_FILES, "release.json", "manifest.json", "source.json")
JSON_LIMIT = 4 * 1024 * 1024
SOURCE_FIELDS = (
    "id",
    "owner_id",
    "name",
    "status",
    "updated_at",
    "manifest",
    "job_sha256",
    "initial_sha256",
    "acceptance",
    "release_prefix",
    "result",
)


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > JSON_LIMIT:
        raise ValueError("Release metadata must be a bounded regular file")
    return json.loads(path.read_text())


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    path.chmod(0o600)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch("[a-f0-9]{64}", value):
        raise ValueError("Invalid release hash")
    return value


def _accepted(job):
    identifier(job["id"])
    identifier(job["owner_id"])
    if job["status"] != "completed" or job["result"]["delivery"]["status"] != "accepted":
        raise ValueError("Only a completed accepted job can become a release")
    model = models.job_model(job)
    if model not in (models.JEVK5, models.IMAJEV):
        raise ValueError("This profile cannot publish a customer API release")
    spec = models.spec(model)
    if job["manifest"].get("model") != spec or job["result"].get("model") != spec:
        raise ValueError("Evaluated model differs from its frozen job")
    if model == models.IMAJEV:
        from .image_jobs import task_contract, verify_binding

        if job.get("model_profile") != spec:
            raise ValueError("An image release requires its frozen job profile")
        verify_binding(job)
        task_contract(job["manifest"])
        if any(
            asset["owner_id"] != job["owner_id"] or asset["job_id"] != job["id"]
            for asset in job["manifest"]["assets"].values()
        ):
            raise ValueError("Image release owner or job binding changed")
    if job["manifest"]["acceptance"] != job["acceptance"]:
        raise ValueError("Job acceptance changed")
    selection = job["manifest"].get("selection")
    if selection is not None or "previous_job_id" in job["acceptance"]:
        version_selection.validate(
            selection, job["acceptance"].get("previous_job_id"), current_job_id=job["id"]
        )
        previous = selection["previous"]
        expected = previous["sha256"] if previous else job["initial_sha256"]
        if (
            job["result"].get("selection") != selection
            or job["result"].get("baseline_reference_sha256") != expected
        ):
            raise ValueError("Evaluation did not use the frozen comparison model")
    for field in ("job_sha256", "initial_sha256"):
        _sha(job[field])
    prefix = job["release_prefix"]
    if not isinstance(prefix, str) or not prefix.startswith(job["id"] + "/releases/"):
        raise ValueError("Invalid accepted release location")
    identifier(prefix.removeprefix(job["id"] + "/releases/"))
    result, delivery = job["result"], job["result"]["delivery"]
    _sha(delivery["sha256"])
    uids = [r["uid"] for r in result["miners"]]
    if any(type(uid) is not int or uid < 0 for uid in uids) or len(set(uids)) != len(uids):
        raise ValueError("Candidate identities must be unique nonnegative integers")
    rows = [result["baseline"], *result["miners"]]
    for row in rows:
        if row.get("status") == "evaluated":
            for key, minimum, maximum in (("accuracy", 0, 1), ("brier", 0, 2), ("skill", -3, 1)):
                value = row[key]
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not minimum <= value <= maximum
                ):
                    raise ValueError("Invalid recorded evaluation")
    candidates = [
        {**r, "sha256": delivery["sha256"] if r["uid"] == delivery["uid"] else "0" * 64}
        for r in result["miners"]
    ]
    if jobs.select(result["baseline"], candidates, job["acceptance"], model=model) != delivery:
        raise ValueError("Recorded candidate does not satisfy the frozen acceptance policy")
    return delivery


def _manifest(root):
    job = _json(root / "source.json")
    delivery = _accepted(job)
    model = models.job_model(job)
    spec = models.spec(model)
    manifest = _json(root / "manifest.json")
    if _hash(root / "manifest.json") != job["job_sha256"] or manifest != job["manifest"]:
        raise ValueError("Frozen training manifest changed")
    checkpoint = zils.checkpoint_hash(root)
    meta = models.metadata(root)
    if checkpoint != delivery["sha256"] or meta is None or meta["kind"] != "adapter":
        raise ValueError("Accepted adapter bytes or model identity changed")
    report = _json(root / "release.json")
    expected = {
        "job_id": job["id"],
        "job_sha256": job["job_sha256"],
        "round_id": job["release_prefix"].rsplit("/", 1)[1],
        "model": spec,
        "base": spec["base"],
        "base_revision": spec["base_revision"],
        "initial_sha256": job["initial_sha256"],
        "baseline_brier": job["result"]["baseline"]["brier"],
        **delivery,
    }
    if "selection" in manifest:
        expected.update(
            selection=manifest["selection"],
            baseline_reference_sha256=job["result"]["baseline_reference_sha256"],
        )
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("Accepted release provenance differs from the completed job")
    _sha(report["submitted_sha256"])
    date = datetime.fromisoformat(job["updated_at"])
    if date.tzinfo is None:
        raise ValueError("Completion timestamp requires a timezone")
    files = (*models.candidate_files(model), "release.json", "manifest.json", "source.json")
    extra = {}
    if model == models.IMAJEV:
        from .image_jobs import task_contract
        from .image_metrics import public_metrics
        from .imajev import validate_checkpoint

        validate_checkpoint(root)
        calibrated = meta.get("calibration") or {}
        if (
            calibrated.get("raw_checkpoint_sha256") != report["submitted_sha256"]
            or calibrated.get("benchmark_manifest_sha256") != job["job_sha256"]
            or calibrated.get("fit_split") != "calibration"
            or calibrated.get("fit_cases") != manifest["counts"]["calibration"]
        ):
            raise ValueError("Image calibration provenance changed")
        winner = next(row for row in job["result"]["miners"] if row["uid"] == delivery["uid"])
        metrics = {
            "baseline": public_metrics(job["result"]["baseline"]),
            "candidate": public_metrics(winner),
        }
        if report.get("image_metrics") != metrics:
            raise ValueError("Image release quality evidence changed")
        extra = {
            "task": task_contract(manifest),
            "calibration": calibrated,
            "image_metrics": metrics,
        }
    for name in files:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Release files must be regular files")
    value = {
        "version": VERSION,
        "release_id": f"zils-adapter-{job['id']}-{checkpoint}",
        "owner_id": job["owner_id"],
        "job_id": job["id"],
        "job_sha256": job["job_sha256"],
        "model": spec,
        "checkpoint_sha256": checkpoint,
        "temperature": meta["temperature"],
        "runtime_revision": spec["runtime_revision"]
        if model == models.IMAJEV
        else RUNTIME_REVISION,
        "prompt_version": spec["prompt"] if model == models.IMAJEV else PROMPT_VERSION,
        "release_date": date.date().isoformat(),
        "files": {n: _hash(root / n) for n in files},
        **extra,
    }
    if "selection" in manifest:
        value["selection"] = manifest["selection"]
    value["fingerprint"] = _digest(value)
    return value


def read_release(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Release directory must not be a symlink")
    recorded = _json(root / "serving.json")
    expected = _manifest(root)
    if recorded != expected or root.name != recorded["release_id"]:
        raise ValueError("Serving release content or identity changed")
    return recorded
