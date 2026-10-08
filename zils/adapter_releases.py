"""Publish accepted training artifacts for the existing API model-registry contract."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import zils

from . import jobs, models, version_selection
from .api import Registry
from .cloud import DATA_BUCKET, MODEL_BUCKET, APIError, Supabase, trusted_url
from .coordinator import identifier, prepared_path
from .runtime import locked

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


def publish(store, job_id, destination):
    """Read existing queue/storage only. Rejected jobs leave the destination untouched."""
    job_id = identifier(job_id)
    rows = store.rows("fez_training_jobs", f"id=eq.{job_id}")
    if len(rows) != 1:
        raise ValueError("Training job was not found")
    job = rows[0]
    if (
        job["status"] != "completed"
        or (job.get("result") or {}).get("delivery", {}).get("status") != "accepted"
    ):
        return None
    delivery = _accepted(job)
    model = models.job_model(job)
    source = {key: job[key] for key in SOURCE_FIELDS}
    if model == models.IMAJEV:
        source["model_profile"] = job["model_profile"]
    # Operational activation progress is mutable; the evaluation and completion time are not.
    source["result"] = {k: v for k, v in source["result"].items() if k != "workflow"}
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    release_id = f"zils-adapter-{job_id}-{delivery['sha256']}"
    with locked(root / ".publish.lock", wait=True):
        target = root / release_id
        if target.exists() or target.is_symlink():
            value = read_release(target)
            if _json(target / "source.json") != source:
                raise ValueError("Published job provenance changed")
            return value
        temporary = Path(tempfile.mkdtemp(prefix=".stage-", dir=root))
        try:
            store.download(
                DATA_BUCKET,
                prepared_path(job, "manifest.json"),
                temporary / "manifest.json",
                JSON_LIMIT,
            )
            remaining = zils.MAX_ARTIFACT_BYTES
            for name in (*models.candidate_files(model), "release.json"):
                limit = remaining if name.endswith(".safetensors") else min(remaining, JSON_LIMIT)
                remaining -= store.download(
                    MODEL_BUCKET, job["release_prefix"] + "/" + name, temporary / name, limit
                )
            _write(temporary / "source.json", source)
            value = _manifest(temporary)
            _write(temporary / "serving.json", value)
            for path in temporary.iterdir():
                path.chmod(0o400)
            temporary.rename(target)
            return value
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def registry_entry(release, url, token_env, alias=None):
    if not isinstance(token_env, str) or not re.fullmatch("[A-Z][A-Z0-9_]*", token_env):
        raise ValueError("Supply an environment-variable name, not a runtime secret")
    selection = release.get("selection")
    if selection:
        task_alias = "zils-task-" + selection["root_job_id"]
        if alias is not None and alias != task_alias:
            raise ValueError("Versioned releases must retain their task alias")
        alias = task_alias
    entry = {
        "id": release["release_id"],
        "fingerprint": release["fingerprint"],
        "aliases": [alias] if alias else [],
        "owners": [identifier(release["owner_id"])],
        "url": trusted_url(url),
        "token_env": token_env,
        "release_date": release["release_date"],
        "description": "Customer JevK5 adapter, accepted by held-out training evaluation",
    }
    if release["model"] == models.spec(models.IMAJEV):
        from .image_contract import IMAGE_CAPABILITIES

        entry.update(
            capabilities=IMAGE_CAPABILITIES,
            profile=release["model"],
            task=release["task"],
            description="Private image model, accepted by held-out training evaluation",
        )
    Registry([entry])
    return entry


def merge_registry(current, entry):
    """Keep immutable releases; move an optional alias only within the same owner."""
    Registry(current["models"])
    Registry([entry])
    if not entry["owners"] or len(entry["owners"]) != 1:
        raise ValueError("Customer releases require exactly one owner")
    output, found = [], False
    for old in current["models"]:
        if old["id"] == entry["id"]:
            normalized_old = Registry([old]).entries[0]
            normalized_entry = Registry([entry]).entries[0]
            if {k: v for k, v in normalized_old.items() if k != "aliases"} != {
                k: v for k, v in normalized_entry.items() if k != "aliases"
            }:
                raise ValueError("An immutable release entry cannot be replaced")
            output.append(
                {**entry, "aliases": list(dict.fromkeys([*old["aliases"], *entry["aliases"]]))}
            )
            found = True
            continue
        if old["id"] in entry["aliases"]:
            raise ValueError("An alias cannot replace another release ID")
        overlap = set(old["aliases"]) & set(entry["aliases"])
        if overlap and old["owners"] != entry["owners"]:
            raise ValueError("An alias cannot move between customer or shared scopes")
        output.append({**old, "aliases": [a for a in old["aliases"] if a not in overlap]})
    if not found:
        output.append(entry)
    Registry(output)
    return {"models": output}


def register(path, entry, *, selection=None):
    if selection is None and any(alias.startswith("zils-task-") for alias in entry["aliases"]):
        raise ValueError("Task aliases require frozen version selection")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with locked(path.with_name(path.name + ".lock"), wait=True):
        current = _json(path) if path.exists() else {"models": []}
        if selection is not None:
            from .version_selection import check_promotion

            check_promotion(current, entry, selection)
        result = merge_registry(current, entry)
        fd, name = tempfile.mkstemp(prefix=".registry-", dir=path.parent)
        try:
            if path.exists():
                stat = path.stat()
                os.fchmod(fd, stat.st_mode & 0o777)
                os.fchown(fd, stat.st_uid, stat.st_gid)
            with os.fdopen(fd, "w") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="Private release root owned by the operator"
    )
    parser.add_argument(
        "--registry", type=Path, required=True, help="Prepared API registry; loaded on API startup"
    )
    parser.add_argument("--runtime-url", required=True)
    parser.add_argument("--token-env", default="ZILS_ADAPTER_RUNTIME_TOKEN")
    parser.add_argument("--alias", help="Optional stable model name scoped to this customer")
    args = parser.parse_args()
    try:
        release = publish(Supabase(), args.job, args.out)
        if release is None:
            print(json.dumps({"status": "not_accepted", "job_id": args.job}))
            return
        entry = registry_entry(release, args.runtime_url, args.token_env, args.alias)
        register(args.registry, entry, selection=release.get("selection"))
        print(
            json.dumps(
                {"status": "published", "model": entry["id"], "fingerprint": entry["fingerprint"]}
            )
        )
    except (ValueError, KeyError, TypeError, OSError, APIError) as error:
        parser.exit(
            1,
            f"Adapter publication failed ({type(error).__name__}); verify the accepted job and local release.\n",
        )


if __name__ == "__main__":
    main()
