"""Claim approved Supabase-backed jobs without receiving a Supabase service credential."""

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path

import requests
from bittensor_wallet import Keypair

import zils
from miner.worker import train_candidate
from zils import image_jobs, miner_grading, models, protocol, queue_protocol, settings
from zils.cloud import MAX_DATA_BYTES, APIError, download, trusted_url, upload
from zils.miner_presence import presence_loop
from zils.queue_protocol import identifier, lease_heartbeat
from zils.runtime import (
    CapacityUnavailable,
    digest,
    gpu_ready,
    gpu_total_mib,
    locked,
    prepare_base,
)


class Client:
    def __init__(self, url, key):
        self.url, self.key = trusted_url(url), key

    def call(self, action, body=None):
        path = "/v1/workers/" + action
        message = queue_protocol.sign(self.key, self.url, path, body or {})
        try:
            response = requests.post(
                self.url + path,
                json=message,
                timeout=(10, 660 if action == "submit" else 30),
                allow_redirects=False,
            )
        except requests.RequestException:
            raise APIError(
                503, "Coordinator unavailable; retrying with the saved candidate."
            ) from None
        try:
            if response.status_code != 200:
                raise APIError(response.status_code, "Coordinator rejected the worker request.")
            if len(response.content) > 64 * 1024:
                raise APIError(503, "Coordinator response exceeds its size limit.")
            return response.json()
        finally:
            response.close()


def admit(client, config):
    """Chain admission (docs/miner-admission.md): once at startup, send the terms version this
    config accepted and the host's qualification report. Nodes an operator registered by hand
    carry no terms_sha256 and skip this; a coordinator without admission answers 404 or 409."""
    terms = config.get("terms_sha256")
    if terms is None:
        return None
    if not isinstance(terms, str) or not re.fullmatch("[a-f0-9]{64}", terms):
        raise ValueError("terms_sha256 must be the sha256 hex digest of docs/mining-terms.md")
    report = {}
    if "qualification" in config:
        report = json.loads(Path(config["qualification"]).read_text())
    try:
        return client.call("admit", {"terms_sha256": terms, "qualification": report})
    except APIError as error:
        if error.status not in (404, 409):
            raise
        print(f"Chain admission not applied ({error.status}); continuing.", file=sys.stderr)
        return None


def run_once(
    client,
    state,
    reference,
    runtime,
    device,
    *,
    references=None,
    busy=None,
    minimum_mib=0,
    terms_sha256=None,
    budget_mib=None,
    qualifications=None,
):
    if not gpu_ready(device):
        return False
    claim = None
    if references is not None:
        available = [model for model in references if gpu_ready(device, model=model)]
        if not available:
            return False
        approved = client.call("profiles")["profiles"]
        available = [
            model
            for model in available
            if any(
                q["profile_id"] == model
                and all(q[k] == v for k, v in models.profile_identity(model).items())
                and gpu_ready(device, model=model, minimum_mib=q["min_free_mib"])
                for q in approved
            )
        ]
        if not available:
            return False
        claim = {"supported_profiles": available}
        if terms_sha256:
            claim["terms_sha256"] = (
                terms_sha256  # the accepted terms version travels in every claim
            )
    assignment = client.call("claim", claim)["assignment"]
    if assignment is None:
        return False
    if busy is not None:
        busy.set()
    job_id = identifier(assignment["job_id"])
    token = identifier(assignment["lease_token"])
    auth = {"job_id": job_id, "lease_token": token}
    try:
        with lease_heartbeat(lambda: client.call("renew", auth)) as check_lease:
            model = models.validate_spec(assignment.get("model", models.spec(models.KEV)))
            if references is not None:
                if model not in claim["supported_profiles"]:
                    raise ValueError("Worker received an unadvertised model profile")
                reference = Path(references[model])
            if models.checkpoint_model(reference) != model:
                raise ValueError("Installed reference differs from the assigned profile")
            if model == models.IMAJEV:
                from zils.imajev import verify_starting_checkpoint

                verify_starting_checkpoint(reference)
            if assignment["base_revision"] != models.spec(model)["base_revision"] or assignment.get(
                "model", models.spec(models.KEV)
            ) != models.spec(model):
                raise ValueError("job uses an unsupported base revision")
            if zils.checkpoint_hash(reference) != assignment["initial_sha256"]:
                raise ValueError("job starting checkpoint differs from the installed reference")
            if not fits(
                model,
                assignment.get("workload"),
                device,
                budget_mib,
                qualification=(qualifications or {}).get(model),
            ):
                client.call("refuse", auth)
                print(
                    f"Refused job {job_id}: its longest example does not fit this GPU.", flush=True
                )
                return False
            directory = Path(state) / job_id
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            config = {
                "uid": assignment["uid"],
                "hotkey": client.key.ss58_address,
                "training_authority": client.url,
                **{
                    k: assignment[k]
                    for k in ("base_revision", "initial_sha256", "training_sha256", "job_sha256")
                },
                "job_id": job_id,
            }
            saved = directory / "assignment.json"
            if saved.exists() and json.loads(saved.read_text()) != config:
                raise ValueError("saved job configuration differs from the assignment")
            if not saved.exists():
                protocol.write_json(saved, config)
            if not (directory / "reference").exists():
                zils.stage(zils.submission(reference, config["uid"]), directory / "reference")
            training = directory / "miner-training.jsonl"
            if not training.exists():
                temporary = directory / ("download-" + uuid.uuid4().hex)
                try:
                    download(assignment["training"]["url"], temporary, MAX_DATA_BYTES)
                    if digest(temporary) != config["training_sha256"]:
                        raise ValueError("training data hash mismatch")
                    temporary.rename(training)
                finally:
                    temporary.unlink(missing_ok=True)
            job = {**config, "round_id": uuid.UUID(job_id).hex, "min_free_mib": minimum_mib}
            if model in models.PINS_FILES:
                job["min_free_mib"] = assignment["min_free_mib"]
                job["max_seconds"] = assignment["max_seconds"]
            if image_jobs.takes_images(model):
                download_images(client, auth, training, directory / "images", model)
            if model == models.IMAJEV:
                runtime = settings.required("ZILS_IMAGE_RUNTIME_PYTHON")
                job["min_free_mib"] = assignment["min_free_mib"]
                job["max_seconds"] = assignment["max_seconds"]
            entry = train_candidate(
                config,
                directory,
                job,
                runtime,
                device,
                check_lease=check_lease,
            )
            check_lease()
            urls = client.call("uploads", auth)["uploads"]
            for name in models.artifact_files(entry["checkpoint"]):
                check_lease()
                if not urls[name].get("uploaded"):
                    upload(
                        urls[name]["url"],
                        Path(entry["checkpoint"]) / name,
                        urls[name].get("headers"),
                    )
            check_lease()
            client.call("submit", {**auth, "sha256": entry["sha256"]})
            print(f"Submitted candidate for job {job_id}.", flush=True)
    except CapacityUnavailable:
        client.call("defer", auth)
        return False
    except Exception:
        # A successful submission with a lost response is protected by the DB state check.
        try:
            client.call("fail", auth)
        except Exception:
            pass
        raise
    finally:
        if busy is not None:
            busy.clear()
    return True


def fits(model, workload, device, budget_mib=None, *, qualification=None):
    """The miner's own fit check: the job's longest example against this GPU's memory budget.
    A job too long for the whole budget is refused so another miner takes it; momentary
    shortage stays a capacity deferral."""
    need = miner_grading.training_mib(model, (workload or {}).get("max_tokens"), qualification)
    if need is None and model in models.d3_family() and models.pool_id(model) != "d3-mini":
        return False
    if need is None or device != "cuda":
        return True
    budget = budget_mib if budget_mib is not None else gpu_total_mib()
    return budget is None or need <= budget


def download_images(client, auth, training, destination, model=models.IMAJEV):
    from datetime import datetime, timezone

    from zils.imajev import image_path

    destination = Path(destination)
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    bindings = {}
    for line in Path(training).read_text().splitlines():
        row = json.loads(line)
        images = row.get("images", [])
        if model == models.IMAJEV and len(images) != 1:
            raise ValueError("Training requires one frozen image per row")
        for image in images:
            image_jobs.export_path(image)
            aid = identifier(image["asset_id"])
            if aid in bindings and bindings[aid] != image["sha256"]:
                raise ValueError("Training image binding changed")
            bindings[aid] = image["sha256"]
    # Small refresh batches keep URLs short lived, even on slow connections.
    for aid, sha in bindings.items():
        target = destination / (sha + ".png")
        if target.exists():
            image_path(destination, {"sha256": sha})
            continue
        response = client.call("image-downloads", {**auth, "asset_ids": [aid]})["images"]
        if len(response) != 1 or response[0]["id"] != aid or response[0]["sha256"] != sha:
            raise ValueError("Coordinator returned a different training image")
        ref = response[0]
        if datetime.fromisoformat(ref["expires_at"].replace("Z", "+00:00")) <= datetime.now(
            timezone.utc
        ):
            raise ValueError("Training image grant expired")
        temporary = destination / ("download-" + uuid.uuid4().hex)
        try:
            download(ref["url"], temporary, 10 * 1024**2)
            if digest(temporary) != sha or temporary.stat().st_size != ref["bytes"]:
                raise ValueError("Training image bytes changed")
            temporary.rename(target)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="JSON with coordinator and seed or wallet/hotkey"
    )
    parser.add_argument("--state", required=True)
    parser.add_argument(
        "--reference", action="append", help="Repeat for each installed pinned checkpoint"
    )
    parser.add_argument("--runtime-python", default=sys.executable)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text())
        if "wallet" in config:
            from bittensor.wallet import Wallet

            key = Wallet(**config["wallet"]).hotkey
            if key.ss58_address != config["hotkey"]:
                raise ValueError("wallet does not match configured hotkey")
        else:
            key = Keypair.create_from_seed(config["seed"])
        client = Client(config["coordinator"], key)
        admit(client, config)
        state = Path(args.state)
        state.mkdir(mode=0o700, parents=True, exist_ok=True)
        references = {}
        for path in args.reference or ["models/reference"]:
            path = Path(path).resolve(strict=True)  # a missing dir would read as Kev
            model = models.checkpoint_model(path)
            if model in references:
                raise ValueError("Install only one reference per model profile")
            if env := models.runtime_python_env(model):
                settings.required(env)  # fail before claiming a job this node cannot run
            references[model] = path
            if not args.no_download:
                prepare_base(model)
        enabled = config.get("graded_scheduling", False)
        if type(enabled) is not bool:
            raise ValueError("graded_scheduling must be true or false")
        minimum = config.get("graded_min_free_mib", 0) if enabled else 0
        budget = config.get("gpu_budget_mib")
        if budget is not None and (type(budget) is not int or budget < 1):
            raise ValueError("gpu_budget_mib must be a positive integer")
        qualifications = {
            model: json.loads(Path(path).read_text())
            for model, path in config.get("qualifications", {}).items()
        }
        with (
            locked(state / "queue-worker.lock"),
            presence_loop(
                client, references, args.device, enabled=enabled, minimum_mib=minimum
            ) as busy,
        ):
            while True:
                try:
                    run_once(
                        client,
                        state,
                        None,
                        args.runtime_python,
                        args.device,
                        references=references,
                        busy=busy,
                        minimum_mib=minimum,
                        terms_sha256=config.get("terms_sha256"),
                        budget_mib=budget,
                        qualifications=qualifications,
                    )
                except (
                    APIError,
                    ValueError,
                    OSError,
                    RuntimeError,
                    requests.RequestException,
                ) as error:
                    print(
                        f"Queued worker attempt failed ({type(error).__name__}).",
                        file=sys.stderr,
                        flush=True,
                    )
                    if args.once:
                        raise
                if args.once:
                    break
                time.sleep(10)
    except KeyboardInterrupt:
        pass
    except (APIError, ValueError, OSError, KeyError, RuntimeError) as error:
        parser.exit(1, f"zils queued miner: {error}\n")


if __name__ == "__main__":
    main()
