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
from zils import models, protocol, queue_protocol, settings
from zils.cloud import MAX_DATA_BYTES, APIError, download, trusted_url, upload
from zils.miner_presence import presence_loop
from zils.queue_protocol import identifier, lease_heartbeat
from zils.runtime import CapacityUnavailable, digest, gpu_ready, locked, prepare_base


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


def run_once(
    client, state, reference, runtime, device, *, references=None, busy=None, minimum_mib=0
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
            if model == models.IMAJEV:
                download_images(client, auth, training, directory / "images")
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


def download_images(client, auth, training, destination):
    from datetime import datetime, timezone

    from zils.imajev import image_path

    destination = Path(destination)
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    bindings = {}
    for line in Path(training).read_text().splitlines():
        row = json.loads(line)
        if len(row["images"]) != 1:
            raise ValueError("Training requires one frozen image per row")
        image = row["images"][0]
        aid = identifier(image["asset_id"])
        if not isinstance(image.get("sha256"), str) or not re.fullmatch(
            "[a-f0-9]{64}", image["sha256"]
        ):
            raise ValueError("Invalid frozen training image hash")
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
        state = Path(args.state)
        state.mkdir(mode=0o700, parents=True, exist_ok=True)
        references = {}
        for path in args.reference or ["models/reference"]:
            model = models.checkpoint_model(path)
            if model in references:
                raise ValueError("Install only one reference per model profile")
            references[model] = Path(path).resolve()
            if not args.no_download:
                prepare_base(model)
        enabled = config.get("graded_scheduling", False)
        if type(enabled) is not bool:
            raise ValueError("graded_scheduling must be true or false")
        minimum = config.get("graded_min_free_mib", 0) if enabled else 0
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
