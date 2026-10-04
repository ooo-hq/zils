"""Claim approved Supabase-backed jobs without receiving a Supabase service credential."""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import requests
from bittensor_wallet import Keypair

import fez
from fez import models, protocol, queue_protocol
from fez.cloud import MAX_DATA_BYTES, APIError, download, trusted_url, upload
from fez.coordinator import identifier, lease_heartbeat
from fez.runtime import digest, locked, prepare_base
from miner.worker import train_candidate


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


def run_once(client, state, reference, runtime, device):
    assignment = client.call("claim")["assignment"]
    if assignment is None:
        return False
    job_id = identifier(assignment["job_id"])
    token = identifier(assignment["lease_token"])
    auth = {"job_id": job_id, "lease_token": token}
    try:
        with lease_heartbeat(lambda: client.call("renew", auth)):
            model = models.checkpoint_model(reference)
            if assignment["base_revision"] != models.spec(model)["base_revision"] or assignment.get(
                "model", models.spec(models.KEV)
            ) != models.spec(model):
                raise ValueError("job uses an unsupported base revision")
            if fez.checkpoint_hash(reference) != assignment["initial_sha256"]:
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
                fez.stage(fez.submission(reference, config["uid"]), directory / "reference")
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
            job = {**config, "round_id": uuid.UUID(job_id).hex}
            entry = train_candidate(config, directory, job, runtime, device)
            urls = client.call("uploads", auth)["uploads"]
            for name in models.artifact_files(entry["checkpoint"]):
                if not urls[name].get("uploaded"):
                    upload(
                        urls[name]["url"],
                        Path(entry["checkpoint"]) / name,
                        urls[name].get("headers"),
                    )
            client.call("submit", {**auth, "sha256": entry["sha256"]})
            print(f"Submitted candidate for job {job_id}.", flush=True)
    except Exception:
        # A successful submission with a lost response is protected by the DB state check.
        try:
            client.call("fail", auth)
        except Exception:
            pass
        raise
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="JSON with coordinator and seed or wallet/hotkey"
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--reference", default="models/reference")
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
        if not args.no_download:
            prepare_base(models.checkpoint_model(args.reference))
        with locked(state / "queue-worker.lock"):
            while True:
                try:
                    run_once(client, state, Path(args.reference), args.runtime_python, args.device)
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
        parser.exit(1, f"fez queued miner: {error}\n")


if __name__ == "__main__":
    main()
