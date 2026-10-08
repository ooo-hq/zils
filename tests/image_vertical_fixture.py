"""Isolated HTTP/storage harness. SQL is verified independently in disposable PostgreSQL.

The optional real mode uses production queue, trainer, evaluator and gateway with CUDA;
only account/database/object storage are local test services. Never a production deployment.
"""

import base64
import gc
import json
import os
import sys
import threading
import time
import uuid
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import requests
import torch
from bittensor_wallet import Keypair
from PIL import Image
from safetensors.torch import save_file

from miner.queue import Client, run_once
from scripts.rehearse_image_vertical import rehearse
from tests.image_flow_fixture import SCRIPT, ImageStore as QueueStore
from tests.test_api import FixtureStore
from tests.test_image_releases import SCHEMA
from tests.test_queue import OTHER, OWNER, server
from zils import api, cloud, coordinator, decision_http, imajev, models
from zils.adapter_releases import read_release, register, registry_entry
from zils.decisions import DecisionError
from zils.image_contract import IMAGE_CAPABILITIES
from zils.image_server import ImageRuntime
from zils.image_store import ImageStore
from zils.runtime import gpu_ready, run_child as real_child
from zils.workflow import Workflow


class Storage(QueueStore):
    def matches(self, row, query):
        accepted = "result->delivery->>status=eq.accepted"
        if accepted in query:
            if (row.get("result") or {}).get("delivery", {}).get("status") != "accepted":
                return False
            query = query.replace(accepted, "")
        return super().matches(row, query)

    def rpc(self, name, args):
        if not name.startswith("zils_image_"):
            result = super().rpc(name, args)
            if name == "fez_finish_processing":
                next(j for j in self.tables[coordinator.JOBS] if j["id"] == args["p_job"])[
                    "updated_at"
                ] = datetime.now(timezone.utc).isoformat()
            return result
        owner = args.get("p_owner")
        aid = args.get("p_asset")
        row = self.assets.get(aid)
        if name == "zils_image_create":
            if args["p_job"] and not any(
                j["id"] == args["p_job"] and j["owner_id"] == owner and j["status"] == "uploading"
                for j in self.tables[coordinator.JOBS]
            ):
                return None
            aid = str(uuid.uuid4())
            base = f"{owner}/{aid}"
            row = {
                "id": aid,
                "owner_id": owner,
                "job_id": args["p_job"],
                "purpose": args["p_purpose"],
                "filename": args["p_filename"],
                "source_bytes": args["p_source_bytes"],
                "source_sha256": args["p_source_sha256"],
                "source_path": base + "/source",
                "canonical_path": base + "/canonical.png",
                "state": "uploading",
                "uploadable": True,
                "expires_at": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
                "preprocessor": models.spec(models.IMAJEV)["preprocessor"],
            }
            self.assets[aid] = row
            return dict(row)
        if row is None or row["owner_id"] != owner or row["state"] in ("deleted", "expired"):
            return None
        if name == "zils_image_get":
            return dict(row)
        if name == "zils_image_grant":
            return None
        if name == "zils_image_claim_finalize":
            row.update(state="verifying", finalize_token=str(uuid.uuid4()))
            return dict(row)
        if name == "zils_image_finish":
            if row["finalize_token"] != args["p_token"]:
                return None
            row.update(
                state="ready",
                canonical_sha256=args["p_sha256"],
                pixel_sha256=args["p_pixels"],
                canonical_bytes=args["p_bytes"],
                width=args["p_width"],
                height=args["p_height"],
            )
            return dict(row)
        if name == "zils_image_delete":
            if row.get("referenced"):
                return False
            row["state"] = "deleted"
            return True
        return super().rpc(name, args)

    def signed(self, bucket, path, *, upload=False):
        ticket = str(uuid.uuid4())
        self.tickets[ticket] = (bucket, path, upload)
        payload = (
            base64.urlsafe_b64encode(json.dumps({"exp": int(time.time()) + 600}).encode())
            .decode()
            .rstrip("=")
        )
        prefix = "upload/sign" if upload else "sign"
        result = {
            "url": f"{self.url}/storage/v1/object/{prefix}/{bucket}/{path}?token=x.{payload}.x&ticket={ticket}"
        }
        if upload:
            result.update(
                method="PUT",
                headers={"Content-Type": "application/octet-stream", "x-upsert": "false"},
            )
        return result

    def download(self, bucket, path, destination, limit, **kwargs):
        return cloud.download(self.signed(bucket, path)["url"], destination, limit, **kwargs)

    def handler(self):
        parent = super().handler()

        class Files(parent):
            def handle_file(self):
                self.path = "/" + parse_qs(urlsplit(self.path).query).get("ticket", ["missing"])[0]
                return super().handle_file()

            do_GET = handle_file
            do_PUT = handle_file

        return Files


class Accounts(FixtureStore):
    def __init__(self, store):
        super().__init__()
        self.store = store

    def session_owner(self, token):
        return self.store.user(token)

    def ensure_account(self, owner):
        pass


class FixtureEngine:
    def activate_release(self, path, release):
        if "task" in release:
            read_release(path)

    def prepare(self, image, state, question):
        return {"keys": [*question["criteria"], "__unknown__"], "input_tokens": 442}

    def billable_input(self, request, prepared):
        return 173

    def predict(self, prepared, temperature=1):
        return {
            "probabilities": dict(zip(prepared["keys"], [0.9, 0.05, 0.05], strict=True)),
            "input_tokens": 442,
        }


def fixture_data(root):
    root.mkdir()
    for split in ("train", "calibration", "test", "fresh"):
        rows = []
        for index, label in enumerate(("normal", "damaged")):
            path = root / f"{split}-{index}.png"
            Image.new("RGB", (8, 8), (index, len(split), ord(split[0]))).save(path)
            rows.append(
                {
                    "id": f"{split}-{index}",
                    "group_id": f"{split}-{index}",
                    "family": "inspection",
                    "state": {},
                    "question": {
                        "type": "choice",
                        "instructions": "Inspect the connector.",
                        "criteria": {"normal": None, "damaged": None},
                    },
                    "label": label,
                    "image": path.name,
                }
            )
        (root / (split + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))


def run(
    root,
    *,
    mode="accepted",
    dataset=None,
    reference=None,
    stock=None,
    health=None,
    runtime_executable=None,
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    real = reference is not None
    if dataset is None:
        dataset = root / "dataset"
        fixture_data(dataset)
    store = Storage()
    accounts = Accounts(store)
    key = Keypair.create_from_seed("0x" + "39" * 32)
    text_key = Keypair.create_from_seed("0x" + "42" * 32)
    for uid, k, model in ((1, key, models.IMAJEV), (2, text_key, models.JEVK5)):
        store.tables["fez_training_workers"].append(
            {"hotkey": k.ss58_address, "enabled": True, "uid": uid}
        )
        store.tables["zils_worker_profiles"].append(
            {
                "hotkey": k.ss58_address,
                "enabled": True,
                "profile_id": model,
                "min_free_mib": 12288,
                "evidence": {"max_seconds": 240},
                **models.profile_identity(model),
            }
        )
    evidence = {
        "storage": "isolated in-memory HTTP fixture; PostgreSQL checked separately",
        "model_execution": "real CUDA" if real else "fixture only",
        "errors": [],
    }
    state = {"runtime": None}
    if not real:
        reference = root / "reference"
        reference.mkdir()
        for name, value in SCHEMA["json"].items():
            (reference / name).write_text(json.dumps(value))
        save_file(
            {"language_model.lora_A.weight": torch.ones(2, 2)},
            str(reference / "adapter_model.safetensors"),
        )
        save_file({"weight": torch.ones(3, 2)}, str(reference / "decision_readout.safetensors"))
        models.write_metadata(reference, model=models.IMAJEV)
        stock = reference
        stock_release = {
            "release_id": "image-stock-fixture",
            "fingerprint": "a" * 64,
            "temperature": 1,
        }
        script = SCRIPT.replace(
            "    for name in models.IMAJEV_FILES:\n        if name!='model.json':(out/name).write_text('trained')",
            "    import shutil,torch\n    from safetensors.torch import save_file\n    for name in ('adapter_config.json','decision_readout.json'):shutil.copyfile(checkpoint/name,out/name)\n    save_file({'language_model.lora_A.weight':torch.full((2,2),2.)},str(out/'adapter_model.safetensors'))\n    save_file({'weight':torch.full((3,2),2.)},str(out/'decision_readout.safetensors'))",
        )
        script = script.replace(
            "trained=(checkpoint/'adapter_model.safetensors').read_text()=='trained'",
            "from safetensors.torch import load_file\n    trained=bool(next(iter(load_file(str(checkpoint/'adapter_model.safetensors')).values())).mean()>1)",
        )
        runtime_python = root / "fixture-runtime"
        runtime_python.write_text(f"#!{sys.executable}\n" + script)
        runtime_python.chmod(0o700)
    else:
        runtime_python = Path(runtime_executable or sys.executable)
        stock_release = imajev.verify_reference(stock)
    service = coordinator.Service(store, "http://127.0.0.1:1", models.IMAJEV)
    stop = threading.Event()

    def child(command, log, device, timeout=3600, **kwargs):
        return real_child(command, log, "cpu", timeout, check_lease=kwargs.get("check_lease"))

    with ExitStack() as stack:
        if not real:
            for target in ("zils.imajev.verify_starting_checkpoint",):
                stack.enter_context(patch(target))
            stack.enter_context(patch("zils.imajev.CHECKPOINT_SCHEMA", SCHEMA))
            stack.enter_context(patch("miner.queue.gpu_ready", return_value=True))
            stack.enter_context(patch("zils.coordinator.gpu_ready", return_value=True))
            stack.enter_context(patch("miner.worker.run_child", side_effect=child))
            stack.enter_context(patch("zils.validator.run_child", side_effect=child))
        stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "ZILS_IMAGES_ENABLED": "1",
                    "ZILS_IMAGE_TRAINING_ENABLED": "1",
                    "ZILS_IMAGE_RUNTIME_PYTHON": str(runtime_python),
                    "FIXTURE_IMAGE_REJECT": "1" if mode == "negative" else "0",
                    "REHEARSAL_OWNER": "owner-token",
                    "REHEARSAL_OTHER": "other-token",
                    "REHEARSAL_RUNTIME": "private-fixture-runtime",
                    "REHEARSAL_TEXT": "private-fixture-text",
                },
            )
        )
        storage = stack.enter_context(server(store.handler()))
        store.url = storage

        def runtime_dispatch(method, path, token, body, rid):
            if state["runtime"]:
                return state["runtime"].dispatch(method, path, token, body, rid)
            if method == "GET" and path == "/health" and token == "private-fixture-runtime":
                return 200, {
                    "models": {stock_release["release_id"]: stock_release},
                    "available": False,
                }
            raise DecisionError(503, "capacity", "Image runtime is unloaded during training")

        runtime_url = stack.enter_context(server(decision_http.make_handler(runtime_dispatch)))
        registry_path = root / "registry.json"
        entry = {
            "id": stock_release["release_id"],
            "fingerprint": stock_release["fingerprint"],
            "owners": None,
            "aliases": [],
            "url": runtime_url,
            "token_env": "REHEARSAL_RUNTIME",
            "description": "Private image rehearsal",
            "release_date": "2026-10-08",
            "capabilities": IMAGE_CAPABILITIES,
        }
        registry_path.write_text(json.dumps({"models": [entry]}))
        registry = api.FileRegistry(registry_path)
        gateway = api.Gateway(accounts, registry, image_store=ImageStore(store))
        gateway_url = stack.enter_context(server(decision_http.make_handler(gateway.dispatch)))
        coordinator_url = stack.enter_context(
            server(coordinator.handler(service, "https://app.example"))
        )
        service.audience = coordinator_url
        text_url = stack.enter_context(
            server(
                decision_http.make_handler(
                    lambda *args: (200, {"release_id": "text-fixture", "fingerprint": "d" * 64})
                )
            )
        )
        processor = coordinator.Processor(
            service,
            root / "processor",
            reference,
            SimpleNamespace(runtime_python=str(runtime_python), device="cuda" if real else "cpu"),
        )

        def start_runtime():
            if state["runtime"]:
                return
            if real:
                torch.set_num_threads(6)
                torch.cuda.set_per_process_memory_fraction(0.5)
            engine = imajev.ImageEngine(stock, "cuda") if real else FixtureEngine()
            state["runtime"] = ImageRuntime(
                engine,
                {stock_release["release_id"]: stock_release},
                storage,
                token="private-fixture-runtime",
                reference=stock,
                release_root=root / "releases",
                timeout=60,
            )
            if real:
                evidence["training_admitted_with_image_runtime_loaded"] = gpu_ready(
                    "cuda", model=models.IMAJEV
                )
                evidence["serving_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()

        def activate(release):
            if mode == "activation_failed":
                raise ValueError("Deliberate fixture activation failure")
            start_runtime()
            verified = requests.get(
                runtime_url + "/health",
                headers={"Authorization": "Bearer private-fixture-runtime"},
                timeout=30,
            ).json()
            assert (
                verified["models"][release["release_id"]]["fingerprint"] == release["fingerprint"]
            )
            register(
                registry_path,
                registry_entry(release, runtime_url, "REHEARSAL_RUNTIME"),
                selection=release.get("selection"),
            )
            return release["release_id"]

        workflow = Workflow(
            store,
            text_key.ss58_address,
            root / "text-releases",
            lambda: False,
            lambda r: None,
            image_releases=root / "releases",
            image_activate=activate,
        )
        checked = set()

        def pump():
            try:
                while not stop.is_set():
                    processor.tick()
                    for job in list(store.tables[coordinator.JOBS]):
                        if job["status"] == "awaiting_approval" and job["id"] not in checked:
                            checked.add(job["id"])
                            store.rpc(
                                "fez_approve_training_job",
                                {"p_job": job["id"], "p_hotkeys": [key.ss58_address]},
                            )
                            tc = Client(coordinator_url, text_key)
                            assert (
                                tc.call("claim", {"supported_profiles": [models.JEVK5]})[
                                    "assignment"
                                ]
                                is None
                            )
                            try:
                                tc.call("claim", {"supported_profiles": [models.IMAJEV]})
                            except cloud.APIError as e:
                                assert e.status == 409
                            else:
                                raise AssertionError("Text worker claimed image job")
                            client = Client(coordinator_url, key)
                            assignment = client.call(
                                "claim", {"supported_profiles": [models.IMAJEV]}
                            )["assignment"]
                            heldout = [
                                aid
                                for aid, a in job["manifest"]["assets"].items()
                                if a["split"] != "train"
                            ][0]
                            try:
                                client.call(
                                    "image-downloads",
                                    {
                                        "job_id": job["id"],
                                        "lease_token": assignment["lease_token"],
                                        "asset_ids": [heldout],
                                    },
                                )
                            except cloud.APIError as e:
                                assert e.status in (403, 404, 409)
                            else:
                                raise AssertionError("Worker received a holdout")
                            evidence["worker_holdouts_denied"] = True
                            evidence["text_worker_denied"] = True
                            assert run_once(
                                client,
                                root / "miner",
                                None,
                                str(runtime_python),
                                "cuda" if real else "cpu",
                                references={models.IMAJEV: reference},
                            )
                    workflow.tick()
                    stop.wait(0.1)
            except Exception as error:
                evidence["errors"].append(type(error).__name__)
                for job in store.tables[coordinator.JOBS]:
                    if job["status"] not in ("completed", "failed"):
                        job.update(status="failed", error="Private rehearsal failed")
                (root / "worker-failure.txt").write_text(type(error).__name__ + "\n")
                stop.set()

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        config = {
            "isolated": True,
            "owner_id": OWNER,
            "other_owner_id": OTHER,
            "api_url": gateway_url,
            "coordinator_url": coordinator_url,
            "storage_url": storage,
            "owner_token_env": "REHEARSAL_OWNER",
            "other_token_env": "REHEARSAL_OTHER",
            "runtime_token_env": "REHEARSAL_RUNTIME",
            "text_token_env": "REHEARSAL_TEXT",
            "text_runtime": health
            or {
                "url": text_url + "/health",
                "token_config": "text_token_env",
                "fingerprint": "d" * 64,
            },
            "image_runtime": {
                "url": runtime_url + "/health",
                "token_config": "runtime_token_env",
                "fingerprint": stock_release["fingerprint"],
            },
            "max_seconds": 240 if real else 60,
            "acceptance": {
                "min_accuracy": 0.8,
                "min_brier_improvement": 0.01,
                "positive_class": "damaged",
                "min_positive_recall": 0.8,
                "max_false_positive_rate": 0.2,
            },
        }
        if health:
            config["live_text_token_env"] = "REHEARSAL_LIVE_TEXT_TOKEN"
        try:
            result = rehearse(config, dataset, root / "evidence")
            if real and not state["runtime"]:
                (root / "releases").mkdir(exist_ok=True)
                start_runtime()
            result["harness"] = evidence
            if evidence["errors"]:
                raise RuntimeError("Harness worker failed")
            (root / "completion.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        finally:
            stop.set()
            thread.join(timeout=30)
            if state["runtime"]:
                state["runtime"].close()
                state["runtime"] = None
            gc.collect()
            if real:
                torch.cuda.empty_cache()
