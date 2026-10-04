"""Opt-in GPU plumbing check with synthetic acceptance records, never customer releases."""

import copy
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from fez import decision_http, models
from fez.api import Gateway, Registry
from fez.decisions import make_response
from tests.test_adapter_releases import OTHER, OWNER, Source, fixture
from tests.test_adapter_server import Accounts
from tests.test_decision_http import server


@unittest.skipUnless(
    os.environ.get("ZILS_TEST_ADAPTER"), "set ZILS_TEST_ADAPTER for GPU verification"
)
class AdapterGPUCheck(unittest.TestCase):
    def test_real_adapter_switching_matches_training_and_existing_gateway(self):
        import torch
        from safetensors.torch import load_file, save_file

        from fez.adapter_releases import publish, registry_entry
        from fez.adapter_server import AdapterEngine, AdapterRuntime
        from fez.jev_server import SerialEngine

        checkpoint = Path(os.environ["ZILS_TEST_ADAPTER"])
        with tempfile.TemporaryDirectory(prefix="zils-adapter-gpu-") as tmp:
            root = Path(tmp)
            zero = root / "zero"
            zero.mkdir()
            for name in models.JEVK5_FILES:
                shutil.copyfile(checkpoint / name, zero / name)
            tensors = load_file(str(zero / "adapter_model.safetensors"))
            self.assertTrue(any(t.count_nonzero() for k, t in tensors.items() if "lora_B" in k))
            for key in tensors:
                if "lora_B" in key:
                    tensors[key] = torch.zeros_like(tensors[key])
            save_file(tensors, str(zero / "adapter_model.safetensors"))
            models.write_metadata(zero, temperature=0.65)
            releases = []
            for owner, source_checkpoint in ((OWNER, checkpoint), (OTHER, zero)):
                source = root / owner
                job = fixture(source, owner, checkpoint=source_checkpoint)
                releases.append(publish(Source(job, source), job["id"], root / "releases"))
            torch.cuda.reset_peak_memory_stats()
            engine = AdapterEngine(root / "releases")
            base_identity = id(engine.backend.model.base)
            serial = SerialEngine(engine)
            runtime = AdapterRuntime(engine, serial, "gpu-test-only-token")
            body = {
                "state": {
                    "decision": "Choose the team that owns the main customer problem.",
                    "ticket": "I cannot sign in to see an invoice that might be incorrect.",
                },
                "questions": {
                    "route": {
                        "type": "choice",
                        "criteria": {"billing": "Billing", "access": "Account access"},
                    },
                    "billing": {"type": "noul", "instructions": "Is this a billing problem?"},
                    "urgency": {
                        "type": "score",
                        "instructions": "Rate the urgency of this support ticket.",
                        "criteria": ["Low", "Medium", "High"],
                    },
                },
            }
            predictions, elapsed = [], []
            try:
                with patch.dict(os.environ, {"ZILS_ADAPTER_GPU_TOKEN": "gpu-test-only-token"}):
                    with server(decision_http, runtime.dispatch) as port:
                        entries = [
                            registry_entry(r, f"http://127.0.0.1:{port}", "ZILS_ADAPTER_GPU_TOKEN")
                            for r in releases
                        ]
                        gateway = Gateway(Accounts(), Registry(entries))
                        with server(decision_http, gateway.dispatch) as api_port:
                            for index in (0, 1, 0):
                                request = {
                                    **copy.deepcopy(body),
                                    "model": releases[index]["release_id"],
                                }
                                start = time.perf_counter()
                                response = requests.post(
                                    f"http://127.0.0.1:{api_port}/v1/systemone",
                                    headers={
                                        "Authorization": "Bearer "
                                        + (OWNER if index == 0 else OTHER)
                                    },
                                    json=request,
                                    timeout=45,
                                )
                                elapsed.append(time.perf_counter() - start)
                                self.assertEqual(response.status_code, 200, response.text)
                                result = response.json()
                                predictions.append(result["answers"])
                                self.assertEqual(id(engine.backend.model.base), base_identity)
                                # Independent training prediction path checks prompt, option order,
                                # and this release's calibrated temperature after each switch.
                                model = engine.backend.model
                                previous = model.meta["temperature"]
                                model.meta["temperature"] = releases[index]["temperature"]
                                try:
                                    expected = {}
                                    for qid, question in request["questions"].items():
                                        ids, _ = engine.backend.encode(request["state"], question)
                                        expected[qid] = {
                                            "input_tokens": len(ids),
                                            "probabilities": model.predict(
                                                request["state"], question
                                            ),
                                        }
                                    self.assertEqual(
                                        result, make_response(request["model"], request, expected)
                                    )
                                finally:
                                    model.meta["temperature"] = previous
                            self.assertEqual(predictions[0], predictions[2])
                            self.assertNotEqual(predictions[0], predictions[1])
            finally:
                serial.close()
            print(
                json.dumps(
                    {
                        "check": "customer_adapter_gpu",
                        "synthetic_acceptance_records": True,
                        "gateway_requests": len(predictions),
                        "decisions_per_request": len(body["questions"]),
                        "one_base_instance": True,
                        "switch_back_reproduced": True,
                        "matches_training_readout": True,
                        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                        "request_seconds": [round(v, 3) for v in elapsed],
                    }
                ),
                flush=True,
            )
