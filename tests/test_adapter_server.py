"""Customer releases cross the existing gateway unchanged; adapters never overlap."""

import os
import tempfile
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from tests.test_adapter_releases import OTHER, OWNER, Source, fixture
from tests.test_decision_http import server
from zils import decision_http
from zils.api import Gateway, Registry
from zils.decisions import DecisionError


class Backend:
    """Tiny deterministic model double; real GPU validation is a separate check."""

    def __init__(self):
        self.current = None
        self.activations = []
        self.temperatures = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def encode(self, state, question):
        from zils import options

        return [1, 2, 3], options(question)

    def activate(self, path, release):
        self.current = release["owner_id"]
        self.activations.append(self.current)

    def probabilities(self, ids, keys, temperature):
        current = self.current
        self.temperatures.append(temperature)
        self.entered.set()
        self.release.wait(3)
        assert current == self.current, "Another request switched the adapter during inference"
        index = 0 if current == OWNER else 1
        return {key: float(i == index) for i, key in enumerate(keys)}


class Accounts:
    def authenticate(self, token):
        if token not in (OWNER, OTHER):
            raise DecisionError(401, "invalid_credentials", "Invalid test key")
        return {"owner_id": token, "id": token}

    def admit(self, *args):
        pass

    def finish_usage(self, *args):
        pass


class AdapterServerTest(unittest.TestCase):
    def test_existing_gateway_calls_only_the_owners_adapter_and_keeps_identity(self):
        from zils.adapter_releases import publish, registry_entry
        from zils.adapter_server import AdapterEngine, AdapterRuntime
        from zils.jev_server import SerialEngine

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            releases = []
            for owner in (OWNER, OTHER):
                source = root / owner
                job = fixture(source, owner)
                releases.append(publish(Source(job, source), job["id"], root / "releases"))
            backend = Backend()
            engine = AdapterEngine(root / "releases", backend)
            serial = SerialEngine(engine)
            runtime = AdapterRuntime(engine, serial, "runtime-secret")
            os.environ["ZILS_ADAPTER_TEST_TOKEN"] = "runtime-secret"
            try:
                with server(decision_http, runtime.dispatch) as runtime_port:
                    entries = [
                        registry_entry(
                            r, f"http://127.0.0.1:{runtime_port}", "ZILS_ADAPTER_TEST_TOKEN"
                        )
                        for r in releases
                    ]
                    gateway = Gateway(Accounts(), Registry(entries))
                    with server(decision_http, gateway.dispatch) as gateway_port:
                        url = f"http://127.0.0.1:{gateway_port}"

                        def request(index):
                            r = requests.post(
                                url + "/v1/systemone",
                                json={
                                    "model": releases[index]["release_id"],
                                    "state": {"decision": "Route this case"},
                                    "questions": {
                                        "route": {
                                            "type": "choice",
                                            "criteria": {
                                                "billing": "Billing",
                                                "support": "Support",
                                            },
                                        }
                                    },
                                },
                                headers={
                                    "Authorization": "Bearer " + (OWNER if index == 0 else OTHER)
                                },
                            )
                            self.assertEqual(r.status_code, 200, r.text)
                            return r.json()

                        with ThreadPoolExecutor(max_workers=2) as pool:
                            first = pool.submit(request, 0)
                            self.assertTrue(backend.entered.wait(2))
                            second = pool.submit(request, 1)
                            try:
                                deadline = time.monotonic() + 2
                                while serial.pending_count != 1 and time.monotonic() < deadline:
                                    time.sleep(0.001)
                                self.assertEqual(serial.pending_count, 1)
                            finally:
                                backend.release.set()
                            results = [first.result(), second.result()]
                        for index, result in enumerate(results):
                            self.assertEqual(result["model"], releases[index]["release_id"])
                            self.assertEqual(
                                result["answers"]["route"]["choice"],
                                "billing" if index == 0 else "support",
                            )
                            self.assertEqual(result["usage"]["input_tokens"], 3)
                        self.assertEqual(backend.temperatures, [0.75, 0.75])
                        count = len(backend.activations)
                        with self.assertRaises(DecisionError) as error:
                            gateway.dispatch(
                                "POST",
                                "/v1/systemone",
                                OTHER,
                                {
                                    "model": releases[0]["release_id"],
                                    "state": {},
                                    "questions": {"q": {"type": "noul"}},
                                },
                                str(uuid.uuid4()),
                            )
                        self.assertEqual(error.exception.status, 404)
                        self.assertEqual(len(backend.activations), count)
                        listing = requests.get(
                            url + "/v1/models", headers={"Authorization": "Bearer " + OWNER}
                        ).json()
                        self.assertEqual(
                            [r["name"] for r in listing["models"]], [releases[0]["release_id"]]
                        )
            finally:
                serial.close()
                os.environ.pop("ZILS_ADAPTER_TEST_TOKEN", None)

    def test_preflight_limits_unknown_models_and_runtime_auth_do_not_activate_weights(self):
        from zils.adapter_releases import publish
        from zils.adapter_server import AdapterEngine, AdapterRuntime
        from zils.jev_server import SerialEngine

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            release = publish(Source(job, root / "source"), job["id"], root / "releases")
            backend = Backend()
            engine = AdapterEngine(root / "releases", backend, max_request_tokens=5)
            serial = SerialEngine(engine)
            runtime = AdapterRuntime(engine, serial, "runtime-secret")
            body = {
                "model": release["release_id"],
                "state": {},
                "questions": {"q": {"type": "noul"}},
            }
            try:
                status, prepared = runtime.dispatch(
                    "POST", "/v1/prepare", "runtime-secret", {"request": body}, "rid"
                )
                self.assertEqual(status, 200)
                self.assertEqual(prepared["fingerprint"], release["fingerprint"])
                self.assertEqual(prepared["reserved_tokens"], 3)
                cases = [
                    ({**body, "model": "unknown"}, 404),
                    (
                        {
                            **body,
                            "questions": {
                                "q": {
                                    "type": "choice",
                                    "criteria": {str(i): None for i in range(17)},
                                }
                            },
                        },
                        422,
                    ),
                    ({**body, "questions": {"q": {"type": "noul"}, "r": {"type": "noul"}}}, 413),
                ]
                for request, expected in cases:
                    with self.assertRaises(DecisionError) as error:
                        engine.prepare(request)
                    self.assertEqual(error.exception.status, expected)
                with self.assertRaises(DecisionError) as error:
                    runtime.dispatch(
                        "POST", "/v1/systemone", "wrong-token", {"request": body}, "rid"
                    )
                self.assertEqual(error.exception.status, 401)
                self.assertEqual(backend.activations, [])
            finally:
                serial.close()

    def test_activation_failure_never_returns_another_customers_prediction(self):
        from zils.adapter_releases import publish
        from zils.adapter_server import AdapterEngine

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            release = publish(Source(job, root / "source"), job["id"], root / "releases")

            class Failing(Backend):
                def activate(self, *args):
                    raise ValueError("Rejected tensors")

                def probabilities(self, *args):
                    raise AssertionError("Prediction ran after activation failed")

            engine = AdapterEngine(root / "releases", Failing())
            prepared = engine.prepare(
                {"model": release["release_id"], "state": {}, "questions": {"q": {"type": "noul"}}}
            )
            with self.assertRaises(DecisionError) as error:
                engine.predict(prepared)
            self.assertEqual(error.exception.status, 503)
