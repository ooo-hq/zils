"""Customer HTTP requests cross a real private-runtime HTTP boundary."""

import importlib
import importlib.util
import os
import unittest
import uuid

import requests

from tests.test_decision_http import server
from zils import decision_http
from zils.decisions import DecisionError, option_descriptions

OWNER = "11111111-1111-4111-8111-111111111111"
RELEASE = "zils-test-r1"
FINGERPRINT = "a" * 64
BODY = {
    "model": "zils-shared",
    "state": {"product": "USB cable"},
    "questions": {"match": {"type": "noul"}},
}


class FixtureStore:
    def __init__(self):
        self.admissions, self.usage = [], []
        self.limit = False

    def authenticate(self, token):
        if token != "valid-key":
            raise DecisionError(401, "invalid_credentials", "Invalid key.")
        return {"owner_id": OWNER, "id": str(uuid.uuid4())}

    def admit(self, *args):
        if self.limit:
            raise DecisionError(429, "rate_limit", "Budget reached.", retry_after=1)
        self.admissions.append(args)

    def finish_usage(self, *args):
        self.usage.append(args)


class GatewayTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("zils.api"), "gateway missing")
        return importlib.import_module("zils.api")

    def runtime(self, method, path, token, body, rid):
        self.assertEqual(token, "runtime-secret")
        identity = {
            "release_id": RELEASE,
            "fingerprint": self.fingerprint,
            "limits": {"max_pass_tokens": 4096, "max_request_tokens": 65536},
        }
        if path == "/v1/prepare":
            prepared = {**identity, "reserved_tokens": 100 * len(body["request"]["questions"])}
            if hasattr(self, "billable_tokens"):
                prepared["billable_tokens"] = self.billable_tokens
            return 200, prepared
        self.executions += 1
        predictions = {}
        for key, q in body["request"]["questions"].items():
            opts = option_descriptions(q)
            predictions[key] = {
                "input_tokens": getattr(self, "resource_tokens", 10),
                "probabilities": {name: 1 / len(opts) for name in opts},
            }
        return 200, {**identity, "predictions": predictions}

    def setUp(self):
        self.fingerprint, self.executions = FINGERPRINT, 0
        os.environ["ZILS_TEST_RUNTIME_TOKEN"] = "runtime-secret"

    def tearDown(self):
        os.environ.pop("ZILS_TEST_RUNTIME_TOKEN", None)

    def metered_gateway(self, store, port):
        m = self.module()
        return m.Gateway(
            store,
            m.Registry(
                [
                    {
                        "id": RELEASE,
                        "fingerprint": FINGERPRINT,
                        "aliases": ["zils-shared"],
                        "owners": None,
                        "url": f"http://127.0.0.1:{port}",
                        "token_env": "ZILS_TEST_RUNTIME_TOKEN",
                        "release_date": "2026-10-04",
                        "description": "Fixture",
                    }
                ]
            ),
        )

    def test_billable_input_is_public_and_admitted_separately_from_resource_tokens(self):
        store = FixtureStore()
        self.billable_tokens = 73
        with server(decision_http, self.runtime) as port:
            gateway = self.metered_gateway(store, port)
            for lane in ("realtime", "bulk"):
                with self.subTest(lane=lane):
                    request_id = str(uuid.uuid4())
                    result = gateway.evaluate(OWNER, OWNER, BODY, request_id, lane=lane)
                    self.assertEqual(
                        store.admissions[-1],
                        (OWNER, OWNER, request_id, 100, 73, RELEASE, "zils-shared"),
                    )
                    self.assertEqual(
                        result["usage"],
                        {
                            "input_tokens": 10,
                            "output_tokens": 0,
                            "billable_input_tokens": 73,
                        },
                    )
            self.assertEqual(len(store.usage), 1)
            self.assertEqual(store.usage[0][1:], (10, "completed"))

    def test_malformed_billable_meter_fails_before_admission_or_inference(self):
        store = FixtureStore()
        with server(decision_http, self.runtime) as port:
            gateway = self.metered_gateway(store, port)
            for value in (None, True, False, -1, 0, 1.5, "73", [], {}, 2**31 + 1):
                with self.subTest(value=value):
                    self.billable_tokens = value
                    with self.assertRaises(DecisionError) as error:
                        gateway.evaluate(OWNER, OWNER, BODY, str(uuid.uuid4()))
                    self.assertEqual(error.exception.status, 502)
            self.assertEqual(store.admissions, [])
            self.assertEqual(store.usage, [])
            self.assertEqual(self.executions, 0)

    def test_billable_meter_does_not_replace_model_resource_limit(self):
        store = FixtureStore()
        self.billable_tokens, self.resource_tokens = 1000, 101
        with server(decision_http, self.runtime) as port:
            gateway = self.metered_gateway(store, port)
            request_id = str(uuid.uuid4())
            with self.assertRaises(DecisionError) as error:
                gateway.evaluate(OWNER, OWNER, BODY, request_id)
            self.assertEqual(error.exception.code, "token_accounting_error")
            self.assertEqual(store.usage, [(request_id, None, "failed")])

    def test_missing_billable_meter_cannot_execute_when_database_requires_billing(self):
        from zils.api_store import Store

        class BillingDB:
            def rpc(self, name, values):
                assert name == "zils_api_admit"
                assert "p_billable_tokens" not in values
                return "billing_meter_unavailable"

        with server(decision_http, self.runtime) as port:
            gateway = self.metered_gateway(Store(BillingDB()), port)
            with self.assertRaises(DecisionError) as error:
                gateway.evaluate(OWNER, OWNER, BODY, str(uuid.uuid4()))
            self.assertEqual(error.exception.status, 503)
            self.assertEqual(self.executions, 0)

    def test_alias_release_acl_usage_and_limits(self):
        m = self.module()
        store = FixtureStore()
        with server(decision_http, self.runtime) as runtime_port:
            entry = {
                "id": RELEASE,
                "fingerprint": FINGERPRINT,
                "aliases": ["zils-shared"],
                "owners": None,
                "url": f"http://127.0.0.1:{runtime_port}",
                "token_env": "ZILS_TEST_RUNTIME_TOKEN",
                "release_date": "2026-10-04",
                "description": "Fixture",
            }
            registry = m.Registry(
                [entry, {**entry, "id": "private-r1", "aliases": [], "owners": [str(uuid.uuid4())]}]
            )
            gateway = m.Gateway(store, registry)
            with server(decision_http, gateway.dispatch) as port:
                url, headers = f"http://127.0.0.1:{port}", {"Authorization": "Bearer valid-key"}
                models = requests.get(url + "/v1/models", headers=headers).json()["models"]
                self.assertEqual({x["name"] for x in models}, {RELEASE, "zils-shared"})
                r = requests.post(url + "/v1/systemone", json=BODY, headers=headers)
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()["model"], RELEASE)
                self.assertEqual(r.json()["answers"]["match"]["noul"], 0.5)
                self.assertEqual(store.admissions[0][3], 100)
                self.assertEqual(store.usage[0][1:], (10, "completed"))
                for token, model, status in [
                    ("invalid", RELEASE, 401),
                    ("valid-key", "private-r1", 404),
                    ("valid-key", "missing", 404),
                ]:
                    r = requests.post(
                        url + "/v1/systemone",
                        json={**BODY, "model": model},
                        headers={"Authorization": "Bearer " + token},
                    )
                    self.assertEqual(r.status_code, status)
                store.limit = True
                r = requests.post(url + "/v1/systemone", json=BODY, headers=headers)
                self.assertEqual(r.status_code, 429)
                self.assertEqual(r.headers["Retry-After"], "1")
                self.assertEqual(self.executions, 1)
                store.limit = False
                self.fingerprint = "b" * 64
                r = requests.post(url + "/v1/systemone", json=BODY, headers=headers)
                self.assertEqual(r.status_code, 503)
                self.assertEqual(self.executions, 1)

    def test_many_questions_and_structured_score_legend(self):
        m = self.module()
        store = FixtureStore()
        with server(decision_http, self.runtime) as port:
            entry = {
                "id": RELEASE,
                "fingerprint": FINGERPRINT,
                "aliases": ["zils-shared"],
                "owners": None,
                "url": f"http://127.0.0.1:{port}",
                "token_env": "ZILS_TEST_RUNTIME_TOKEN",
                "release_date": "2026-10-04",
                "description": "Fixture",
            }
            gateway = m.Gateway(store, m.Registry([entry]))
            qs = {
                str(i): {"type": "choice", "criteria": {str(n): None for n in range(255)}}
                for i in range(12)
            }
            qs["score"] = {"type": "score", "criteria": [{"level": "low"}, ["high"]]}
            code, result = gateway.dispatch(
                "POST", "/v1/systemone", "valid-key", {**BODY, "questions": qs}, str(uuid.uuid4())
            )
            self.assertEqual(code, 200)
            self.assertEqual(len(result["answers"]), 13)
            self.assertEqual(
                result["answers"]["score"]["legend"], {"0": {"level": "low"}, "1": ["high"]}
            )

    def test_private_envelope_preserves_utf8_and_depth_budget(self):
        m = self.module()
        from zils.decisions import MAX_BODY, MAX_DEPTH

        with server(
            decision_http, self.runtime, body_limit=MAX_BODY + 1024, max_depth=MAX_DEPTH + 1
        ) as port:
            entry = {
                "id": RELEASE,
                "fingerprint": FINGERPRINT,
                "aliases": ["zils-shared"],
                "owners": None,
                "url": f"http://127.0.0.1:{port}",
                "token_env": "ZILS_TEST_RUNTIME_TOKEN",
                "release_date": "2026-10-04",
                "description": "Fixture",
            }
            gateway = m.Gateway(FixtureStore(), m.Registry([entry]))
            nested = "x"
            for _ in range(31):
                nested = [nested]
            for state in ("€" * 200000, nested):
                status, result = gateway.dispatch(
                    "POST",
                    "/v1/systemone",
                    "valid-key",
                    {**BODY, "state": state},
                    str(uuid.uuid4()),
                )
                self.assertEqual(status, 200)
                self.assertEqual(result["model"], RELEASE)
