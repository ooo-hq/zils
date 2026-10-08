"""Image requests share authentication/usage while keeping model and asset tenant boundaries."""

import copy
import importlib.util
import json
import os
import unittest
import uuid
from unittest.mock import patch

import requests

from tests.test_api import OWNER, FixtureStore
from tests.test_decision_http import server
from tests.test_image_contract import BODY
from zils import api, decision_http
from zils.decisions import DecisionError
from zils.image_contract import IMAGE_CAPABILITIES

OTHER = "22222222-2222-4222-8222-222222222222"
ASSET = BODY["images"][0]["asset_id"]


class Accounts(FixtureStore):
    def __init__(self):
        super().__init__()
        self.sessions = []
        self.disabled = False

    def session_owner(self, token):
        self.sessions.append(token)
        if token != "owner-session":
            raise DecisionError(401, "invalid_credentials", "Invalid session")
        return OWNER

    def ensure_account(self, owner):
        if self.disabled:
            raise DecisionError(403, "account_disabled", "Account unavailable")

    def authenticate(self, token):
        if token != "zils_sk_valid":
            raise DecisionError(401, "invalid_credentials", "Revoked key")
        self.ensure_account(OWNER)
        return {"owner_id": OWNER, "id": "33333333-3333-4333-8333-333333333333"}


class Assets:
    def __init__(self):
        self.owner = OWNER
        self.expired = False
        self.grants = 0
        self.deleted = []

    def read_reference(self, owner, asset_id, purpose=None):
        if owner != self.owner or asset_id != ASSET or self.expired:
            raise DecisionError(404, "not_found", "Image unavailable")
        self.grants += 1
        return {
            "id": ASSET,
            "sha256": "b" * 64,
            "url": "https://storage.example/image?grant=" + str(self.grants),
        }

    def delete_unused(self, owner, asset_id):
        self.read_reference(owner, asset_id)
        self.deleted.append(asset_id)


class ImageApiTest(unittest.TestCase):
    def setUp(self):
        self.accounts, self.assets = Accounts(), Assets()
        self.calls = []
        self.fail_predict = False
        self.tokens = 442
        self.env = patch.dict(
            os.environ, {"ZILS_IMAGES_ENABLED": "1", "IMAGE_TEST_TOKEN": "secret"}
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def gateway(self):
        self.assertIsNotNone(importlib.util.find_spec("zils.image_api"), "image gateway missing")

        def runtime(method, path, bearer, body, rid):
            self.assertEqual(bearer, "secret")
            self.calls.append(copy.deepcopy(body))
            self.assertEqual(body["fingerprint"], "a" * 64)
            self.assertEqual(body["image"]["sha256"], "b" * 64)
            identity = {"release_id": "image-release", "fingerprint": "a" * 64}
            if path == "/v1/prepare":
                return 200, {**identity, "reserved_tokens": 442}
            if self.fail_predict:
                return 422, {"error": {"code": "expired_image"}}
            return 200, {
                **identity,
                "predictions": {
                    "inspection": {
                        "probabilities": {"normal": 0.2, "damaged": 0.3, "__unknown__": 0.5},
                        "input_tokens": self.tokens,
                    }
                },
            }

        context = server(decision_http, runtime)
        port = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        entry = {
            "id": "image-release",
            "fingerprint": "a" * 64,
            "aliases": ["images"],
            "owners": None,
            "url": f"http://127.0.0.1:{port}",
            "token_env": "IMAGE_TEST_TOKEN",
            "release_date": "2026-10-08",
            "description": "Image fixture",
            "capabilities": IMAGE_CAPABILITIES,
        }
        registry = api.Registry(
            [
                entry,
                {**entry, "id": "private-image", "aliases": [], "owners": [OTHER]},
                {
                    k: v
                    for k, v in {**entry, "id": "text-release", "aliases": []}.items()
                    if k != "capabilities"
                },
            ]
        )
        return api.Gateway(self.accounts, registry, image_store=self.assets)

    def call(self, gateway, body=BODY, token="owner-session", path="/v1/image-decisions"):
        return gateway.dispatch("POST", path, token, copy.deepcopy(body), str(uuid.uuid4()))

    def test_session_prediction_has_no_key_and_preserves_unknown(self):
        gateway = self.gateway()
        status, result = self.call(gateway)
        self.assertEqual(status, 200)
        self.assertTrue(result["answers"]["inspection"]["abstained"])
        self.assertIsNone(self.accounts.admissions[0][1])
        self.assertEqual(self.accounts.admissions[0][-1], 442)
        self.assertEqual(self.calls[0]["image"], self.calls[1]["image"])
        self.assertEqual(self.accounts.usage[-1][1:], (442, "completed"))
        status, _ = self.call(gateway, token="zils_sk_valid", path="/v1/systemone")
        self.assertEqual(status, 200)

    def test_cross_owner_expired_and_forged_inputs_never_admit_usage(self):
        gateway = self.gateway()
        for mutate, body, want in [
            (lambda: setattr(self.assets, "owner", OTHER), BODY, 404),
            (lambda: setattr(self.assets, "owner", OWNER), {**BODY, "model": "private-image"}, 404),
            (lambda: setattr(self.assets, "expired", True), BODY, 404),
            (lambda: setattr(self.assets, "expired", False), {**BODY, "owner": OTHER}, 422),
            (lambda: None, {**BODY, "model": "text-release"}, 422),
            (lambda: None, {k: v for k, v in BODY.items() if k != "images"}, 422),
        ]:
            mutate()
            with self.subTest(body=body), self.assertRaises(DecisionError) as ctx:
                self.call(gateway, body)
            self.assertEqual(ctx.exception.status, want)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.accounts.admissions, [])

    def test_expiry_after_prepare_records_failure_and_retry_gets_new_grant(self):
        gateway = self.gateway()
        self.fail_predict = True
        with self.assertRaises(DecisionError):
            self.call(gateway)
        self.assertEqual(self.accounts.usage[-1][1:], (None, "failed"))
        first = self.calls[-1]["image"]
        self.fail_predict = False
        self.call(gateway)
        self.assertNotEqual(first["url"], self.calls[-1]["image"]["url"])
        self.assertEqual(first["sha256"], self.calls[-1]["image"]["sha256"])
        self.tokens = 443
        with self.assertRaises(DecisionError):
            self.call(gateway)
        self.assertEqual(self.accounts.usage[-1][1:], (None, "failed"))

    def test_revoked_key_never_falls_back_to_browser_session_and_flags_deny(self):
        gateway = self.gateway()
        with self.assertRaises(DecisionError):
            gateway.dispatch("DELETE", "/v1/image-assets/" + ASSET, "zils_sk_revoked", {}, "test")
        self.assertEqual(self.accounts.sessions, [])
        self.accounts.disabled = True
        with self.assertRaises(DecisionError):
            self.call(gateway)
        self.accounts.disabled = False
        with patch.dict(os.environ, {"ZILS_IMAGES_ENABLED": "0"}), self.assertRaises(DecisionError):
            self.call(gateway)
        self.assertEqual(self.calls, [])

    def test_delete_cors_is_explicit_and_bulk_images_are_rejected_before_queueing(self):
        gateway = self.gateway()
        with server(
            decision_http,
            gateway.dispatch,
            origin="https://dashboard.example",
            allowed_methods=("GET", "POST", "DELETE"),
        ) as port:
            response = requests.delete(
                f"http://127.0.0.1:{port}/v1/image-assets/{ASSET}",
                headers={
                    "Authorization": "Bearer owner-session",
                    "Origin": "https://dashboard.example",
                },
                timeout=3,
            )
            self.assertEqual(response.status_code, 204)
            self.assertIn("DELETE", response.headers["Access-Control-Allow-Methods"])
            self.assertEqual(self.assets.deleted, [ASSET])
        from zils.batches import parse_input

        with self.assertRaises(DecisionError) as ctx:
            parse_input((json.dumps({"custom_id": "image", "body": BODY}) + "\n").encode(), {})
        self.assertEqual(ctx.exception.code, "image_bulk_unsupported")


class ImageBatchAdmissionTest(unittest.TestCase):
    def test_image_upload_cannot_transition_to_queued_or_reserve_usage(self):
        from pathlib import Path

        from zils.batches import Batches

        class Storage:
            def __init__(self):
                self.rpc_calls = []

            def rows(self, table, query):
                return [{"id": ASSET, "status": "uploading", "input_path": "private/input.jsonl"}]

            def exists(self, *args):
                return True

            def download(self, bucket, path, destination, limit):
                Path(destination).write_text(
                    json.dumps({"custom_id": "photo", "body": BODY}) + "\n"
                )

            def rpc(self, *args):
                self.rpc_calls.append(args)

        class Catalog:
            def listing(self, owner):
                return {"models": []}

        db = Storage()
        with self.assertRaises(DecisionError) as ctx:
            Batches(db).dispatch("POST", "/v1/batches/" + ASSET + "/submit", OWNER, {}, Catalog())
        self.assertEqual(ctx.exception.code, "image_bulk_unsupported")
        self.assertEqual(db.rpc_calls, [])
