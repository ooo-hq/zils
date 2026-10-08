"""Signed HTTP payment events settle the real PostgreSQL ledger through Stripe SDK.

Invoked by scripts.check_queue_db after the billing migration. Only Stripe's
network boundary and Supabase session verification use local fixtures.
"""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import requests
import stripe

from tests.api_database import Database, literal
from tests.test_billing import SECRET, signed
from tests.test_decision_http import server
from zils import decision_http
from zils.api import Gateway, Registry
from zils.api_store import Store
from zils.billing import Billing
from zils.cloud import APIError


class SessionDatabase(Database):
    def __init__(self, command):
        super().__init__(command)
        self.sessions = {"billing-owner": str(uuid.uuid4()), "billing-other": str(uuid.uuid4())}
        for owner in self.sessions.values():
            self.sql(f"insert into auth.users values ({literal(owner)})")

    def user(self, token):
        if token not in self.sessions:
            raise APIError(401, "Sign in to continue.")
        return self.sessions[token]


class StripeNetwork:
    def __init__(self):
        self.sessions, self.charges, self.keys = {}, {}, {}
        self.creates = 0

    def request(self, method, url, headers, post_data=None):
        parsed = urlsplit(url)
        assert parsed.scheme == "https" and parsed.netloc == "api.stripe.com", url
        params = parse_qs(post_data if method == "post" else parsed.query)
        path = parsed.path
        if method == "post" and path == "/v1/checkout/sessions":
            self.creates += 1
            key = headers["Idempotency-Key"]
            if key in self.keys:
                result = self.sessions[self.keys[key]]
            else:
                sid = "cs_test_" + uuid.uuid4().hex
                metadata = {
                    name: params[f"metadata[{name}]"][0]
                    for name in ("zils_purchase_id", "zils_owner_id", "zils_mode")
                }
                assert params["mode"] == ["payment"]
                assert params["payment_method_types[0]"] == ["card"]
                assert params["payment_intent_data[metadata][zils_owner_id]"] == [
                    metadata["zils_owner_id"]
                ]
                result = {
                    "object": "checkout.session",
                    "id": sid,
                    "livemode": False,
                    "mode": "payment",
                    "status": "open",
                    "payment_status": "unpaid",
                    "amount_total": int(params["line_items[0][price_data][unit_amount]"][0]),
                    "currency": params["line_items[0][price_data][currency]"][0],
                    "metadata": metadata,
                    "client_reference_id": params["client_reference_id"][0],
                    "payment_intent": None,
                    "url": "https://checkout.stripe.com/c/pay/" + sid,
                }
                self.sessions[sid], self.keys[key] = result, sid
        elif method == "get" and path == "/v1/checkout/sessions":
            payment_id = params["payment_intent"][0]
            result = {
                "object": "list",
                "data": [
                    session
                    for session in self.sessions.values()
                    if (session.get("payment_intent") or {}).get("id") == payment_id
                ],
                "has_more": False,
                "url": path,
            }
        elif method == "get" and path.startswith("/v1/checkout/sessions/"):
            result = self.sessions[path.rsplit("/", 1)[-1]]
        elif method == "get" and path.startswith("/v1/charges/"):
            result = self.charges[path.rsplit("/", 1)[-1]]
        else:
            raise AssertionError(f"Unexpected Stripe request: {method} {path}")
        return json.dumps(result).encode(), 200, {"Request-Id": "req_fixture"}

    def pay(self, sid, *, refunded=0):
        session = self.sessions[sid]
        payment_id, charge_id = "pi_" + uuid.uuid4().hex, "ch_" + uuid.uuid4().hex
        charge = {
            "object": "charge",
            "id": charge_id,
            "livemode": False,
            "payment_intent": payment_id,
            "amount": session["amount_total"],
            "amount_refunded": refunded,
            "currency": "usd",
            "paid": True,
            "disputed": False,
            "receipt_url": "https://pay.stripe.com/receipts/fixture",
        }
        self.charges[charge_id] = charge
        session.update(
            status="complete",
            payment_status="paid",
            payment_intent={
                "object": "payment_intent",
                "id": payment_id,
                "livemode": False,
                "status": "succeeded",
                "amount": session["amount_total"],
                "amount_received": session["amount_total"],
                "currency": "usd",
                "metadata": session["metadata"],
                "latest_charge": charge,
            },
        )
        return charge


def run(command):
    db, network = SessionDatabase(command), StripeNetwork()
    db.sql("update zils_billing_settings set mode='test'")
    owner, other = db.sessions["billing-owner"], db.sessions["billing-other"]
    billing = Billing(
        db,
        mode="test",
        secret_key="sk_test_fixture",
        webhook_secret=SECRET,
        return_origin="https://billing-test.example",
    )
    gateway = Gateway(Store(db), Registry([]), billing=billing)

    def auth(token):
        return {"Authorization": "Bearer " + token}

    with patch.object(stripe.RequestsClient, "request", side_effect=network.request):
        with server(decision_http, gateway.dispatch, webhook=billing.webhook) as port:
            base = f"http://127.0.0.1:{port}"

            def summary(token="billing-owner"):
                response = requests.get(base + "/v1/billing", headers=auth(token), timeout=15)
                assert response.status_code == 200, response.text
                return response.json()

            def checkout(token, key):
                response = requests.post(
                    base + "/v1/billing/checkout",
                    headers=auth(token),
                    json={"amount_cents": 500, "idempotency_key": key},
                    timeout=15,
                )
                assert response.status_code == 200, response.text
                return response.json()

            def event(kind, obj, **changes):
                return {
                    "id": "evt_" + uuid.uuid4().hex,
                    "object": "event",
                    "type": kind,
                    "livemode": False,
                    "created": int(time.time()),
                    "data": {"object": obj},
                    **changes,
                }

            def webhook(value, expected=200):
                raw, headers = signed(value)
                response = requests.post(
                    base + "/v1/billing/webhook", data=raw, headers=headers, timeout=15
                )
                assert response.status_code == expected, response.text

            # API authentication and account isolation traverse the real gateway.
            for token in ("wrong-owner-session", "zils_sk_invalid"):
                response = requests.get(base + "/v1/billing", headers=auth(token), timeout=15)
                assert response.status_code == 401, response.text
            assert summary()["balance_nanos"] == "0"
            key = str(uuid.uuid4())
            purchase = checkout("billing-owner", key)
            assert checkout("billing-owner", key) == purchase
            assert network.creates == 1
            sid = purchase["url"].rsplit("/", 1)[-1]
            network.pay(sid)
            paid = event("checkout.session.completed", network.sessions[sid])
            webhook(paid)
            with ThreadPoolExecutor(max_workers=4) as workers:
                list(workers.map(webhook, [paid, paid, {**paid, "id": "evt_" + uuid.uuid4().hex}]))
            account = summary()
            assert account["balance_nanos"] == "5000000000", account
            assert account["available_nanos"] == "5000000000", account
            assert account["free_training_runs"] == 1, account
            assert len(account["transactions"]) == 1, account
            assert account["payments"][0]["id"] == purchase["purchase_id"], account
            assert summary("billing-other")["balance_nanos"] == "0"
            assert summary("billing-other")["payments"] == []

            # Reusing a client UUID as a different account creates its own purchase.
            reversed_purchase = checkout("billing-other", key)
            assert reversed_purchase["purchase_id"] != purchase["purchase_id"]
            reverse_sid = reversed_purchase["url"].rsplit("/", 1)[-1]
            charge = network.pay(reverse_sid, refunded=500)
            reversed_event = event("charge.refunded", charge)

            def observe():
                return [
                    db.rpc("zils_billing_summary", {"p_owner": other, "p_mode": "test"})[
                        "available_nanos"
                    ]
                    for _ in range(20)
                ]

            with ThreadPoolExecutor(max_workers=2) as workers:
                observed = workers.submit(observe)
                webhook(reversed_event)
                assert set(observed.result()) == {"0"}
            webhook(event("checkout.session.completed", network.sessions[reverse_sid]))
            webhook(reversed_event)
            reversed_account = summary("billing-other")
            assert reversed_account["balance_nanos"] == "0", reversed_account
            assert reversed_account["free_training_runs"] == 0, reversed_account
            assert len(reversed_account["transactions"]) == 2, reversed_account
            assert reversed_account["payments"][0]["status"] == "refunded", reversed_account
            assert summary()["balance_nanos"] == "5000000000"

            # Verified but wrong-mode events cannot touch the ledger.
            webhook({**paid, "livemode": True}, expected=400)
            assert summary()["balance_nanos"] == "5000000000"
            try:
                db.sql("update zils_billing_settings set mode='live'")
                response = requests.get(
                    base + "/v1/billing", headers=auth("billing-owner"), timeout=15
                )
                assert response.status_code == 503, response.text
                created_before = network.creates
                response = requests.post(
                    base + "/v1/billing/checkout",
                    headers=auth("billing-owner"),
                    json={"amount_cents": 500, "idempotency_key": str(uuid.uuid4())},
                    timeout=15,
                )
                assert response.status_code == 503, response.text
                assert network.creates == created_before
                live = db.rpc("zils_billing_summary", {"p_owner": owner, "p_mode": "live"})
                assert live["balance_nanos"] == "0", live
            finally:
                db.sql("update zils_billing_settings set mode='test'")
    print(
        "Signed Stripe HTTP + PostgreSQL: checkout, replay, refund ordering, account and mode isolation passed."
    )
