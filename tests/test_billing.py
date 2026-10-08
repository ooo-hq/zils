"""Signed Stripe fixtures cross the real HTTP transport; SDK requests stay local."""

import copy
import hashlib
import hmac
import json
import socket
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from urllib.parse import parse_qs

import requests
import stripe

from tests.test_decision_http import server
from zils import decision_http
from zils.api import Gateway, Registry
from zils.api_store import Store
from zils.billing import Billing
from zils.cloud import APIError
from zils.decisions import DecisionError

OWNER = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"
KEY = "33333333-3333-4333-8333-333333333333"
SECRET = "whsec_fixture_secret"
URL = "https://checkout.stripe.com/c/pay/cs_test_fixture"


class Database:
    def __init__(self):
        self.purchases, self.calls = {}, []
        self.mode = "test"

    def user(self, token):
        if token != "supabase-session":
            raise APIError(401, "Sign in to continue.")
        return OWNER

    def rows(self, table, query):
        assert table == "zils_billing_purchases"
        return [p for p in self.purchases.values() if f"id=eq.{p['id']}" in query]

    def rpc(self, name, values):
        self.calls.append((name, copy.deepcopy(values)))
        if values.get("p_mode") != self.mode:
            raise APIError(409, "Mode mismatch.")
        if name == "zils_billing_summary":
            return {
                "mode": self.mode,
                "currency": "usd",
                "balance_nanos": "0",
                "reserved_nanos": "0",
                "available_nanos": "0",
                "free_training_runs": 0,
                "topup_amounts_cents": [500, 2000, 5000, 10000],
                "transactions": [],
                "payments": [],
            }
        if name == "zils_billing_checkout":
            purchase = self.purchases.setdefault(
                values["p_purchase"],
                {
                    "id": values["p_purchase"],
                    "owner_id": values["p_owner"],
                    "mode": values["p_mode"],
                    "amount_cents": values["p_amount"],
                    "session_id": None,
                    "checkout_url": None,
                    "payment_id": None,
                    "status": "pending",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            if purchase["amount_cents"] != values["p_amount"]:
                raise APIError(409, "Purchase conflict.")
            return copy.deepcopy(purchase)
        if name == "zils_billing_attach_checkout":
            self.purchases[values["p_purchase"]].update(
                session_id=values["p_session"], checkout_url=values["p_url"]
            )
        if name == "zils_billing_expire":
            self.purchases[values["p_purchase"]]["status"] = "expired"
        if name == "zils_billing_fulfill":
            self.purchases[values["p_purchase"]].update(
                status="paid", payment_id=values["p_payment"]
            )


def signed(event, timestamp=None, secret=SECRET):
    raw = json.dumps(event, ensure_ascii=False, indent=2).encode()
    timestamp = int(time.time()) if timestamp is None else timestamp
    digest = hmac.new(
        secret.encode(), str(timestamp).encode() + b"." + raw, hashlib.sha256
    ).hexdigest()
    return raw, {
        "Content-Type": "application/json",
        "Stripe-Signature": f"t={timestamp},v1={digest}",
    }


class BillingTest(unittest.TestCase):
    def setUp(self):
        self.db = Database()
        self.client = Mock()
        self.billing = Billing(
            self.db,
            mode="test",
            secret_key="sk_test_fixture",
            webhook_secret=SECRET,
            return_origin="https://app.example",
            client=self.client,
        )
        self.purchase = None
        self.client.v1.checkout.sessions.create.side_effect = self.create_session

    def create_session(self, params, options):
        self.purchase = params["metadata"]["zils_purchase_id"]
        return {
            "id": "cs_test_fixture",
            "url": URL,
            "livemode": False,
            "status": "open",
            "metadata": params["metadata"],
            "mode": "payment",
        }

    def checkout(self, owner=OWNER, amount=500):
        return self.billing.checkout(owner, {"amount_cents": amount, "idempotency_key": KEY})

    def paid(self):
        self.checkout()
        metadata = {"zils_purchase_id": self.purchase, "zils_owner_id": OWNER, "zils_mode": "test"}
        charge = {
            "id": "ch_fixture",
            "object": "charge",
            "livemode": False,
            "payment_intent": "pi_fixture",
            "amount": 500,
            "amount_refunded": 0,
            "currency": "usd",
            "paid": True,
            "disputed": False,
            "receipt_url": "https://pay.stripe.com/receipts/fixture",
            "metadata": metadata,
        }
        payment = {
            "id": "pi_fixture",
            "object": "payment_intent",
            "livemode": False,
            "status": "succeeded",
            "amount": 500,
            "amount_received": 500,
            "currency": "usd",
            "metadata": metadata,
            "latest_charge": charge,
        }
        session = {
            "id": "cs_test_fixture",
            "object": "checkout.session",
            "livemode": False,
            "status": "complete",
            "payment_status": "paid",
            "mode": "payment",
            "amount_total": 500,
            "currency": "usd",
            "metadata": metadata,
            "client_reference_id": OWNER,
            "payment_intent": payment,
        }
        self.client.v1.checkout.sessions.retrieve.return_value = session
        self.client.v1.checkout.sessions.list.return_value = {"data": [session], "has_more": False}
        self.client.v1.charges.retrieve.return_value = charge
        self.client.v1.disputes.list.return_value = {"data": [], "has_more": False}
        return session, payment, charge

    def event(self, kind="checkout.session.completed", obj=None, **changes):
        if obj is None:
            obj = self.paid()[0]
        return {
            "id": "evt_fixture",
            "object": "event",
            "type": kind,
            "livemode": False,
            "created": int(time.time()),
            "data": {"object": obj},
            **changes,
        }

    def post(self, port, event, **signing):
        raw, headers = signed(event, **signing)
        return requests.post(
            f"http://127.0.0.1:{port}/v1/billing/webhook", data=raw, headers=headers
        )

    def fulfillments(self):
        return [v for n, v in self.db.calls if n == "zils_billing_fulfill"]

    def test_checkout_is_fixed_and_scoped_idempotently(self):
        first = self.checkout()
        self.client.v1.checkout.sessions.retrieve.return_value = {
            "status": "open",
            "livemode": False,
            "id": "cs_test_fixture",
            "url": URL,
        }
        self.assertEqual(self.checkout(), first)
        self.assertEqual(first, {"url": URL, "purchase_id": self.purchase})
        self.assertEqual(self.client.v1.checkout.sessions.create.call_count, 1)
        params, options = self.client.v1.checkout.sessions.create.call_args.args
        self.assertEqual(params.get("allowed_payment_method_types"), ["card"])
        self.assertNotIn("payment_method_types", params)
        self.assertEqual(params["line_items"][0]["price_data"]["unit_amount"], 500)
        self.assertEqual(params["success_url"], "https://app.example/billing?checkout=success")
        self.assertEqual(params["cancel_url"], "https://app.example/billing?checkout=cancelled")
        self.assertEqual(params["payment_intent_data"]["metadata"], params["metadata"])
        self.assertEqual(params["client_reference_id"], OWNER)
        self.assertIn(first["purchase_id"], options["idempotency_key"])
        with self.assertRaises(DecisionError) as error:
            self.checkout(amount=2000)
        self.assertEqual(error.exception.status, 409)
        self.checkout(owner=OTHER)
        self.assertEqual(len(self.db.purchases), 2)

    def test_customer_routes_require_supabase_session(self):
        gateway = Gateway(Store(self.db), Registry([]), billing=self.billing)
        with server(decision_http, gateway.dispatch, webhook=self.billing.webhook) as port:
            base = f"http://127.0.0.1:{port}"
            for token in (None, "zils_sk_123", "invalid"):
                headers = {"Authorization": "Bearer " + token} if token else {}
                self.assertEqual(
                    requests.get(base + "/v1/billing", headers=headers).status_code, 401
                )
            headers = {"Authorization": "Bearer supabase-session"}
            response = requests.get(base + "/v1/billing", headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["balance_nanos"], "0")
            for body in (
                {"amount_cents": True, "idempotency_key": KEY},
                {"amount_cents": 1, "idempotency_key": KEY},
                {"amount_cents": 500, "idempotency_key": "bad"},
                {"amount_cents": 500, "idempotency_key": KEY, "owner_id": OTHER},
            ):
                response = requests.post(base + "/v1/billing/checkout", json=body, headers=headers)
                self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(self.client.v1.checkout.sessions.create.call_count, 0)

    def test_signature_raw_bytes_staleness_mode_and_exact_route(self):
        event = self.event()
        with server(decision_http, lambda *a: (200, {}), webhook=self.billing.webhook) as port:
            for signing in (
                {"secret": "wrong"},
                {"timestamp": int(time.time()) - 301},
                {"timestamp": int(time.time()) + 301},
            ):
                self.assertEqual(self.post(port, event, **signing).status_code, 400)
            raw, headers = signed(event)
            response = requests.post(
                f"http://127.0.0.1:{port}/v1/billing/webhook", data=raw + b" ", headers=headers
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(self.post(port, {**event, "livemode": True}).status_code, 400)
            self.assertEqual(self.post(port, {**event, "account": "acct_other"}).status_code, 400)
            self.assertEqual(self.fulfillments(), [])
            for path in ("/v1/billing/webhook?x=1", "/v1/billing/webhook/", "/v1/systemone"):
                self.assertEqual(
                    requests.post(
                        f"http://127.0.0.1:{port}" + path, data=raw, headers=headers
                    ).status_code,
                    401,
                )
            self.assertEqual(self.post(port, event).status_code, 200)
            self.assertEqual(self.fulfillments()[0]["p_payment"], "pi_fixture")

    def test_paid_only_and_metadata_amount_currency_checks(self):
        for field, value in (
            ("payment_status", "unpaid"),
            ("amount_total", 501),
            ("currency", "eur"),
            ("client_reference_id", OTHER),
            ("livemode", True),
        ):
            with self.subTest(field=field):
                self.setUp()
                session, _, _ = self.paid()
                session[field] = value
                raw, headers = signed(self.event(obj=session))
                try:
                    self.billing.webhook(raw, headers["Stripe-Signature"])
                except DecisionError:
                    pass
                self.assertEqual(self.fulfillments(), [])
        self.setUp()
        session, payment, _ = self.paid()
        payment["metadata"] = {**payment["metadata"], "zils_owner_id": OTHER}
        with self.assertRaises(DecisionError):
            raw, headers = signed(self.event(obj=session))
            self.billing.webhook(raw, headers["Stripe-Signature"])
        self.assertEqual(self.fulfillments(), [])

    def test_refund_before_completion_uses_latest_atomic_reversal(self):
        session, _, charge = self.paid()
        charge["amount_refunded"] = 300
        stale = {**charge, "amount_refunded": 100}
        event = self.event("charge.refunded", stale)
        with server(decision_http, lambda *a: (200, {}), webhook=self.billing.webhook) as port:
            for fixture in (event, event, self.event(obj=session)):
                response = self.post(port, fixture)
                self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(all(row["p_refunded"] == 300 for row in self.fulfillments()))
        self.assertEqual(self.client.v1.charges.retrieve.call_count, 2)

    def test_dispute_latest_state_freezes_or_reverses_lost_payment(self):
        session, _, charge = self.paid()
        charge["disputed"] = True
        dispute = {
            "id": "du_fixture",
            "object": "dispute",
            "livemode": False,
            "payment_intent": "pi_fixture",
            "charge": "ch_fixture",
            "status": "needs_response",
        }
        self.client.v1.disputes.list.return_value = {"data": [dispute], "has_more": False}
        self.client.v1.disputes.retrieve.return_value = dispute
        for status, frozen, reversal in (
            ("needs_response", True, 0),
            ("won", False, 0),
            ("lost", False, 500),
        ):
            dispute["status"] = status
            raw, headers = signed(self.event("charge.dispute.updated", dispute))
            self.billing.webhook(raw, headers["Stripe-Signature"])
            self.assertEqual(self.fulfillments()[-1]["p_disputed"], frozen)
            self.assertEqual(self.fulfillments()[-1]["p_refunded"], reversal)
            self.assertGreater(self.fulfillments()[-1]["p_event_created"], 0)

    def test_configuration_mode_failures_and_safe_sdk_errors(self):
        for settings in (
            {"mode": "live", "secret_key": "sk_live_secret"},
            {"mode": "test", "secret_key": "sk_live_secret"},
            {"mode": "test", "return_origin": "https://app.example/evil"},
        ):
            with self.assertRaises(ValueError):
                Billing(self.db, webhook_secret=SECRET, **settings)
        off = Billing(self.db)
        self.db.mode = "off"
        self.assertEqual(off.summary(OWNER)["mode"], "off")
        with self.assertRaises(DecisionError) as error:
            off.checkout(OWNER, {"amount_cents": 500, "idempotency_key": KEY})
        self.assertEqual(error.exception.status, 503)
        self.db.mode = "live"
        with self.assertRaises(DecisionError) as error:
            self.checkout()
        self.assertEqual(error.exception.status, 503)
        self.client.v1.checkout.sessions.create.assert_not_called()
        self.db.mode = "test"
        self.client.v1.checkout.sessions.create.side_effect = stripe.APIConnectionError(
            "private secret"
        )
        with self.assertRaises(DecisionError) as error:
            self.checkout()
        self.assertEqual(error.exception.status, 503)
        self.assertNotIn("private secret", str(error.exception))

    def test_expired_checkout_and_unsafe_checkout_url(self):
        self.checkout()
        self.client.v1.checkout.sessions.retrieve.return_value = {
            "id": "cs_test_fixture",
            "livemode": False,
            "status": "expired",
            "url": None,
        }
        with self.assertRaises(DecisionError) as error:
            self.checkout()
        self.assertEqual(error.exception.status, 410)
        self.assertEqual(self.db.purchases[self.purchase]["status"], "expired")
        self.setUp()
        self.client.v1.checkout.sessions.create.side_effect = None
        self.client.v1.checkout.sessions.create.return_value = {
            "id": "cs_test_fixture",
            "livemode": False,
            "url": "https://evil.example/pay",
        }
        with self.assertRaises(DecisionError):
            self.checkout()

    def test_real_sdk_objects_are_recursively_decoded(self):
        session, _, _ = self.paid()
        self.client.v1.checkout.sessions.retrieve.return_value = (
            stripe.checkout.Session.construct_from(session, "sk_test_fixture")
        )
        raw, headers = signed(self.event(obj=session))
        self.billing.webhook(raw, headers["Stripe-Signature"])
        self.assertEqual(self.fulfillments()[0]["p_amount"], 500)

    def test_real_sdk_checkout_serialization_at_network_boundary(self):
        self.billing = Billing(
            self.db,
            mode="test",
            secret_key="sk_test_fixture",
            webhook_secret=SECRET,
            return_origin="https://app.example",
        )
        result = {
            "object": "checkout.session",
            "id": "cs_test_fixture",
            "livemode": False,
            "url": URL,
        }
        with patch.object(
            stripe.RequestsClient, "request", return_value=(json.dumps(result), 200, {})
        ) as network:
            self.checkout()
        args, _ = network.call_args
        self.assertEqual(args[:2], ("post", "https://api.stripe.com/v1/checkout/sessions"))
        params = parse_qs(args[3])
        self.assertEqual(params["line_items[0][price_data][unit_amount]"], ["500"])
        self.assertEqual(params["payment_intent_data[metadata][zils_owner_id]"], [OWNER])
        self.assertEqual(params["allowed_payment_method_types[0]"], ["card"])
        self.assertNotIn("payment_method_types[0]", params)

    def test_failed_payment_ignored_and_webhook_upstream_failure_retries(self):
        event = self.event("checkout.session.async_payment_failed")
        raw, headers = signed(event)
        self.assertEqual(self.billing.webhook(raw, headers["Stripe-Signature"])[0], 200)
        self.assertEqual(self.fulfillments(), [])
        self.client.v1.checkout.sessions.retrieve.side_effect = stripe.APIConnectionError(
            "private secret"
        )
        event["type"] = "checkout.session.completed"
        with server(decision_http, lambda *a: (200, {}), webhook=self.billing.webhook) as port:
            response = self.post(port, event)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private secret", response.text)

    def test_webhook_duplicate_signature_and_oversized_body_rejected(self):
        event = self.event()
        raw, headers = signed(event)
        with server(decision_http, lambda *a: (200, {}), webhook=self.billing.webhook) as port:
            response = requests.post(
                f"http://127.0.0.1:{port}/v1/billing/webhook",
                data=raw,
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(response.status_code, 400)
            for extra, length, status in (
                ("Stripe-Signature: " + headers["Stripe-Signature"] + "\r\n", len(raw), 400),
                ("", 256 * 1024 + 1, 413),
            ):
                with socket.create_connection(("127.0.0.1", port)) as connection:
                    request = (
                        "POST /v1/billing/webhook HTTP/1.1\r\nHost: localhost\r\n"
                        "Content-Type: application/json\r\n"
                        f"Stripe-Signature: {headers['Stripe-Signature']}\r\n"
                        f"{extra}Content-Length: {length}\r\n\r\n"
                    ).encode() + raw
                    connection.sendall(request)
                    connection.shutdown(socket.SHUT_WR)
                    response = b""
                    while chunk := connection.recv(65536):
                        response += chunk
                    self.assertIn(f" {status} ".encode(), response.split(b"\r\n")[0])
        self.assertEqual(self.fulfillments(), [])


if __name__ == "__main__":
    unittest.main()
