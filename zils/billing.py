"""Manual Stripe Checkout top-ups; PostgreSQL is the authority for money and replay."""

import os
import re
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlsplit

import stripe

from .api_store import identifier
from .cloud import APIError, trusted_url
from .decisions import DecisionError, decode_body

TOPUPS = (500, 2000, 5000, 10000)
WEBHOOK_LIMIT = 256 * 1024
EVENTS = {
    "checkout.session.completed",
    "checkout.session.async_payment_succeeded",
    "charge.refunded",
    "charge.dispute.created",
    "charge.dispute.updated",
    "charge.dispute.closed",
    "charge.dispute.funds_withdrawn",
    "charge.dispute.funds_reinstated",
}


def unavailable():
    return DecisionError(503, "billing_unavailable", "Billing is unavailable; please retry later.")


def invalid_event():
    return DecisionError(400, "invalid_webhook", "Payment event could not be verified.")


def stripe_id(value, prefix):
    if isinstance(value, dict):
        value = value.get("id")
    if not isinstance(value, str) or not re.fullmatch(prefix + r"[A-Za-z0-9_]{1,200}", value):
        raise invalid_event()
    return value


def checkout_url(value):
    if not isinstance(value, str):
        raise unavailable()
    parts = urlsplit(value)
    if (
        parts.scheme != "https"
        or parts.hostname != "checkout.stripe.com"
        or parts.username
        or parts.password
        or parts.port not in (None, 443)
    ):
        raise unavailable()
    return value


class Billing:
    def __init__(
        self,
        db,
        *,
        mode="off",
        secret_key=None,
        webhook_secret=None,
        return_origin=None,
        allow_live=False,
        client=None,
    ):
        if mode not in ("off", "test", "live"):
            raise ValueError("Billing mode must be off, test, or live")
        if mode == "live" and not allow_live:
            raise ValueError("Live payments require explicit operator enablement")
        if (
            mode != "off"
            and secret_key
            and not secret_key.startswith((f"sk_{mode}_", f"rk_{mode}_"))
        ):
            raise ValueError("Stripe key does not match the configured billing mode")
        if webhook_secret and not webhook_secret.startswith("whsec_"):
            raise ValueError("Stripe webhook secret must be an endpoint signing secret")
        if return_origin:
            parts = urlsplit(trusted_url(return_origin))
            if parts.path or parts.query or parts.fragment:
                raise ValueError("Billing return origin must be an exact origin without a path")
            if mode == "live" and parts.scheme != "https":
                raise ValueError("Live billing return origin requires HTTPS")
            return_origin = return_origin.rstrip("/")
        self.db, self.mode = db, mode
        self.webhook_secret, self.return_origin = webhook_secret, return_origin
        self.client = None
        if mode != "off" and secret_key and webhook_secret and return_origin:
            self.client = client or stripe.StripeClient(
                secret_key,
                max_network_retries=0,
                http_client=stripe.RequestsClient(timeout=(3, 8)),
            )

    @classmethod
    def from_env(cls, db):
        return cls(
            db,
            mode=os.environ.get("ZILS_BILLING_MODE", "off"),
            secret_key=os.environ.get("STRIPE_SECRET_KEY"),
            webhook_secret=os.environ.get("STRIPE_WEBHOOK_SECRET"),
            return_origin=os.environ.get("ZILS_BILLING_RETURN_ORIGIN"),
            allow_live=os.environ.get("ZILS_BILLING_ALLOW_LIVE") == "true",
        )

    def rpc(self, name, **values):
        try:
            return self.db.rpc("zils_billing_" + name, {"p_mode": self.mode, **values})
        except APIError as error:
            if error.status == 409 and name in ("checkout", "attach_checkout", "fulfill"):
                raise DecisionError(
                    409, "billing_conflict", "Payment conflicts with its saved state."
                ) from None
            raise unavailable() from None

    def summary(self, owner):
        owner = identifier(owner)
        summary = self.rpc("summary", p_owner=owner)
        if not isinstance(summary, dict) or summary.get("mode") != self.mode:
            raise unavailable()
        return summary

    def configured(self):
        if self.client is None:
            raise unavailable()

    def request(self, function, *args, **kwargs):
        try:
            result = function(*args, **kwargs)
            return result.to_dict() if isinstance(result, stripe.StripeObject) else result
        except stripe.StripeError:
            raise unavailable() from None

    def checkout(self, owner, body):
        self.configured()
        if (
            not isinstance(body, dict)
            or set(body) != {"amount_cents", "idempotency_key"}
            or type(body.get("amount_cents")) is not int
            or body["amount_cents"] not in TOPUPS
        ):
            raise DecisionError(422, "invalid_topup", "Choose one of the available top-up amounts.")
        try:
            key = str(uuid.UUID(body["idempotency_key"]))
        except (ValueError, TypeError, AttributeError):
            raise DecisionError(
                422, "invalid_idempotency_key", "A UUID idempotency key is required."
            ) from None
        owner = identifier(owner)
        self.summary(owner)  # Mode must match the database before creating any Stripe object.
        purchase_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"zils:billing:{self.mode}:{owner}:{key}"))
        purchase = self.rpc(
            "checkout", p_owner=owner, p_purchase=purchase_id, p_amount=body["amount_cents"]
        )
        if (
            purchase.get("owner_id") != owner
            or purchase.get("mode") != self.mode
            or purchase.get("id") != purchase_id
            or purchase.get("amount_cents") != body["amount_cents"]
        ):
            raise unavailable()
        if purchase.get("status") in ("paid", "refunded"):
            raise DecisionError(409, "checkout_paid", "This top-up has already been credited.")
        if purchase.get("status") == "expired":
            raise DecisionError(410, "checkout_expired", "Checkout expired; start a new top-up.")
        if purchase.get("session_id"):
            session = self.request(
                self.client.v1.checkout.sessions.retrieve, purchase["session_id"]
            )
            self.check_mode(session)
            if session.get("id") != purchase["session_id"]:
                raise unavailable()
            if session.get("status") == "expired":
                self.rpc("expire", p_purchase=purchase_id)
                raise DecisionError(
                    410, "checkout_expired", "Checkout expired; start a new top-up."
                )
            if session.get("status") != "open":
                raise DecisionError(
                    409,
                    "checkout_processing",
                    "Payment is processing; refresh your balance shortly.",
                )
            return {"url": checkout_url(session.get("url")), "purchase_id": purchase_id}
        # Stripe can prune idempotency keys after 24 hours. Never create a second
        # payable session for an unattached purchase after that safety window.
        try:
            age = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(purchase["created_at"].replace("Z", "+00:00"))
            ).total_seconds()
        except (KeyError, TypeError, ValueError):
            raise unavailable() from None
        if not -60 <= age < 23 * 3600:
            raise DecisionError(
                409, "checkout_pending", "This checkout requires payment reconciliation."
            )
        metadata = {"zils_purchase_id": purchase_id, "zils_owner_id": owner, "zils_mode": self.mode}
        session = self.request(
            self.client.v1.checkout.sessions.create,
            {
                "mode": "payment",
                "allowed_payment_method_types": ["card"],
                "client_reference_id": owner,
                "metadata": metadata,
                "payment_intent_data": {"metadata": metadata},
                "line_items": [
                    {
                        "quantity": 1,
                        "price_data": {
                            "currency": "usd",
                            "unit_amount": body["amount_cents"],
                            "product_data": {"name": "Zils prepaid credit"},
                        },
                    }
                ],
                "success_url": self.return_origin + "/billing?checkout=success",
                "cancel_url": self.return_origin + "/billing?checkout=cancelled",
            },
            {"idempotency_key": f"zils-topup-{self.mode}-{purchase_id}"},
        )
        self.check_mode(session)
        session_id, url = stripe_id(session, "cs_"), checkout_url(session.get("url"))
        self.rpc("attach_checkout", p_purchase=purchase_id, p_session=session_id, p_url=url)
        return {"url": url, "purchase_id": purchase_id}

    def check_mode(self, obj):
        if not isinstance(obj, dict) or obj.get("livemode") is not (self.mode == "live"):
            raise invalid_event()

    def purchase(self, session):
        metadata = session.get("metadata") or {}
        try:
            purchase_id = str(uuid.UUID(metadata["zils_purchase_id"]))
        except (ValueError, TypeError, KeyError, AttributeError):
            raise invalid_event() from None
        try:
            rows = self.db.rows(
                "zils_billing_purchases", f"id=eq.{purchase_id}&mode=eq.{self.mode}&limit=1"
            )
        except APIError:
            raise unavailable() from None
        if len(rows) != 1:
            raise invalid_event()
        purchase = rows[0]
        expected = {
            "zils_purchase_id": purchase_id,
            "zils_owner_id": purchase["owner_id"],
            "zils_mode": self.mode,
        }
        if (
            any(metadata.get(k) != v for k, v in expected.items())
            or session.get("client_reference_id") != purchase["owner_id"]
            or purchase["mode"] != self.mode
            or purchase["amount_cents"] not in TOPUPS
            or purchase.get("session_id") not in (None, session.get("id"))
        ):
            raise invalid_event()
        self.summary(purchase["owner_id"])
        return purchase, expected

    def reconcile(
        self, session_id, event_id, created, *, expected_payment=None, expected_charge=None
    ):
        session = self.request(
            self.client.v1.checkout.sessions.retrieve,
            session_id,
            {"expand": ["payment_intent.latest_charge"]},
        )
        self.check_mode(session)
        if session.get("id") != session_id or session.get("mode") != "payment":
            raise invalid_event()
        if session.get("payment_status") != "paid":
            return  # Completing a session does not establish that payment succeeded.
        purchase, metadata = self.purchase(session)
        payment = session.get("payment_intent")
        self.check_mode(payment)
        payment_id = stripe_id(payment, "pi_")
        charge = payment.get("latest_charge")
        self.check_mode(charge)
        charge_id = stripe_id(charge, "ch_")
        amount = purchase["amount_cents"]
        if (
            session.get("status") != "complete"
            or session.get("amount_total") != amount
            or session.get("currency") != "usd"
            or payment.get("status") != "succeeded"
            or payment.get("amount") != amount
            or payment.get("amount_received") != amount
            or payment.get("currency") != "usd"
            or charge.get("currency") != "usd"
            or charge.get("paid") is not True
            or type(charge.get("disputed")) is not bool
            or charge.get("amount") != amount
            or stripe_id(charge.get("payment_intent"), "pi_") != payment_id
            or (expected_payment and payment_id != expected_payment)
            or (expected_charge and charge_id != expected_charge)
            or any((payment.get("metadata") or {}).get(k) != v for k, v in metadata.items())
        ):
            raise invalid_event()
        refunded = charge.get("amount_refunded")
        if type(refunded) is not int or not 0 <= refunded <= amount:
            raise invalid_event()
        disputed = False
        if charge.get("disputed") is True:
            disputes = self.request(
                self.client.v1.disputes.list, {"payment_intent": payment_id, "limit": 100}
            )
            if disputes.get("has_more") or not disputes.get("data"):
                raise unavailable()
            for dispute in disputes["data"]:
                self.check_mode(dispute)
                if (
                    stripe_id(dispute.get("payment_intent"), "pi_") != payment_id
                    or stripe_id(dispute.get("charge"), "ch_") != charge_id
                ):
                    raise invalid_event()
                status = dispute.get("status")
                if status == "lost":
                    refunded = amount
                elif status not in ("won", "warning_closed"):
                    disputed = True
        receipt = charge.get("receipt_url")
        if receipt is not None:
            parts = urlsplit(receipt)
            if (
                parts.scheme != "https"
                or parts.hostname != "pay.stripe.com"
                or parts.username
                or parts.password
            ):
                receipt = None
        # A single DB transaction credits, reverses current refunds, and freezes
        # unresolved disputes, including when a reversal event arrives first.
        self.rpc(
            "fulfill",
            p_purchase=purchase["id"],
            p_session=session_id,
            p_payment=payment_id,
            p_amount=amount,
            p_event=event_id,
            p_receipt=receipt,
            p_refunded=refunded,
            p_disputed=disputed,
            p_event_created=created,
        )

    def webhook(self, raw, signature):
        self.configured()
        if not isinstance(raw, bytes) or len(raw) > WEBHOOK_LIMIT:
            raise invalid_event()
        try:
            stripe.Webhook.construct_event(raw, signature, self.webhook_secret, tolerance=300)
            # The SDK checks old timestamps; also reject signatures far in the future.
            timestamps = [int(part[2:]) for part in signature.split(",") if part.startswith("t=")]
            if len(timestamps) != 1 or abs(time.time() - timestamps[0]) > 300:
                raise ValueError("signature timestamp")
            event = decode_body(raw, limit=WEBHOOK_LIMIT)
        except (
            stripe.SignatureVerificationError,
            ValueError,
            TypeError,
            AttributeError,
            UnicodeError,
            DecisionError,
        ):
            raise invalid_event() from None
        self.check_mode(event)
        if event.get("account") or event.get("context"):
            raise invalid_event()  # This service handles its own Stripe account only.
        if event.get("type") not in EVENTS:
            return 200, {"received": True}
        event_id = stripe_id(event, "evt_")
        created = event.get("created")
        if type(created) is not int or created <= 0:
            raise invalid_event()
        obj = event.get("data", {}).get("object")
        self.check_mode(obj)
        if event["type"].startswith("checkout.session."):
            self.reconcile(stripe_id(obj, "cs_"), event_id, created)
        else:
            if event["type"].startswith("charge.dispute."):
                dispute = self.request(self.client.v1.disputes.retrieve, stripe_id(obj, "du_"))
                self.check_mode(dispute)
                charge_id = stripe_id(dispute.get("charge"), "ch_")
            else:
                charge_id = stripe_id(obj, "ch_")
            charge = self.request(self.client.v1.charges.retrieve, charge_id)
            self.check_mode(charge)
            if stripe_id(charge, "ch_") != charge_id:
                raise invalid_event()
            payment_id = stripe_id(charge.get("payment_intent"), "pi_")
            sessions = self.request(
                self.client.v1.checkout.sessions.list, {"payment_intent": payment_id, "limit": 2}
            )
            if sessions.get("has_more") or len(sessions.get("data", [])) != 1:
                raise invalid_event()
            self.reconcile(
                stripe_id(sessions["data"][0], "cs_"),
                event_id,
                created,
                expected_payment=payment_id,
                expected_charge=charge_id,
            )
        return 200, {"received": True}
