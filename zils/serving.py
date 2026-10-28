"""Signed messages for miner-served inference: endpoint manifest, request and answer."""

import hashlib
import json
import re
import uuid

from bittensor_wallet import Keypair

from . import models, protocol

VERSION = "zils-serving/v1"
FRESH_SECONDS = 120
MAX_MANIFEST_BYTES = 65536

_FIELDS = {
    "manifest": {"version", "hotkey", "endpoint", "issued_at", "models"},
    "request": {"caller", "audience", "timestamp", "nonce", "model", "weights_sha256", "body"},
    "answer": {"request_sha256", "model", "weights_sha256", "predictions"},
}
# Payload field naming the signer, and the timestamp checked for freshness (answers have neither time nor signer field).
_SIGNER = {"manifest": "hotkey", "request": "caller"}
_CLOCK = {"manifest": "issued_at", "request": "timestamp"}
_HEX64 = re.compile(r"[a-f0-9]{64}")
_SIG = re.compile(r"[a-f0-9]{128}")


def canonical(kind, payload):
    return (
        f"{VERSION}/{kind}\0".encode()
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def manifest_body(hotkey, endpoint, served, now):
    return {
        "version": VERSION,
        "hotkey": hotkey,
        "endpoint": endpoint,
        "issued_at": now,
        "models": served,
    }


def request_payload(caller, audience, model, weights_sha256, body, now):
    return {
        "caller": caller,
        "audience": audience,
        "timestamp": now,
        "nonce": str(uuid.uuid4()),
        "model": model,
        "weights_sha256": weights_sha256,
        "body": body,
    }


def request_digest(payload):
    return hashlib.sha256(canonical("request", payload)).hexdigest()


def answer_payload(request_sha256, model, weights_sha256, predictions):
    return {
        "request_sha256": request_sha256,
        "model": model,
        "weights_sha256": weights_sha256,
        "predictions": predictions,
    }


def sign(key, kind, payload):
    return {"payload": payload, "signature": key.sign(canonical(kind, payload)).hex()}


def verify(message, kind, hotkey, now):
    try:
        if set(message) != {"payload", "signature"}:
            raise ValueError()
        payload = message["payload"]
        if set(payload) != _FIELDS[kind]:
            raise ValueError()
        if kind in _SIGNER and payload[_SIGNER[kind]] != hotkey:
            raise ValueError()
        if kind in _CLOCK:
            stamp = payload[_CLOCK[kind]]
            if type(stamp) is not int or abs(now - stamp) > FRESH_SECONDS:
                raise ValueError()
        sig = message["signature"]
        if not isinstance(sig, str) or not _SIG.fullmatch(sig):
            raise ValueError()
        if not Keypair(ss58_address=hotkey).verify(canonical(kind, payload), bytes.fromhex(sig)):
            raise ValueError()
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ValueError("invalid serving message") from None
    return payload


def validate_manifest(payload):
    """Check a verified manifest payload and return its served models."""
    served = payload.get("models")
    if payload.get("version") != VERSION:
        raise ValueError("unsupported serving version")
    address, _ = protocol.endpoint_port(payload.get("endpoint"))
    if not address.is_global or address.is_multicast:
        raise ValueError("endpoint must be a public IPv4 address")
    family = models.d3_family()
    if not isinstance(served, list) or not 1 <= len(served) <= len(family):
        raise ValueError("manifest must list 1 to %d models" % len(family))
    for item in served:
        if not isinstance(item, dict) or set(item) != {"model", "weights_sha256", "max_input_tokens"}:
            raise ValueError("invalid manifest model entry")
        if item["model"] not in family:
            raise ValueError("unsupported model")
        if not isinstance(item["weights_sha256"], str) or not _HEX64.fullmatch(item["weights_sha256"]):
            raise ValueError("weights_sha256 must be 64 hex characters")
        limit = models.spec(item["model"]).get("max_input_tokens", 2048)
        tokens = item["max_input_tokens"]
        if type(tokens) is not int or not 1 <= tokens <= limit:
            raise ValueError("max_input_tokens out of range")
    if len({item["model"] for item in served}) != len(served):
        raise ValueError("duplicate model")
    return served
