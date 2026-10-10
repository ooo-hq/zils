"""Hotkey authentication for the training queue, bound to one service and request."""

import json
import threading
import time
import uuid
from contextlib import contextmanager

from bittensor_wallet import Keypair

from .cloud import APIError


def canonical(payload):
    return (
        b"fez-training-queue/v1\0"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def sign(key, audience, path, body):
    payload = {
        "hotkey": key.ss58_address,
        "audience": audience,
        "path": path,
        "timestamp": int(time.time()),
        "nonce": str(uuid.uuid4()),
        "body": body,
    }
    return {"payload": payload, "signature": key.sign(canonical(payload)).hex()}


def verify(message, audience, path, store):
    try:
        if set(message) != {"payload", "signature"}:
            raise ValueError()
        payload = message["payload"]
        if set(payload) != {"hotkey", "audience", "path", "timestamp", "nonce", "body"}:
            raise ValueError()
        if (
            payload["audience"] != audience
            or payload["path"] != path
            or not isinstance(payload["body"], dict)
        ):
            raise ValueError()
        if type(payload["timestamp"]) is not int or abs(time.time() - payload["timestamp"]) > 120:
            raise ValueError()
        if str(uuid.UUID(payload["nonce"])) != payload["nonce"]:
            raise ValueError()
        signature = bytes.fromhex(message["signature"])
        if len(signature) != 64 or not Keypair(ss58_address=payload["hotkey"]).verify(
            canonical(payload), signature
        ):
            raise ValueError()
    except (ValueError, TypeError, KeyError):
        raise APIError(401, "Invalid worker signature or request timestamp.") from None
    store.rpc("fez_worker_nonce", {"p_hotkey": payload["hotkey"], "p_nonce": payload["nonce"]})
    return payload["hotkey"], payload["body"]


def identifier(value):
    try:
        if str(uuid.UUID(value)) != value:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise APIError(400, "Invalid job or lease identifier.") from None
    return value


@contextmanager
def lease_heartbeat(renew, interval=60):
    stopped, errors = threading.Event(), []

    def keep_alive():
        while not stopped.wait(interval):
            try:
                renew()
            except Exception as error:
                errors.append(error)
                return

    renew()
    thread = threading.Thread(target=keep_alive, daemon=True)
    thread.start()

    def check_lease():
        if errors:
            raise APIError(409, "Lease renewal failed; this attempt must be retried.")

    try:
        yield check_lease
        check_lease()
    finally:
        stopped.set()
        thread.join(timeout=35)
