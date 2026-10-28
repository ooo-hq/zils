"""Signed messages for miner-served inference: endpoint manifest, request and answer.

A manifest's `weights_sha256` is `checkpoint_hash(dir)` of the served checkpoint: for a base
(`kind: "base"`) that covers only its model.json, because a base's weights are the pinned release
(the worker refuses to start when a candidate-base override would swap them); for a candidate it
covers the full weights."""

import hashlib
import http.client
import json
import re
import socket
import threading
import time
import uuid

from bittensor_wallet import Keypair

from . import decisions, models, protocol

VERSION = "zils-serving/v1"
FRESH_SECONDS = 120
MAX_MANIFEST_BYTES = 65536
MAX_ANSWER_BYTES = 1 << 20

_FIELDS = {
    "manifest": {"version", "hotkey", "endpoint", "issued_at", "models"},
    "request": {"caller", "audience", "timestamp", "nonce", "model", "weights_sha256", "body"},
    "answer": {"miner", "request_sha256", "model", "weights_sha256", "predictions"},
}
# Payload field naming the signer, and the timestamp checked for freshness (answers carry no time).
_SIGNER = {"manifest": "hotkey", "request": "caller", "answer": "miner"}
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


def answer_payload(miner, request_sha256, model, weights_sha256, predictions):
    return {
        "miner": miner,
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


def public_endpoint(endpoint):
    """Return `endpoint` if it is a globally routable IPv4 endpoint; nothing may connect to anything else."""
    address, _ = protocol.endpoint_port(endpoint)
    if not address.is_global or address.is_multicast:
        raise ValueError("endpoint must be a public IPv4 address")
    return endpoint


def validate_manifest(payload):
    """Check a verified manifest payload and return its served models."""
    served = payload.get("models")
    if payload.get("version") != VERSION:
        raise ValueError("unsupported serving version")
    public_endpoint(payload.get("endpoint"))
    family = models.d3_family()
    if not isinstance(served, list) or not 1 <= len(served) <= len(family):
        raise ValueError("manifest must list 1 to %d models" % len(family))
    for item in served:
        if not isinstance(item, dict) or set(item) != {
            "model",
            "weights_sha256",
            "max_input_tokens",
        }:
            raise ValueError("invalid manifest model entry")
        if item["model"] not in family:
            raise ValueError("unsupported model")
        if not isinstance(item["weights_sha256"], str) or not _HEX64.fullmatch(
            item["weights_sha256"]
        ):
            raise ValueError("weights_sha256 must be 64 hex characters")
        limit = models.spec(item["model"]).get("max_input_tokens", 2048)
        tokens = item["max_input_tokens"]
        if type(tokens) is not int or not 1 <= tokens <= limit:
            raise ValueError("max_input_tokens out of range")
    if len({item["model"] for item in served}) != len(served):
        raise ValueError("duplicate model")
    return served


def fetch_manifest(endpoint, timeout=5.0):
    """GET the manifest from a public endpoint; `timeout` bounds the whole fetch (connect, headers, body)."""
    return _fetch_manifest(public_endpoint(endpoint), timeout)


def _fetch_manifest(endpoint, timeout):
    """`fetch_manifest` without the public-address check; tests reach 127.0.0.1 through this."""
    address, port = protocol.endpoint_port(endpoint)  # numeric host: no DNS, no proxy, no redirects
    deadline = time.monotonic() + timeout
    conn = http.client.HTTPConnection(str(address), port, timeout=timeout)
    fired = threading.Event()

    def expire():
        fired.set()
        try:  # shutdown wakes a blocked read; the response may have taken conn.sock, so keep our own reference
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()

    watchdog = None
    try:
        conn.connect()  # bounded by the same budget
        sock = conn.sock
        watchdog = threading.Timer(max(deadline - time.monotonic(), 0), expire)
        watchdog.start()
        data = b""
        conn.request("GET", "/v1/serving")
        response = conn.getresponse()
        if response.status != 200:
            raise ValueError("manifest fetch failed: HTTP %d" % response.status)
        while len(data) <= MAX_MANIFEST_BYTES:
            chunk = response.read1(min(4096, MAX_MANIFEST_BYTES + 1 - len(data)))
            if not chunk:
                break
            data += chunk
    except (OSError, http.client.HTTPException) as error:
        if fired.is_set():
            raise ValueError("manifest fetch exceeded its deadline") from None
        raise ValueError("manifest fetch failed: %s" % (error or type(error).__name__)) from None
    finally:
        if watchdog:
            watchdog.cancel()
        conn.close()
    if fired.is_set() or time.monotonic() > deadline:
        raise ValueError("manifest fetch exceeded its deadline")
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("manifest too large")
    try:
        return json.loads(data)
    except ValueError:
        raise ValueError("manifest is not JSON") from None


def discover(axons, clock=time.time, fetch=fetch_manifest):
    """{uid: {"hotkey", "endpoint", "models"} or {"error"}}; one bad miner never raises. The clock is
    read after each fetch, so a slow scan never ages out the manifests fetched last."""
    found = {}
    for uid, (hotkey, endpoint) in axons.items():
        try:
            message = fetch(public_endpoint(endpoint))
            payload = verify(message, "manifest", hotkey, int(clock()))
            served = validate_manifest(payload)
            if payload["endpoint"] != endpoint:
                raise ValueError("manifest endpoint does not match the on-chain axon")
            found[uid] = {"hotkey": hotkey, "endpoint": endpoint, "models": served}
        except Exception as error:  # fetch can fail in many ways (OSError, http.client, JSON)
            found[uid] = {"error": str(error) or type(error).__name__}
    return found


class MinerError(Exception):
    """`reason` is one of busy (HTTP 503/429), unavailable (503 worker failure),
    contradicts_manifest (404 model_not_found / 409 weights_mismatch), refused (401/403),
    rejected (other 4xx), timeout, http, invalid, unverified; never carries request or answer contents."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _error_code(response):
    """The `error.code` of a JSON error body (read to an 8 KiB cap), else None."""
    try:
        return json.loads(response.read(8192))["error"]["code"]
    except (ValueError, KeyError, TypeError, OSError, http.client.HTTPException):
        return None


def _post(endpoint, data, timeout):
    """POST `data` to the miner's /v1/systemone under a hard total deadline; return the body bytes."""
    address, port = protocol.endpoint_port(endpoint)  # numeric host: no DNS, no proxy, no redirects
    deadline = time.monotonic() + timeout
    conn = http.client.HTTPConnection(str(address), port, timeout=timeout)
    conn._http_vsn, conn._http_vsn_str = 10, "HTTP/1.0"
    fired = threading.Event()
    sock = None

    def expire():
        fired.set()
        if sock is not None:
            try:  # shutdown wakes a blocked read; the response may have taken conn.sock
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    watchdog = threading.Timer(max(timeout, 0), expire)
    watchdog.start()
    body = bytearray()
    try:
        conn.connect()
        sock = conn.sock
        if fired.is_set():
            raise OSError()
        conn.request(
            "POST",
            "/v1/systemone",
            body=data,
            headers={"Content-Type": "application/json", "Content-Length": str(len(data))},
        )
        response = conn.getresponse()
        if response.status in (429, 503, 404, 409):
            code = _error_code(response)
            if fired.is_set() or time.monotonic() >= deadline:
                raise MinerError("timeout")
            if response.status == 503 and code == "unavailable":
                raise MinerError("unavailable")  # the miner's worker failed
            if (response.status, code) in (
                (404, "model_not_found"),
                (409, "weights_mismatch"),
            ):
                raise MinerError("contradicts_manifest")  # contradicts its own signed manifest
            raise MinerError("busy" if response.status in (429, 503) else "rejected")
        if response.status in (401, 403):
            raise MinerError("refused")
        if 400 <= response.status < 500:
            raise MinerError("rejected")
        if response.status != 200:
            raise MinerError("http")
        while len(body) <= MAX_ANSWER_BYTES:
            chunk = response.read1(min(4096, MAX_ANSWER_BYTES + 1 - len(body)))
            if not chunk:
                break
            body += chunk
    except (OSError, http.client.HTTPException):
        late = fired.is_set() or time.monotonic() >= deadline
        raise MinerError("timeout" if late else "http") from None
    finally:
        watchdog.cancel()
        conn.close()
    if fired.is_set() or time.monotonic() >= deadline:
        raise MinerError("timeout")
    if len(body) > MAX_ANSWER_BYTES:
        raise MinerError("invalid")
    return bytes(body)


def ask(
    endpoint,
    miner_hotkey,
    key,
    model,
    weights_sha256,
    body,
    timeout,
    *,
    clock=time.time,
    check=public_endpoint,
):
    """Send one signed request to a miner; return its verified signed answer message or raise MinerError."""
    try:
        try:
            check(endpoint)
        except ValueError:
            raise MinerError("invalid") from None
        miner_body = {**body, "model": model}  # the miner checks body model == signed model
        payload = request_payload(
            key.ss58_address,
            miner_hotkey,
            model,
            weights_sha256,
            miner_body,
            int(clock()),
        )
        data = json.dumps(sign(key, "request", payload)).encode()
        raw = _post(endpoint, data, timeout)
        try:
            message = json.loads(raw)
            answer = verify(message, "answer", miner_hotkey, int(clock()))
        except ValueError:  # includes bad JSON
            raise MinerError("unverified") from None
        if (
            answer["request_sha256"] != request_digest(payload)
            or answer["model"] != model
            or answer["weights_sha256"] != weights_sha256
        ):
            raise MinerError("invalid")
        try:  # the exact check the gateway applies to the answer before the customer sees it
            decisions.make_response(model, miner_body, answer["predictions"])
        except decisions.DecisionError:
            raise MinerError("invalid") from None
        return message
    except MinerError:
        raise
    except Exception:  # a miner payload must never crash the caller with another type
        raise MinerError("invalid") from None
