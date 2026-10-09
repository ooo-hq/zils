"""Signed checkpoint announcements, private-LAN transport, and atomic state."""

import ipaddress
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from bittensor_wallet import Keypair

import zils

from . import models

BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
MAX_ANNOUNCEMENT = 8192


def canonical(claim):
    return (
        b"fez-local-checkpoint/v1\0"
        + json.dumps(claim, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def endpoint_ok(endpoint, allowed=None):
    match = (
        re.fullmatch(r"http://([0-9.]+):([1-9][0-9]{0,4})", endpoint)
        if isinstance(endpoint, str)
        else None
    )
    if not match or int(match[2]) > 65535:
        raise ValueError("endpoint must be http://<numeric-IPv4>:<port>")
    address = ipaddress.IPv4Address(match[1])
    if allowed is None:
        if str(address) != "127.0.0.1":
            raise ValueError("remote endpoints require an explicit pin")
    elif (
        endpoint not in allowed
        or not (address.is_private or address.is_loopback)
        or address.is_link_local
        or address.is_unspecified
        or address.is_multicast
        or address.is_reserved
    ):
        raise ValueError("endpoint must match a pinned private IPv4 address")


def register(message, round_id, members, registry, endpoints=None, job_sha256=None):
    if not isinstance(message, dict) or set(message) != {"claim", "signature"}:
        raise ValueError("announcement requires claim and signature")
    c, signature = message["claim"], message["signature"]
    fields = {"round_id", "uid", "hotkey", "sha256", "endpoint"}
    if job_sha256 is not None:
        fields.add("job_sha256")
    if not isinstance(c, dict) or set(c) != fields:
        raise ValueError("invalid claim fields")
    if job_sha256 is not None and c["job_sha256"] != job_sha256:
        raise ValueError("announcement belongs to another customer job")
    uid = c["uid"]
    if type(uid) is not int or uid not in members or c["hotkey"] != members[uid]:
        raise ValueError("identity is not in this rehearsal's allowlist")
    if c["round_id"] != round_id:
        raise ValueError("announcement belongs to another round")
    if not isinstance(c["sha256"], str) or not re.fullmatch("[a-f0-9]{64}", c["sha256"]):
        raise ValueError("invalid checkpoint hash")
    endpoint_ok(c["endpoint"], allowed=[endpoints[uid]] if endpoints is not None else None)
    if not isinstance(signature, str) or not re.fullmatch("[a-f0-9]{128}", signature):
        raise ValueError("invalid signature format")
    if not Keypair(ss58_address=c["hotkey"]).verify(canonical(c), bytes.fromhex(signature)):
        raise ValueError("signature verification failed")
    if uid in registry and registry[uid]["claim"] != c:
        raise ValueError("a miner cannot change its submission within a round")
    if any(
        other != uid and entry["claim"]["sha256"] == c["sha256"]
        for other, entry in registry.items()
    ):
        raise ValueError("duplicate checkpoint from another miner")
    registry[uid] = {"claim": dict(c), "signature": signature}
    return c


def write_json(path, data):
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        # Publish complete state atomically; link refuses to overwrite an earlier round.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *args):
        pass

    def reply(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def local_server(handler, host="127.0.0.1", port=0):
    with HTTPServer((host, port), handler) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )
        thread.start()
        try:
            yield f"http://{host}:{server.server_port}"
        finally:
            server.shutdown()
            thread.join(timeout=6)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("artifact and announcement redirects are forbidden")


def opener():
    return build_opener(ProxyHandler({}), NoRedirect())


def fetch_checkpoint(
    claim, destination, expected_endpoint=None, round_scoped=False, *, model=models.KEV
):
    endpoint_ok(
        claim["endpoint"], allowed=[expected_endpoint] if expected_endpoint is not None else None
    )
    if round_scoped and not re.fullmatch("[a-f0-9]{32}", claim["round_id"]):
        raise ValueError("invalid artifact round id")
    prefix = "/artifacts/" + (claim["round_id"] + "/" if round_scoped else "")
    destination = Path(destination)
    destination.mkdir()
    started, total = time.monotonic(), 0
    files = models.candidate_files(model)
    for name in files:
        with opener().open(claim["endpoint"] + prefix + name, timeout=5) as response:
            size = int(response.headers.get("Content-Length", "-1"))
            if not 0 <= size <= zils.MAX_ARTIFACT_BYTES - total:
                raise ValueError("invalid artifact size or checkpoint exceeds byte budget")
            received = 0
            with (destination / name).open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > size or time.monotonic() - started > (
                        300 if model == models.IMAJEV else 30
                    ):
                        raise ValueError("artifact exceeds declared size or download deadline")
                    output.write(chunk)
            if received != size:
                raise ValueError("truncated artifact")
            total += received
    if models.checkpoint_model(destination) != model or models.artifact_files(destination) != files:
        raise ValueError("Downloaded adapter differs from the required model profile")
    if zils.checkpoint_hash(destination) != claim["sha256"]:
        raise ValueError("downloaded checkpoint hash differs from signed announcement")
    return {"bytes": total, "download_ms": (time.monotonic() - started) * 1000}
