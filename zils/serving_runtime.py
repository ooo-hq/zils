"""Answer `zils-miner serve` requests from a box's adapter runtime instead of a second model copy.

A box that already runs the zils-platform adapter runtime with `--serve-base` holds the pinned base
and answers it as `<profile>-base`. `connect` checks that the runtime serves exactly the base of the
`--checkpoint` reference, then returns a `RuntimeServed` that `serving_server.make_server` uses in
place of `serving_server.Served`. The contract is the runtime's loopback HTTP API, mirrored here:

  GET  /health        -> {"profile", "profile_sha256", "runtime_sha256", "ready",
                          "models": [{"release_id", "fingerprint"}]}
  POST /v1/systemone  {"request": {"model", "state", "questions"}, "lane": "bulk"}
                      -> {"release_id", "fingerprint", "limits",
                          "predictions": {qid: {"probabilities", "input_tokens"}}}
  errors              {"error": {"code", "message"}} with an HTTP status (401, 404, 413, 422, 503,
                      504, 529 overloaded)

Every request names the base release only, never a customer adapter, and uses the bulk lane so the
runtime's paid traffic goes first. The runtime must be on loopback; its token is never logged.

The identity checks bind the pinned profile and runtime files, not the runtime process's environment:
a runtime started with ZILS_D3_CANDIDATE_BASE (unpinned base weights) would still pass them. That is
accepted because the same operator runs both processes, and validators recompute answers anyway."""

import hashlib
import http.client
import json
import re
import sys
import threading
import urllib.error
import urllib.request

from . import models, protocol, serving_worker
from .serving_server import WorkerError

LOOPBACK = re.compile(r"http://(127\.0\.0\.1|localhost):([1-9][0-9]{0,4})")
MAX_REPLY = 4 * 1024 * 1024
HEALTH_SECONDS = 60.0
# Runtime statuses that blame the request; every other failure is the node's (WorkerError "runtime").
# 401 (credential), 404 (base not loaded) and 408 mean this box's runtime, not the caller's input.
CALLER_FAULTS = {400: "invalid", 413: "too_long", 422: "invalid"}
OVERLOADED = 529


def log(text):
    """Operator log on stderr: fixed text and status codes only, never the token or request content."""
    print(f"zils-miner serve: {text}", file=sys.stderr, flush=True)


def loopback(runtime_url):
    """`http://127.0.0.1:<port>` for a loopback runtime URL; anything else is refused."""
    match = LOOPBACK.fullmatch(runtime_url) if isinstance(runtime_url, str) else None
    if not match or int(match[2]) > 65535:
        raise ValueError("--runtime must be http://127.0.0.1:<port> or http://localhost:<port>")
    return f"http://127.0.0.1:{match[2]}"  # localhost dials 127.0.0.1: no resolver in the path


def base_identity(model):
    """The runtime's `<model>-base` release, computed as zils_platform.adapter_server.base_release does."""
    identity = models.profile_identity(model)
    fingerprint = hashlib.sha256(
        (identity["profile_sha256"] + identity["runtime_sha256"]).encode()
    ).hexdigest()
    return {**identity, "release_id": f"{model}-base", "fingerprint": fingerprint}


def identity_error(model, health):
    """Why `health` is not a runtime serving exactly `model`'s pinned base, or None. Ignores `ready`."""
    if not isinstance(health, dict):
        return "the runtime's health reply is not an object"
    expected = base_identity(model)
    for key in ("profile_sha256", "runtime_sha256"):
        if health.get(key) != expected[key]:
            return f"the runtime's {key} differs from {model}'s pinned profile"
    listed = health.get("models")
    if not isinstance(listed, list) or not any(
        isinstance(entry, dict)
        and entry.get("release_id") == expected["release_id"]
        and entry.get("fingerprint") == expected["fingerprint"]
        for entry in listed
    ):
        return f"the runtime does not serve {expected['release_id']} with the pinned fingerprint (start it with --serve-base)"
    return None


class RuntimeServed:
    """One base the runtime holds: model, weights_sha256, max_input_tokens, predict(), watch()."""

    def __init__(self, url, token, model, weights_sha256, max_input_tokens, timeout):
        self.url, self._token, self.timeout = url, token, timeout
        self.model, self.weights_sha256 = model, weights_sha256
        self.max_input_tokens = max_input_tokens
        identity = base_identity(model)
        # Fixed at construction: predict() names this release whatever later happens to .model.
        self._release, self._fingerprint = identity["release_id"], identity["fingerprint"]
        self.changed = None  # set by watch() to why the runtime's identity changed

    def __repr__(self):
        return f"RuntimeServed({self.url!r}, {self.model!r})"

    def _call(self, method, path, body=None):
        """(status, parsed JSON or None); OSError when the runtime cannot be reached or read."""
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.url + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                **({"Content-Type": "application/json"} if data is not None else {}),
            },
        )
        try:
            try:
                response = protocol.opener().open(request, timeout=self.timeout)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                status, raw = response.status, response.read(MAX_REPLY + 1)
        except ValueError:  # a redirect, refused by the opener
            raise OSError("the runtime redirected") from None
        except http.client.HTTPException as error:  # BadStatusLine, IncompleteRead: not OSErrors
            raise OSError(f"the runtime sent a malformed reply ({type(error).__name__})") from None
        if len(raw) > MAX_REPLY:
            raise OSError("the runtime's reply is too large")
        try:
            return status, json.loads(raw)
        except ValueError:
            return status, None

    def health(self):
        """The runtime's /health object; ValueError naming the status, OSError if unreachable."""
        status, value = self._call("GET", "/health")
        if status != 200:
            raise ValueError(f"the runtime's /health answered HTTP {status}")
        if not isinstance(value, dict):
            raise ValueError("the runtime's /health reply is not a JSON object")
        return value

    def check(self):
        """Refuse (ValueError) unless the runtime is ready and serves exactly this base."""
        try:
            value = self.health()
        except OSError as error:
            raise ValueError(
                f"the runtime at {self.url} is unreachable ({type(error).__name__})"
            ) from None
        reason = identity_error(self.model, value)
        if reason:
            raise ValueError(reason)
        if value.get("ready") is not True:
            raise ValueError("the runtime is not ready")

    def predict(self, state, questions):
        """{qid: {"probabilities", "input_tokens"}}; WorkerError(kind) on any refusal or failure."""
        body = {
            "request": {"model": self._release, "state": state, "questions": questions},
            "lane": "bulk",
        }
        try:
            status, reply = self._call("POST", "/v1/systemone", body)
        except (OSError, ValueError) as error:
            log(f"runtime request failed ({type(error).__name__})")
            raise WorkerError("runtime") from None
        if status != 200:
            if status == OVERLOADED:
                raise WorkerError("busy")
            kind = CALLER_FAULTS.get(status)
            if kind is None:
                log(f"runtime answered HTTP {status}")
                raise WorkerError("runtime")
            raise WorkerError(kind)
        predictions = self._predictions(reply, questions)
        if predictions is None:
            log("runtime reply did not match the request")
            raise WorkerError("runtime")
        return predictions

    def _predictions(self, reply, questions):
        if (
            not isinstance(reply, dict)
            or reply.get("release_id") != self._release
            or reply.get("fingerprint") != self._fingerprint
        ):
            return None
        found = reply.get("predictions")
        if not isinstance(found, dict) or set(found) != set(questions):
            return None
        out = {}
        for qid, item in found.items():
            if not isinstance(item, dict):
                return None
            probabilities, tokens = item.get("probabilities"), item.get("input_tokens")
            if not isinstance(probabilities, dict) or type(tokens) is not int:
                return None
            out[qid] = {"probabilities": probabilities, "input_tokens": tokens}
        return out

    def watch(self, stop, interval=HEALTH_SECONDS):
        """Check /health every `interval` seconds on a daemon thread. A paused or unreachable runtime
        is logged and serving goes on (each request answers unavailable); an identity change sets
        `self.changed` and `stop` so `zils-miner serve` exits for its supervisor."""

        def probe():
            """The runtime's state as fixed text, or None after an identity change (stop is set)."""
            try:
                value = self.health()
            except OSError as error:
                return f"runtime is unreachable ({type(error).__name__})"
            except ValueError as error:  # an HTTP status: fixed text, no token
                return f"runtime is unreachable ({error})"
            reason = identity_error(self.model, value)
            if reason:
                self.changed = reason
                log(f"runtime identity changed: {reason}; stopping")
                stop.set()
                return None
            return "ok" if value.get("ready") is True else "runtime is not ready"

        def run():
            last = None
            while not stop.wait(interval):
                try:
                    state = probe()
                except Exception as error:  # the check must outlive any surprise; class name only
                    state = f"runtime health check failed ({type(error).__name__})"
                if state is None:
                    return
                if state != last and not (last is None and state == "ok"):
                    log(state if state != "ok" else "runtime is ready again")
                last = state

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread


def check(runtime_url, token, model, timeout=10.0):
    """`zils-miner doctor --runtime`: the runtime is on loopback, reachable, ready and serves `model`'s
    pinned base; ValueError with the reason otherwise."""
    url = loopback(runtime_url)
    if not isinstance(token, str) or not token:
        raise ValueError("the runtime token is empty; export it in --runtime-token-env")
    RuntimeServed(url, token, model, None, None, timeout).check()


def connect(runtime_url, token, checkpoint, timeout=120.0):
    """[RuntimeServed] for the base `checkpoint` (the reference the runtime loaded), after checking
    the URL, the checkpoint, its temperature and the runtime's identity; ValueError otherwise."""
    url = loopback(runtime_url)
    if not isinstance(token, str) or not token:
        raise ValueError("the runtime token is empty; export it in --runtime-token-env")
    meta = models.metadata(checkpoint)
    if meta is None or meta["kind"] != "base":
        raise ValueError(
            f"{checkpoint}: --runtime serves the pinned base only; pass the runtime's base reference"
        )
    if meta["temperature"] != 1.0:
        raise ValueError(
            f"{checkpoint}: temperature {meta['temperature']} differs from the runtime's base"
            " temperature 1.0"
        )
    entry = serving_worker.describe(checkpoint)
    served = RuntimeServed(url, token, timeout=timeout, **entry)
    served.check()
    return [served]
