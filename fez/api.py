"""Zils authenticated decision gateway. Training and onboarding remain separate services."""

import argparse
import json
import os
import re
import threading
from pathlib import Path
from urllib.parse import urlsplit

import requests

from .api_store import Store, identifier, safe_cloud
from .cloud import APIError, trusted_url
from .decision_http import Server, make_handler
from .decisions import DecisionError, decode_body, make_response, validate_request


class Registry:
    """Operator-approved releases. Aliases are unique and never imply adapter promotion."""

    def __init__(self, entries):
        self.entries, self.names = [], {}
        for entry in entries:
            required = {
                "id",
                "fingerprint",
                "aliases",
                "owners",
                "url",
                "token_env",
                "release_date",
                "description",
            }
            if not isinstance(entry, dict) or set(entry) != required:
                raise ValueError("Invalid model registry entry")
            if not re.fullmatch("[a-f0-9]{64}", entry["fingerprint"]):
                raise ValueError("Pin the model release fingerprint")
            if not isinstance(entry["aliases"], list) or not all(
                isinstance(x, str) and x for x in entry["aliases"]
            ):
                raise ValueError("Invalid aliases")
            if entry["owners"] is not None:
                if not isinstance(entry["owners"], list):
                    raise ValueError("Owners must be a list or null for a shared model")
                entry = {**entry, "owners": [identifier(x) for x in entry["owners"]]}
            entry = {**entry, "url": trusted_url(entry["url"])}
            for name in [entry["id"], *entry["aliases"]]:
                if not isinstance(name, str) or not 1 <= len(name) <= 128 or name in self.names:
                    raise ValueError("Model names must be unique nonempty strings")
                self.names[name] = entry
            self.entries.append(entry)

    def resolve(self, name, owner):
        entry = self.names.get(name)
        if entry and (entry["owners"] is None or owner in entry["owners"]):
            return entry
        raise DecisionError(404, "model_not_found", "Model is unavailable to this account.")

    def snapshot(self, owner):
        return {
            name: {"id": entry["id"], "fingerprint": entry["fingerprint"]}
            for name, entry in self.names.items()
            if entry["owners"] is None or owner in entry["owners"]
        }

    def listing(self, owner):
        return {
            "models": [
                {
                    "name": name,
                    "description": entry["description"],
                    "release_date": entry["release_date"],
                }
                for name, entry in self.names.items()
                if entry["owners"] is None or owner in entry["owners"]
            ]
        }


class FileRegistry:
    """Atomic catalog reloads; malformed updates leave the last verified catalog usable."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.signature = None
        self.current = None
        self._read()

    def _read(self):
        with self.lock:
            try:
                with self.path.open() as stream:
                    stat = os.fstat(stream.fileno())
                    signature = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
                    if signature == self.signature:
                        return self.current
                    candidate = Registry(json.load(stream)["models"])
                if self.current is not None:
                    if self.current.names.keys() - candidate.names.keys():
                        raise ValueError("Existing model names must be preserved")
                    for old in self.current.entries:
                        new = candidate.names.get(old["id"])
                        if new is None or any(old[k] != new[k] for k in old if k != "aliases"):
                            raise ValueError("Existing immutable releases must be preserved")
                    for name in self.current.names.keys() & candidate.names.keys():
                        if self.current.names[name]["owners"] != candidate.names[name]["owners"]:
                            raise ValueError("A model name cannot move between owners")
                self.current, self.signature = candidate, signature
            except (OSError, ValueError, KeyError, TypeError, DecisionError):
                if self.current is None:
                    raise
            return self.current

    def resolve(self, name, owner):
        return self._read().resolve(name, owner)

    def snapshot(self, owner):
        return self._read().snapshot(owner)

    def listing(self, owner):
        return self._read().listing(owner)


class RuntimeClient:
    def __init__(self, entry):
        self.entry = entry
        self.token = os.environ[entry["token_env"]]

    def call(self, path, body):
        try:
            with requests.post(
                self.entry["url"] + path,
                data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                headers={
                    "Authorization": "Bearer " + self.token,
                    "Accept-Encoding": "identity",
                    "Content-Type": "application/json",
                },
                timeout=(5, 40),
                allow_redirects=False,
                stream=True,
            ) as response:
                raw = bytearray()
                for chunk in response.iter_content(65536):
                    raw.extend(chunk)
                    if len(raw) > 8 * 1024 * 1024:
                        raise DecisionError(
                            502, "invalid_model_response", "Model response exceeds its size limit."
                        )
                if response.status_code != 200:
                    if response.status_code in (413, 422, 429, 529, 504):
                        raise DecisionError(
                            response.status_code,
                            "runtime_rejected",
                            "Model could not process this request; check its limits or retry.",
                            retry_after=1 if response.status_code in (429, 529) else None,
                        )
                    raise DecisionError(
                        503, "runtime_unavailable", "Model is unavailable.", retry_after=1
                    )
                try:
                    result = decode_body(bytes(raw), limit=8 * 1024 * 1024)
                except DecisionError:
                    raise DecisionError(
                        502, "invalid_model_response", "Model returned an invalid response."
                    ) from None
        except requests.RequestException:
            raise DecisionError(
                503, "runtime_unavailable", "Model connection failed.", retry_after=1
            ) from None
        if (
            not isinstance(result, dict)
            or result.get("release_id") != self.entry["id"]
            or result.get("fingerprint") != self.entry["fingerprint"]
        ):
            raise DecisionError(
                503, "release_mismatch", "The approved model release is unavailable."
            )
        return result

    def prepare(self, body):
        result = self.call("/v1/prepare", {"request": body})
        tokens = result.get("reserved_tokens")
        if type(tokens) is not int or not 0 <= tokens <= 2**31:
            raise DecisionError(
                502, "invalid_model_response", "Model returned invalid token accounting."
            )
        return tokens

    def predict(self, body, lane):
        return self.call("/v1/systemone", {"request": body, "lane": lane}).get("predictions")


class Gateway:
    def __init__(self, store, registry, batches=None):
        self.store, self.registry, self.batches = store, registry, batches

    def evaluate(self, owner, key_id, body, request_id, *, lane="realtime", frozen=None):
        validate_request(body)
        entry = self.registry.resolve(frozen["id"] if frozen else body["model"], owner)
        if frozen is not None and (
            entry["id"] != frozen["id"] or entry["fingerprint"] != frozen["fingerprint"]
        ):
            raise DecisionError(503, "release_mismatch", "The batch model release is unavailable.")
        request = {**body, "model": entry["id"]}
        client = RuntimeClient(entry)
        reserved = client.prepare(request)
        self.store.admit(owner, key_id, request_id, reserved)
        try:
            result = make_response(entry["id"], request, client.predict(request, lane))
            if result["usage"]["input_tokens"] > reserved:
                raise DecisionError(
                    502, "token_accounting_error", "Model exceeded its token reservation."
                )
        except Exception:
            self.store.finish_usage(request_id, None, "failed")
            raise
        if lane == "realtime":
            self.store.finish_usage(request_id, result["usage"]["input_tokens"], "completed")
        # Bulk commits the item result and usage together under its database lease.
        return result

    def dispatch(self, method, path, bearer, body, request_id):
        try:
            if path == "/v1/keys" or re.fullmatch(r"/v1/keys/[^/]+/revoke", path):
                owner = self.store.session_owner(bearer)
                if path == "/v1/keys" and method == "GET":
                    return 200, {"keys": self.store.list_keys(owner)}
                if path == "/v1/keys" and method == "POST":
                    if not isinstance(body, dict) or set(body) != {"name"}:
                        raise DecisionError(422, "invalid_request", "A key name is required.")
                    return 201, self.store.create_key(owner, body["name"])
                if path.endswith("/revoke") and method == "POST" and body == {}:
                    return 200, self.store.revoke_key(owner, path.split("/")[3])
                raise DecisionError(405, "method_not_allowed", "Method is not supported.")
            principal = self.store.authenticate(bearer)
            owner = principal["owner_id"]
            if path == "/v1/models" and method == "GET":
                return 200, self.registry.listing(owner)
            if path == "/v1/systemone" and method == "POST":
                return 200, self.evaluate(owner, principal["id"], body, request_id)
            if urlsplit(path).path.startswith("/v1/batches") and self.batches:
                return self.batches.dispatch(method, path, owner, body, self.registry)
            raise DecisionError(404, "not_found", "Resource not found.")
        except APIError as error:
            raise safe_cloud(error) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8920)
    parser.add_argument("--origin", help="Exact optional dashboard CORS origin")
    args = parser.parse_args()
    from .batches import Batches

    store = Store()
    registry = FileRegistry(args.registry)
    gateway = Gateway(store, registry, Batches(store.db))
    service = Server(("127.0.0.1", args.port), make_handler(gateway.dispatch, args.origin))
    print(f"Zils API ready: loopback port {args.port}", flush=True)
    try:
        service.serve_forever()
    finally:
        service.server_close()


if __name__ == "__main__":
    main()
