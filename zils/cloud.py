"""Small Supabase REST client; server credentials never travel to workers or browsers."""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

DATA_BUCKET = "fez-training-data"
MODEL_BUCKET = "fez-training-models"
MAX_DATA_BYTES = 128 * 1024 * 1024


class APIError(Exception):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


def trusted_url(url):
    parsed = urlsplit(url)
    if parsed.username or parsed.password or parsed.fragment or not parsed.hostname:
        raise ValueError("invalid service URL")
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")
    ):
        raise ValueError("remote service URLs require HTTPS")
    return url.rstrip("/")


def download(url, destination, limit, headers=None, *, max_seconds=600):
    """Terminate the complete network operation at its wall-clock deadline."""
    trusted_url(url)
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("invalid transfer-time limit")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("download destination already exists")
    # A separate process can be killed even during DNS, headers, or a trickling read.
    # Keep partial bytes private, and publish exclusively only after full validation.
    with tempfile.TemporaryDirectory(prefix=".zils-download-", dir=destination.parent) as temp:
        staged = Path(temp) / "payload"
        payload = dict(url=url, destination=str(staged), limit=limit, headers=headers)
        try:
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--download"],
                input=json.dumps(payload),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=max_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise APIError(503, "File download timed out; please retry.") from None
        try:
            result = json.loads(process.stdout)
        except ValueError:
            result = {}
        if process.returncode or type(result.get("size")) is not int:
            if result.get("error") == "invalid_file":
                raise ValueError("file exceeds its size limit or is truncated")
            raise APIError(503, "File download interrupted; please retry.")
        os.link(staged, destination)
        return result["size"]


def _download_stream(url, destination, limit, headers=None, *, max_seconds=600):
    trusted_url(url)
    try:
        started, size = time.monotonic(), 0
        with requests.get(
            url,
            headers={"Accept-Encoding": "identity", **(headers or {})},
            stream=True,
            timeout=(10, 30),
            allow_redirects=False,
        ) as response:
            if response.status_code != 200:
                raise APIError(503, "File download failed; retry or check storage configuration.")
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > limit:
                raise ValueError("file exceeds the allowed size")
            with Path(destination).open("xb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise ValueError("file exceeds its size limit")
                    if time.monotonic() - started > max_seconds:
                        raise APIError(503, "File download timed out; please retry.")
                    output.write(chunk)
            if declared is not None and size != int(declared):
                raise ValueError("truncated file download")
        Path(destination).chmod(0o600)
        return size
    except requests.RequestException:
        raise APIError(503, "File download interrupted; please retry.") from None


def upload(url, source, headers=None, method="PUT"):
    trusted_url(url)
    try:
        with Path(source).open("rb") as stream:
            response = requests.request(
                method,
                url,
                data=stream,
                headers=headers
                or {
                    "Content-Type": "application/octet-stream",
                    "x-upsert": "false",
                },
                timeout=(10, 300),
                allow_redirects=False,
            )
        try:
            if not 200 <= response.status_code < 300:
                raise APIError(503, "File upload failed; retry or check storage configuration.")
        finally:
            response.close()
    except requests.RequestException:
        raise APIError(503, "File upload interrupted; please retry.") from None


class Supabase:
    def __init__(self, url=None, key=None):
        self.url = trusted_url(url or os.environ["SUPABASE_URL"])
        self.key = key or os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        self.headers = {"apikey": self.key, "Authorization": f"Bearer {self.key}"}

    def request(self, method, path, body=None, headers=None):
        try:
            response = requests.request(
                method,
                self.url + path,
                json=body,
                headers={**self.headers, **(headers or {})},
                timeout=(10, 30),
                allow_redirects=False,
            )
        except requests.RequestException:
            raise APIError(503, "Supabase is unavailable; please retry.") from None
        try:
            if not 200 <= response.status_code < 300:
                # Never send database details, storage tokens or upstream URLs to a caller.
                if response.status_code in (400, 409, 422):
                    try:
                        detail = response.json()
                    except ValueError:
                        detail = None
                    if isinstance(detail, dict) and detail.get("code") == "P0402":
                        raise APIError(
                            402,
                            "Add credit or resolve your payment issue before submitting this job.",
                        )
                    raise APIError(409, "Operation conflicts with the current job or lease state.")
                raise APIError(503, "Supabase request failed; check service configuration.")
            return response.json() if response.content else None
        finally:
            response.close()

    def user(self, token):
        if not isinstance(token, str) or not token or len(token) > 8192:
            raise APIError(401, "Sign in to continue.")
        try:
            response = requests.get(
                self.url + "/auth/v1/user",
                headers={
                    "apikey": self.key,
                    "Authorization": "Bearer " + token,
                },
                timeout=(10, 20),
                allow_redirects=False,
            )
        except requests.RequestException:
            raise APIError(503, "Authentication is unavailable; please retry.") from None
        try:
            if response.status_code in (401, 403):
                raise APIError(401, "Your session has expired; sign in again.")
            if response.status_code != 200:
                raise APIError(503, "Authentication is unavailable; please retry.")
            user = response.json()
            if not user.get("id") or user.get("is_anonymous"):
                raise APIError(401, "Sign in with a verified account.")
            return user["id"]
        finally:
            response.close()

    def rows(self, table, query=""):
        return self.request("GET", f"/rest/v1/{table}?{query}")

    def patch(self, table, query, values):
        return self.request(
            "PATCH", f"/rest/v1/{table}?{query}", values, {"Prefer": "return=representation"}
        )

    def rpc(self, name, values):
        return self.request("POST", "/rest/v1/rpc/" + name, values)

    @property
    def storage(self):
        if not hasattr(self, "_storage"):
            from .storage import storage_for

            self._storage = storage_for(self)
        return self._storage

    def signed(self, bucket, path, *, upload=False):
        return self.storage.signed(bucket, path, upload=upload)

    def download(self, bucket, path, destination, limit, *, max_seconds=600):
        return self.storage.download(bucket, path, destination, limit, max_seconds=max_seconds)

    def upload(self, bucket, path, source):
        return self.storage.upload(bucket, path, source)

    def exists(self, bucket, path):
        return self.storage.exists(bucket, path)

    def remove(self, bucket, paths):
        return self.storage.remove(bucket, paths)


def _download_worker():
    try:
        values = json.loads(sys.stdin.read())
        result = {"size": _download_stream(**values)}
    except ValueError:
        result = {"error": "invalid_file"}
    except (APIError, OSError):
        result = {"error": "unavailable"}
    print(json.dumps(result))


if __name__ == "__main__" and sys.argv[1:] == ["--download"]:
    _download_worker()
