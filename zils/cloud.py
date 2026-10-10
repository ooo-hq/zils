"""Bounded worker file transfers; no hosted storage or account dependencies."""

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
