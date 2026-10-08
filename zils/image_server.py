"""Private image serving with verified canonical input and one serialized execution owner."""

import argparse
import copy
import hmac
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .cloud import APIError, download, trusted_url
from .decision_http import Server, make_handler
from .decisions import MAX_BODY, DecisionError
from .image_assets import MAX_CANONICAL_BYTES, PREPROCESSOR, canonicalize
from .image_contract import IMAGE_CAPABILITIES, make_image_response, validate_image_request
from .jev_server import SerialEngine


class _Execution:
    def __init__(self, owner):
        self.owner = owner

    def prepare(self, call):
        request, image, release, operation = call
        with tempfile.TemporaryDirectory(prefix="zils-image-request-") as temp:
            path = Path(temp) / "image.png"
            try:
                download(image["url"], path, MAX_CANONICAL_BYTES, max_seconds=15)
                raw = path.read_bytes()
                canonical = canonicalize(raw)
                if (
                    len(raw) != image["bytes"]
                    or canonical.source_sha256 != image["sha256"]
                    or canonical.sha256 != image["sha256"]
                    or (canonical.width, canonical.height) != (image["width"], image["height"])
                ):
                    raise ValueError("canonical image mismatch")
                qid, question = next(iter(request["questions"].items()))
                prepared = self.owner.engine.prepare(path, request["state"], question)
            except (ValueError, APIError):
                raise DecisionError(
                    422, "invalid_image", "The verified image could not be read; retry the upload."
                ) from None
        count = prepared.get("input_tokens")
        if type(count) is not int or count < 0 or count > 4096:
            raise DecisionError(
                413, "context_limit", "The image and question exceed this model context limit."
            )
        return request, release, operation, qid, prepared

    def predict(self, prepared):
        request, release, operation, qid, inputs = prepared
        if operation == "prepare":
            return {"reserved_tokens": inputs["input_tokens"]}
        try:
            row = self.owner.engine.predict(inputs, release["temperature"])
        except (ValueError, RuntimeError):
            raise DecisionError(
                503, "image_unavailable", "Image model is unavailable; retry shortly."
            ) from None
        predictions = {qid: row}
        make_image_response(request["model"], request, predictions)
        return {"predictions": predictions}


class ImageRuntime:
    def __init__(self, engine, releases, image_store_origin, *, token=None, timeout=30):
        self.engine = engine
        self.releases = copy.deepcopy(releases)
        self.token = token if token is not None else os.environ["ZILS_IMAGE_RUNTIME_TOKEN"]
        if not isinstance(self.token, str) or not self.token:
            raise ValueError("Image runtime requires a private token")
        self.origin = trusted_url(image_store_origin)
        parsed = urlsplit(self.origin)
        if parsed.path or parsed.query:
            raise ValueError("Image storage origin cannot contain a path")
        self.serial = SerialEngine(_Execution(self), timeout=timeout)

    def close(self):
        self.serial.close()

    def dispatch(self, method, path, token, body, request_id):
        if not isinstance(token, str) or not hmac.compare_digest(
            token.encode(), self.token.encode()
        ):
            raise DecisionError(401, "invalid_credentials", "Invalid runtime credential.")
        if method == "GET" and path == "/health":
            return 200, {
                "models": {
                    k: {"release_id": k, "fingerprint": v["fingerprint"]}
                    for k, v in self.releases.items()
                },
                "capabilities": IMAGE_CAPABILITIES,
            }
        if method != "POST" or path not in ("/prepare", "/v1/prepare", "/v1/systemone"):
            raise DecisionError(404, "not_found", "Resource not found.")
        if (
            not isinstance(body, dict)
            or set(body) - {"request", "image", "fingerprint", "lane"}
            or not {"request", "image", "fingerprint"} <= set(body)
        ):
            raise DecisionError(422, "invalid_request", "Expected a verified image envelope.")
        request = validate_image_request(body["request"])
        release = self.releases.get(request["model"])
        if not release or release["fingerprint"] != body["fingerprint"]:
            raise DecisionError(
                503, "release_mismatch", "The approved image release is unavailable."
            )
        if body.get("lane", "realtime") != "realtime":
            raise DecisionError(
                422, "image_bulk_unsupported", "Image models support realtime requests only."
            )
        image = body["image"]
        try:
            if not isinstance(image, dict) or image["id"] != request["images"][0]["asset_id"]:
                raise ValueError()
            parsed = urlsplit(image["url"])
            if (
                trusted_url(image["url"]).split("?", 1)[0] != self.origin + parsed.path
                or not re.fullmatch(
                    r"/storage/v1/object/sign/zils-images/[a-f0-9-]{36}/"
                    + re.escape(image["id"])
                    + r"/canonical\.png",
                    parsed.path,
                )
                or not re.fullmatch("[a-f0-9]{64}", image["sha256"])
                or type(image["bytes"]) is not int
                or not 0 < image["bytes"] <= MAX_CANONICAL_BYTES
                or image["preprocessor"] != PREPROCESSOR
                or datetime.fromisoformat(image["expires_at"]) <= datetime.now(timezone.utc)
            ):
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise DecisionError(
                422, "invalid_image_reference", "Verified image reference is unavailable."
            ) from None
        operation = "predict" if path == "/v1/systemone" else "prepare"
        result = self.serial.evaluate(
            (copy.deepcopy(request), copy.deepcopy(image), release, operation)
        )
        return 200, {
            "release_id": request["model"],
            "fingerprint": release["fingerprint"],
            **result,
        }


def main():
    from .imajev import ImageEngine, verify_reference

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--image-store-origin", required=True)
    parser.add_argument("--port", type=int, default=8931)
    args = parser.parse_args()
    manifest = verify_reference(args.reference)
    engine = ImageEngine(args.reference, "cuda")
    runtime = ImageRuntime(engine, {manifest["release_id"]: manifest}, args.image_store_origin)
    server = Server(
        ("127.0.0.1", args.port), make_handler(runtime.dispatch, body_limit=MAX_BODY + 4096)
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        runtime.close()


if __name__ == "__main__":
    main()
