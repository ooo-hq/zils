"""Authenticated image asset and browser routes sharing the existing account boundary."""

import os
import re

from .decisions import DecisionError, invalid
from .image_contract import validate_image_request


def enabled(name="ZILS_IMAGES_ENABLED"):
    return os.environ.get(name, "").lower() in ("1", "true")


def require_images():
    if not enabled():
        raise DecisionError(503, "images_unavailable", "Image decisions are not enabled yet.")


class ImageApi:
    def __init__(self, api_store, image_store, registry, evaluate):
        self.store, self.images, self.registry, self.evaluate = (
            api_store,
            image_store,
            registry,
            evaluate,
        )

    def dispatch(self, method, path, bearer, body, request_id):
        require_images()
        if path in ("/v1/image-models", "/v1/image-decisions"):
            owner = self.store.session_owner(bearer)
        elif isinstance(bearer, str) and bearer.startswith("zils_sk_"):
            owner = self.store.authenticate(bearer)["owner_id"]
        else:
            owner = self.store.session_owner(bearer)
        self.store.ensure_account(owner)
        if path == "/v1/image-models" and method == "GET":
            return 200, {
                "models": [
                    {
                        **entry,
                        "stock": self.registry.resolve(entry["name"], owner)["owners"] is None,
                    }
                    for entry in self.registry.listing(owner)["models"]
                    if "image" in entry.get("capabilities", {}).get("modalities", [])
                ],
                "training_enabled": enabled("ZILS_IMAGE_TRAINING_ENABLED"),
            }
        if path == "/v1/image-decisions" and method == "POST":
            validate_image_request(body)
            return 200, self.evaluate(owner, None, body, request_id)
        if path == "/v1/image-assets" and method == "POST":
            if (
                not isinstance(body, dict)
                or set(body) - {"purpose", "job_id", "filename", "source_bytes", "source_sha256"}
                or not {"purpose", "filename", "source_bytes", "source_sha256"} <= set(body)
            ):
                raise invalid(["body"])
            if body["purpose"] == "training" and not enabled("ZILS_IMAGE_TRAINING_ENABLED"):
                raise DecisionError(
                    503, "image_training_unavailable", "Image training is not enabled yet."
                )
            return 201, self.images.create(
                owner,
                body["purpose"],
                body.get("job_id"),
                body["filename"],
                body["source_bytes"],
                body["source_sha256"],
            )
        match = re.fullmatch(r"/v1/image-assets/([^/]+)(/complete)?", path)
        if match and body == {}:
            if method == "POST" and match[2]:
                return 200, self.images.complete(owner, match[1])
            if method == "DELETE" and not match[2]:
                self.images.delete_unused(owner, match[1])
                return 204, {}
        raise DecisionError(404, "not_found", "Resource not found.")
