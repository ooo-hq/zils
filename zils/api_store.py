"""Service-role persistence for API credentials and account-wide admission."""

import hashlib
import hmac
import re
import secrets
import uuid
from datetime import datetime, timezone

from .cloud import APIError, Supabase
from .decisions import DecisionError

KEY_PATTERN = re.compile(r"zils_sk_([0-9a-f]{32})_([A-Za-z0-9_-]{43})\Z")
KEY_FIELDS = ("id", "name", "prefix", "created_at", "revoked_at")


def identifier(value):
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise DecisionError(404, "not_found", "Resource not found.") from None


def metadata(row):
    return {key: row.get(key) for key in KEY_FIELDS}


class Store:
    def __init__(self, db=None):
        self.db = db or Supabase()

    def session_owner(self, token):
        owner = identifier(self.db.user(token))
        return owner

    def ensure_account(self, owner):
        owner = identifier(owner)
        try:
            self.db.rpc("zils_image_ensure_account", {"p_owner": owner})
        except APIError as error:
            if error.status == 409:
                raise DecisionError(403, "account_disabled", "Account is unavailable.") from None
            raise

    def create_key(self, owner, name):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise DecisionError(422, "invalid_name", "Use a key name of 1–80 characters.")
        key_id = uuid.uuid4()
        token = f"zils_sk_{key_id.hex}_{secrets.token_urlsafe(32)}"
        row = self.db.rpc(
            "zils_api_create_key",
            {
                "p_key": {
                    "id": str(key_id),
                    "owner_id": identifier(owner),
                    "name": name.strip(),
                    "prefix": token[:20],
                    "digest": hashlib.sha256(token.encode()).hexdigest(),
                }
            },
        )
        return {**metadata(row), "key": token}

    def list_keys(self, owner):
        owner = identifier(owner)
        result, cursor = [], None
        while True:
            query = f"owner_id=eq.{owner}&select={','.join(KEY_FIELDS)}&order=id&limit=100"
            if cursor:
                query += f"&id=gt.{cursor}"
            rows = self.db.rows("zils_api_keys", query)
            if not rows:
                return result
            result.extend(metadata(row) for row in rows)
            cursor = identifier(rows[-1]["id"])

    def revoke_key(self, owner, key_id):
        rows = self.db.patch(
            "zils_api_keys",
            f"id=eq.{identifier(key_id)}&owner_id=eq.{identifier(owner)}",
            {
                "revoked_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if not rows:
            raise DecisionError(404, "not_found", "Key not found.")
        return metadata(rows[0])

    def authenticate(self, token):
        match = KEY_PATTERN.fullmatch(token) if isinstance(token, str) else None
        if match:
            rows = self.db.rpc("zils_api_auth", {"p_id": str(uuid.UUID(hex=match[1]))})
            if rows and hmac.compare_digest(
                rows[0]["digest"], hashlib.sha256(token.encode()).hexdigest()
            ):
                return {"owner_id": rows[0]["owner_id"], "id": rows[0]["id"]}
        raise DecisionError(401, "invalid_credentials", "API key is invalid or revoked.")

    def admit(self, owner, key_id, request_id, tokens, billable_tokens=None):
        values = {
            "p_owner": identifier(owner),
            "p_key": identifier(key_id) if key_id else None,
            "p_request": identifier(request_id),
            "p_tokens": tokens,
        }
        if billable_tokens is not None:
            values["p_billable_tokens"] = billable_tokens
        status = self.db.rpc("zils_api_admit", values)
        if status == "allowed":
            return
        if status == "limited":
            raise DecisionError(
                429, "rate_limit", "Account throughput budget reached; retry later.", retry_after=1
            )
        if status == "disabled":
            raise DecisionError(401, "invalid_credentials", "Account or credential is disabled.")
        if status == "billing_meter_unavailable":
            raise DecisionError(503, status, "Model input billing is temporarily unavailable.")
        if status == "insufficient_credit":
            raise DecisionError(402, status, "Add credit to your account to run this request.")
        raise DecisionError(409, "request_conflict", "Request has already been admitted.")

    def finish_usage(self, request_id, tokens, status):
        self.db.rpc(
            "zils_api_finish_usage",
            {"p_request": identifier(request_id), "p_tokens": tokens, "p_status": status},
        )


def safe_cloud(error):
    """Translate existing cloud errors without leaking provider details."""
    if not isinstance(error, APIError):
        raise error
    return DecisionError(
        error.status,
        "storage_conflict" if error.status == 409 else "service_unavailable",
        str(error),
        retry_after=1 if error.status == 503 else None,
    )
