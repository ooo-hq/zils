"""Customer admission is separate from authentication and model ownership."""

from .cloud import APIError


def require_access(db, owner):
    if db.rpc("zils_access_allowed", {"p_owner": owner}) is not True:
        raise APIError(403, "Early access is required. Request access at zils.ai/early-access.")
