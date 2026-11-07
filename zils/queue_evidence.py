"""The adapter track from paid API spend on delivered customer adapters (signed platform evidence).

The platform exports, per hotkey, the live API spend on open-tier adapters it delivered, credited for
EARNING_DAYS after delivery (zils_platform.adapter_spend), plus the window's TAO/USD price. This
module applies the payout rule over the window:

    paid   = min(1, revenue_cap * revenue / adapter budget value)
    weight = paid * spend / revenue for each registered hotkey; the rest of the share burns

Budget value is the adapter share of the miners' emission over the window, at the subnet's moving
alpha price read from the chain at the round block. A miner posing as a customer must pay at least
1 / revenue_cap of what it earns back. Unregistered hotkeys' spend burns too.
"""

import json
import math
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

from .runtime import verified

SCHEMA = "zils-adapter-evidence/v2"
MAX_BYTES = 8 * 1024**2
SS58 = re.compile("[1-9A-HJ-NP-Za-km-z]{40,64}")
HEX64 = re.compile("[a-f0-9]{64}")
ROW = {"hotkey", "spend_nanos"}
MINER_SPLIT = 0.41  # dTAO: miners take 41% of a subnet's alpha emission (owner 18%, validators 41%)
BLOCKS_PER_HOUR = 300  # 12-second blocks


def validate_track(track):
    if not isinstance(track, dict) or set(track) != {
        "url",
        "signer",
        "window_hours",
        "revenue_cap",
    }:
        raise ValueError("adapter_evidence needs exactly url, signer, window_hours and revenue_cap")
    url, signer, hours = track["url"], track["signer"], track["window_hours"]
    cap = track["revenue_cap"]
    if not isinstance(url, str) or not (
        url.startswith("https://") or url.startswith("http://127.0.0.1:")
    ):
        raise ValueError("adapter_evidence.url must be https")
    if not isinstance(signer, str) or not SS58.fullmatch(signer):
        raise ValueError("adapter_evidence.signer must be the platform evidence ss58 hotkey")
    if type(hours) is not int or not 1 <= hours <= 720:
        raise ValueError("adapter_evidence.window_hours must be an integer in 1..720")
    if type(cap) not in (int, float) or not 0 < cap <= 1:
        raise ValueError("adapter_evidence.revenue_cap must be a number in (0, 1]")
    return track


def window_end(now=None):
    """The current whole UTC hour: every validator asking in that hour gets the same document."""
    return (now or datetime.now(UTC)).astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


def fetch(track, until):
    from .protocol import opener

    query = urlencode({"until": iso(until), "hours": track["window_hours"]})
    with opener().open(f"{track['url']}?{query}", timeout=30) as response:
        body = response.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError("adapter evidence is too large")
    message = json.loads(body)
    verify(message, track, until)
    return message


def verify(message, track, until=None):
    """The pinned signer's payload for exactly this track's window, or ValueError."""
    payload = verified(message, track["signer"])
    if payload.get("schema") != SCHEMA or payload.get("signer") != track["signer"]:
        raise ValueError("adapter evidence schema or signer differs")
    try:
        end = datetime.fromisoformat(payload["until"].replace("Z", "+00:00"))
        start = datetime.fromisoformat(payload["since"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError):
        raise ValueError("adapter evidence window is invalid") from None
    if (until is not None and end != until) or end - start != timedelta(
        hours=track["window_hours"]
    ):
        raise ValueError("adapter evidence is for another window")
    price = payload.get("tao_usd")
    if type(price) not in (int, float) or not math.isfinite(price) or price <= 0:
        raise ValueError("adapter evidence has no TAO/USD price")
    rows = payload.get("miners")
    if not isinstance(rows, list):
        raise ValueError("adapter evidence has no miners")
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != ROW
            or not isinstance(row["hotkey"], str)
            or not SS58.fullmatch(row["hotkey"])
            or type(row["spend_nanos"]) is not int
            or row["spend_nanos"] <= 0
        ):
            raise ValueError("adapter evidence row is invalid")
    if len({row["hotkey"] for row in rows}) != len(rows):
        raise ValueError("a hotkey appears twice in the adapter evidence")
    return payload


def paid_fraction(payload, track, share, chain):
    """How much of the adapter share the window's revenue pays: revenue_cap times revenue over the
    share's value, at most all of it. `chain` holds the round block's alpha price and emission."""
    revenue = sum(row["spend_nanos"] for row in payload["miners"]) / 1e9
    blocks = track["window_hours"] * BLOCKS_PER_HOUR
    value = (
        share
        * MINER_SPLIT
        * chain["alpha_per_block"]
        * blocks
        * chain["alpha_price_tao"]
        * payload["tao_usd"]
    )
    return min(1.0, track["revenue_cap"] * revenue / value) if value > 0 else 0.0


def track_weights(payload, members, paid, burn_uid):
    """{uid: weight}: each registered hotkey's slice of spend times `paid`, the rest at `burn_uid`.
    The hotkey at the snapshot block decides the uid. Empty when no registered hotkey earns."""
    uid_of = {item["hotkey"]: int(uid) for uid, item in members.items()}
    revenue = sum(row["spend_nanos"] for row in payload["miners"])
    weights = {
        uid_of[row["hotkey"]]: paid * row["spend_nanos"] / revenue
        for row in payload["miners"]
        if row["hotkey"] in uid_of and paid > 0
    }
    if not weights:
        return {}
    burned = 1 - sum(weights.values())
    return dict(sorted({**weights, **({burn_uid: burned} if burned > 1e-12 else {})}.items()))
