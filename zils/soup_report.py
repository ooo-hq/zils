"""A miner's private soup round report: where each of its bodies stopped, why, and how far it was
from the bar. Built once per round from the validator's private record and served only to the
hotkey that signs for it. Reasons are codes from a fixed list and scores are rounded to half a
point; copy distances, the body a copy matched, family scores, items and raw error text stay out."""

import json
import math
import re
import time
import uuid
from pathlib import Path

HEX64 = re.compile("[a-f0-9]{64}")
KIND = "soup-report"
WINDOW = 120  # seconds a signed request stays valid, and its nonce remembered
FILE = "reports.json"
DEFERRED = ("round_limit", "download_failed", "backing_off")  # retried without a new commitment
# The first match names the reason; `check_failed` is any other failed check.
_REASONS = (
    ("retried next round", "round_limit"),
    ("download failed", "download_failed"),
    ("backs off", "backing_off"),
    ("committed first by another hotkey", "committed_first_by_another"),
    ("rejected in an earlier round", "rejected_earlier"),
    ("failed in an earlier round", "failed_earlier"),
    ("behavioural twin", "twin"),
    ("copy: house re-upload", "house_reupload"),
    ("copy: no change", "no_change"),
    ("copy:", "near_copy"),
    ("zils-owner.json", "owner_file"),
    ("license", "license"),
    ("hash differs", "hash_mismatch"),
    ("change from the base", "change_cap"),
    ("not finite", "not_finite"),
    ("byte limit", "too_large"),
    ("commitment", "bad_commitment"),
    ("body ", "bad_format"),
    ("safetensors", "bad_format"),
)
REASONS = (
    *dict.fromkeys(code for _, code in _REASONS),
    "check_failed",
    "grade_failed",
    "below_screen",
    "pool_full",
    "no_soup_gain",
    "soup_full",
    "current_soup_kept",
    "below_floor",
    "fine_tune_gate",
    "unpaid_rate",
)


_CHECK = "run zils-miner commit-body --check --path <folder> to see which."
_OWN = "commit a body with your own training."
# reason -> (what happened, what to do next or None), as the miner CLI prints them.
EXPLAIN = {
    None: ("Your body is in the soup and its seat pays.", None),
    "round_limit": (
        "The round reached its limit on new bodies, claims, downloads or disk before your turn.",
        "nothing to do; it is tried again next round.",
    ),
    "download_failed": (
        "The validator could not download your repo.",
        "check that the repo is public; it is tried again in a later round.",
    ),
    "backing_off": (
        "Your hotkey's last claim failed, so it sits out a few rounds.",
        "nothing to do; it is tried again after the wait.",
    ),
    "committed_first_by_another": ("Another hotkey committed this exact body first.", _OWN),
    "rejected_earlier": ("This body was refused in an earlier round.", "commit a new body."),
    "failed_earlier": (
        "This commitment failed in an earlier round.",
        "upload a new revision and commit it.",
    ),
    "twin": (
        "Your body answers almost exactly like a body already in the pool.",
        "commit a new revision; it is also tried again when that body leaves the pool.",
    ),
    "house_reupload": (
        "Your body is a re-upload of a house ingredient.",
        "fine-tune it first; a fine-tune of a house ingredient is welcome.",
    ),
    "no_change": ("Your body is the pool's parent, unchanged.", _OWN),
    "near_copy": (
        "Your body is a near-copy of an earlier body, or mostly built from other miners' bodies.",
        _OWN,
    ),
    "owner_file": (
        "zils-owner.json is missing or names another hotkey.",
        "run zils-miner commit-body --write-owner --path <folder>, upload, then commit.",
    ),
    "license": (
        "The model card has no accepted license.",
        "add a license to the README.md front matter, upload, then commit.",
    ),
    "hash_mismatch": (
        "The downloaded body does not hash to your commitment.",
        "commit the revision you uploaded with zils-miner commit-body.",
    ),
    "change_cap": (
        "Your body moved further from the pool's parent than the change bound allows.",
        _CHECK.replace("which", "the change and the bound"),
    ),
    "not_finite": ("A weight in your body is NaN or infinite.", _CHECK),
    "too_large": ("Your body is over the pool's size limit.", _CHECK),
    "bad_commitment": (
        "The commitment text is malformed or names an unknown pool.",
        "commit again with zils-miner commit-body.",
    ),
    "bad_format": ("A tensor is missing, extra, misshapen or of the wrong type.", _CHECK),
    "check_failed": ("Your body failed a check.", _CHECK),
    "grade_failed": (
        "Your body could not be graded this round.",
        "nothing to do yet; after three such rounds it is refused.",
    ),
    "below_screen": (
        "Your screening score was below the lowest score that entered the pool.",
        "train a stronger body; only the best new bodies are graded in full.",
    ),
    "pool_full": (
        "An earlier body of yours was left out of this round's full pool.",
        "commit a stronger body.",
    ),
    "no_soup_gain": (
        "Adding your body did not raise the soup's score by the margin needed.",
        "train where the soup is weak, not where it is already strong.",
    ),
    "soup_full": (
        "The soup reached its member limit before your body's turn.",
        "raise your score alone; bodies are tried in that order.",
    ),
    "current_soup_kept": (
        "Your body was in the new soup, but that soup did not beat the current one by the margin.",
        "nothing is wrong with the body; the new soup needs a larger gain.",
    ),
    "below_floor": (
        "Your body is in the soup, but its contribution is under the seat floor.",
        "add more of what the soup lacks.",
    ),
    "fine_tune_gate": (
        "Your body is a fine-tune that did not beat its base alone by the margin.",
        "train until your body alone beats what you started from.",
    ),
    "unpaid_rate": ("Your body is in the soup at rate 0, which pays nothing.", None),
}


def reason_code(text):
    return next((code for part, code in _REASONS if part in text), "check_failed")


def _half(value):
    return math.floor(float(value) * 2 + 0.5) / 2


def _short(value, needed):
    """``value`` at half a point, kept below a bar it missed: 0.3 against 0.5 shows 0.0, not 0.5."""
    shown = _half(value)
    return shown if value >= needed or shown < needed else math.ceil(needed * 2 - 1) / 2


def claims(snapshot):
    """{uid: {"hotkey", "sha256"}} for the round snapshot's commitments by roster hotkeys."""
    out = {}
    for uid, item in snapshot.get("commitments", {}).items():
        hotkey = snapshot.get("members", {}).get(str(uid))
        if hotkey is None:
            continue
        parts = str(item.get("data") or "").split(":")
        sha = next((p for p in parts if HEX64.fullmatch(p)), None)
        out[str(uid)] = {"hotkey": hotkey, "sha256": sha}
    return out


def _tried(private):
    """{body: greedy's trace row for it}. A row's ``try`` is the soup so far plus the body tried;
    the soup starts as the one body of the first row that greedy's final soup holds."""
    rows = [row for row in private.get("trace", []) if "versus" not in row]
    if not rows:
        return {}
    kept = next((row for row in private["trace"] if row.get("versus") == "current"), None)
    final = (
        set(kept["try"])
        if kept is not None and not kept["accepted"]
        else {m["body_sha256"] for m in private["members"]}
    )
    first = [sha for sha in rows[0]["try"] if sha in final] or rows[0]["try"]
    soup, out = {max(first, key=lambda sha: private["solo"].get(sha, -math.inf))}, {}
    for row in rows:
        for body in set(row["try"]) - soup:
            out[body] = row
        if row["accepted"]:
            soup = set(row["try"])
    return out


def build(private, claims, owners):
    """{hotkey: [one entry per body or refused commitment]} for one round.

    ``private`` is the round's private record, ``claims`` the snapshot's commitments (``claims``)
    and ``owners`` {body: hotkey} for the pool's known bodies (None for a house body). An entry is
    {"body_sha256", "stage", "reason"} plus, for a graded body, the numbers behind the stage.
    """
    from .soup_round import BASE_MARGIN, FLOOR, MARGIN

    members = {m["body_sha256"]: m for m in private["members"]}
    solo, screen = private["solo"], private.get("screen", {})
    tried = _tried(private)
    kept = next((row for row in private.get("trace", []) if row.get("versus") == "current"), None)
    pooled = [score for sha, score in screen.items() if sha in solo]
    screen_bar = 100 * min(pooled) if pooled else None
    soup_categories = private.get("soup_categories", {})
    reports = {}

    def entry(hotkey, sha, stage, reason, **numbers):
        reports.setdefault(hotkey, []).append(
            {"body_sha256": sha, "stage": stage, "reason": reason, **numbers}
        )

    def graded(sha):
        numbers = {}
        if sha in private.get("rates", {}):
            numbers["rate"] = private["rates"][sha]
        if sha in screen and screen_bar is not None:
            numbers["screen"] = {
                "score": _short(100 * screen[sha], screen_bar),
                "needed": _half(screen_bar),
            }
        if sha in solo and private.get("soup_score") is not None:
            numbers["solo"] = {"score": _half(solo[sha]), "soup": _half(private["soup_score"])}
        own = private.get("solo_categories", {}).get(sha)
        if own:
            numbers["categories"] = {
                name: {"score": _half(score), "soup": _half(soup_categories[name])}
                for name, score in own.items()
                if name in soup_categories
            }
        return numbers

    def trial(row):
        return {"gain": _short(row["score"] - row["best_before"], MARGIN), "needed": MARGIN}

    refused = set()
    for uid, text in private.get("refused", {}).items():
        claim = claims.get(str(uid))
        if claim is None:
            continue
        code = reason_code(text)
        refused.add((claim["hotkey"], claim["sha256"]))
        entry(claim["hotkey"], claim["sha256"], "deferred" if code in DEFERRED else "refused", code)
    bodies = dict(owners)
    for claim in claims.values():  # a body rejected at intake is known only by its commitment
        if claim["sha256"] in private.get("rejected", {}):
            bodies.setdefault(claim["sha256"], claim["hotkey"])
    for sha, hotkey in bodies.items():
        if hotkey is None or (hotkey, sha) in refused:
            continue
        if sha in members:
            member = members[sha]
            seat = {"contribution": _short(member["contribution"], FLOOR), "floor": FLOOR}
            if member["seat_weight"] > 0:
                entry(hotkey, sha, "seated", None, **graded(sha), seat=seat)
            elif sha in private.get("gated", {}):
                base = re.search(r"base (house|soup|release)", private["gated"][sha])
                gate = {"base": base[1] if base else None, "needed": BASE_MARGIN}
                entry(hotkey, sha, "seated_unpaid", "fine_tune_gate", **graded(sha), seat=seat,
                      gate=gate)  # fmt: skip
            else:
                reason = "unpaid_rate" if member["rate"] == 0 else "below_floor"
                entry(hotkey, sha, "seated_unpaid", reason, **graded(sha), seat=seat)
        elif sha in solo:
            row = tried.get(sha)
            if row is not None and not row["accepted"]:
                entry(hotkey, sha, "not_selected", "no_soup_gain", **graded(sha), trial=trial(row))
            elif kept is not None and sha in kept["try"]:  # in greedy's soup, which lost
                entry(hotkey, sha, "not_selected", "current_soup_kept", **graded(sha),
                      trial=trial(kept))  # fmt: skip
            else:  # the soup was full before greedy reached it
                entry(hotkey, sha, "not_selected", "soup_full", **graded(sha))
        elif sha in private.get("twins", {}):
            entry(hotkey, sha, "refused", "twin")
        elif sha in private.get("grade_failures", {}):
            entry(hotkey, sha, "deferred", "grade_failed")
        elif sha in private.get("rejected", {}):
            entry(hotkey, sha, "refused", reason_code(private["rejected"][sha]))
        elif sha in screen:
            entry(hotkey, sha, "screened_out", "below_screen", **graded(sha))
        else:  # graded in an earlier round and left out of this round's full pool
            entry(hotkey, sha, "screened_out", "pool_full")
    return reports


def _read(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def read(config, directory, hotkey):
    """{pool: {"round_id", "bodies"}} for every enabled soup pool: the hotkey's entries in the
    pool's newest finished round (``round_id`` None before the first)."""
    from . import model_pools, soup_pool

    pools = {}
    if "base_pools" not in config:
        return pools
    for pool, item in model_pools.validate(config["base_pools"])["pools"].items():
        if not (item["enabled"] and model_pools.is_soup(item)):
            continue
        state = soup_pool.state_dir(Path(directory), pool)
        round_id = _read(state / "pool.json").get("round_id")
        found = _read(state / "rounds" / round_id / FILE) if isinstance(round_id, str) else {}
        bodies = found.get(hotkey)
        pools[pool] = {"round_id": round_id, "bodies": bodies if isinstance(bodies, list) else []}
    return pools


def sign(key, validator_hotkey):
    """A miner's report request, signed by its hotkey and bound to one validator."""
    from .runtime import signed

    payload = {
        "kind": KIND,
        "hotkey": key.ss58_address,
        "validator": validator_hotkey,
        "timestamp": int(time.time()),
        "nonce": str(uuid.uuid4()),
    }
    return signed(payload, key)


def answer(config, directory, message, seen, now=None):
    """(status, payload) for a report request: the signing hotkey's own report, or 401.

    ``seen`` ({nonce: timestamp}, kept by the caller) refuses a replay inside the window."""
    from .runtime import verified

    now = time.time() if now is None else now
    for old in [n for n, at in seen.items() if now - at > WINDOW]:
        del seen[old]
    try:
        payload = verified(message, message["payload"]["hotkey"])
        if set(payload) != {"kind", "hotkey", "validator", "timestamp", "nonce"}:
            raise ValueError()
        if payload["kind"] != KIND or payload["validator"] != config["validator_hotkey"]:
            raise ValueError()
        stamp, nonce = payload["timestamp"], payload["nonce"]
        if type(stamp) is not int or abs(now - stamp) > WINDOW:
            raise ValueError()
        if str(uuid.UUID(nonce)) != nonce or nonce in seen:
            raise ValueError()
    except (ValueError, TypeError, KeyError, AttributeError):
        return 401, {"error": "invalid report request signature or timestamp"}
    seen[nonce] = stamp
    hotkey = payload["hotkey"]
    return 200, {"kind": KIND, "hotkey": hotkey, "pools": read(config, directory, hotkey)}
