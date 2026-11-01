"""Soup seat rates. House ingredients (open decision models known at launch) earn nothing; a body within
a cosine threshold of a known public model's sketch is a scout re-upload and earns the scout rate; any
other body earns the trainer rate. These rates must match the ones `zils.soup_round` applies to seats."""

import json
import re
from pathlib import Path

from . import commitments
from .soup_copies import cosine, explained_share

HOUSE_RATE = 0.0
SCOUT_RATE = 0.25
TRAINER_RATE = 1.0
COSINE_SCOUT = 0.9
HOUSE_MIX_MAX = 0.05  # a body whose change house bodies explain to within this share is a house mix
HOUSE_PATH = Path(__file__).with_name("data") / "soup-house.json"
FIELDS = {"name", "repo", "revision", "body_sha256"}


def load_house(path=None):
    """The pinned house registry: a list of {"name", "repo", "revision", "body_sha256" (or null)}."""
    return validate_house(json.loads(Path(path or HOUSE_PATH).read_text()))


def validate_house(value):
    """Check a house registry list (as `load_house` reads it) and return it."""
    if not isinstance(value, list):
        raise ValueError("house registry must be a list")
    names = set()
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != FIELDS:
            raise ValueError(f"house entries need exactly {', '.join(sorted(FIELDS))}")
        if not isinstance(entry["name"], str) or not entry["name"] or entry["name"] in names:
            raise ValueError("house entry names must be unique non-empty strings")
        names.add(entry["name"])
        commitments.check_repo(entry["repo"])
        commitments.check_revision(entry["revision"])
        sha = entry["body_sha256"]
        if sha is not None and not (isinstance(sha, str) and re.fullmatch("[a-f0-9]{64}", sha)):
            raise ValueError(f"house entry {entry['name']} body_sha256 must be 64 hex or null")
    return value


def _house_hashes(house):
    out = set()
    for entry in house:
        sha = entry.get("body_sha256") if isinstance(entry, dict) else entry
        if sha is not None:
            out.add(sha)
    return out


def rate(
    body_sha256,
    sketch,
    *,
    house,
    known,
    hotkey,
    cosine_scout=COSINE_SCOUT,
    house_sketches=(),
    house_mix_max=HOUSE_MIX_MAX,
    base_residual=None,
) -> float:
    """HOUSE_RATE for a house body hash; SCOUT_RATE when the sketch's cosine to any known public model is at
    least `cosine_scout`; else TRAINER_RATE. `house` holds registry entries or hashes; `known` holds
    {"body_sha256", "sketch", "hotkey" (None for a public model), "repo"}. House entries never make a
    scout: a fine-tune of a house model is a trainer's work (an exact or near-exact re-upload is
    rejected as a copy before rating). Known entries committed by the submitting `hotkey` itself are
    skipped, so a trainer's own earlier upload never makes it a scout, whatever its repo.

    A house mix is paid as a scout: a body whose change from the base is almost entirely a
    non-negative mix of the house bodies' changes (`house_sketches`) and the listed public models'
    (the `known` entries with no hotkey), with a residual share of its squared norm under
    `house_mix_max`. A light fine-tune leaves more than that and is a trainer's.
    So is a body whose `base_residual` (``soup_copies.base_residual`` against the house bodies, the
    soup and the release, measured when it was admitted) is under `house_mix_max`: a near-pure
    re-weighting of the soup or release. A real fine-tune of either keeps the trainer rate."""
    house_hashes = _house_hashes(house)
    if body_sha256 in house_hashes:
        return HOUSE_RATE
    public = [
        e["sketch"]
        for e in known
        if e.get("hotkey") is None and e.get("body_sha256") not in house_hashes
    ]
    mixable = [*house_sketches, *public]
    if mixable and 1 - explained_share(sketch, mixable) < house_mix_max - 1e-9:
        return SCOUT_RATE
    if base_residual is not None and base_residual < house_mix_max - 1e-9:
        return SCOUT_RATE
    for entry in known:
        if entry.get("body_sha256") in house_hashes:
            continue
        if entry.get("hotkey") is not None and entry.get("hotkey") == hotkey:
            continue
        if cosine(sketch, entry["sketch"]) >= cosine_scout:
            return SCOUT_RATE
    return TRAINER_RATE
