"""Soup round exams drawn from Zils Current dated exports (``zils.current`` ``export --bank``).

A soup pool with ``exam: {"source": "current", "current_bank": DIR}`` builds each round's subset from
two parts. The **Current part** holds recent real-world items from the dated folders under ``DIR``
(``<YYYY-MM-DD>/items.jsonl`` and ``manifest.json``). The **bank part** comes from the fixed round
bank. Current streams are families in the base score's category and block cells. In every cell the
Current part covers, its streams replace the fixed bank's families, and the fixed bank fills the other
cells, so the base score keeps all its cells.

The rules for the Current part:

* A folder is used only when its manifest checks out: the ``items.jsonl`` hash and row count match,
  the overlap check ran, and the mode is ``local`` or ``publishable``. Any other folder is skipped and
  logged. A bad folder never stops the round.
* An item is eligible when it resolved within ``max_age_days`` before the round's commit deadline,
  has been drawn in fewer than ``max_uses`` earlier rounds (the use ledger), and fits the pool
  model's input cap.
* The draw takes ``current_items`` items. No stream may hold more than ``stream_max`` of them, and at
  least ``min_streams`` streams must contribute. Within a stream the labels are balanced. Newer
  resolution days come first. Within a day, items are ranked by an HMAC of the item id under this
  validator's local secret (``soup_round.sketch_seed``), because anyone can rebuild Current items.
* If the Current part cannot meet these rules, the whole round falls back to the fixed bank. The
  draw is then exactly the one the fixed bank alone would give, and the fallback is recorded.

``prepare`` writes the round's ``subset.json`` (which ``soup_round.run`` then reads), the Current
items it drew (``current-items.jsonl``) and the record (``bank-source.json``) into the round
folder. A resumed round reuses them. The bank it returns also holds the Current items that earlier
rounds drew, so a later round can still score those items after they age out.
"""

import hashlib
import hmac
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

SOURCES = ("bank", "current")
DEFAULTS = {
    "max_age_days": 14,
    "max_uses": 3,
    "current_items": 1000,
    "min_items": 300,
    "min_streams": 3,
    "stream_max": 0.4,
}
KEEP_ROUNDS = 8  # earlier rounds whose drawn Current items stay in the bank (pooled decisions)
MODES = ("local", "publishable")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ITEM_KEYS = ("id", "family", "category", "block", "state", "question", "label", "resolved_at")


def validate(exam):
    """The ``exam`` setting of a soup pool, with defaults filled in. Without it: the fixed bank."""
    if exam is None:
        return {"source": "bank"}
    if not isinstance(exam, dict) or exam.get("source") not in SOURCES:
        raise ValueError('soup exam source must be "bank" or "current"')
    if exam["source"] == "bank":
        if set(exam) != {"source"}:
            raise ValueError("soup exam settings apply only to the current source")
        return {"source": "bank"}
    unknown = set(exam) - {"source", "current_bank"} - set(DEFAULTS)
    if unknown:
        raise ValueError("unknown soup exam settings: " + ", ".join(sorted(unknown)))
    path = exam.get("current_bank")
    if not isinstance(path, str) or not path:
        raise ValueError("soup exam source current needs current_bank (a directory)")
    result = {"source": "current", "current_bank": path, **DEFAULTS}
    result.update({k: exam[k] for k in DEFAULTS if k in exam})
    for key in ("max_age_days", "max_uses", "current_items", "min_items", "min_streams"):
        if type(result[key]) is not int:
            raise ValueError(f"soup exam {key} must be an integer")
    if not 1 <= result["max_age_days"] <= 90:
        raise ValueError("soup exam max_age_days must be in [1, 90]")
    if not 1 <= result["max_uses"] <= 24:
        raise ValueError("soup exam max_uses must be in [1, 24]")
    if not 1 <= result["min_streams"] <= 20:
        raise ValueError("soup exam min_streams must be in [1, 20]")
    if not 1 <= result["min_items"] <= result["current_items"]:
        raise ValueError("soup exam min_items must be in [1, current_items]")
    share = result["stream_max"]
    if type(share) not in (int, float) or not 0 < share <= 1:
        raise ValueError("soup exam stream_max must be in (0, 1]")
    if share * result["min_streams"] < 1:
        raise ValueError("soup exam stream_max x min_streams must reach 1, or no draw can fill")
    return result


def _parse_time(text):
    moment = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def verify(folder):
    """(manifest sha256, manifest, items) of one dated folder; ValueError(reason) if it fails."""
    folder = Path(folder)
    try:
        raw = (folder / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        data = (folder / "items.jsonl").read_bytes()
    except (OSError, ValueError) as error:
        raise ValueError(f"unreadable: {type(error).__name__}") from None
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not an object")
    if manifest.get("date") != folder.name:
        raise ValueError("manifest date differs from the folder name")
    if manifest.get("mode") not in MODES:
        raise ValueError(f"mode {manifest.get('mode')!r} is not local or publishable")
    overlap = manifest.get("overlap")
    if not isinstance(overlap, dict) or overlap.get("checked") is not True:
        raise ValueError("the overlap check did not run")
    entry = (manifest.get("files") or {}).get("items.jsonl")
    if not isinstance(entry, dict) or entry.get("sha256") != hashlib.sha256(data).hexdigest():
        raise ValueError("items.jsonl does not match the manifest hash")
    try:
        items = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    except ValueError:
        raise ValueError("items.jsonl is not JSON lines") from None
    if entry.get("rows") != len(items):
        raise ValueError("items.jsonl row count differs from the manifest")
    return hashlib.sha256(raw).hexdigest(), manifest, items


def _valid_item(item):
    from . import core

    if not isinstance(item, dict) or any(k not in item for k in _ITEM_KEYS):
        return False
    try:
        core.validate_cases([item])
        _parse_time(item["resolved_at"])
    except (TypeError, ValueError):
        return False
    return isinstance(item["state"].get("text"), str)


def _tokens(item):
    from . import current_items

    return current_items.estimated_tokens(item["state"]["text"]) + current_items.QUESTION_TOKENS


def load(bank_dir, *, deadline, max_age_days, max_tokens, uses, max_uses, log=print):
    """Eligible items from the dated folders under ``bank_dir`` and what was read and excluded."""
    root = Path(bank_dir)
    start = deadline - timedelta(days=max_age_days)
    folders, skipped, excluded = [], {}, Counter()
    eligible, seen = {}, set()
    names = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if not root.is_dir():
        log(f"soup exam: Zils Current bank {root} is not a directory")
    for name in names:
        if not _DATE.fullmatch(name):
            continue
        try:
            day = datetime.strptime(name, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            continue
        if day + timedelta(days=1) <= start or day > deadline:
            continue  # the folder's items all resolved outside the window
        try:
            manifest_sha, manifest, items = verify(root / name)
        except ValueError as error:
            skipped[name] = str(error)
            log(f"soup exam: skipped Zils Current folder {name}: {error}")
            continue
        used = 0
        for item in items:
            if not _valid_item(item):
                excluded["invalid"] += 1
                continue
            resolved = _parse_time(item["resolved_at"])
            key = item.get("content_sha256") or item["id"]
            if item["id"] in eligible or item["id"] in seen or key in seen:
                excluded["duplicate"] += 1
            elif not start <= resolved <= deadline:
                excluded["outside_window"] += 1
            elif uses.get(item["id"], 0) >= max_uses:
                excluded["used_up"] += 1
            elif _tokens(item) > max_tokens:
                excluded["too_long"] += 1
            else:
                eligible[item["id"]] = item
                used += 1
            seen.update((item["id"], key))
        folders.append(
            {
                "date": name,
                "manifest_sha256": manifest_sha,
                "items_sha256": manifest["files"]["items.jsonl"]["sha256"],
                "mode": manifest["mode"],
                "rows": len(items),
                "eligible": used,
            }
        )
    return {
        "items": list(eligible.values()),
        "folders": folders,
        "skipped": skipped,
        "excluded": dict(sorted(excluded.items())),
    }


def _fill(capacity, size):
    """Spread ``size`` over groups as evenly as their ``capacity`` allows (water-filling)."""
    quota = dict.fromkeys(capacity, 0)
    left = size
    while left > 0:
        open_ = sorted(k for k in capacity if quota[k] < capacity[k])
        if not open_:
            break
        share = max(1, left // len(open_))
        for key in open_:
            add = min(share, capacity[key] - quota[key], left)
            quota[key] += add
            left -= add
            if left == 0:
                break
    return quota


def draw(items, secret, settings):
    """(chosen items, per-stream counts) for the Current part; ValueError when it cannot fill."""
    size = settings["current_items"]
    cap = int(settings["stream_max"] * size)
    streams = defaultdict(lambda: defaultdict(list))
    for item in items:
        streams[item["family"]][item["label"]].append(item)
    usable = {s: labels for s, labels in streams.items() if len(labels) >= 2}
    if len(usable) < settings["min_streams"]:
        raise ValueError(
            f"{len(usable)} usable streams (two labels or more), {settings['min_streams']} needed"
        )

    def rank(item):
        day = item["resolved_at"][:10]
        tag = hmac.new(secret, item["id"].encode(), hashlib.sha256).digest()
        return (-int(day.replace("-", "")), tag)

    def balanced(labels):  # how many items a stream can give with labels kept balanced
        floor = min(len(rows) for rows in labels.values())
        return sum(min(len(rows), floor) for rows in labels.values())

    quota = _fill({s: min(cap, balanced(labels)) for s, labels in usable.items()}, size)
    chosen, counts = [], {}
    for stream, labels in sorted(usable.items()):
        if not quota[stream]:
            continue
        per_label = _fill({k: len(v) for k, v in labels.items()}, quota[stream])
        picked = []
        for label, rows in sorted(labels.items()):
            picked += sorted(rows, key=rank)[: per_label[label]]
        chosen += picked
        counts[stream] = {
            "drawn": len(picked),
            "available": sum(map(len, labels.values())),
            "labels": dict(sorted(Counter(c["label"] for c in picked).items())),
            "resolved_from": min(c["resolved_at"] for c in picked),
            "resolved_until": max(c["resolved_at"] for c in picked),
        }
    contributing = [s for s in counts if counts[s]["drawn"]]
    if len(contributing) < settings["min_streams"]:
        raise ValueError(
            f"{len(contributing)} streams contribute, {settings['min_streams']} needed"
        )
    if len(chosen) < settings["min_items"]:
        raise ValueError(
            f"{len(chosen)} items under the stream cap, {settings['min_items']} needed"
        )
    return chosen, counts


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    partial.replace(path)


def _earlier_items(state_dir, round_id):
    """Current items drawn by the latest ``KEEP_ROUNDS`` earlier rounds, from their round folders."""
    rounds = Path(state_dir) / "rounds"
    paths = sorted(
        (p for p in rounds.glob("*/current-items.jsonl") if p.parent.name != round_id),
        key=lambda p: p.parent.name,
    )
    items = {}
    for path in paths[-KEEP_ROUNDS:]:
        for item in _read_jsonl(path):
            items.setdefault(item["id"], item)
    return items


def _deadline(config, round_id):
    day = datetime.fromisoformat(round_id[-10:]).replace(tzinfo=UTC)
    hours, minutes = map(int, config["schedule"]["deadline_utc"].split(":"))
    return day + timedelta(hours=hours, minutes=minutes)


def _max_tokens(config):
    from . import models

    model = config["model"]
    if isinstance(model, dict) and type(model.get("max_input_tokens")) is int:
        return model["max_input_tokens"]
    return models.spec(model if isinstance(model, str) else model.get("id"))["max_input_tokens"]


def _merge(bank, extra):
    by_id = {c["id"]: c for c in bank}
    for identity, item in extra.items():
        by_id.setdefault(identity, item)
    return list(by_id.values())


def _parts(items, bank, secret, settings, *, size, seed, retired):
    """(Current part, its stream counts, the cells it covers, bank part) filling ``size`` items.

    In the cells the Current part covers, its streams replace the fixed bank's families. When the
    rest of the bank cannot fill the remainder (Current covers most or all cells), the Current
    part grows to take the remainder, up to the whole subset."""
    from . import soup_exam

    target = min(settings["current_items"], size)
    for _ in range(4):  # a larger draw can only cover more cells, so this settles quickly
        current, streams = draw(items, secret, {**settings, "current_items": target})
        cells = sorted({(c["category"], c["block"]) for c in current})
        rest = [c for c in bank if (c.get("category"), c.get("block")) not in set(cells)]
        need = size - len(current)
        free = sum(c["id"] not in retired for c in rest)
        if need <= free or target >= size:
            break
        target = size - free
    if need > 0:
        part = soup_exam.subset(rest, seed, need, exclude=retired, whole=False)
    elif need == 0:
        part = []
    else:
        raise ValueError("internal: the Current part is larger than the subset")
    return current, streams, cells, part


def prepare(config, state_dir, *, round_id, block_hash, bank, ledger, log=print):
    """(bank for ``soup_round.run``, bank-source record) for a pool whose exam source is current.

    ``bank`` is the fixed round bank. Writes ``subset.json``, ``current-items.jsonl`` and
    ``bank-source.json`` into the round folder, unless the folder already holds them (a resumed
    round). If even the fixed bank cannot fill the subset, nothing is written and the fixed bank is
    returned, so ``soup_round.run`` skips the round as before.
    """
    from . import base_metrics, soup_exam, soup_round

    settings = validate(config.get("exam"))
    round_dir = Path(state_dir) / "rounds" / round_id
    earlier = _earlier_items(state_dir, round_id)
    record_path = round_dir / "bank-source.json"
    if record_path.exists() and (round_dir / "subset.json").exists():
        own = round_dir / "current-items.jsonl"
        drawn = {c["id"]: c for c in _read_jsonl(own)} if own.exists() else {}
        return _merge(bank, {**earlier, **drawn}), json.loads(record_path.read_text())
    size = config.get("subset_size", soup_round.SUBSET_SIZE)
    seed = soup_round.round_seed(block_hash, config["pool_id"])
    retired = ledger.retired()
    record = {
        "source": "current",
        "fallback": None,
        "settings": {k: v for k, v in settings.items() if k != "current_bank"},
        "folders": [],
        "skipped_folders": {},
        "excluded": {},
        "streams": {},
        "current_items": 0,
        "bank_items": 0,
        "replaced_cells": [],
    }
    chosen = []
    try:
        deadline = _deadline(config, round_id)
        uses = ledger._uses()
        loaded = load(
            settings["current_bank"],
            deadline=deadline,
            max_age_days=settings["max_age_days"],
            max_tokens=_max_tokens(config),
            uses=uses,
            max_uses=settings["max_uses"],
            log=log,
        )
        record.update(
            folders=loaded["folders"],
            skipped_folders=loaded["skipped"],
            excluded=loaded["excluded"],
        )
        secret = hmac.new(
            soup_round.sketch_seed(config, state_dir), b"current-draw:" + seed, hashlib.sha256
        ).digest()
        current, streams, cells, part = _parts(
            loaded["items"], bank, secret, settings, size=size, seed=seed, retired=retired
        )
        chosen = sorted([*part, *current], key=lambda c: c["id"])
        base_metrics.baselines(chosen)
        record.update(
            streams=streams,
            current_items=len(current),
            bank_items=len(part),
            replaced_cells=[f"{c}/{b}" for c, b in cells],
        )
        drawn = {c["id"]: c for c in current}
    except (OSError, ValueError, KeyError, TypeError) as error:
        reason = f"{type(error).__name__}: {error}"[:400]
        log(f"soup exam: round {round_id} falls back to the fixed bank: {reason}")
        record["fallback"] = reason
        record["streams"] = {}
        drawn = {}
        try:
            chosen = soup_exam.subset(bank, seed, size, exclude=retired)
        except ValueError:
            return list(bank), record  # soup_round.run records the skipped round
        record["bank_items"] = len(chosen)
    round_dir.mkdir(parents=True, exist_ok=True)
    _write(
        round_dir / "current-items.jsonl",
        "".join(json.dumps(c, sort_keys=True) + "\n" for c in drawn.values()),
    )
    _write(round_dir / "subset.json", json.dumps(sorted(c["id"] for c in chosen)) + "\n")
    _write(record_path, json.dumps(record, sort_keys=True) + "\n")
    return _merge(bank, {**earlier, **drawn}), record
