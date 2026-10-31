"""Seeded round subsets of the private round bank and a per-item use ledger."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from . import base_metrics

RETIRE_AFTER = 24
FAMILY_MINIMUM = 10


def _allocate(counts, size, minimum):
    # Every group gets min(minimum, available); the rest follows the group's share of the pool.
    quota = {key: min(n, minimum) for key, n in counts.items()}
    total = sum(counts.values())
    if sum(quota.values()) > size or total < size:
        raise ValueError("round bank cannot fill the requested subset")
    while sum(quota.values()) < size:
        key = max(
            (k for k in counts if quota[k] < counts[k]),
            key=lambda k: (size * counts[k] / total - quota[k], k),
        )
        quota[key] += 1
    return quota


def subset(bank, seed: bytes, size: int, *, exclude=(), minimum=FAMILY_MINIMUM, whole=True):
    """Return a deterministic, family- and label-stratified subset sorted by id.

    ``whole=False`` skips the check that the subset is a whole base exam (every category and block
    cell), for a part that is completed by other items (``soup_current``)."""
    if type(seed) is not bytes or not seed or type(size) is not int or size < 1:
        raise ValueError("seed must be non-empty bytes and size a positive integer")
    exclude = set(exclude)
    families = defaultdict(lambda: defaultdict(list))
    for case in bank:
        if case["id"] not in exclude:
            families[case["family"]][case["label"]].append(case)
    counts = {f: sum(map(len, labels.values())) for f, labels in families.items()}
    chosen = []
    for family, quota in _allocate(counts, size, minimum).items():
        labels = families[family]
        per_label = _allocate({k: len(v) for k, v in labels.items()}, quota, 1)
        for label, rows in labels.items():
            rows = sorted(
                rows, key=lambda c: hashlib.sha256(seed + b"\0" + c["id"].encode()).digest()
            )
            chosen += rows[: per_label[label]]
    chosen.sort(key=lambda c: c["id"])
    if whole:
        base_metrics.baselines(chosen)
    return chosen


class UseLedger:
    """Append-only JSONL of the item ids each published round graded."""

    def __init__(self, path):
        self.path = Path(path)

    def _rounds(self):
        if not self.path.exists():
            return {}
        rows = (json.loads(line) for line in self.path.read_text().splitlines() if line)
        return {row["round"]: row["ids"] for row in rows}

    def rounds(self):
        """[(round id, item ids)] in the order the rounds were recorded."""
        return list(self._rounds().items())

    def record(self, round_id, ids):
        """Count each id once for this round; a round already recorded adds nothing."""
        if round_id in self._rounds():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps({"round": round_id, "ids": sorted(set(ids))}) + "\n")
        return True

    def _uses(self):
        uses = defaultdict(int)
        for ids in self._rounds().values():
            for identity in ids:
                uses[identity] += 1
        return uses

    def exposure(self, identity):
        return self._uses().get(identity, 0)

    def retired(self, cap=RETIRE_AFTER):
        return {identity for identity, n in self._uses().items() if n >= cap}
