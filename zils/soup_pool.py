"""Soup pools in the validator: the daily schedule, the seed block, and one soup round per pool per day.

A soup pool's round for day D runs once the chain passes D's commit deadline (``schedule.deadline_utc``).
Its seed is the hash of the first block after the deadline; commitments made at or after that block wait
for the next day's round. The coordinator stays torch-free: ``epoch`` decides what is due and hands a
request to ``run_round``, which on a validator runs ``python -m zils.soup_pool <request>`` in the GPU
runtime (``child_round``). State lives under ``<validator dir>/base-pools/<pool>/soup``.
"""

import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import model_pools

SOUP_PATHS = {
    "round_bank",
    "base",
}  # base_pool_rounds keys for a soup pool (the bank and vendor base)
BLOCK_SECONDS = 12


def state_dir(directory, pool):
    return Path(directory) / "base-pools" / pool / "soup"


def decision(directory, pool):
    """The latest published soup round record for `pool`, or None before its first round."""
    path = state_dir(directory, pool) / "decision.json"
    return json.loads(path.read_text()) if path.exists() else None


def recorded(directory, pool, round_id):
    """True once the round's ledger entry exists (the last write of a finished round)."""
    path = state_dir(directory, pool) / "uses.jsonl"
    if not path.exists():
        return False
    return any(
        line and json.loads(line)["round"] == round_id for line in path.read_text().splitlines()
    )


def due(schedule, head_time):
    """(day, deadline timestamp) of the latest daily commit deadline at or before `head_time`."""
    hours, minutes = map(int, schedule["deadline_utc"].split(":"))
    now = datetime.fromtimestamp(head_time, UTC)
    deadline = now.replace(hour=hours, minute=minutes, second=0, microsecond=0)
    if deadline > now:
        deadline -= timedelta(days=1)
    return deadline.date().isoformat(), deadline.timestamp()


def _time(info):
    return info.timestamp.timestamp()


def seed_block(block_info, head, deadline):
    """(number, hash) of the first block whose timestamp is after `deadline`, searching back from `head`.

    `block_info(n)` returns an object with `number`, `hash` and a UTC `timestamp` (bittensor's
    `Subtensor.block_info`).
    """
    if _time(block_info(head)) <= deadline:
        raise ValueError("the chain has not passed the commit deadline")
    high, step = head, 64
    low = max(0, head - step)
    while low > 0 and _time(block_info(low)) > deadline:
        high, step = low, step * 2
        low = max(0, high - step)
    if _time(block_info(low)) > deadline:
        high = low
    while high - low > 1:
        middle = (low + high) // 2
        if _time(block_info(middle)) > deadline:
            high = middle
        else:
            low = middle
    return high, block_info(high).hash


def soup_pools(config):
    """{pool: validated entry} for each enabled soup pool."""
    if "base_pools" not in config:
        return {}
    pools = model_pools.validate(config["base_pools"])["pools"]
    return {p: e for p, e in pools.items() if e["enabled"] and model_pools.is_soup(e)}


def paths(config, directory, pool):
    """The pool's operator paths from base_pool_rounds, resolved against the validator directory."""
    value = config.get("base_pool_rounds", {}).get(pool)
    if (
        not isinstance(value, dict)
        or set(value) != SOUP_PATHS
        or any(not isinstance(v, str) or not v for v in value.values())
    ):
        raise ValueError("a soup pool's base_pool_rounds entry needs round_bank and base")
    return {k: str((Path(directory) / v).resolve()) for k, v in value.items()}


def house(entry, directory):
    """The pool's house registry: inline, or read from a path relative to the validator directory."""
    from . import soup_rates

    if isinstance(entry["house"], str):
        return soup_rates.load_house(Path(directory) / entry["house"])
    return soup_rates.validate_house(entry["house"])


def epoch(config, directory, snapshot, *, block_info, commitments_at, run_round):
    """Run each enabled soup pool's due round once. Returns {pool: summary}; a failed round keeps the
    pool's last decision (yesterday's soup stays paid) and is retried on the next epoch.

    `commitments_at(block)` returns the roster's commitments ({uid: {"block", "data"}}) as of
    `block`; the round reads them at its seed block, so every validator sees the same set however
    late it runs (the live snapshot only holds each hotkey's latest commitment)."""
    summary = {}
    for pool, entry in soup_pools(config).items():
        try:
            head = int(snapshot["block"])
            day, deadline = due(entry["schedule"], _time(block_info(head)))
            round_id = f"{pool}-{day}"
            summary[pool] = {"round_id": round_id}
            if recorded(directory, pool, round_id):
                continue
            block, block_hash = seed_block(block_info, head, deadline)
            commitments = {
                uid: item
                for uid, item in commitments_at(block).items()
                if int(item["block"]) < block
            }
            secret = config["base_pools"]["pools"][pool].get(model_pools.SOUP_SECRET)
            request = {
                "round_id": round_id,
                "seed_block": block,
                "block_hash": block_hash,
                "snapshot": {"members": snapshot["members"], "commitments": commitments},
                "config": {
                    **{k: v for k, v in entry.items() if k != "house"},
                    "pool_id": pool,
                    "house": house(entry, directory),
                    **({model_pools.SOUP_SECRET: secret} if secret else {}),
                },
                "state_dir": str(state_dir(directory, pool)),
                **paths(config, directory, pool),
            }
            run_round(pool, request)
            summary[pool]["decision"] = decision(directory, pool) is not None
        except (ValueError, OSError, RuntimeError) as error:
            summary.setdefault(pool, {})["error"] = str(error)[:400]
    return summary


def child_round(args, work):
    """A `run_round` for the validator loop: the round runs in the GPU runtime, under the shared
    compute lock, as `python -m zils.soup_pool <request.json>`."""
    from . import protocol as wire, runtime

    def run_round(pool, request):
        path = Path(work) / f"soup-{pool}-request.json"
        wire.write_json(path, request)
        while True:
            try:
                runtime.run_child(
                    [args.runtime_python, "-m", "zils.soup_pool", str(path)],
                    Path(work) / f"soup-{pool}.log",
                    args.device,
                    timeout=12 * 3600,
                    model=request["config"]["model"]["id"],
                )
                return
            except runtime.CapacityUnavailable:
                time.sleep(max(1, args.poll))

    return run_round


def _bank(path):
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def fetch_body(claim, dest):
    """Download a body repo's weights, index, model card and ownership file at the committed revision."""
    from . import bodies, commitments

    listed = commitments.revision_files(claim["repo"], claim["revision"])
    names = sorted(
        n
        for n in listed
        if "/" not in n
        and (
            n.endswith(".safetensors")
            or n in ("model.safetensors.index.json", "README.md", "zils-owner.json")
        )
    )
    commitments.download_files(claim, dest, names, listed, bodies.MAX_BYTES)
    return Path(dest)


def main(argv=None):
    """Run one soup round from a request written by `epoch` (GPU runtime entry point)."""
    from . import d3, soup_exam, soup_grade, soup_round

    argv = sys.argv[1:] if argv is None else argv
    request = json.loads(Path(argv[0]).read_text())
    config, root = request["config"], Path(request["state_dir"])
    model = config["model"]["id"]
    device = "cuda"
    record = soup_round.run(
        config,
        root,
        round_id=request["round_id"],
        snapshot=request["snapshot"],
        block_hash=request["block_hash"],
        bank=_bank(request["round_bank"]),
        ledger=soup_exam.UseLedger(root / "uses.jsonl"),
        cache=soup_grade.PredictionCache(root / "cache"),
        runner_for=lambda folder: soup_grade.production_runner(folder, device),
        fetch_body=fetch_body,
        chassis=d3.pinned_base_path(model=model),
        base=request["base"],
    )
    print(f"soup round {record['round_id']}: {len(record['members'])} members", flush=True)


if __name__ == "__main__":
    main()
