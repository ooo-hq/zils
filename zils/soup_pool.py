"""Soup pools in the validator: the daily schedule, the seed block, and one soup round per pool per day.

A soup pool's round for day D runs once the chain passes D's commit deadline (``schedule.deadline_utc``).
Its seed is the hash of the first block after the deadline; commitments made at or after that block wait
for the next day's round. The coordinator stays torch-free: ``epoch`` decides what is due and hands a
request to ``run_round``. On a validator that is ``background_round``: it starts a detached
``python -m zils.soup_pool --run <request>`` and returns at once, so weight setting never waits for a
round; the runner takes the compute lock and runs ``python -m zils.soup_pool <request>`` in the GPU
runtime under a hard deadline. Later epochs see the round running, finished (its ledger entry) or dead
(restarted, at most ``MAX_ATTEMPTS`` times per round). Until a round finishes the previous decision
pays. State lives under ``<validator dir>/base-pools/<pool>/soup``.

Each pool reads only the body commitments addressed to it (``zilsb2`` for d3-mini, ``zilsb3:<pool>``
for the others). A pool with a schedule runs no round before its opening block. A validator can
declare in ``soup_grading`` which pools it grades (``pools``) and its GPU memory (``gpu_mib``); a
pool it does not grade is skipped and logged, and its share burns in this validator's weights.
"""

import functools
import json
import os
import subprocess
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
ROUND_SECONDS = 6 * 3600  # hard deadline for one soup round in the GPU runtime
MAX_ATTEMPTS = 3  # starts of one day's round before it is skipped until the next day
CAPACITY_TRIES = 120  # GPU-busy retries (one poll apart) before a runner gives up
# GPU memory (MiB) one soup grading needs per pool: d3-mini measured about 9.5 GB on an RTX 6000 Ada;
# the others are estimates from bf16 weights (lite 1.7 GB, flash 16.8 GB, z3 52.2 GB) plus room for
# activations at the pool's token cap. FP8 grading is not supported.
GRADING_MIB = {"d3-lite": 4096, "d3-mini": 12288, "d3-flash": 20480, "d3": 57344}


def state_dir(directory, pool):
    return Path(directory) / "base-pools" / pool / "soup"


def decision(directory, pool):
    """The latest published soup round record for `pool`, or None before its first round."""
    path = state_dir(directory, pool) / "decision.json"
    return json.loads(path.read_text()) if path.exists() else None


def fresh_decision(directory, pool, entry, *, now):
    """The latest decision while it is current: None before the first round, and None once its
    round's commit deadline is more than ``max_decision_age_days`` old, so a pool whose rounds keep
    failing burns its share instead of paying an old soup forever."""
    record = decision(directory, pool)
    if record is None:
        return None
    try:
        day = datetime.fromisoformat(str(record["round_id"])[-10:]).replace(tzinfo=UTC)
    except (KeyError, ValueError):
        return None
    hours, minutes = map(int, entry["schedule"]["deadline_utc"].split(":"))
    deadline = day + timedelta(hours=hours, minutes=minutes)
    age = now - deadline.timestamp()
    return record if age <= entry["max_decision_age_days"] * 86400 else None


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


def grading(config):
    """The validator's local ``soup_grading`` setting: {"pools": [pool, ...] or None (every soup
    pool), "gpu_mib": GPU memory in MiB or None (not checked)}."""
    value = config.get("soup_grading", {})
    if not isinstance(value, dict) or set(value) - {"pools", "gpu_mib"}:
        raise ValueError("soup_grading takes only pools and gpu_mib")
    pools, memory = value.get("pools"), value.get("gpu_mib")
    if pools is not None and (
        not isinstance(pools, list) or any(p not in GRADING_MIB for p in pools)
    ):
        raise ValueError("soup_grading.pools must list soup pools: " + ", ".join(GRADING_MIB))
    if memory is not None and (type(memory) is not int or memory < 1):
        raise ValueError("soup_grading.gpu_mib must be a positive integer")
    return {"pools": pools, "gpu_mib": memory}


def not_graded(config, pools=None):
    """{pool: reason} for the enabled soup pools this validator does not grade."""
    if pools is None:
        if "base_pools" not in config:
            return {}
        pools = model_pools.validate(config["base_pools"])["pools"]
    local, out = grading(config), {}
    for pool, entry in pools.items():
        if not (entry["enabled"] and model_pools.is_soup(entry)):
            continue
        need = GRADING_MIB.get(pool)
        if local["pools"] is not None and pool not in local["pools"]:
            out[pool] = "not in soup_grading.pools"
        elif need is None:
            out[pool] = "no soup grading support for this pool"
        elif local["gpu_mib"] is not None and need > local["gpu_mib"]:
            out[pool] = (
                f"needs {need} MiB of GPU memory; soup_grading.gpu_mib is {local['gpu_mib']}"
            )
    return out


def closed(entry, block):
    """Why a soup pool runs no round at `block`, or None once it is open."""
    if model_pools.is_open(entry, block):
        return None
    if entry["opens_block"] is None:
        return "closed: no opening block is set"
    return f"closed until block {entry['opens_block']}"


def soup_pools(config, block=None):
    """{pool: validated entry} for each enabled soup pool this validator grades. With `block`, a pool
    that has not reached its opening block is left out too."""
    if "base_pools" not in config:
        return {}
    pools = model_pools.validate(config["base_pools"])["pools"]
    skipped = not_graded(config, pools)
    return {
        p: e
        for p, e in pools.items()
        if e["enabled"]
        and model_pools.is_soup(e)
        and p not in skipped
        and (block is None or model_pools.is_open(e, block))
    }


def pool_commitments(commitments, pool):
    """The commitments addressed to `pool`, as the zilsb2 text soup intake reads."""
    from . import bodies

    return {
        uid: {**item, "data": bodies.as_zilsb2(item.get("data", ""))}
        for uid, item in commitments.items()
        if bodies.body_pool(item.get("data", "")) == pool
    }


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


def house(entry, directory, key="house"):
    """The pool's house registry (or, with ``key="known_public"``, its known-public list, empty when
    absent): inline, or read from a path relative to the validator directory."""
    from . import soup_rates

    value = entry.get(key, []) if key == model_pools.SOUP_KNOWN else entry[key]
    if isinstance(value, str):
        return soup_rates.load_house(Path(directory) / value)
    return soup_rates.validate_house(value)


def known_public(entry, directory):
    """The pool's known-public list, or ([], the reason) when it cannot be read: the list only lowers
    the pay of re-uploads, so a bad file never stops the round."""
    try:
        return house(entry, directory, model_pools.SOUP_KNOWN), None
    except (OSError, ValueError) as error:
        return [], f"{type(error).__name__}: {error}"[:400]


def _started(pid):
    """Process `pid`'s start time as `ps` reports it (Linux and macOS), or None."""
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _alive(pid, started=None):
    """True while process `pid` runs (a finished child of this process is reaped, not counted).
    With `started`, the process must also have that start time, so a reused pid is not the round."""
    if type(pid) is not int or pid <= 0:
        return False
    try:
        if os.waitpid(pid, os.WNOHANG)[0]:
            return False
    except ChildProcessError:
        pass  # not our child: ask the kernel
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return started is None or _started(pid) == started


def epoch(config, directory, snapshot, *, block_info, commitments_at, run_round):
    """Start each enabled soup pool's due round once; never waits for one. Returns {pool: summary}.

    `run_round(pool, request)` starts the round and returns its process id (or None). A started
    round is recorded in `<state>/job.json`; while it runs later epochs only report it, and a round
    that died unrecorded is restarted up to `MAX_ATTEMPTS` times. Any failure is reported in the
    summary and the pool's last decision (yesterday's soup) stays paid.

    `commitments_at(block)` returns the roster's commitments ({uid: {"block", "data"}}) as of
    `block`; the round reads them at its seed block, so every validator sees the same set however
    late it runs (the live snapshot only holds each hotkey's latest commitment)."""
    summary = {}
    if "base_pools" in config:
        validated = model_pools.validate(config["base_pools"])["pools"]
        for pool, reason in not_graded(config, validated).items():
            summary[pool] = {"skipped": reason}
        for pool, entry in validated.items():
            if pool not in summary and entry["enabled"] and model_pools.is_soup(entry):
                if reason := closed(entry, int(snapshot["block"])):
                    summary[pool] = {"closed": reason}
    for pool, entry in soup_pools(config, int(snapshot["block"])).items():
        try:
            head = int(snapshot["block"])
            day, deadline = due(entry["schedule"], _time(block_info(head)))
            round_id = f"{pool}-{day}"
            summary[pool] = {"round_id": round_id}
            summary[pool]["decision"] = decision(directory, pool) is not None
            if recorded(directory, pool, round_id):
                continue
            job_path = state_dir(directory, pool) / "job.json"
            job = json.loads(job_path.read_text()) if job_path.exists() else {}
            attempts = job.get("attempts", 0) if job.get("round_id") == round_id else 0
            if (
                attempts
                and _alive(job.get("pid"), job.get("pid_started"))
                and time.time() < job["started"] + ROUND_SECONDS + 3600
            ):
                summary[pool]["running"] = True
                continue
            if attempts >= MAX_ATTEMPTS:
                raise RuntimeError(f"round skipped after {attempts} failed attempts")
            block, block_hash = seed_block(block_info, head, deadline)
            commitments = {
                uid: item
                for uid, item in pool_commitments(commitments_at(block), pool).items()
                if int(item["block"]) < block
            }
            secret = config["base_pools"]["pools"][pool].get(model_pools.SOUP_SECRET)
            exam = entry.get("exam")
            if exam:  # a relative Zils Current bank is read from under the validator directory
                exam = {**exam, "current_bank": str((Path(directory) / exam["current_bank"]))}
            known, known_error = known_public(entry, directory)
            if known_error:
                summary[pool]["known_public_error"] = known_error
            request = {
                "round_id": round_id,
                "seed_block": block,
                "block_hash": block_hash,
                "snapshot": {"members": snapshot["members"], "commitments": commitments},
                "config": {
                    **{k: v for k, v in entry.items() if k != "house"},
                    **({"exam": exam} if exam else {}),
                    "pool_id": pool,
                    "house": house(entry, directory),
                    model_pools.SOUP_KNOWN: known,
                    **({model_pools.SOUP_SECRET: secret} if secret else {}),
                },
                "state_dir": str(state_dir(directory, pool)),
                **paths(config, directory, pool),
            }
            job = {"round_id": round_id, "attempts": attempts + 1, "started": time.time()}
            _write(job_path, {**job, "pid": None})  # an attempt counts even if starting fails
            pid = run_round(pool, request)
            _write(job_path, {**job, "pid": pid, "pid_started": _started(pid) if pid else None})
            summary[pool]["started"] = True
            summary[pool]["decision"] = decision(directory, pool) is not None
        except Exception as error:  # one pool's round must never stop the validator's weights
            summary.setdefault(pool, {})["error"] = f"{type(error).__name__}: {error}"[:400]
    return summary


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(partial, path)


def background_round(args):
    """A `run_round` for the validator loop: writes the request under the pool's state directory and
    starts a detached `python -m zils.soup_pool --run <request>` (``run_locked``). Returns its pid."""
    from . import ROOT

    def run_round(pool, request):
        root = Path(request["state_dir"])
        path = root / f"request-{request['round_id']}.json"
        runner = {"runtime_python": args.runtime_python, "device": args.device, "poll": args.poll}
        _write(path, {**request, "runner": runner})
        with (root / "round.log").open("ab") as output:
            process = subprocess.Popen(
                [sys.executable, "-m", "zils.soup_pool", "--run", str(path)],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=ROOT,
            )
        return process.pid

    return run_round


def run_locked(path, *, run_child=None, sleep=time.sleep):
    """Run the round in `path` in the GPU runtime under the shared compute lock, within
    `ROUND_SECONDS`. Waits for GPU capacity at most `CAPACITY_TRIES` polls. Returns True on success;
    a failure leaves the round unrecorded for the next epoch to restart."""
    from . import runtime

    run_child = run_child or runtime.run_child
    request = json.loads(Path(path).read_text())
    runner = request["runner"]
    for _ in range(CAPACITY_TRIES):
        try:
            run_child(
                [runner["runtime_python"], "-m", "zils.soup_pool", str(path)],
                Path(request["state_dir"]) / "round.log",
                runner["device"],
                timeout=ROUND_SECONDS,
                model=request["config"]["model"]["id"],
            )
            return True
        except runtime.CapacityUnavailable:
            sleep(max(1, runner["poll"]))
        except subprocess.TimeoutExpired:
            print(f"soup round passed its {ROUND_SECONDS} s deadline", flush=True)
            return False
    print("soup round found no GPU capacity", flush=True)
    return False


def _bank(path):
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def fetch_body(claim, dest, pool=None):
    """Download a body repo's weights, index, model card and ownership file at the committed revision,
    within the pool's body byte cap and download deadline (d3-mini's without a pool). A claim the
    validator builds itself for a known public model may carry its own ``max_bytes``; a miner's
    claim is parsed from its commitment and never does."""
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
    if pool is None:
        commitments.download_files(claim, dest, names, listed, bodies.MAX_BYTES)
    else:
        commitments.download_files(
            claim, dest, names, listed, claim.get("max_bytes") or bodies.max_bytes(pool),
            seconds=bodies.download_seconds(pool),
        )  # fmt: skip
    return Path(dest)


def fit_bank(bank, model, *, encoder=None, log=print):
    """The bank items the pool's model accepts: within its token cap (``max_input_tokens``, the
    d3 prompt as the runtime renders it) and otherwise valid input. Others are dropped and counted,
    so one long item never fails a whole round. The encoder is the chassis tokenizer (no weights)."""
    from . import models

    cap = models.spec(model)["max_input_tokens"]
    kept = []
    for case in bank:
        if encoder is None:
            from . import d3

            encoder = d3.Encoder(d3.base_path(model=model), model=model)
        try:
            ids, _ = encoder.encode(case["state"], case["question"])
        except ValueError:
            continue
        if len(ids) <= cap:
            kept.append(case)
    if len(kept) < len(bank):
        log(
            f"soup bank: {len(bank) - len(kept)} of {len(bank)} items dropped for {model} ({cap} tokens)"
        )
    return kept


def main(argv=None):
    """Run one soup round from a request written by `epoch` (GPU runtime entry point)."""
    from . import d3, soup_exam, soup_grade, soup_round

    argv = sys.argv[1:] if argv is None else argv
    if argv[0] == "--run":
        raise SystemExit(0 if run_locked(argv[1]) else 1)
    request = json.loads(Path(argv[0]).read_text())
    config, root = request["config"], Path(request["state_dir"])
    model = config["model"]["id"]
    device = "cuda"
    deadline = time.time() + ROUND_SECONDS  # run_locked kills this process at that point
    bank, ledger = _bank(request["round_bank"]), soup_exam.UseLedger(root / "uses.jsonl")
    bank = fit_bank(bank, model)  # items within the pool's token cap
    bank_source = None
    if config.get("exam", {}).get("source") == "current":
        from . import soup_current

        bank, bank_source = soup_current.prepare(
            config,
            root,
            round_id=request["round_id"],
            block_hash=request["block_hash"],
            bank=bank,
            ledger=ledger,
        )
    record = soup_round.run(
        config,
        root,
        round_id=request["round_id"],
        snapshot=request["snapshot"],
        block_hash=request["block_hash"],
        bank=bank,
        ledger=ledger,
        cache=soup_grade.PredictionCache(root / "cache"),
        runner_for=lambda folder: soup_grade.production_runner(folder, device),
        fetch_body=functools.partial(fetch_body, pool=config["pool_id"]),
        chassis=d3.pinned_base_path(model=model),
        base=request["base"],
        bank_source=bank_source,
        deadline=deadline,
    )
    if record is not None:
        print(f"soup round {record['round_id']}: {len(record['members'])} members", flush=True)


if __name__ == "__main__":
    main()
