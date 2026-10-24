"""Persistent, resource-budgeted base evaluation outside the weight-publishing loop."""

import argparse
import contextlib
import hashlib
import http.client
import json
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError

from . import base_round, commitments, model_pools, models, protocol as wire, validator
from .runtime import CapacityUnavailable, locked

FIELDS = {
    "window_seconds",
    "max_candidates",
    "max_seconds",
    "max_download_bytes",
    "max_cache_bytes",
}


def policy(config):
    if "base_pools" not in config:
        raise ValueError("base_validation_queue requires configured base_pools")
    value = config.get("base_validation_queue")
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("base_validation_queue requires " + ", ".join(sorted(FIELDS)))
    if any(type(v) is not int or v <= 0 for v in value.values()):
        raise ValueError("base validation budgets must be positive integers")
    pools = model_pools.validate(config["base_pools"])["pools"]
    largest = max(
        (
            models.artifact_budget("candidate", models.validate_spec(p["model"]))
            for p in pools.values()
            if p["enabled"]
        ),
        default=0,
    )
    if value["max_seconds"] > value["window_seconds"]:
        raise ValueError("evaluation seconds cannot exceed the budget window")
    if min(value["max_download_bytes"], value["max_cache_bytes"]) < largest:
        raise ValueError("download and cache budgets must each fit one enabled candidate")
    return value


def context(config):
    return hashlib.sha256(
        json.dumps(
            {k: config[k] for k in ("base_pools", "base_pool_rounds")}, sort_keys=True
        ).encode()
    ).hexdigest()


@contextlib.contextmanager
def state_at(directory):
    root = Path(directory) / "base-queue"
    root.mkdir(mode=0o700, exist_ok=True)
    with locked(root / "state.lock", wait=True):
        path = root / "state.json"
        state = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "version": 1,
                "context": None,
                "entries": {},
                "next_ticket": 0,
                "decisions": {},
                "budget": None,
                "active": None,
                "owners": {},
            }
        )
        if state["version"] != 1:
            raise ValueError("unsupported base queue state")
        before = json.dumps(state, sort_keys=True)
        yield state
        if path.exists() and json.dumps(state, sort_keys=True) == before:
            return
        temporary = path.with_name("state-" + uuid.uuid4().hex + ".tmp")
        try:
            wire.write_json(temporary, state)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def ticket(state):
    value = state["next_ticket"]
    state["next_ticket"] += 1
    return value


def new_nomination(row):
    latest, previous = row["latest"], row["claim"]
    return latest and (
        latest["sha256"] != previous["sha256"]
        or row.get("error")
        and any(latest[k] != previous[k] for k in ("repo", "revision"))
    )


def enqueue(config, directory, snapshot):
    """Only local parsing and an atomic state update; no HTTP, model loading or GPU work."""
    policy(config)
    binding = context(config)
    claims = commitments.candidates(snapshot, None)
    roster = snapshot["members"]
    nominations = {
        roster[str(uid)]: {**claim, "uid": uid, "hotkey": roster[str(uid)]}
        for uid, claim in claims.items()
        if "error" not in claim and str(uid) in roster
    }
    with state_at(directory) as state:
        if state["context"] != binding:
            state.update(context=binding, entries={}, decisions={})
        # Remember first ownership even after the original miner nominates new weights.
        for uid, item in snapshot.get("commitments", {}).items():
            if str(uid) not in roster or type(item.get("block")) is not int or item["block"] < 0:
                continue
            try:
                claim = {**commitments.parse_commitment(item.get("data")), "block": item["block"]}
            except ValueError:
                continue
            hotkey = roster[str(uid)]
            owner = state["owners"].get(claim["sha256"])
            if owner is None or claim["block"] < owner["block"]:
                state["owners"][claim["sha256"]] = {"block": claim["block"], "hotkey": hotkey}
            elif claim["block"] == owner["block"] and hotkey != owner["hotkey"]:
                owner["hotkey"] = None
        nominations = {
            h: c for h, c in nominations.items() if state["owners"][c["sha256"]]["hotkey"] == h
        }
        # A new occupant of a UID cannot inherit the previous hotkey's queue/evidence.
        for hotkey, row in list(state["entries"].items()):
            if (
                roster.get(str(row["claim"]["uid"])) != hotkey
                or state["owners"][row["claim"]["sha256"]]["hotkey"] != hotkey
            ):
                del state["entries"][hotkey]
        for hotkey, row in list(state["entries"].items()):
            row["latest"] = nominations.get(hotkey)
            if row["status"] == "done":
                if new_nomination(row):
                    row.update(
                        claim=row["latest"],
                        status="pending",
                        frozen=False,
                        ticket=ticket(state),
                        ready_at=0,
                    )
            elif not row["frozen"]:
                if row["latest"]:
                    row["claim"] = row["latest"]  # Replacement keeps the original queue position.
                else:
                    del state["entries"][hotkey]
        for hotkey, claim in nominations.items():
            if hotkey not in state["entries"]:
                state["entries"][hotkey] = {
                    "claim": claim,
                    "latest": claim,
                    "status": "pending",
                    "frozen": False,
                    "ticket": ticket(state),
                    "ready_at": 0,
                }
        return {"queued": sum(r["status"] != "done" for r in state["entries"].values())}


def decisions(config, directory):
    with state_at(directory) as state:
        return dict(state["decisions"]) if state["context"] == context(config) else {}


def reserve(config, directory, now):
    limits = policy(config)
    with state_at(directory) as state:
        # Called only by the holder of worker.lock. A lost worker's reservation stays charged.
        if active := state["active"]:
            row = state["entries"].get(active["claim"]["hotkey"])
            if row and row["claim"] == active["claim"]:
                row.update(status="pending", ticket=ticket(state))
            state["active"] = None
        if state["context"] != context(config):
            return None
        budget = state["budget"]
        if budget is None or now >= budget["started_at"] + limits["window_seconds"]:
            budget = {"started_at": now, "candidates": 0, "seconds": 0, "download_bytes": 0}
            state["budget"] = budget
        if now < budget["started_at"] or budget["candidates"] >= limits["max_candidates"]:
            return None
        seconds = min(
            limits["max_seconds"] - budget["seconds"],
            budget["started_at"] + limits["window_seconds"] - now,
        )
        if seconds < 1:
            return None
        ready = [
            r
            for r in state["entries"].values()
            if r["status"] == "pending" and r["ready_at"] <= now
        ]
        if not ready:
            return None
        row = min(ready, key=lambda r: r["ticket"])
        row.update(status="running", frozen=True)
        job = {
            "id": uuid.uuid4().hex,
            "claim": dict(row["claim"]),
            "context": state["context"],
            "seconds": seconds,
            "window": budget["started_at"],
        }
        state["active"] = job
        budget["candidates"] += 1
        budget["seconds"] += seconds
        return job


class BudgetExhausted(Exception):
    pass


class EvaluationDeadline(Exception):
    pass


@contextlib.contextmanager
def deadline(seconds):
    def expire(signum, frame):
        raise EvaluationDeadline("base evaluation wall-time budget exhausted")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def reserve_download(config, directory, job, size):
    with state_at(directory) as state:
        if state["context"] != job["context"] or state["active"] != job:
            raise BudgetExhausted("queue context changed")
        budget = state["budget"]
        if budget["download_bytes"] + size > policy(config)["max_download_bytes"]:
            raise BudgetExhausted("download budget exhausted")
        # Charge the profile's maximum before fetching, including failed/partial downloads.
        budget["download_bytes"] += size


BLOCK_SECONDS = 12


def admit_submission(directory, claim, pool, cooldown):
    """A hotkey's distinct submissions to one pool must be committed at least `cooldown` apart,
    measured in chain blocks so every validator reaches the same answer. A commitment inside the
    cooldown is refused and recorded; the miner commits again once it has passed."""
    blocks = -(-cooldown // BLOCK_SECONDS)
    with state_at(directory) as state:
        pools = state.setdefault("submissions", {}).setdefault(claim["hotkey"], {})
        last = pools.get(pool)
        if last and last["sha256"] == claim["sha256"]:
            return  # a retry or the confirmation run of the accepted submission
        if last and claim["block"] < last["block"] + blocks:
            raise ValueError(
                f"committed at block {claim['block']}, inside the {pool} cooldown; commit again"
                f" at or after block {last['block'] + blocks}"
            )
        pools[pool] = {"block": claim["block"], "sha256": claim["sha256"]}


def make_room(directory, target, size, maximum, *, protected=()):
    """Evict only reproducible candidate weights; retain all evaluation and release evidence."""
    caches = []
    # worker.lock excludes any other queue fetch, including recovery after SIGKILL.
    for path in Path(directory).glob("base-pools/*/base-tier/store/*/fetch-*"):
        if path.is_symlink() or not re.fullmatch("fetch-[a-f0-9]{32}", path.name):
            raise ValueError("unexpected partial download path")
        shutil.rmtree(path)
    for path in Path(directory).glob("base-pools/*/base-tier/store/*/checkpoint"):
        if not re.fullmatch("[a-f0-9]{64}", path.parent.name) or path.is_symlink():
            raise ValueError("unexpected candidate cache path")
        amount = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        caches.append((path, amount))
    total = sum(n for _, n in caches)
    needed = 0 if target.exists() else size
    for path, amount in sorted(caches, key=lambda item: item[0].stat().st_mtime):
        if total + needed <= maximum:
            break
        if path != target and path.resolve() not in protected:
            shutil.rmtree(path)
            total -= amount
    if total + needed > maximum:
        raise BudgetExhausted("candidate cache budget exhausted")
    # Fetch + evaluator staging may temporarily duplicate a checkpoint.
    if shutil.disk_usage(directory).free < 2 * size + 2 * 1024**3:
        raise BudgetExhausted("insufficient free space for candidate staging")


def evaluate(config, directory, job, args, work):
    claim = job["claim"]
    model = commitments.parent_model(claim)
    pool = models.pool_id(model)
    enabled = model_pools.validate(config["base_pools"])["pools"].get(pool)
    if not enabled or not enabled["enabled"] or enabled["model"] != models.spec(model):
        raise ValueError("candidate parent is not an enabled pool profile")
    if claim["sha256"] == enabled["incumbent_sha256"]:
        raise ValueError("incumbent is not a candidate")
    admit_submission(directory, claim, pool, enabled.get("cooldown_seconds", 0))
    settings, root, draws = validator.pool_rounds(config, Path(directory))[pool]
    target = root / "base-tier/store" / claim["sha256"] / "checkpoint"
    size = models.artifact_budget("candidate", model)
    make_room(
        directory,
        target,
        size,
        policy(config)["max_cache_bytes"],
        protected={
            (Path(directory) / p["base_reference"]).resolve()
            for p in config["base_pool_rounds"].values()
        },
    )
    if not target.exists():
        reserve_download(config, directory, job, size)
    result = validator.base_tier_epoch(
        settings, root, work, {claim["uid"]: claim}, args, draws, wait_for_capacity=False
    )
    if target.exists():
        target.touch()
    candidate = next((c for c in result["candidates"] if c["sha256"] == claim["sha256"]), None)
    ready_at = 0
    if candidate and candidate["eligible"] and "confirmed_candidates" not in result:
        ready_at = (
            max(
                v["completed_at"]
                for r in (result["incumbent"], candidate)
                for v in r["evaluation"].values()
            )
            + 12 * 3600
        )
    return pool, result, ready_at


def finish(config, directory, job, elapsed, *, result=None, ready_at=0, error=None, retry=False):
    with state_at(directory) as state:
        if state["active"] != job:
            return
        state["active"] = None
        # Refund unused reserved time only on a cleanly observed completion.
        state["budget"]["seconds"] -= max(0, job["seconds"] - elapsed)
        if state["context"] != job["context"]:
            return
        row = state["entries"].get(job["claim"]["hotkey"])
        if not row or row["claim"] != job["claim"]:
            return
        row.update(
            status="pending" if retry or ready_at else "done",
            ready_at=ready_at,
            ticket=ticket(state),
            error=error,
        )
        if result:
            pool, decision = result
            state["decisions"][pool] = base_round.public_decision(decision)
        if row["status"] == "done" and new_nomination(row):
            row.update(claim=row["latest"], status="pending", frozen=False, ready_at=0)


def step(config, directory, args):
    """One bounded attempt. Other validator processes only contend on short state transactions."""
    directory = Path(directory)
    root = directory / "base-queue"
    root.mkdir(mode=0o700, exist_ok=True)
    with locked(root / "worker.lock"):
        job = reserve(config, directory, time.time())
        if not job:
            return False
        work = root / "attempts" / job["id"]
        work.mkdir(mode=0o700, parents=True)
        wire.write_json(work / "job.json", job)
        started = time.monotonic()
        result, ready_at, error, retry = None, 0, None, False
        try:
            with deadline(job["seconds"]):
                pool, decision, ready_at = evaluate(config, directory, job, args, work)
                result = (pool, decision)
        except ValueError as exc:
            error = str(exc)[:400]
        except HTTPError as exc:
            error = str(exc)[:400]
            retry = exc.code in (408, 429) or exc.code >= 500
        except (
            BudgetExhausted,
            EvaluationDeadline,
            CapacityUnavailable,
            OSError,
            RuntimeError,
            http.client.HTTPException,
            subprocess.TimeoutExpired,
        ) as exc:
            error, retry = str(exc)[:400], True
        if retry:
            ready_at = job["window"] + policy(config)["window_seconds"]
        elapsed = time.monotonic() - started
        wire.write_json(
            work / "result.json",
            {"elapsed": elapsed, "error": error, "retry": retry, "ready_at": ready_at},
        )
        finish(
            config,
            directory,
            job,
            elapsed,
            result=result,
            ready_at=ready_at,
            error=error,
            retry=retry,
        )
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-python", default=sys.executable)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    parser.add_argument("--poll", type=int, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll <= 0:
        parser.error("poll must be positive")
    path = Path(args.config).resolve()
    while True:
        config = json.loads(path.read_text())
        policy(config)
        step(config, path.parent, args)
        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
