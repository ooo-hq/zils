"""Grade serving miners: fresh seeded questions signed with the shared request key, a secret subset
recomputed with the same pinned model, a rolling per-miner per-model score. Reads the chain, never writes it.

Our own failures (`reference_unavailable`, `lock_timeout`) go to evidence only: an honest miner is never
scored down by the validator's side."""

import argparse
import fcntl
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import answer_check, serving, serving_items, serving_server, settings

ASK_TIMEOUT = 30.0
DEFAULTS = {
    "round_seconds": 300,
    "questions": 8,
    "verified": 3,
    "window_hours": 24,
    "floor": 0.9,
    "lock_wait_seconds": 60,
}
ANSWERED = {"pass", "marginal", "fail", "mismatch"}
STRIKES = {"fail", "mismatch"}
OURS = {"reference_unavailable", "lock_timeout"}


class ReferenceMismatch(Exception):
    """The reference worker does not serve the configured models at the configured weights."""


def log(text):
    print(f"zils serving-grader: {text}", file=sys.stderr, flush=True)


def finite(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def write_atomic(path, value):
    """Temp file in the same directory, fsync, replace; a failed write leaves the old file."""
    path = Path(path)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def lock_path():
    """The lookup zils.runtime uses, so the grader shares the validator's and training's lock."""
    return settings.get(
        "ZILS_COMPUTE_LOCK",
        str(Path(tempfile.gettempdir()) / f"fez-compute-{os.getuid()}-cuda.lock"),
    )


@contextmanager
def compute_lock(wait_seconds):
    """Yield True holding the shared compute lock, or False if it stayed busy for `wait_seconds`."""
    try:
        fd = os.open(lock_path(), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    except OSError as error:  # missing directory, EACCES, a symlink: treat as busy
        log(f"compute lock unavailable ({type(error).__name__})")
        yield False
        return
    try:
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    yield False
                    return
                time.sleep(0.1)
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class Scores:
    """Rolling round results per (hotkey, model), pruned to `window_hours`, persisted atomically."""

    def __init__(self, path, window_hours=24, floor=0.9):
        self.path, self.window, self.floor = Path(path), window_hours * 3600, floor
        self.rounds = json.loads(self.path.read_text()) if self.path.exists() else {}

    def record(self, hotkey, model, result):
        if result["outcome"] in OURS:
            raise ValueError("our own failures are never scored")
        rows = self.rounds.setdefault(hotkey, {}).setdefault(model, [])
        rows.append({k: finite(v) if isinstance(v, float) else v for k, v in result.items()})
        self.prune(result["at"])
        write_atomic(self.path, self.rounds)

    def prune(self, now):
        for hotkey in list(self.rounds):
            for model in list(self.rounds[hotkey]):
                rows = [r for r in self.rounds[hotkey][model] if r["at"] >= now - self.window]
                if rows:
                    self.rounds[hotkey][model] = rows
                else:
                    del self.rounds[hotkey][model]
            if not self.rounds[hotkey]:
                del self.rounds[hotkey]

    def summary(self, hotkey, model, now):
        rows = [
            r for r in self.rounds.get(hotkey, {}).get(model, []) if r["at"] >= now - self.window
        ]
        answered = [r for r in rows if r["outcome"] in ANSWERED]
        latencies = [r["latency_s"] for r in answered if r["latency_s"] is not None]
        pass_rate = sum(r["outcome"] == "pass" for r in answered) / len(answered) if answered else 0
        availability = len(answered) / len(rows) if rows else 0
        struck_until = max(
            (r["at"] + self.window for r in rows if r["outcome"] in STRIKES), default=0
        )
        struck = now < struck_until
        return {
            "rounds": len(rows),
            "pass_rate": pass_rate,
            "availability": availability,
            "median_latency_s": statistics.median(latencies) if latencies else None,
            "struck_until": struck_until,
            "score": 0 if pass_rate < self.floor or struck else pass_rate * availability,
        }

    def summaries(self, now):
        return {
            hotkey: {model: self.summary(hotkey, model, now) for model in models}
            for hotkey, models in self.rounds.items()
        }


def unload(served):
    """Close the worker behind a {model: Served} (all share one process)."""
    if served:
        next(iter(served.values())).worker.close()


class Grader:
    """`catalog_read()` -> {uid: (hotkey, endpoint)}; `worker()` -> list of `Served`. The reference model
    is loaded only while holding the compute lock and unloaded before releasing it, every round."""

    def __init__(self, config, catalog_read, worker, key, clock=time.time, ask=serving.ask):
        self.config = {**DEFAULTS, **config}
        if not 1 <= self.config["verified"] <= self.config["questions"]:
            raise ValueError("verified must be between 1 and questions")
        self.models = self.config["models"]
        self.catalog_read, self.worker, self.key = catalog_read, worker, key
        self.clock, self.ask = clock, ask
        self.state = Path(self.config["state"])
        self.evidence_path = self.state / "serving-evidence.jsonl"
        self.scores = Scores(
            self.state / "serving-rounds.json", self.config["window_hours"], self.config["floor"]
        )

    def start(self):
        """{model: Served}, or None if the worker would not start. Call it holding the lock and close
        the result with `unload`; a worker at other weights is closed and raises ReferenceMismatch."""
        try:
            served = {s.model: s for s in self.worker()}
        except (OSError, ValueError, serving_server.WorkerError) as error:
            log(f"reference worker failed to start ({type(error).__name__})")
            return None
        for model, spec in self.models.items():
            if model not in served or served[model].weights_sha256 != spec["weights_sha256"]:
                unload(served)
                raise ReferenceMismatch(
                    f"reference worker does not serve {model} at the configured weights"
                )
        return served

    def check(self):
        """Startup check: load, compare weights, unload, under the lock; skipped if the lock is busy."""
        with compute_lock(self.config["lock_wait_seconds"]) as held:
            if not held:
                log("compute lock busy at startup; weights are checked in the first round")
                return
            unload(self.start() or {})

    def targets(self):
        found = serving.discover(self.catalog_read(), clock=self.clock)
        out = []
        for uid in sorted(found):
            entry = found[uid]
            if "error" in entry:
                continue
            for model, spec in self.models.items():
                if any(
                    item["model"] == model and item["weights_sha256"] == spec["weights_sha256"]
                    for item in entry["models"]
                ):
                    out.append((entry["hotkey"], entry["endpoint"], model))
        return out

    def run_round(self, round_id):
        began = time.monotonic()
        outcomes = {}
        with compute_lock(0) as free:  # probe: a busy GPU means no requests at all this round
            pass
        if not free:
            outcomes = {"skipped": 1}
        else:
            targets = self.targets()
            random.Random(round_id).shuffle(targets)  # the budget must not starve the same uids
            answered = []
            for hotkey, endpoint, model in targets:
                if time.monotonic() - began >= self.config["round_seconds"]:
                    break  # over budget: later miners get no record this round
                seed = serving_items.round_seed(
                    self.config["validator_hotkey"], round_id, hotkey, model
                )
                row = {"hotkey": hotkey, "model": model, "seed": seed}
                row.update(self.collect(hotkey, endpoint, model, seed))
                if row["outcome"] is None:
                    answered.append(row)
                else:
                    self.finish(round_id, row, outcomes)
            if answered:
                self.reference(answered)
                for row in answered:
                    self.finish(round_id, row, outcomes)
        now = self.clock()
        self.scores.prune(now)
        write_atomic(self.state / "serving-scores.json", self.scores.summaries(now))
        return outcomes

    def collect(self, hotkey, endpoint, model, seed):
        """Ask one miner every question, one signed request each; outcome None once all answered
        (graded later under the lock), else the first MinerError's reason."""
        questions, count = self.config["questions"], self.config["verified"]
        sha = self.models[model]["weights_sha256"]
        bodies = serving_items.items(seed, questions)
        messages, began = [], time.monotonic()
        for body in bodies:
            try:
                messages.append(self.ask(endpoint, hotkey, self.key, model, sha, body, ASK_TIMEOUT))
            except serving.MinerError as error:
                return {
                    "outcome": error.reason,
                    "latency_s": time.monotonic() - began,
                    "messages": messages or None,
                }
        return {
            "outcome": None,
            "latency_s": time.monotonic() - began,
            "messages": messages,
            "bodies": bodies,
            # chosen only now that every answer is in
            "verified": sorted(random.Random(seed ^ 1).sample(range(questions), count)),
        }

    def reference(self, answered):
        """One lock hold: load, grade every answered miner's verified subset, unload. Sets each outcome."""
        with compute_lock(self.config["lock_wait_seconds"]) as held:
            if not held:
                for row in answered:
                    row["outcome"] = "lock_timeout"
                return
            served = self.start()
            try:
                for row in answered:
                    if served is None:
                        row["outcome"] = "reference_unavailable"
                        continue
                    try:
                        row.update(self.compare(served[row["model"]], row))
                    except serving_server.WorkerError as error:  # one refusal; the worker lives
                        log(f"reference worker refused a request ({error.kind})")
                        row["outcome"] = "reference_unavailable"
                    except OSError as error:
                        log(f"reference worker failed ({type(error).__name__})")
                        row["outcome"] = "reference_unavailable"
                        unload(served)
                        served = None
            finally:
                unload(served or {})

    def compare(self, served, row):
        bodies, messages, ours, theirs = row.pop("bodies"), row["messages"], {}, {}
        for i in row["verified"]:
            reference = served.predict(bodies[i]["state"], bodies[i]["questions"])
            predictions = messages[i]["payload"]["predictions"]
            for qid in bodies[i]["questions"]:
                ours[f"{i}:{qid}"] = reference[qid]["probabilities"]
                theirs[f"{i}:{qid}"] = predictions[qid]["probabilities"]
        verdict = answer_check.compare(ours, theirs)
        return {"outcome": verdict.outcome, "mean": verdict.mean, "worst": verdict.worst}

    def finish(self, round_id, row, outcomes):
        """Evidence first, then Scores (never for our own failures)."""
        row.pop("bodies", None)
        row = {"verified": None, "messages": None, "mean": None, "worst": None, **row}
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
        at = self.clock()
        self.write_evidence({"kind": "validator", "at": at, "round_id": round_id, **row})
        if row["outcome"] not in OURS:
            self.scores.record(
                row["hotkey"],
                row["model"],
                {key: row[key] for key in ("outcome", "mean", "worst", "latency_s")} | {"at": at},
            )

    def write_evidence(self, row):
        row = {k: finite(v) if isinstance(v, float) else v for k, v in row.items()}
        data = (json.dumps({"v": 1, **row}, allow_nan=False) + "\n").encode()
        fd = os.open(self.evidence_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)


def load_key(path):
    """The shared request hotkey, in the wallet file format zils-platform reads (`secretSeed`)."""
    from bittensor_wallet import Keypair

    with open(path) as stream:
        return Keypair.create_from_seed(json.load(stream)["secretSeed"])


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m zils.serving_grader")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if not Path(config["state"]).is_dir():
        raise SystemExit(f"missing state directory {config['state']}")
    for path in [config["request_key_file"], *(m["reference"] for m in config["models"].values())]:
        if not Path(path).exists():
            raise SystemExit(f"missing {path}")
    key = load_key(config["request_key_file"])

    from . import miner, testnet

    chain = {"chain": config["chain"]}
    testnet.check_config(chain)

    def catalog_read():
        with testnet.connect(chain) as sub:
            return testnet.serving_axons(chain, sub)

    checkpoints = [spec["reference"] for spec in config["models"].values()]
    grader = Grader(
        config,
        catalog_read,
        lambda: serving_server.start_worker(checkpoints, miner.d3_python()),
        key,
    )
    seconds = grader.config["round_seconds"]
    grader.check()  # wrong weights stop startup
    while True:
        began = time.monotonic()
        try:
            outcomes = grader.run_round(uuid.uuid4().hex)
            counts = " ".join(f"{k}={v}" for k, v in sorted(outcomes.items())) or "no miners"
            log(f"round {counts} in {time.monotonic() - began:.1f}s")
        except ReferenceMismatch:
            raise
        except Exception as error:  # a chain read or disk hiccup: try again next round
            log(f"round failed ({type(error).__name__})")
        time.sleep(seconds * random.uniform(0.8, 1.2))


if __name__ == "__main__":
    main()
