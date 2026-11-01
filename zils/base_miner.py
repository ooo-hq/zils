"""Run one scheduled public-data base attempt using the normal miner contract.

No election or channel changes. Interrupted attempts require operator review;
a timer must never replace an uncertain publication with another run.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace


def save(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def next_attempt(state, *, now, interval):
    path = state / "attempt.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["status"] != "committed":
            raise RuntimeError("Previous attempt needs operator review; no new training started")
        if now - previous["finished_at"] < interval:
            return None
    return str(int(now))


def chain_state(config, hotkey):
    from . import testnet

    with testnet.connect(config) as sub:
        return testnet.hotkey_state(config, sub, hotkey)


def write_chain(config, keys, text):
    from . import testnet

    with testnet.connect(config, raw_calls=True) as sub:
        testnet.write_commitment(config, sub, keys, text)


def commit_once(config, keys, hotkey, text):
    before = chain_state(config, hotkey).get("commitment")
    if before and before["data"] == text:
        return before
    try:
        write_chain(config, keys, text)
    except Exception:
        # A response can be lost after inclusion. Read back; never repeat the write here.
        pass
    after = chain_state(config, hotkey).get("commitment")
    if not after or after["data"] != text:
        raise RuntimeError("Uncertain chain outcome: operator review required before retry")
    return after


def commands(config, directory, seed):
    """(train, finish) commands for the configured recipe. `adapter` (default): one epoch of the
    production LoRA on labeled public data, merged into the parent. `relay` (Zils Relay): a merged
    LoRA fitted to teacher answers, optionally with the answer layer, then souped with the parent."""
    python = config["runtime_python"]
    if config.get("recipe", "adapter") == "adapter":
        return (
            [python, "-m", "zils.d3", "train", "--data", config["data"],
             "--reference", config["reference"], "--out", str(directory / "adapter"),
             "--seed", str(seed)],
            [python, "-m", "zils.d3", "candidate", "--adapter", str(directory / "adapter"),
             "--out", str(directory / "candidate")],
        )  # fmt: skip
    relay = config["relay"]
    train = [python, "-m", "zils.teacher", "train", "--answers", config["data"],
             "--reference", config["reference"], "--out", str(directory / "student"),
             "--seed", str(seed), "--lora", str(relay["lora"])]  # fmt: skip
    if relay.get("answer_layer"):
        train += [
            "--train-answer-layer",
            "--answer-layer-lr",
            str(relay.get("answer_layer_lr", 1e-5)),
        ]
    if relay.get("limit"):
        train += ["--limit", str(relay["limit"])]
    weight = relay.get("soup", 0.5)
    soup = [python, "-m", "zils.teacher", "soup", "--model", config["model"],
            "--out", str(directory / "candidate"),
            f"stock:{1 - weight:g}", f"{directory / 'student'}:{weight:g}"]  # fmt: skip
    return train, soup


def validate_recipe(config):
    recipe = config.get("recipe", "adapter")
    if recipe == "adapter":
        return
    relay = config.get("relay")
    if recipe != "relay" or not isinstance(relay, dict):
        raise ValueError("recipe must be adapter, or relay with a relay section")
    if set(relay) - {"lora", "answer_layer", "answer_layer_lr", "limit", "soup"}:
        raise ValueError("unknown relay setting")
    if type(relay.get("lora")) is not int or not 1 <= relay["lora"] <= 256:
        raise ValueError("relay lora must be an integer rank in [1, 256]")
    if "limit" in relay and (type(relay["limit"]) is not int or relay["limit"] < 1):
        raise ValueError("relay limit must be a positive integer")
    weight = relay.get("soup", 0.5)
    if type(weight) not in (int, float) or not 0 < weight <= 1:
        raise ValueError("relay soup must be the student's weight in (0, 1]")


def validate(config):
    from . import miner, models, testnet

    network = config.get("network", "test")
    if network == "test" and config.get("netuid") != miner.NETUID:
        raise ValueError(f"On testnet this runner is restricted to subnet {miner.NETUID}")
    if network not in testnet.ENDPOINTS or type(config.get("netuid")) is not int:
        raise ValueError("network must be test or finney, with an explicit integer netuid")
    interval = config.get("interval_seconds", 0)
    if type(interval) is not int or interval < 0:
        raise ValueError("interval_seconds must be a nonnegative integer")
    if not 1 <= config.get("max_training_seconds", 0) <= 21600:
        raise ValueError("Training timeout must be between 1 and 21600 seconds")
    validate_recipe(config)
    if hashlib.sha256(Path(config["data"]).read_bytes()).hexdigest() != config["data_sha256"]:
        raise ValueError("Public training data changed")
    if models.checkpoint_model(config["reference"]) != config["model"]:
        raise ValueError("Reference differs from configured profile")
    keys = miner.wallet(SimpleNamespace(**config))
    hotkey = keys.hotkey.ss58_address
    if hotkey != config["hotkey"]:
        raise ValueError("Wallet differs from this miner identity")
    chain = miner.chain_config(config["netuid"], network)
    if chain_state(chain, hotkey)["uid"] is None:
        raise ValueError("Miner is not registered")
    miner.commitments.check_repo(config["repo"])
    return keys, chain


def run(config):
    from . import miner
    from .runtime import gpu_ready
    from .serving_yield import yield_gpu

    if time.time() < config.get("start_after", 0):
        print("Waiting for this node's scheduled start", flush=True)
        return
    keys, chain = validate(config)
    state = Path(config["state"])
    state.mkdir(parents=True, exist_ok=True)
    with (state / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt_id = next_attempt(
            state, now=time.time(), interval=config.get("interval_seconds", 0)
        )
        if attempt_id is None:
            print("Not due; current commitment retained", flush=True)
            return
        if shutil.disk_usage(state).free < 30 * 1024**3:
            raise RuntimeError("Less than 30 GiB free; archive completed attempts first")
        from huggingface_hub import HfApi

        HfApi().whoami()
        gpu = os.open(config["compute_lock"], os.O_RDWR | os.O_NOFOLLOW)
        try:
            try:
                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Local GPU is busy; timer will try later", flush=True)
                return
            directory = state / "attempts" / attempt_id
            directory.mkdir(parents=True, exist_ok=False)
            seed = (
                int(hashlib.sha256((config["hotkey"] + attempt_id).encode()).hexdigest()[:8], 16)
                % 2**31
            )
            receipt = {
                "id": attempt_id,
                "status": "training",
                "seed": seed,
                "model": config["model"],
                "hotkey": config["hotkey"],
                "data_sha256": config["data_sha256"],
                "recipe": config.get("recipe", "adapter"),
                "started_at": time.time(),
            }

            def record(status, **extra):
                receipt.update(status=status, **extra)
                save(directory / "receipt.json", receipt)
                save(state / "attempt.json", receipt)

            record("training")
            flag = Path(config["yield_file"])
            flag.touch(exist_ok=False)
            with ExitStack() as cleanup:
                cleanup.callback(flag.unlink, missing_ok=True)
                cleanup.enter_context(yield_gpu("cuda", model=config["model"], minimum_mib=12288))
                deadline = time.monotonic() + 180
                need = config.get("gpu_free_mib", 12288)
                while not gpu_ready("cuda", model=config["model"], minimum_mib=need):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Serving did not yield GPU capacity within 180 seconds")
                    time.sleep(3)
                env = {
                    k: v
                    for k, v in os.environ.items()
                    if k
                    in (
                        "PATH",
                        "HOME",
                        "PYTHONPATH",
                        "HF_HOME",
                        "ZILS_D3_BASE_DIR",
                        "TRITON_CACHE_DIR",
                        "TORCHINDUCTOR_CACHE_DIR",
                        "PYTORCH_CUDA_ALLOC_CONF",
                        "XDG_CACHE_HOME",
                    )
                }
                env.update(
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    TOKENIZERS_PARALLELISM="false",
                    PYTHONUNBUFFERED="1",
                )
                train, finish = commands(config, directory, seed)
                with (directory / "training.log").open("ab") as log:
                    subprocess.run(
                        train,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        pass_fds=(gpu,),
                        timeout=config["max_training_seconds"],
                        check=True,
                    )
        finally:
            os.close(gpu)
        record("merging")
        with (directory / "merge.log").open("ab") as log:
            subprocess.run(
                finish,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
                check=True,
            )
        candidate, digest = miner.local_candidate(directory / "candidate", config["model"])
        record("uploading", checkpoint_sha256=digest)
        revision = miner.upload(candidate, config["repo"], digest)
        record("uploaded", revision=revision, repo=config["repo"])
        miner.check_remote(config["repo"], revision, candidate)
        commitment = miner.commitments.format_commitment(digest, revision, config["repo"])
        record("committing", commitment=commitment)
        after = commit_once(chain, keys, config["hotkey"], commitment)
        record("committed", block=after["block"], finished_at=time.time())
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--check", action="store_true", help="validate identity/data; no training or writes"
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.check:
        validate(config)
        print("Miner identity, registration, public training data and reference verified")
    else:
        run(config)


if __name__ == "__main__":
    main()
