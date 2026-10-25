"""Opt-in local serving handoff for testnet hosts that also train on their GPU."""

import fcntl
import json
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import models, runtime, settings


def directory():
    value = settings.get("ZILS_SERVING_DIRECTORY")
    return Path(value) if value else None


def state(root):
    try:
        value = json.loads((root / "state.json").read_text())
        if 0 <= time.time() - value["checked_at"] < 15:
            return value
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {}


def ready(device, *, model=None, minimum_mib=0):
    if runtime.gpu_ready(device, model=model, minimum_mib=minimum_mib):
        return True
    root = directory()
    if root is None or device != "cuda" or model not in runtime.CAPACITY_ENV:
        return False
    if state(root).get("serving") is not True:
        return False
    # Match the pinned runtime's floor; its actual-free check remains the final gate.
    floor = 12288 if models.pool_id(model) == "d3-mini" else 0
    required = max(floor, minimum_mib, int(settings.get(runtime.CAPACITY_ENV[model], str(floor))))
    reclaim = int(settings.required("ZILS_SERVING_YIELD_MIB"))
    if reclaim < 1 or required < 1:
        raise ValueError("Serving yield and GPU capacity must be measured positive MiB")
    try:
        result = subprocess.run(
            [
                settings.get("ZILS_NVIDIA_SMI", "nvidia-smi"),
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        free, total = (int(v.strip()) for v in result.stdout.split(","))
        return 0 <= free <= total and min(total, free + reclaim) >= required
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def active_requests(root, invocations=()):
    requests = []
    for path in sorted(root.glob("*.request")):
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            continue
        with os.fdopen(fd) as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                requests.append(path.name)
            else:
                # systemd may still be stopping a dead parent's GPU children.
                if stream.read(64).strip() in invocations:
                    requests.append(path.name)
    return requests


@contextmanager
def yield_gpu(device, *, model=None, minimum_mib=0, check_lease=None, timeout=180, poll=1):
    root = directory()
    if root is None:
        yield
        return
    path = root / (uuid.uuid4().hex + ".request")
    staging = path.with_suffix(".pending")
    try:
        fd = os.open(staging, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except OSError:
        raise runtime.CapacityUnavailable() from None
    try:
        with os.fdopen(fd, "w") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stream.write(os.environ.get("INVOCATION_ID", ""))
            stream.flush()
            staging.rename(path)
            deadline = time.monotonic() + timeout
            while True:
                if check_lease:
                    check_lease()
                response = state(root)
                acknowledged = (
                    response.get("serving") is False
                    and isinstance(response.get("yielded"), list)
                    and path.name in response["yielded"]
                )
                if acknowledged and runtime.gpu_ready(device, model=model, minimum_mib=minimum_mib):
                    break
                if time.monotonic() >= deadline:
                    raise runtime.CapacityUnavailable()
                time.sleep(poll)
            yield
    finally:
        path.unlink(missing_ok=True)
        staging.unlink(missing_ok=True)
