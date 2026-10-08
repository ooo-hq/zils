"""Process isolation, signing, and the pinned model cache shared by services."""

import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request

from bittensor_wallet import Keypair

import zils

from . import ROOT, models, protocol as wire, settings
from .cloud import APIError

LIMIT = 64 * 1024


class CapacityUnavailable(APIError):
    def __init__(self):
        super().__init__(503, "Waiting for GPU capacity; this attempt can be deferred.")


def canonical(payload):
    return (
        b"fez-fleet/v1\0"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def signed(payload, key):
    return {"payload": dict(payload), "signature": key.sign(canonical(payload)).hex()}


def verified(message, hotkey):
    if not isinstance(message, dict) or set(message) != {"payload", "signature"}:
        raise ValueError("invalid signed message")
    if not isinstance(message["signature"], str) or not re.fullmatch(
        "[a-f0-9]{128}", message["signature"]
    ):
        raise ValueError("invalid message signature")
    if not isinstance(message["payload"], dict) or not Keypair(ss58_address=hotkey).verify(
        canonical(message["payload"]), bytes.fromhex(message["signature"])
    ):
        raise ValueError("validator or miner signature failed")
    return message["payload"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def locked(path, wait=False):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError("this service is already running") from error
        yield


def gpu_ready(device, *, model=None, minimum_mib=0):
    """Opt-in deployment capacity gate, also checked under the shared compute lock."""
    if model == models.IMAJEV:
        if device != "cuda":
            return False
        minimum = max(12288, int(settings.get("ZILS_IMAGE_MIN_FREE_MIB", "12288")))
    else:
        if device != "cuda" or settings.get("ZILS_GPU_MIN_FREE_MIB") is None:
            return True
        minimum = int(settings.required("ZILS_GPU_MIN_FREE_MIB"))
    minimum = max(minimum, minimum_mib)
    if minimum < 1:
        raise ValueError("GPU memory requirement must be positive")
    try:
        result = subprocess.run(
            [
                settings.get("ZILS_NVIDIA_SMI", "nvidia-smi"),
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return int(result.stdout.strip()) >= minimum
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def run_child(command, log, device, timeout=3600, *, model=None, minimum_mib=0, check_lease=None):
    image_environment = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TMP",
        "TEMP",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "XDG_CACHE_HOME",
        "TRITON_CACHE_DIR",
        "TORCH_HOME",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
        "ZILS_IMAGE_REFERENCE",
        "ZILS_COMPUTE_LOCK",
        "ZILS_NVIDIA_SMI",
    }
    environment = {
        **{
            k: v
            for k, v in os.environ.items()
            if k not in {"SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_DB_URL"}
            and (model != models.IMAJEV or k in image_environment)
        },
        "HF_HOME": os.environ.get("HF_HOME", str(ROOT / ".cache/huggingface")),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    # ponytail: one GPU job per user/device; add device-index locks only when multi-GPU hosts exist.
    # Keep the legacy default path so old and new processes share one compute lock.
    lock = Path(
        settings.get(
            "ZILS_COMPUTE_LOCK",
            str(Path(tempfile.gettempdir()) / f"fez-compute-{os.getuid()}-{device}.lock"),
        )
    )
    # Images defer immediately under contention; they never reserve a training attempt while waiting.
    if model == models.IMAJEV:
        try:
            with locked(lock):
                if not gpu_ready(device, model=model, minimum_mib=minimum_mib):
                    raise CapacityUnavailable()
                _run_child(command, log, environment, timeout, check_lease)
        except RuntimeError as error:
            if str(error) == "this service is already running":
                raise CapacityUnavailable() from None
            raise
        return
    with locked(lock, wait=True):
        deadline = time.monotonic() + timeout
        while not gpu_ready(device):
            if time.monotonic() >= deadline:
                raise CapacityUnavailable()
            time.sleep(5)
        _run_child(command, log, environment, timeout, check_lease)


def _run_child(command, log, environment, timeout, check_lease=None):
    with Path(log).open("ab") as output:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
            cwd=ROOT,
        )
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if check_lease:
                    check_lease()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    process.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise RuntimeError(f"worker failed; inspect {log}")
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=6)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=6)


def signing_key(config):
    if "wallet" in config:
        from bittensor.wallet import Wallet

        key = Wallet(**config["wallet"]).hotkey
        if key.ss58_address != config.get("hotkey", config["validator_hotkey"]):
            raise ValueError("wallet hotkey does not match the configured identity")
        return key
    if "chain" in config:
        raise ValueError("testnet services require a registered wallet hotkey")
    return Keypair.create_from_seed(config["seed"])


def request(config, path, message=None):
    data = json.dumps(message).encode() if message is not None else None
    req = Request(config["validator"] + path, data, {"Content-Type": "application/json"})
    with wire.opener().open(req, timeout=10) as response:
        body = response.read(LIMIT + 1)
    if len(body) > LIMIT:
        raise ValueError("validator response too large")
    return verified(json.loads(body), config["validator_hotkey"])


def prepare_base(model=models.KEV):
    if model == models.IMAJEV:
        from .imajev import resolve_reference

        base, _ = resolve_reference(Path(settings.required("ZILS_IMAGE_REFERENCE")))
        return base
    if model == models.JEVK5:
        from .jevk5 import base_path

        return base_path(download=True)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    cache = Path(os.environ.get("HF_HOME", str(ROOT / ".cache/huggingface"))) / "hub"
    options = {
        "repo_id": zils.BASE,
        "revision": wire.BASE_REVISION,
        "cache_dir": str(cache),
        "allow_patterns": ["*.json", "*.safetensors", "*.txt", "*.jinja"],
    }
    try:
        snapshot_download(**options, local_files_only=True)
    except LocalEntryNotFoundError:
        print("Downloading the pinned base model for this machine's first start.", flush=True)
        snapshot_download(**options)
