"""Beat the base in your tier. A candidate is a merged d3 checkpoint on the miner's own
Hugging Face repo plus one chain commitment of repo, revision and checkpoint hash; every validator
reads the commitments and pulls that revision. No validator URL is needed.

`setup` asks for the wallet, base profile, candidate and repo once and saves them; every flag then
defaults to the saved value. `doctor` checks the setup step by step and prints the one command
that fixes the first failing step. `commit-candidate` validates a candidate, uploads it (or takes
an uploaded revision), prints the commitment and, with --yes, writes it with the hotkey. `update`
reinstalls at the stable release, the commit release/stable on main names. `commit-body` checks a
soup-pool body (a full fine-tune of the pool's Qwen parent in its chassis layout; d3-mini by default,
--pool for lite, flash or z3) already uploaded to Hugging Face, prints its body hash and `zilsb2` (or
`zilsb3:<pool>`) commitment and, with --yes, writes it with the hotkey (docs/mining.md,
"Soup pool"). A hotkey has one commitment slot, shared by `zils1` and `zilsb2`. `announce-serving`
writes the miner's public endpoint on chain and `serve` answers signed requests for public z3
checkpoints there, from its own worker or, with --runtime, from the box's adapter runtime
(docs/serving-miners.md)."""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import zils

from . import bodies, commitments, models, testnet

NETUID = 579
REQUIRED_MODULES = ("bittensor", "bittensor_wallet", "huggingface_hub", "kev", "torch")
PINNED_BITTENSOR = "11.1.0"
WALLET_PATH = "~/.bittensor/wallets"
SAVED = ("wallet_name", "wallet_hotkey", "wallet_path", "model", "candidate", "repo", "revision")
RUNTIME_TOKEN_ENV = "ZILS_RUNTIME_TOKEN"
INSTALL = "curl -fsSL https://zils.ai/install.sh | sh"
STABLE_URL = "https://raw.githubusercontent.com/ooo-hq/zils/main/release/stable"
# Every d3-mini soup body is a fine-tune of this parent; its change from it is bounded
# (bodies.validate_body). Other pools' parents are in bodies.BODY_POOLS.
BODY_PARENT = bodies.parent(bodies.MINI_POOL)
BODY_FILES = ("*.safetensors", "model.safetensors.index.json", "README.md", "zils-owner.json")
SLOT_NOTE = "a hotkey has one commitment slot, shared by zils1 base candidates and soup bodies"


def paint(code, text):
    """ANSI color on a terminal (green 32, red 31, cyan 36, bold 1, dim 2); plain otherwise."""
    if not os.isatty(1) or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


def config_path():
    return Path(os.environ.get("ZILS_MINER_CONFIG", "~/.zils/miner.json")).expanduser()


def load_config():
    path = config_path()
    return json.loads(path.read_text()) if path.is_file() else {}


def write_config(config):
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def d3_python():
    """The isolated d3 training interpreter: the exported variable, else install.sh's venv."""
    name = os.environ.get(models.runtime_python_env(models.D3))
    return Path(name) if name else zils.ROOT / ".venv-d3/bin/python"


def stable_release():
    with urllib.request.urlopen(STABLE_URL, timeout=10) as response:
        return commitments.check_revision(response.read().decode().strip())


def installed_release():
    done = subprocess.run(
        ["git", "-C", str(zils.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return done.stdout.strip() if done.returncode == 0 else None


def behind_release():
    """A warning when this checkout is not the stable release; None when it is or GitHub is down."""
    try:
        stable = stable_release()
    except (OSError, ValueError):
        return None
    current = installed_release()
    if current != stable:
        return (
            f"this install is at {(current or 'an unknown commit')[:12]}, the stable release is"
            f" {stable[:12]}; run zils update"
        )


def update():
    """Rerun the installer at the stable release, keeping the saved setup."""
    stable = stable_release()
    if installed_release() == stable:
        print(f"{paint('32', '✓')} already on the stable release {stable[:12]}", flush=True)
        return
    if zils.ROOT.name != "zils":
        raise ValueError(
            f"{zils.ROOT} is not an installer checkout (~/.zils/zils); update it with git"
        )
    env = {
        **os.environ,
        "ZILS_HOME": str(zils.ROOT.parent),
        "ZILS_REF": stable,
        "ZILS_SKIP_SETUP": "1",
    }
    if subprocess.run(["sh", "-c", INSTALL], env=env).returncode:
        raise RuntimeError("the installer failed; its log path is printed above")


def chain_config(netuid, network="test"):
    return {"chain": {"network": network, "netuid": netuid}}


def resolve_netuid(network, netuid):
    """Testnet defaults to 579; mainnet has no default, so a mainnet write is always explicit."""
    if netuid is not None:
        return netuid
    if network == "test":
        return NETUID
    raise ValueError("--network finney needs an explicit --netuid")


def hotkey_state(netuid, hotkey, network="test"):
    """The hotkey's uid (None when unregistered) and current commitment, read from the chain."""
    config = chain_config(netuid, network)
    with testnet.connect(config) as sub:
        return testnet.hotkey_state(config, sub, hotkey)


def selected_model(model, candidate=None):
    if model is None and candidate:
        model = models.checkpoint_model(Path(candidate).expanduser())
    if model not in models.d3_family():
        raise ValueError("choose a base profile with zils setup or --model (see --help)")
    return model


def local_candidate(path, model):
    from . import d3

    candidate = Path(path).expanduser().resolve()
    parent = models.checkpoint_model(candidate)
    if parent != model:
        raise ValueError(f"candidate is for {parent}, not the selected base {model}")
    d3.validate_candidate(candidate, model=model)
    return candidate, zils.checkpoint_hash(candidate)


def check_remote(repo, revision, candidate):
    """The public repo at this commit holds every candidate file with the local size and LFS digest."""
    listed = commitments.revision_files(repo, revision)
    for name in models.artifact_files(candidate):
        local, remote = candidate / name, listed.get(name)
        if remote is None:
            raise ValueError(f"{repo} at {revision[:12]} has no {name}")
        if remote["size"] not in (None, local.stat().st_size):
            raise ValueError(f"{name} on Hugging Face differs in size from the local candidate")
        if remote["sha256"] is not None:
            with local.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != remote["sha256"]:
                    raise ValueError(f"{name} on Hugging Face differs from the local candidate")


def upload(candidate, repo, sha256):
    """Push only the candidate files to the miner's public model repo; returns the commit sha."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    info = api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=str(candidate),
        allow_patterns=list(models.artifact_files(candidate)),
        commit_message=f"Zils d3 base candidate {sha256[:12]}",
    )
    return commitments.check_revision(info.oid)


def wallet(args):
    from bittensor.wallet import Wallet

    return Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey, path=args.wallet_path)


def commit_command(args, revision=None):
    command = [
        "zils commit-candidate",
        f"--model {args.model}",
        f"--wallet-name {args.wallet_name} --wallet-hotkey {args.wallet_hotkey}"
        f" --wallet-path {args.wallet_path}",
        f"--candidate {args.candidate} --repo {args.repo}"
        + (f" --revision {revision}" if revision else ""),
    ]
    if args.network != "test":
        command.append(f"--network {args.network}")
    if args.netuid != NETUID or args.network != "test":
        command.append(f"--netuid {args.netuid}")
    return " \\\n  ".join(command + ["--yes"])


def registered_state(args, hotkey):
    state = hotkey_state(args.netuid, hotkey, args.network)
    if state["uid"] is None:
        raise ValueError(
            f"hotkey {hotkey} is not registered on {testnet.NAMES[args.network]} subnet {args.netuid}"
        )
    return state


def write_chain(args, keys, hotkey, text):
    """Write `text` as the hotkey's commitment and read it back."""
    config = chain_config(args.netuid, args.network)
    with testnet.connect(config, raw_calls=True) as sub:
        testnet.write_commitment(config, sub, keys, text)
        after = testnet.hotkey_state(config, sub, hotkey)["commitment"]
    if (after or {}).get("data") != text:
        raise ValueError(
            "the chain does not show the new commitment; read it again before retrying"
        )
    return after


def commit_candidate(args):
    candidate, sha256 = local_candidate(args.candidate, args.model)
    commitments.check_repo(args.repo)
    if args.revision is not None:
        commitments.check_revision(args.revision)
    keys = wallet(args)
    hotkey = keys.hotkey.ss58_address
    state = registered_state(args, hotkey)
    current = (state["commitment"] or {}).get("data")
    revision = args.revision
    print(
        f"miner {state['uid']}: candidate {sha256}\n"
        f"  tier {models.pool_id(args.model)}; base profile {args.model}\n"
        f"  repo {args.repo} at {revision or '<the commit the upload creates>'}\n"
        f"  commitment {commitments.format_commitment(sha256, revision or '0' * 40, args.repo)}"
        + ("" if revision else " (revision filled in after the upload)")
        + f"\n  current on chain: {current or 'none'}",
        flush=True,
    )
    if revision and current == commitments.format_commitment(sha256, revision, args.repo):
        # Committing again would move the block later and could hand the hash to a copier.
        print(f"already committed at block {state['commitment']['block']}; nothing to do")
        return state["commitment"]
    if not args.yes:
        print("dry run: nothing uploaded or written; run again with --yes", flush=True)
        return None
    if revision is None:
        revision = upload(candidate, args.repo, sha256)
        print(f"uploaded {args.repo} at {revision}", flush=True)
    check_remote(args.repo, revision, candidate)
    text = commitments.format_commitment(sha256, revision, args.repo)
    if current == text:
        print(f"already committed at block {state['commitment']['block']}; nothing to do")
        return state["commitment"]
    after = write_chain(args, keys, hotkey, text)
    write_config(
        load_config()
        | {
            "model": args.model,
            "candidate": str(candidate),
            "repo": args.repo,
            "revision": revision,
        }
    )
    print(
        f"committed at block {after['block']}; validators pull it at their next round",
        flush=True,
    )
    return after


def hf_cache():
    return str(Path(os.environ.get("HF_HOME", str(zils.ROOT / ".cache/huggingface"))) / "hub")


def body_pool(args):
    return getattr(args, "pool", None) or bodies.MINI_POOL


def body_references(args):
    """The chassis (the pool's pinned release) and parent (Qwen) folders: given paths, else downloaded."""
    pool = body_pool(args)
    if args.chassis:
        chassis = Path(args.chassis).expanduser().resolve(strict=True)
    else:
        from . import d3

        chassis = d3.pinned_base_path(download=True, model=models.D3_POOL_MODELS[pool])
    if args.base:
        return chassis, Path(args.base).expanduser().resolve(strict=True)
    from huggingface_hub import snapshot_download

    repo, revision = bodies.parent(pool)
    base = snapshot_download(
        repo_id=repo,
        revision=revision,
        cache_dir=hf_cache(),
        allow_patterns=["*.safetensors", "model.safetensors.index.json"],
    )
    return chassis, Path(base)


def owner_missing(repo, hotkey):
    return ValueError(
        f"the body has no zils-owner.json. Write one naming hotkey {hotkey} with"
        f" `zils commit-body --repo {repo} --path <local body folder> --write-owner`, upload it"
        " to the repo, then commit the new revision"
    )


def download_body(repo, revision, hotkey, pool=bodies.MINI_POOL):
    """The body files at exactly this commit, after checking the listing for the ownership file and
    the pool's byte budget so a refused body is not downloaded."""
    listed = commitments.revision_files(repo, revision)
    if "zils-owner.json" not in listed:
        raise owner_missing(repo, hotkey)
    total = sum(v["size"] or 0 for k, v in listed.items() if k.endswith(".safetensors"))
    if total > bodies.max_bytes(pool):
        raise ValueError(
            f"{repo} holds {total} bytes of safetensors, over {bodies.max_bytes(pool)}"
        )
    from huggingface_hub import snapshot_download

    folder = snapshot_download(
        repo_id=repo, revision=revision, cache_dir=hf_cache(), allow_patterns=list(BODY_FILES)
    )
    return Path(folder)


def check_remote_body(repo, revision, folder):
    """The repo at this commit holds every local body file with the same size and LFS digest."""
    listed = commitments.revision_files(repo, revision)
    names = sorted(p.name for p in folder.iterdir() if p.name.endswith(".safetensors"))
    names += [name for name in BODY_FILES[1:] if (folder / name).is_file()]
    for name in names:
        local, remote = folder / name, listed.get(name)
        if remote is None:
            raise ValueError(f"{repo} at {revision[:12]} has no {name}")
        if remote["size"] not in (None, local.stat().st_size):
            raise ValueError(f"{name} on Hugging Face differs from the local body")
        if remote["sha256"] is not None:
            with local.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != remote["sha256"]:
                    raise ValueError(f"{name} on Hugging Face differs from the local body")


def commit_body(args):
    """Validate a soup body at an uploaded revision, print its zilsb2 commitment and, with --yes,
    write it. --write-owner only writes zils-owner.json into --path and checks the body locally."""
    if args.write_owner and args.yes:
        raise ValueError("--write-owner writes a file to upload first; commit after the upload")
    if args.write_owner and not args.path:
        raise ValueError("--write-owner needs --path, the local body folder you upload")
    commitments.check_repo(args.repo)
    if not args.write_owner:
        if args.revision is None:
            raise ValueError("--revision is needed: the commit sha of the uploaded body")
        commitments.check_revision(args.revision)
    keys = wallet(args)
    hotkey = keys.hotkey.ss58_address
    if args.path:
        folder = Path(args.path).expanduser().resolve(strict=True)
    else:
        folder = download_body(args.repo, args.revision, hotkey, body_pool(args))
    owner = folder / "zils-owner.json"
    if args.write_owner:
        owner.write_text(json.dumps({"hotkey": hotkey}) + "\n")
        print(f"wrote {owner} naming hotkey {hotkey}", flush=True)
    elif not owner.is_file():
        raise owner_missing(args.repo, hotkey)
    chassis, base = body_references(args)
    # The miner's own folder or snapshot download (symlinks into the HF blob store) is trusted here;
    # validators download the files themselves and check them strictly.
    pool = body_pool(args)
    sha256 = bodies.validate_body(
        folder, chassis, hotkey=hotkey, base=base, trusted=True, max_bytes=bodies.max_bytes(pool)
    )
    if args.write_owner:
        print(
            f"body {sha256} passes the local checks. Upload the folder, then commit that"
            f" revision:\n  hf upload {args.repo} {folder} .\n"
            f"  zils commit-body --pool {pool} --repo {args.repo} --revision <the new commit>"
            f" --path {folder}",
            flush=True,
        )
        return None
    text = bodies.format_pool_commitment(pool, sha256, args.revision, args.repo)
    state = registered_state(args, hotkey)
    current = (state["commitment"] or {}).get("data")
    print(
        f"miner {state['uid']}: body {sha256} for pool {pool}\n"
        f"  repo {args.repo} at {args.revision}\n"
        f"  commitment {text}\n"
        f"  current on chain: {current or 'none'}",
        flush=True,
    )
    if current == text:
        print(f"already committed at block {state['commitment']['block']}; nothing to do")
        return state["commitment"]
    print(f"  note: {SLOT_NOTE}" + (f"; this replaces {current}" if current else ""), flush=True)
    if not args.yes:
        print("dry run: nothing written; run again with --yes", flush=True)
        return None
    if args.path:
        check_remote_body(args.repo, args.revision, folder)
    after = write_chain(args, keys, hotkey, text)
    print(
        f"committed at block {after['block']}; validators admit it at their next soup round",
        flush=True,
    )
    return after


def serving_endpoint(args):
    """The endpoint this miner announces; refuses anything but a public IPv4 before any chain call."""
    from . import serving  # lazy: doctor and update must run where bittensor_wallet is broken

    return serving.public_endpoint(f"http://{args.public_ip}:{args.port}")


def announce_serving(args):
    endpoint = serving_endpoint(args)
    keys = wallet(args)
    hotkey = keys.hotkey.ss58_address
    config = chain_config(args.netuid, args.network)
    with testnet.connect(config) as sub:
        state = testnet.hotkey_state(config, sub, hotkey)
        if state["uid"] is None:
            raise ValueError(
                f"hotkey {hotkey} is not registered on {testnet.NAMES[args.network]} subnet {args.netuid}"
            )
        current = testnet.read_axon(sub, args.netuid, hotkey, state["block"])
        print(
            f"miner {state['uid']}: current axon {current or 'none'}\n  new endpoint {endpoint}",
            flush=True,
        )
        if current == endpoint:
            print("already announced; nothing to do")
            return current
        if not args.yes:
            print("dry run: nothing written; run again with --yes", flush=True)
            return None
        testnet.write_axon(config, sub, keys, endpoint)
        after = testnet.read_axon(sub, args.netuid, hotkey, sub.block)
    if after != endpoint:
        raise ValueError("the chain does not show the new axon; read it again before retrying")
    print(f"announced {endpoint}; validators find it at their next refresh", flush=True)
    return after


def permit_checker(read, gateways):
    """`allowed(hotkey)`: a gateway hotkey or a current permit holder. `read()` returns the permit
    hotkeys (the first call must succeed); `allowed.refresh()` re-reads, keeping the last set on error."""
    holders = [read()]

    def allowed(hotkey):
        return hotkey in gateways or hotkey in holders[0]

    def refresh():
        try:
            holders[0] = read()
        except Exception as error:  # any chain failure keeps the last known set
            print(f"permit refresh failed, keeping the last set: {error}", flush=True)

    allowed.refresh = refresh
    return allowed


def gateway_hotkeys(values):
    from bittensor_wallet import Keypair

    for value in values:
        try:
            Keypair(ss58_address=value)
        except ValueError:
            raise ValueError(f"--gateway-hotkey {value!r} is not an SS58 address") from None
    return set(values)


def runtime_token(args):
    """The adapter runtime's bearer token from --runtime-token-env; never printed."""
    token = os.environ.get(args.runtime_token_env, "")
    if not token:
        raise ValueError(f"set {args.runtime_token_env} to the adapter runtime's token")
    return token


def serve(args, interval=600.0):
    """Exits (non-zero, through main) when the inference worker dies, or with --runtime when the
    runtime's identity changes, so a supervisor restarts it."""
    from . import serving_server

    endpoint = serving_endpoint(args)
    if args.runtime:
        if len(args.checkpoint) != 1:
            raise ValueError("--runtime serves one --checkpoint: the runtime's base reference")
        token = runtime_token(args)
    gateways = gateway_hotkeys(args.gateway_hotkey or ())
    key = wallet(args).hotkey  # the coldkey is never read
    hotkey = key.ss58_address
    config = chain_config(args.netuid, args.network)

    def read_permits():
        with testnet.connect(config) as sub:
            block = sub.block
            permits = sub.query(("SubtensorModule", "ValidatorPermit"), [args.netuid], block=block)
            return {
                holder
                for uid, permit in enumerate(permits)
                if permit
                for holder in [
                    sub.query(("SubtensorModule", "Keys"), [args.netuid, uid], block=block)
                ]
                if holder
            }

    with testnet.connect(config) as sub:
        announced = testnet.read_axon(sub, args.netuid, hotkey, sub.block)
    if announced != endpoint:
        raise ValueError(
            f"the chain shows axon {announced or 'none'} for {hotkey}, not {endpoint}; run"
            f" zils announce-serving --public-ip {args.public_ip} --port {args.port} --yes first"
        )
    if args.runtime:
        from . import serving_runtime

        loaded = serving_runtime.connect(args.runtime, token, args.checkpoint[0])
    else:
        loaded = serving_server.start_worker(args.checkpoint, d3_python())
    allowed = permit_checker(read_permits, gateways)

    def keep_fresh():
        while True:
            time.sleep(interval)
            allowed.refresh()

    threading.Thread(target=keep_fresh, daemon=True).start()
    server = serving_server.make_server(
        key, endpoint, loaded, allowed, ("0.0.0.0", args.port), args.slots
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"serving {', '.join(s.model for s in loaded)} at {endpoint}", flush=True)
    if args.runtime:
        stop = threading.Event()  # set by the health check when the runtime's identity changes
        loaded[0].watch(stop)
        stop.wait()
        server.shutdown()
        raise RuntimeError(
            f"the adapter runtime's identity changed ({loaded[0].changed}); restart zils serve"
        )
    loaded[0].worker.exited.wait()
    server.shutdown()
    raise RuntimeError("the inference worker exited; restart zils serve")


def doctor(args):
    """Run the outsider's checklist in order; stop at the first failing step with its fix."""
    from importlib.metadata import PackageNotFoundError, version

    flags = f"--wallet {args.wallet_name} --wallet-hotkey {args.wallet_hotkey} --wallet-path {args.wallet_path}"
    btcli = f"btcli --network {args.network} {flags}"
    state = {}

    def requirements():
        missing = []
        for name in REQUIRED_MODULES:
            try:
                importlib.import_module(name)
            except ImportError:
                missing.append(name)
        try:
            pinned = version("bittensor") == PINNED_BITTENSOR
        except PackageNotFoundError:
            pinned = False
        if missing or not pinned:
            return (
                f"missing {', '.join(missing) or 'bittensor ' + PINNED_BITTENSOR}; rerun the"
                f" installer\n  {INSTALL}"
            )

    def d3_runtime():
        python = d3_python()
        if not python.exists():
            return (
                f"no d3 training runtime at {python}; it needs Linux x86_64 with an NVIDIA GPU."
                f" On that host rerun the installer\n  {INSTALL}"
            )
        check = (
            "import sys, torch; from zils import d3; d3.verify_runtime(sys.argv[1]);"
            " assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"
            " 'no BF16-capable CUDA GPU is visible'"
        )
        done = subprocess.run(
            [str(python), "-c", check, args.model], cwd=zils.ROOT, capture_output=True, text=True
        )
        if done.returncode:
            reason = (done.stderr.strip().splitlines() or ["the check failed"])[-1]
            return f"{python}: {reason}; rerun the installer\n  {INSTALL}"
        state["d3"] = python

    def serving_runtime():
        from . import serving_runtime

        try:
            serving_runtime.check(args.runtime, runtime_token(args), args.model)
        except ValueError as error:
            return str(error)

    def wallet_hotkey():
        key = wallet(args)
        if not key.hotkey_file.exists_on_device():
            return (
                f"no hotkey file at {key.hotkey_file.path}; create the identity (coldkey stays on"
                f" your admin machine) with\n  {btcli} wallet create"
            )
        state["hotkey"] = key.hotkey.ss58_address

    def registration():
        state.update(hotkey_state(args.netuid, state["hotkey"], args.network))
        if state["uid"] is None:
            return (
                f"hotkey {state['hotkey']} is not registered on {testnet.NAMES[args.network]} subnet {args.netuid};"
                f" fund the coldkey with test TAO, then\n"
                f"  {btcli} tx burned-register --netuid {args.netuid}"
            )

    def candidate():
        d3 = f"{state['d3']} -m zils.d3"
        if args.candidate is None:
            reference = f"models/{args.model}-reference"
            adapter, merged = f"{args.model}-adapter", f"{args.model}-candidate"
            return (
                "no --candidate; train and merge one with the reference recipe\n"
                f"  {d3} reference --model {args.model} --out {reference}\n"
                f"  {d3} train --data train.jsonl --reference {reference}"
                f" --out {adapter} --seed 553 --device cuda\n"
                f"  {d3} candidate --adapter {adapter} --out {merged}"
                f" > {args.model}-provenance.json\n"
                "then run zils setup again, or\n"
                f"  zils doctor --model {args.model} --candidate {merged}"
            )
        try:
            state["path"], state["sha256"] = local_candidate(args.candidate, args.model)
        except (ValueError, OSError) as error:
            return f"{error}; merge again with {d3} candidate"

    def hugging_face():
        if args.repo is None:
            return (
                "no --repo; pick a public model repo id under your Hugging Face account"
                " (commit-candidate creates the repo) and run zils setup again"
            )
        if args.revision is None:
            from huggingface_hub import get_token

            if get_token() is None:
                return "not logged in to Hugging Face; run\n  hf auth login"
            return "not uploaded yet; upload and commit with\n  " + commit_command(args)
        try:
            commitments.check_repo(args.repo)
            commitments.check_revision(args.revision)
            check_remote(args.repo, args.revision, state["path"])
        except (ValueError, OSError) as error:
            return f"{error}; make the repo public, or upload again with\n  " + commit_command(args)

    def commitment():
        wanted = commitments.format_commitment(state["sha256"], args.revision, args.repo)
        current = (state["commitment"] or {}).get("data")
        if current != wanted:
            return (
                f"the chain holds {current or 'no commitment'} for this hotkey; commit with\n  "
                + commit_command(args, args.revision)
            )

    steps = [
        ("requirements", requirements),
        *([("serving runtime", serving_runtime)] if getattr(args, "runtime", None) else []),
        ("d3 runtime", d3_runtime),
        ("wallet hotkey", wallet_hotkey),
        ("registration", registration),
        ("candidate", candidate),
        ("hugging face", hugging_face),
        ("commitment", commitment),
    ]
    print(paint("1", "Zils doctor"), flush=True)
    print(f"Beat the base in your tier: {models.pool_id(args.model)} · {args.model}", flush=True)
    print("Validators decide which tiers are open and which base profile to beat.", flush=True)
    warning = behind_release()
    if warning:
        print(f"{paint('1;33', '⚠')} {warning}", flush=True)
    for name, check in steps:
        fix = check()
        if fix:
            print(f"{paint('1;31', '✗')} {paint('1', name)}", flush=True)
            for line in fix.splitlines():
                # Indented lines are commands to run; the rest explains why.
                indented = line.startswith("  ")
                print(f"    {paint('36', line.strip())}" if indented else f"  {line}", flush=True)
            return False
        print(f"{paint('32', '✓')} {name}", flush=True)
    print(
        f"\n{paint('1;32', 'Ready.')} uid {state['uid']} committed {state['sha256'][:12]} at block"
        f" {state['commitment']['block']}; every validator pulls it at its next round. Results:"
        " the chain weights for your uid (btcli subnets metagraph --netuid"
        f" {args.netuid} --network {args.network})",
        flush=True,
    )
    return True


def ask(question, default=None):
    hint = paint("2", f" [{default}]") if default else ""
    answer = input(f"{paint('36', '?')} {question}{hint}: ").strip()
    return answer or default


def setup(args):
    """Ask for each saved value (Enter keeps the shown one), save them, then run doctor."""
    config = load_config()
    print(
        f"{paint('1;32', '◆ Zils miner setup')}  {paint('2', 'Enter keeps the value in brackets')}"
    )
    print("Beat the base in your tier. Choose the base profile your validator accepts.")
    path = ask("Bittensor wallet folder", config.get("wallet_path", WALLET_PATH))
    wallets = Path(path).expanduser()
    found = sorted(p.name for p in wallets.iterdir() if p.is_dir()) if wallets.is_dir() else []
    if found:
        print(paint("2", f"  wallets there: {', '.join(found)}"))
    name = ask("Wallet name", config.get("wallet_name") or (found[0] if found else None))
    hotkeys = wallets / (name or "") / "hotkeys"
    found_keys = hotkeys.iterdir() if name and hotkeys.is_dir() else ()
    keys = sorted(p.name for p in found_keys if not p.name.endswith("pub.txt"))
    if keys:
        print(paint("2", f"  hotkeys: {', '.join(keys)}"))
    hotkey = ask("Hotkey name", config.get("wallet_hotkey") or (keys[0] if keys else "default"))
    candidate = ask(
        "Candidate folder from zils.d3 candidate (Enter if not trained yet)",
        config.get("candidate"),
    )
    print("Base profiles (availability depends on the validator):")
    for profile in models.d3_family():
        print(f"  {models.pool_id(profile)}: {profile}")
    model = selected_model(
        args.model
        or ask("Base profile (Enter to read it from the candidate)", config.get("model")),
        candidate,
    )
    repo = ask("Public Hugging Face repo for it, owner/name", config.get("repo"))
    if not name:
        raise ValueError("a wallet name is needed; create one with btcli wallet create")
    candidate = str(Path(candidate).expanduser().resolve()) if candidate else None
    if (
        candidate != config.get("candidate")
        or repo != config.get("repo")
        or config.get("model") not in (None, model)
    ):
        config.pop("revision", None)  # a changed target must not reuse the uploaded revision
    config |= {"wallet_path": path, "wallet_name": name, "wallet_hotkey": hotkey}
    config |= {"model": model, "candidate": candidate, "repo": repo}
    write_config(config)
    print(f"{paint('32', '✓')} saved to {config_path()}\n", flush=True)
    for key in SAVED:
        setattr(args, key, config.get(key))
    return doctor(args)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zils", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("setup", help="choose a base profile, wallet, candidate and repo")
    check = commands.add_parser("doctor", help="check the setup and print the next command")
    commands.add_parser("update", help="reinstall at the stable release")
    commit = commands.add_parser(
        "commit-candidate", help="upload a base candidate and commit it on chain"
    )
    body = commands.add_parser(
        "commit-body",
        help="check a soup-pool body on Hugging Face and commit it on chain",
        description="Check a soup-pool body uploaded to Hugging Face (a bf16 fine-tune of the"
        " pool's parent in its chassis layout: "
        + ", ".join(f"{p} {bodies.parent(p)[0]}" for p in bodies.BODY_POOLS)
        + "), print its body hash and commitment (zilsb2 for d3-mini, zilsb3:<pool> otherwise)"
        f" and, with --yes, write it. Note: {SLOT_NOTE}; committing a body"
        " replaces whatever the hotkey committed before.",
    )
    body.add_argument("--repo", required=True, help="your public Hugging Face model repo")
    body.add_argument(
        "--revision", help="40-hex commit sha of the uploaded body (not needed with --write-owner)"
    )
    body.add_argument("--path", help="local copy of the body at --revision (skips the download)")
    body.add_argument(
        "--pool",
        choices=tuple(bodies.BODY_POOLS),
        default=bodies.MINI_POOL,
        help="soup pool the body is for (default %(default)s; d3 is the z3 27B pool)",
    )
    body.add_argument("--chassis", help="local pinned release of the pool (skips the download)")
    body.add_argument("--base", help="local copy of the pool's Qwen parent at its pinned revision")
    body.add_argument(
        "--write-owner",
        action="store_true",
        help="write zils-owner.json naming your hotkey into --path; upload it before committing",
    )
    body.add_argument(
        "--yes", action="store_true", help="write the commitment; without it, a dry run"
    )
    announce = commands.add_parser("announce-serving", help="write your public endpoint on chain")
    host = commands.add_parser("serve", help="serve z3 checkpoints to validators")
    for command in (announce, host):
        command.add_argument("--public-ip", required=True, help="public IPv4 of this machine")
        command.add_argument("--port", type=int, required=True)
    announce.add_argument(
        "--yes", action="store_true", help="write the axon; without it, a dry run"
    )
    host.add_argument(
        "--checkpoint", action="append", required=True, help="z3 checkpoint dir; repeatable"
    )
    host.add_argument(
        "--gateway-hotkey", action="append", help="extra hotkey allowed to call; repeatable"
    )
    host.add_argument("--slots", type=int, default=8)
    for command in (check, host):
        command.add_argument(
            "--runtime",
            help="answer from this box's adapter runtime at http://127.0.0.1:<port> (serves the"
            " base only, on its bulk lane) instead of loading a second model copy",
        )
        command.add_argument(
            "--runtime-token-env",
            default=RUNTIME_TOKEN_ENV,
            help="environment variable holding the runtime's token (default %(default)s)",
        )
    for command in (start, check, commit, body, announce, host):
        command.add_argument(
            "--network",
            choices=tuple(testnet.ENDPOINTS),
            default="test",
            help="finney is mainnet and needs an explicit --netuid",
        )
        command.add_argument("--netuid", type=int)
        if command in (start, check, commit):
            command.add_argument(
                "--model",
                choices=models.d3_family(),
                help="base profile to beat in your tier; saved by setup or read from the candidate",
            )
    for command in (check, commit, body, announce, host):
        command.add_argument("--wallet-name")
        command.add_argument("--wallet-hotkey", help="hotkey name; the coldkey is never read")
        command.add_argument("--wallet-path")
    for command in (check, commit):
        command.add_argument("--candidate", help="directory written by `zils.d3 candidate`")
        command.add_argument("--repo", help="your public Hugging Face model repo")
        command.add_argument("--revision", help="40-hex commit sha of an upload already in --repo")
    commit.add_argument(
        "--yes", action="store_true", help="upload and write the commitment; without it, a dry run"
    )
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "netuid"):
            args.netuid = resolve_netuid(args.network, args.netuid)
        if args.command == "setup":
            if not setup(args):
                parser.exit(1)
            return
        if args.command == "update":
            update()
            return
        config = load_config()
        if args.command == "commit-candidate":
            config.pop("revision", None)  # a fresh upload must not reuse the last one
        if args.command == "commit-body":  # the saved repo and revision are the base candidate's
            config = {k: v for k, v in config.items() if k.startswith("wallet_")}
        for key in SAVED:
            if getattr(args, key, None) is None:
                setattr(args, key, config.get(key))
        args.wallet_path = args.wallet_path or WALLET_PATH
        if args.wallet_name is None or args.wallet_hotkey is None:
            raise ValueError("no wallet configured; run zils setup (or pass --wallet-name)")
        if args.command in ("commit-body", "announce-serving", "serve"):
            run = {"commit-body": commit_body, "announce-serving": announce_serving, "serve": serve}
            run[args.command](args)
            return
        args.model = selected_model(args.model, args.candidate)
        if args.command == "doctor":
            if not doctor(args):
                parser.exit(1)
            return
        if args.candidate is None or args.repo is None:
            raise ValueError("no candidate or repo configured; run zils setup")
        commit_candidate(args)
    except KeyboardInterrupt:
        print("Zils miner stopped.", flush=True)
    except (ValueError, OSError, RuntimeError, KeyError, EOFError) as error:
        parser.exit(1, f"zils miner: {error}\n")


if __name__ == "__main__":
    main()
