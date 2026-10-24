"""Beat the base in your tier. A candidate is a merged d3 checkpoint on the miner's own
Hugging Face repo plus one chain commitment of repo, revision and checkpoint hash; every validator
reads the commitments and pulls that revision. No validator URL is needed.

`setup` asks for the wallet, base profile, candidate and repo once and saves them; every flag then
defaults to the saved value. `doctor` checks the setup step by step and prints the one command
that fixes the first failing step. `commit-candidate` validates a candidate, uploads it (or takes
an uploaded revision), prints the commitment and, with --yes, writes it with the hotkey. `update`
reinstalls at the stable release, the commit release/stable on main names."""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

import zils

from . import commitments, models, testnet

NETUID = 579
REQUIRED_MODULES = ("bittensor", "bittensor_wallet", "huggingface_hub", "kev", "torch")
PINNED_BITTENSOR = "11.1.0"
WALLET_PATH = "~/.bittensor/wallets"
SAVED = ("wallet_name", "wallet_hotkey", "wallet_path", "model", "candidate", "repo", "revision")
INSTALL = "curl -fsSL https://zils.ai/install.sh | sh"
STABLE_URL = "https://raw.githubusercontent.com/ooo-hq/zils/main/release/stable"


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


def commit_candidate(args):
    candidate, sha256 = local_candidate(args.candidate, args.model)
    commitments.check_repo(args.repo)
    if args.revision is not None:
        commitments.check_revision(args.revision)
    keys = wallet(args)
    hotkey = keys.hotkey.ss58_address
    state = hotkey_state(args.netuid, hotkey, args.network)
    if state["uid"] is None:
        raise ValueError(
            f"hotkey {hotkey} is not registered on {testnet.NAMES[args.network]} subnet {args.netuid}"
        )
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
    config = chain_config(args.netuid, args.network)
    with testnet.connect(config, raw_calls=True) as sub:
        testnet.write_commitment(config, sub, keys, text)
        after = testnet.hotkey_state(config, sub, hotkey)["commitment"]
    if (after or {}).get("data") != text:
        raise ValueError(
            "the chain does not show the new commitment; read it again before retrying"
        )
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
    for command in (start, check, commit):
        command.add_argument(
            "--network",
            choices=tuple(testnet.ENDPOINTS),
            default="test",
            help="finney is mainnet and needs an explicit --netuid",
        )
        command.add_argument("--netuid", type=int)
        command.add_argument(
            "--model",
            choices=models.d3_family(),
            help="base profile to beat in your tier; saved by setup or read from the candidate",
        )
    for command in (check, commit):
        command.add_argument("--wallet-name")
        command.add_argument("--wallet-hotkey", help="hotkey name; the coldkey is never read")
        command.add_argument("--wallet-path")
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
        for key in SAVED:
            if getattr(args, key) is None:
                setattr(args, key, config.get(key))
        args.wallet_path = args.wallet_path or WALLET_PATH
        if args.wallet_name is None or args.wallet_hotkey is None:
            raise ValueError("no wallet configured; run zils setup (or pass --wallet-name)")
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
