"""An outside miner's base-tier commands. A candidate is a merged d3 checkpoint on the miner's own
Hugging Face repo plus one chain commitment of repo, revision and checkpoint hash; every validator
reads the commitments and pulls that revision. No validator URL is needed.

`doctor` checks the setup step by step and prints the one command that fixes the first failing
step. `commit-candidate` validates a candidate, uploads it (or takes an uploaded revision), prints the
commitment and, with --yes, writes it with the hotkey."""

import argparse
import hashlib
import importlib
from pathlib import Path

import zils

from . import commitments, models, testnet

NETUID = 579
REQUIRED_MODULES = ("bittensor", "bittensor_wallet", "huggingface_hub", "kev", "torch")
PINNED_BITTENSOR = "11.1.0"


def chain_config(netuid):
    return {"chain": {"network": "test", "netuid": netuid}}


def hotkey_state(netuid, hotkey):
    """The hotkey's uid (None when unregistered) and current commitment, read from the chain."""
    config = chain_config(netuid)
    with testnet.connect(config) as sub:
        return testnet.hotkey_state(config, sub, hotkey)


def local_candidate(path):
    from . import d3

    candidate = Path(path).resolve()
    d3.validate_candidate(candidate)
    return candidate, zils.checkpoint_hash(candidate)


def check_remote(repo, revision, candidate):
    """The public repo at this commit holds every candidate file with the local size and LFS digest."""
    listed = commitments.revision_files(repo, revision)
    for name in models.D3_CANDIDATE_FILES:
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
        allow_patterns=list(models.D3_CANDIDATE_FILES),
        commit_message=f"Zils d3 base candidate {sha256[:12]}",
    )
    return commitments.check_revision(info.oid)


def wallet(args):
    from bittensor.wallet import Wallet

    return Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey, path=args.wallet_path)


def commit_command(args, revision=None):
    command = [
        "python -m zils.miner commit-candidate",
        f"--wallet-name {args.wallet_name} --wallet-hotkey {args.wallet_hotkey}"
        f" --wallet-path {args.wallet_path}",
        f"--candidate {args.candidate} --repo {args.repo}"
        + (f" --revision {revision}" if revision else ""),
    ]
    if args.netuid != NETUID:
        command.append(f"--netuid {args.netuid}")
    return " \\\n  ".join(command + ["--yes"])


def commit_candidate(args):
    candidate, sha256 = local_candidate(args.candidate)
    commitments.check_repo(args.repo)
    if args.revision is not None:
        commitments.check_revision(args.revision)
    keys = wallet(args)
    hotkey = keys.hotkey.ss58_address
    state = hotkey_state(args.netuid, hotkey)
    if state["uid"] is None:
        raise ValueError(f"hotkey {hotkey} is not registered on testnet subnet {args.netuid}")
    current = (state["commitment"] or {}).get("data")
    revision = args.revision
    print(
        f"miner {state['uid']}: candidate {sha256}\n"
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
    config = chain_config(args.netuid)
    with testnet.connect(config, raw_calls=True) as sub:
        testnet.write_commitment(config, sub, keys, text)
        after = testnet.hotkey_state(config, sub, hotkey)["commitment"]
    if (after or {}).get("data") != text:
        raise ValueError(
            "the chain does not show the new commitment; read it again before retrying"
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
    btcli = f"btcli --network test {flags}"
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
                f"missing {', '.join(missing) or 'bittensor ' + PINNED_BITTENSOR}; run\n"
                "  uv pip install --python .venv-kev/bin/python -r requirements/model.txt"
                " -r requirements/rehearsal.txt -r requirements/testnet.txt"
            )

    def wallet_hotkey():
        key = wallet(args)
        if not key.hotkey_file.exists_on_device():
            return (
                f"no hotkey file at {key.hotkey_file.path}; create the identity (coldkey stays on"
                f" your admin machine) with\n  {btcli} wallet create"
            )
        state["hotkey"] = key.hotkey.ss58_address

    def registration():
        state.update(hotkey_state(args.netuid, state["hotkey"]))
        if state["uid"] is None:
            return (
                f"hotkey {state['hotkey']} is not registered on testnet subnet {args.netuid};"
                f" fund the coldkey with test TAO, then\n"
                f"  {btcli} tx burned-register --netuid {args.netuid}"
            )

    def candidate():
        if args.candidate is None:
            return (
                "no --candidate; train and merge one with the reference recipe\n"
                "  python -m zils.d3 reference --out models/d3-reference\n"
                "  python -m zils.d3 train --data train.jsonl --reference models/d3-reference"
                " --out candidate-adapter --seed 553 --device cuda\n"
                "  python -m zils.d3 candidate --adapter candidate-adapter --out candidate"
                " > provenance.json\n"
                "then run doctor again with --candidate candidate"
            )
        try:
            state["path"], state["sha256"] = local_candidate(args.candidate)
        except (ValueError, OSError) as error:
            return f"{error}; merge again with python -m zils.d3 candidate"

    def hugging_face():
        if args.repo is None:
            return (
                "no --repo; pick a public model repo id under your Hugging Face account"
                " (log in with hf auth login; commit-candidate creates the repo) and run doctor"
                " again with --repo owner/name"
            )
        if args.revision is None:
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
        ("wallet hotkey", wallet_hotkey),
        ("registration", registration),
        ("candidate", candidate),
        ("hugging face", hugging_face),
        ("commitment", commitment),
    ]
    for name, check in steps:
        fix = check()
        if fix:
            print(f"fix {name}: {fix}", flush=True)
            return False
        print(f"ok  {name}", flush=True)
    print(
        f"ready: uid {state['uid']} committed {state['sha256'][:12]} at block"
        f" {state['commitment']['block']}; every validator pulls it at its next round. Results:"
        " the chain weights for your uid (btcli subnets metagraph --netuid"
        f" {args.netuid} --network test)",
        flush=True,
    )
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="check the setup and print the next command")
    commit = commands.add_parser(
        "commit-candidate", help="upload a base candidate and commit it on chain"
    )
    for command in (check, commit):
        command.add_argument("--wallet-name", required=True)
        command.add_argument(
            "--wallet-hotkey", required=True, help="hotkey name; the coldkey is never read"
        )
        command.add_argument("--wallet-path", default="~/.bittensor/wallets")
        command.add_argument(
            "--candidate",
            required=command is commit,
            help="directory written by `zils.d3 candidate`",
        )
        command.add_argument(
            "--repo", required=command is commit, help="your public Hugging Face model repo"
        )
        command.add_argument("--revision", help="40-hex commit sha of an upload already in --repo")
        command.add_argument("--netuid", type=int, default=NETUID)
    commit.add_argument(
        "--yes", action="store_true", help="upload and write the commitment; without it, a dry run"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            if not doctor(args):
                parser.exit(1)
            return
        commit_candidate(args)
    except KeyboardInterrupt:
        print("Zils miner stopped.", flush=True)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(1, f"zils miner: {error}\n")


if __name__ == "__main__":
    main()
