"""Pack and commit box adapters. Upload and chain writes require --yes."""

import argparse
import json
import tempfile
from pathlib import Path

from . import commitments, feature_commitments as features, miner, models, testnet


def upload(candidate, repo, sha256):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    info = api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=str(candidate),
        allow_patterns=[*models.artifact_files(candidate), features.MANIFEST],
        commit_message=f"Zils box adapter {sha256[:12]}",
    )
    return commitments.check_revision(info.oid)


def check_remote(repo, revision, candidate, sha256, manifest):
    miner.check_remote(repo, revision, candidate)
    # Non-LFS metadata has no remote SHA-256, so verify the complete pinned bundle.
    with tempfile.TemporaryDirectory(prefix="zils-box-check-") as tmp:
        features.fetch(
            {"sha256": sha256, "revision": revision, "repo": repo},
            Path(tmp) / "bundle",
            track_id=manifest["track_id"],
            model=manifest["model"],
        )


def _receipt(args, manifest, sha256, hotkey, uid, revision, stored):
    receipt = {
        "version": "zils-feature-commit-receipt/v1",
        "network": args.network,
        "netuid": args.netuid,
        "uid": uid,
        "hotkey": hotkey,
        "track_id": manifest["track_id"],
        "model": manifest["model"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "bundle_sha256": sha256,
        "repo": args.repo,
        "revision": revision,
        "block": stored["block"],
        "commitment": stored["data"],
    }
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def commit_candidate(args):
    candidate = Path(args.candidate).expanduser()
    manifest = features.validate_bundle(candidate)
    sha256 = features.bundle_hash(candidate)
    commitments.check_repo(args.repo)
    revision = args.revision
    if revision is not None:
        commitments.check_revision(revision)
    keys = miner.wallet(args)
    hotkey = keys.hotkey.ss58_address
    state = miner.hotkey_state(args.netuid, hotkey, args.network)
    if state["uid"] is None:
        raise ValueError(
            f"hotkey {hotkey} is not registered on {testnet.NAMES[args.network]} subnet {args.netuid}"
        )
    current = (state["commitment"] or {}).get("data")
    print(
        f"miner {state['uid']}: {manifest['track_id']} bundle {sha256}\n"
        f"  parent {manifest['model']['id']}\n"
        f"  repo {args.repo} at {revision or '<the commit the upload creates>'}\n"
        f"  current on chain: {current or 'none'}\n"
        "  one active commitment per hotkey; this replaces any base or feature entry",
        flush=True,
    )
    if revision and current == features.format_commitment(sha256, revision, args.repo):
        print(f"already committed at block {state['commitment']['block']}; nothing to do")
        return _receipt(args, manifest, sha256, hotkey, state["uid"], revision, state["commitment"])
    if not args.yes:
        print("dry run: nothing uploaded or written; run again with --yes", flush=True)
        return None
    if revision is None:
        revision = upload(candidate, args.repo, sha256)
        print(f"uploaded {args.repo} at {revision}", flush=True)
    check_remote(args.repo, revision, candidate, sha256, manifest)
    text = features.format_commitment(sha256, revision, args.repo)
    if current == text:
        return _receipt(args, manifest, sha256, hotkey, state["uid"], revision, state["commitment"])
    config = miner.chain_config(args.netuid, args.network)
    with testnet.connect(config, raw_calls=True) as sub:
        testnet.write_commitment(config, sub, keys, text)
        after = testnet.hotkey_state(config, sub, hotkey)["commitment"]
    if (after or {}).get("data") != text:
        raise ValueError(
            "the chain does not show the new commitment; read it again before retrying"
        )
    print(
        f"committed at block {after['block']}; validators pull it at their next round", flush=True
    )
    return _receipt(args, manifest, sha256, hotkey, state["uid"], revision, after)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m zils.feature_miner", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="package a native-temperature adapter for a box track")
    pack.add_argument("--checkpoint", required=True)
    pack.add_argument("--out", required=True)
    pack.add_argument("--track", required=True, choices=features.TRACKS)
    commit = commands.add_parser(
        "commit", help="upload a feature bundle and commit its pinned revision"
    )
    commit.add_argument("--candidate", required=True, help="bundle written by feature_miner pack")
    commit.add_argument("--repo", required=True, help="public Hugging Face model repo, owner/name")
    commit.add_argument("--revision", help="full 40-hex commit SHA of an existing upload")
    commit.add_argument("--wallet-name")
    commit.add_argument("--wallet-hotkey")
    commit.add_argument("--wallet-path")
    commit.add_argument(
        "--network",
        choices=tuple(testnet.ENDPOINTS),
        default="test",
        help="finney is mainnet and needs an explicit --netuid",
    )
    commit.add_argument("--netuid", type=int)
    commit.add_argument(
        "--yes", action="store_true", help="allow upload and chain write; default is dry run"
    )
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "netuid"):
            args.netuid = miner.resolve_netuid(args.network, args.netuid)
        if args.command == "pack":
            manifest = features.pack(args.checkpoint, args.out, args.track)
            print(
                json.dumps(
                    {"manifest": manifest, "bundle_sha256": features.bundle_hash(args.out)},
                    sort_keys=True,
                )
            )
            return manifest
        saved = miner.load_config()
        for key in ("wallet_name", "wallet_hotkey", "wallet_path"):
            if getattr(args, key) is None:
                setattr(args, key, saved.get(key))
        args.wallet_path = args.wallet_path or miner.WALLET_PATH
        if not args.wallet_name or not args.wallet_hotkey:
            raise ValueError(
                "pass --wallet-name and --wallet-hotkey, or configure them with zils setup"
            )
        return commit_candidate(args)
    except KeyboardInterrupt:
        parser.exit(1, "Zils feature miner stopped.\n")
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(1, f"zils feature miner: {error}\n")


if __name__ == "__main__":
    main()
