"""Closed Zils rehearsal on Bittensor testnet. Local mode never imports the SDK."""

import argparse
import hashlib
import json
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import zils

ENDPOINT = "wss://test.finney.opentensor.ai:443"


def check_config(config):
    chain = config.get("chain")
    if (
        not isinstance(chain, dict)
        or set(chain) != {"network", "netuid"}
        or chain["network"] != "test"
        or type(chain["netuid"]) is not int
        or not 1 <= chain["netuid"] < 4096
    ):
        raise ValueError("chain mode requires network test and an explicit subnet ID in 1..4095")
    return chain["netuid"]


@contextmanager
def connect(config):
    netuid = check_config(config)
    if version("bittensor") != "11.1.0":
        raise ValueError("install requirements/testnet.txt (bittensor 11.1.0)")
    import bittensor as bt

    with bt.Subtensor(
        network="test",
        fallback_endpoints=[],
        archive_endpoints=[],
        policy=bt.Policy(allowed_netuids=[netuid]),
    ) as sub:
        yield sub


def preflight(config, sub):
    """Pin reads to one block; a replaced UID must never inherit a miner's score."""
    netuid = check_config(config)
    if sub.network != "test" or sub.endpoint != ENDPOINT:
        raise ValueError("unexpected chain connection; only the pinned testnet endpoint is allowed")
    block = sub.block

    def query(item, *params):
        return sub.query(("SubtensorModule", item), list(params), block=block)

    if not query("NetworksAdded", netuid):
        raise ValueError(f"testnet subnet {netuid} does not exist")
    validator = config["validator_hotkey"]
    validator_uid = query("Uids", netuid, validator)
    if validator_uid is None or query("Keys", netuid, validator_uid) != validator:
        raise ValueError("validator registration is missing or changed")
    members = config.get("members", {str(config.get("uid")): {"hotkey": config.get("hotkey")}})
    seen = {validator}
    for uid, item in members.items():
        hotkey = item["hotkey"]
        if (
            not isinstance(uid, str)
            or not uid.isdecimal()
            or str(int(uid)) != uid
            or not 0 <= int(uid) <= 65535
        ):
            raise ValueError("invalid member UID")
        if hotkey in seen or not hotkey:
            raise ValueError("duplicate miner or validator identity")
        seen.add(hotkey)
        if query("Uids", netuid, hotkey) != int(uid) or query("Keys", netuid, int(uid)) != hotkey:
            raise ValueError(
                f"miner {uid} registration is missing or changed; refresh the fleet config"
            )
    permits = query("ValidatorPermit", netuid)
    owner = query("SubnetOwnerHotkey", netuid)
    if owner != validator and (validator_uid >= len(permits) or not permits[validator_uid]):
        raise ValueError("validator has no permit and is not the subnet owner hotkey")
    last = query("LastUpdate", netuid)
    last_update = last[validator_uid] if validator_uid < len(last) else 0
    rate = query("WeightsSetRateLimit", netuid)
    return {
        **config["chain"],
        "block": block,
        "validator_uid": validator_uid,
        "validator_hotkey": validator,
        "members": {uid: m["hotkey"] for uid, m in members.items()},
        "commit_reveal": bool(query("CommitRevealWeightsEnabled", netuid)),
        "min_allowed_weights": query("MinAllowedWeights", netuid),
        "max_weight_limit": query("MaxWeightsLimit", netuid),
        "required_version": query("WeightsVersionKey", netuid),
        "last_update": last_update,
        "retry_after_block": last_update + rate if last_update else block,
    }


def publish_round(config, work, report, sub, wallet, *, publish=False):
    """Persist intent before signing. An interrupted submission is never retried automatically."""
    import bittensor as bt

    from . import models, protocol as wire

    work = Path(work)
    report_hash = hashlib.sha256(
        json.dumps(report, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    receipt_path, attempt_path = work / "chain-receipt.json", work / "chain-attempt.json"
    netuid = check_config(config)
    identities = {uid: item["hotkey"] for uid, item in config["members"].items()}
    if report.get("chain") != config["chain"] or report.get("identities") != identities:
        raise ValueError("report was not evaluated for this registered testnet fleet")
    model = config.get("model", models.spec(models.KEV))
    models.validate_spec(model)
    if report.get("model", models.spec(models.KEV)) != model:
        raise ValueError("Report differs from the testnet fleet model profile")
    for path in (receipt_path, attempt_path):
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["report_sha256"] != report_hash:
                raise ValueError("round report changed after the chain attempt")
            return (
                saved
                if path == receipt_path
                else {
                    **saved,
                    "status": "unknown",
                    "chain_write": None,
                    "detail": "interrupted submission; inspect the chain before any retry",
                }
            )
    state = preflight(config, sub)
    if wallet.hotkey.ss58_address != config["validator_hotkey"]:
        raise ValueError("wallet does not match the configured validator")
    rows = report["miners"]
    if any(str(row["uid"]) not in config["members"] for row in rows):
        raise ValueError("report contains a miner outside this registered fleet")
    weights = zils.weight_vector(
        [{**row, "skill": row["skill"] if row["status"] == "evaluated" else 0.0} for row in rows]
    )
    record = {
        **state,
        "report_sha256": report_hash,
        "requested_weights": {str(k): v for k, v in weights.items()},
        "chain_write": False,
        "weights_verified": False,
    }
    if not weights:
        return {
            **record,
            "status": "no_weights",
            "detail": "nothing sent; previous chain weights remain",
        }
    if state["required_version"] > 1:
        raise ValueError("subnet requires a newer Zils weights version")
    # Let the pinned SDK clip/quantize and enforce minimum count, stake and chain dispatch errors.
    intent = bt.SetWeights(netuid=netuid, weights=weights, version_key=1)
    if state["block"] < state["retry_after_block"]:
        return {
            **record,
            "status": "rate_limited",
            "detail": "nothing sent; previous chain weights remain",
        }
    if not publish:
        return {**record, "status": "dry_run", "plan": sub.plan(intent, wallet).to_dict()}
    wire.write_json(attempt_path, {**record, "status": "submitting", "chain_write": None})
    try:
        result = sub.execute(
            intent, wallet, wait_for_inclusion=True, wait_for_finalization=True, retries=0
        )
        # The SDK can return a failed result after losing the inclusion subscription.
        uncertain = not result.success and (
            result.error is None or result.error.code == bt.ErrorCode.UNKNOWN
        )
        record.update(
            receipt=result.to_dict(),
            chain_write=None if uncertain else bool(result.success),
            status="unknown"
            if uncertain
            else (
                ("committed" if state["commit_reveal"] else "included")
                if result.success
                else "failed"
            ),
        )
    except bt.ChainError as error:
        uncertain = error.code == bt.ErrorCode.UNKNOWN
        record.update(
            status="unknown" if uncertain else "failed",
            chain_write=None if uncertain else False,
            error=error.to_dict(),
        )
    except Exception as error:
        # A lost connection may follow a successful submission. Keep the durable attempt for inspection.
        record.update(status="unknown", chain_write=None, detail=str(error)[:400])
    wire.write_json(receipt_path, record)
    return record


def verify_receipt(config, record, sub):
    """Read current weights; a successful timelock commit alone is not a revealed weight vector."""
    state = preflight(config, sub)
    if {k: record[k] for k in ("network", "netuid")} != config["chain"]:
        raise ValueError("receipt is not for the configured testnet subnet")
    if (
        record["validator_hotkey"] != state["validator_hotkey"]
        or record["members"] != state["members"]
    ):
        raise ValueError("receipt identities do not match the current registered fleet")
    pairs = sub.query(
        ("SubtensorModule", "Weights"),
        [config["chain"]["netuid"], state["validator_uid"]],
        block=state["block"],
    )
    actual = {str(uid): value for uid, value in pairs if value > 0}
    from bittensor.intents.weights import clip_to_max_weight, normalize

    wanted = record["requested_weights"]
    uids = sorted(int(k) for k in wanted)
    floats = [wanted[str(uid)] for uid in uids]
    if record["max_weight_limit"] < 65535:
        floats = clip_to_max_weight(floats, record["max_weight_limit"] / 65535)
    ids, vals = normalize(uids, floats)
    expected = {str(uid): val for uid, val in zip(ids, vals)}
    verified = (
        record.get("chain_write") is True
        and bool(expected)
        and set(actual) == set(expected)
        and state["last_update"] > record["block"]
        and all(
            abs(actual[k] / sum(actual.values()) - expected[k] / sum(expected.values())) < 2 / 65535
            for k in expected
        )
    )
    return {
        "weights_verified": verified,
        "block": state["block"],
        "last_update": state["last_update"],
        "on_chain_weights": actual,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "publish", "verify"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--round", help="validator state/rounds/<round-id> directory")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    try:
        with connect(config) as sub:
            if args.action == "preflight":
                result = preflight(config, sub)
            else:
                if not args.round:
                    raise ValueError("--round is required")
                from .runtime import locked

                work = Path(args.round)
                with locked(work / "chain.lock"):
                    if args.action == "publish":
                        from bittensor.wallet import Wallet

                        result = publish_round(
                            config,
                            work,
                            json.loads((work / "report.json").read_text()),
                            sub,
                            Wallet(**config["wallet"]),
                            publish=True,
                        )
                    else:
                        result = verify_receipt(
                            config, json.loads((work / "chain-receipt.json").read_text()), sub
                        )
            print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"zils testnet: {error}\n")


if __name__ == "__main__":
    main()
