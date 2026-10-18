"""An outside miner's commands against a public Zils validator.

`announce-candidate` serves a merged base candidate over HTTP, announces it to a validator with a
base-tier intake every round, and prints the aggregate result the validator publishes."""

import argparse
import http.client
import ipaddress
import json
import math
import os
import re
import shutil
import signal
import socket
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

import zils

from . import models, protocol as wire, testnet
from .runtime import request

NETUID = 579


def hotkey_uid(netuid, hotkey, validator_hotkey):
    """The miner's uid on the subnet, from the same chain read the validator admits members with."""
    config = {
        "chain": {"network": "test", "netuid": netuid},
        "validator_hotkey": validator_hotkey,
        "admission": "chain",
    }
    with testnet.connect(config) as sub:
        members = testnet.preflight(config, sub)["members"]
    for uid, registered in members.items():
        if registered == hotkey:
            return int(uid)
    raise ValueError(f"hotkey {hotkey} is not registered on testnet subnet {netuid}")


def own_endpoint(validator, port, advertise=None):
    """The address the validator will see this miner connect from, or an explicit one; globally
    routable whenever the validator itself is, since the validator pins downloads to that address."""
    address, validator_port = wire.endpoint_port(validator)
    if advertise is None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect((str(address), validator_port))
            advertise = route.getsockname()[0]
    endpoint = f"http://{ipaddress.IPv4Address(advertise)}:{port}"
    public = address.is_global
    try:
        wire.endpoint_ok(endpoint, allowed=[endpoint], public=public)
    except ValueError as error:
        hint = (
            "the validator is public, so the artifact endpoint must be a globally routable IPv4 "
            "address the validator can reach; give --advertise-host when this host sits behind 1:1 NAT"
            if public
            else "a private validator needs a private artifact endpoint"
        )
        raise ValueError(f"{error}: {endpoint}; {hint}") from error
    return endpoint


def artifact_handler(candidate, files):
    class Artifacts(wire.Handler):
        def do_GET(self):
            match = re.fullmatch(r"/artifacts/[a-f0-9]{32}/([^/]+)", self.path)
            if not match or match[1] not in files:
                self.reply(404, {"error": "unknown artifact"})
                return
            with (candidate / match[1]).open("rb") as source:
                self.send_response(200)
                self.send_header("Content-Length", str(os.fstat(source.fileno()).st_size))
                self.end_headers()
                shutil.copyfileobj(source, self.wfile)

    return Artifacts


def refusal(error):
    try:
        body = json.loads(error.read())
        return str(body.get("error") or body.get("payload", {}).get("error", ""))[:200]
    except (ValueError, OSError, AttributeError):
        return ""


def summary(uid, result):
    """The aggregate base-tier rows a miner is entitled to see."""
    tier = result.get("base_tier")
    if tier is None:
        return "base tier: no decision this round"
    rejected = tier.get("rejected", {}).get(str(uid))
    if rejected:
        return f"base tier: candidate rejected: {rejected}"
    mine = next((c for c in tier.get("candidates", []) if c.get("uid") == uid), None)
    if mine is None:
        return "base tier: candidate not scored this round"
    winner = (tier.get("pending_winner") or {}).get("uid")
    confirmed = (tier.get("confirmed_winner") or {}).get("uid")
    incumbent = tier.get("incumbent", {})
    return (
        f"base tier: brier {mine.get('brier')} vs incumbent {incumbent.get('brier')}; "
        f"accuracy {mine.get('accuracy')} vs {incumbent.get('accuracy')}; "
        f"eligible {mine.get('eligible')}; pending winner uid {winner}; "
        f"confirmed winner uid {confirmed}; weight {result.get('weights', {}).get(str(uid), 0)}"
    )


def announce_candidate(args):
    from bittensor.wallet import Wallet

    from . import d3

    candidate = Path(args.candidate).resolve()
    d3.validate_candidate(candidate)
    sha256 = zils.checkpoint_hash(candidate)
    files = models.artifact_files(candidate)
    key = Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey, path=args.wallet_path).hotkey
    endpoint = own_endpoint(args.validator, args.port, args.advertise_host)
    uid = hotkey_uid(args.netuid, key.ss58_address, args.validator_hotkey)
    config = {"validator": args.validator, "validator_hotkey": args.validator_hotkey}
    chain = {"network": "test", "netuid": args.netuid}
    announced, finished, last_error = set(), set(), None
    host = str(wire.endpoint_port(endpoint)[0])
    with wire.local_server(artifact_handler(candidate, files), host, args.port):
        print(
            f"miner {uid}: serving candidate {sha256[:12]} at {endpoint}; validator {args.validator}",
            flush=True,
        )
        while not args.rounds or len(finished) < args.rounds:
            try:
                job = request(config, "/round")
                rid = job.get("round_id")
                if job.get("kind") != "round" or rid in finished:
                    time.sleep(args.poll)
                    continue
                if not re.fullmatch("[a-f0-9]{32}", rid or ""):
                    raise ValueError("invalid validator round")
                if "chain" in job and job["chain"] != chain:
                    raise ValueError(f"validator round is for another subnet: {job['chain']}")
                if job["status"] == "collecting" and rid not in announced:
                    claim = {
                        "round_id": rid,
                        "uid": uid,
                        "hotkey": key.ss58_address,
                        "sha256": sha256,
                        "endpoint": endpoint,
                        "kind": "candidate",
                    }
                    try:
                        reply = request(
                            config,
                            "/submit",
                            {"claim": claim, "signature": key.sign(wire.canonical(claim)).hex()},
                        )
                    except HTTPError as error:
                        detail = refusal(error)
                        if error.code == 409 and "base candidates" in detail:
                            raise ValueError(
                                "this validator has no base-tier intake; announce to one that does"
                            )
                        if error.code == 409:
                            time.sleep(args.poll)  # round changed under us; ask again
                            continue
                        raise ValueError(
                            f"validator refused the claim (HTTP {error.code}): {detail}"
                        ) from error
                    if reply.get("status") != "accepted" or reply.get("round_id") != rid:
                        raise ValueError("validator did not acknowledge this round's announcement")
                    announced.add(rid)
                    print(f"miner {uid}: announced {sha256[:12]} for round {rid}", flush=True)
                if rid in announced:
                    result = request(config, "/results/" + rid)
                    if result.get("kind") == "result" and result.get("round_id") == rid:
                        finished.add(rid)
                        print(
                            f"miner {uid}: round {rid} complete; {summary(uid, result)}", flush=True
                        )
                last_error = None
            except HTTPError as error:
                if error.code not in (404, 409, 503):
                    raise ValueError(
                        f"validator rejected request (HTTP {error.code}): {refusal(error)}"
                    ) from error
            except (URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
                message = str(error)
                if message != last_error:
                    print(f"miner {uid}: validator unavailable; retrying: {message}", flush=True)
                    last_error = message
            time.sleep(args.poll)
    return {"uid": uid, "sha256": sha256, "endpoint": endpoint, "rounds": sorted(finished)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    announce = commands.add_parser(
        "announce-candidate", help="serve and announce a base candidate every round"
    )
    announce.add_argument("--validator", required=True, help="public URL, http://<IPv4>:<port>")
    announce.add_argument(
        "--validator-hotkey", required=True, help="SS58 hotkey the validator signs its replies with"
    )
    announce.add_argument("--wallet-name", required=True)
    announce.add_argument(
        "--wallet-hotkey", required=True, help="hotkey name; the coldkey is never read"
    )
    announce.add_argument("--wallet-path", default="~/.bittensor/wallets")
    announce.add_argument(
        "--candidate", required=True, help="directory written by `zils.d3 candidate`"
    )
    announce.add_argument("--port", type=int, default=8901, help="this host's artifact port")
    announce.add_argument(
        "--advertise-host",
        help="public IPv4 of this host when it differs from the interface address",
    )
    announce.add_argument("--netuid", type=int, default=NETUID)
    announce.add_argument(
        "--rounds", type=int, default=0, help="0 keeps announcing; N stops after N round results"
    )
    announce.add_argument("--poll", type=float, default=5)
    args = parser.parse_args(argv)

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        if args.rounds < 0 or not math.isfinite(args.poll) or args.poll <= 0:
            raise ValueError("rounds must be nonnegative and poll must be positive")
        wire.endpoint_port(args.validator)
        announce_candidate(args)
        print("Zils miner stopped.", flush=True)
    except KeyboardInterrupt:
        print("Zils miner stopped.", flush=True)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(1, f"zils miner: {error}\n")


if __name__ == "__main__":
    main()
