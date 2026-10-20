"""Verify private two-draw evidence and write a new configuration for operator review."""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zils import (  # noqa: E402
    base_round,
    box,
    feature_commitments as bundles,
    feature_round,
    feature_tracks,
    testnet,
)


def _read(path):
    return json.loads(feature_round._regular(Path(path)).read_text())


def _predictions(prepared, bundle, checkpoint, uid, hotkey, draw, cases):
    role = "incumbent-" if uid == feature_round.INCUMBENT_UID else ""
    cache = prepared["root"] / "store" / bundle / (role + draw["manifest_sha256"] + "-scores.json")
    saved = _read(cache)
    expected = {
        "bundle_sha256": bundle,
        "checkpoint_sha256": checkpoint,
        "draw": draw,
        "model": prepared["entry"]["model"],
        "decoder": box.DECODER,
        "hotkey": hotkey,
    }
    if not isinstance(saved, dict) or any(saved.get(k) != v for k, v in expected.items()):
        raise ValueError("private feature cache binding changed")
    evaluated_at = feature_round._time(saved.get("evaluated_at"))
    rows = saved.get("miners")
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or rows[0].get("uid") != uid
        or rows[0].get("sha256") != checkpoint
        or rows[0].get("status") != "evaluated"
        or rows[0].get("runtime", {}).get("model") != expected["model"]
    ):
        raise ValueError("private feature cache requires a successful bound model evaluation")
    verified = base_round.rows_for(saved, box.expand_cases(cases))
    return verified[uid]["predictions"], evaluated_at


def propose(config_path, track, decision_path, candidate, snapshot_path, out):
    """Write only a new proposal; never install it or change registration or model releases."""
    config_path, candidate = Path(config_path).expanduser(), Path(candidate).expanduser()
    config = _read(config_path)
    testnet.check_config(config)
    prepared = feature_round.rounds(config, config_path.resolve().parent)
    if track not in prepared:
        raise ValueError("crown requires an enabled configured feature track")
    prepared = prepared[track]
    entry = prepared["entry"]
    decision = _read(decision_path)
    feature_tracks.validate_decision(track, entry, decision)
    winner, confirmation = decision.get("confirmed_winner"), decision.get("confirmation")
    if (
        not isinstance(winner, dict)
        or set(winner) != {"uid", "hotkey", "sha256"}
        or type(winner["uid"]) is not int
        or not 0 <= winner["uid"] < feature_round.INCUMBENT_UID
        or not isinstance(winner["hotkey"], str)
        or not winner["hotkey"]
        or not isinstance(confirmation, dict)
        or feature_round.confirm(decision, confirmation) != winner
    ):
        raise ValueError("decision has no matching confirmed winner from two eligible draws")
    manifest = bundles.validate_bundle(candidate)
    candidate_hash = bundles.bundle_hash(candidate)
    incumbent = bundles.validate_bundle(prepared["reference"], allow_base=True)
    if (
        manifest["track_id"] != track
        or manifest["model"] != entry["model"]
        or candidate_hash != winner["sha256"]
        or candidate_hash == entry["incumbent_sha256"]
        or manifest["checkpoint_sha256"] == incumbent["checkpoint_sha256"]
    ):
        raise ValueError("candidate differs from confirmed bundle, track or exact parent")
    snapshot = _read(snapshot_path)
    if any(snapshot.get(k) != config["chain"][k] for k in ("network", "netuid")):
        raise ValueError("snapshot belongs to a different network or subnet")
    if snapshot.get("members", {}).get(str(winner["uid"])) != winner["hotkey"]:
        raise ValueError("confirmed winner is not registered under its evaluated UID and hotkey")
    claim = bundles.candidates(snapshot).get(winner["uid"], {})
    if "error" in claim or claim.get("sha256") != candidate_hash:
        raise ValueError("registered winner no longer holds the confirmed feature bundle claim")

    receipts = []
    for draw, path in prepared["draws"]:
        _, cases = feature_round.audit(path)
        baseline, baseline_time = _predictions(
            prepared,
            entry["incumbent_sha256"],
            incumbent["checkpoint_sha256"],
            feature_round.INCUMBENT_UID,
            None,
            draw,
            cases,
        )
        predictions, evaluated_at = _predictions(
            prepared,
            candidate_hash,
            manifest["checkpoint_sha256"],
            winner["uid"],
            winner["hotkey"],
            draw,
            cases,
        )
        comparison = box.compare(cases, baseline, predictions, entry["policy"])
        receipts.append(
            {
                **{key: decision[key] for key in feature_round.BINDINGS},
                "evaluated_draw": draw,
                "evaluated_at": max(baseline_time, evaluated_at),
                "candidates": [{**winner, "evaluated_at": evaluated_at, **comparison}],
                "pending_winner": winner if comparison["eligible"] else None,
            }
        )
    if feature_round.confirm(*receipts) != winner:
        raise ValueError("candidate is not confirmed by both private qualifying exams")

    proposal = copy.deepcopy(config)
    selected = proposal["feature_tracks"]["tracks"][track]
    selected["incumbent_sha256"] = candidate_hash
    selected["kings"] = [winner["hotkey"], *selected["kings"]]
    proposal["feature_track_rounds"][track]["reference"] = str(candidate.resolve())
    encoded = json.dumps(proposal, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(out).expanduser().open("x") as stream:
        stream.write(encoded)
    return proposal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="current validator configuration")
    parser.add_argument("--track", required=True, choices=bundles.TRACKS)
    parser.add_argument("--decision", required=True, help="confirmed private track decision JSON")
    parser.add_argument(
        "--candidate", required=True, help="verified feature adapter bundle directory"
    )
    parser.add_argument("--snapshot", required=True, help="current frozen chain snapshot JSON")
    parser.add_argument("--out", required=True, help="new configuration proposal file")
    args = parser.parse_args(argv)
    try:
        result = propose(
            args.config, args.track, args.decision, args.candidate, args.snapshot, args.out
        )
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"crown refused: {error}\n")
    entry = result["feature_tracks"]["tracks"][args.track]
    print(
        json.dumps(
            {
                "proposal": str(Path(args.out).expanduser().resolve()),
                "track_id": args.track,
                "incumbent_sha256": entry["incumbent_sha256"],
                "king_hotkey": entry["kings"][0],
                "installed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
