"""Base-tier round decisions: score full-weight candidates against the incumbent on a sealed holdout.

Inputs are ordinary ``zils evaluate`` reports for the sealed calibration and test splits, one row per
submission plus the incumbent. This module fits one temperature per model on calibration, compares each
candidate with the incumbent on test by equal-family Brier loss, and requires a minimum improvement whose
paired bootstrap lower bound is positive. Promotion additionally needs the same result on a second,
disjoint draw. No training, downloading or chain interaction happens here.
"""

import argparse
import hashlib
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import zils

from . import benchmark

VERSION = "zils-base-holdout/v1"
SPLITS = ("calibration", "test")
DEFAULT_POLICY = {
    "min_brier_improvement": 0.01,
    "bootstrap_resamples": 2000,
    "max_accuracy_drop": 0.0,
}


def audit(root):
    """Verify the sealed round files against the public manifest; return cases per split."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("version") != VERSION or set(manifest["files"]) != {
        f"{s}.jsonl" for s in SPLITS
    }:
        raise ValueError("unsupported holdout manifest")
    cases = {}
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        if benchmark.file_hash(path) != manifest["files"][path.name]:
            raise ValueError(f"sealed {split} split differs from its manifest hash")
        cases[split] = benchmark.read_jsonl(path)
        zils.validate_cases(cases[split])
    if {c["family"] for c in cases["calibration"]} != {c["family"] for c in cases["test"]}:
        raise ValueError("calibration and test must share task families")
    if {c["id"] for c in cases["calibration"]} & {c["id"] for c in cases["test"]}:
        raise ValueError("case ids overlap splits")
    return manifest, cases


def dataset_hash(cases):
    return hashlib.sha256(json.dumps(cases, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rows_for(report, cases):
    """Evaluated prediction lists by uid, verified against the split they claim to describe."""
    if report["dataset_sha256"] != dataset_hash(cases):
        raise ValueError("report dataset does not match the sealed split")
    rows = {}
    for row in report["miners"]:
        if row["status"] == "evaluated":
            zils.score(cases, row["predictions"])
            rows[row["uid"]] = row
    return rows


def _fit_rows(cases, predictions):
    by_id = {p["id"]: p for p in predictions}
    return [
        {
            "p": [by_id[c["id"]]["probabilities"][k] for k in zils.options(c["question"])],
            "label": zils.options(c["question"]).index(c["label"]),
            "task": c["family"],
            "keys": zils.options(c["question"]),
            "source": c["family"],
            "inference_temperature": 1.0,
            "variant": "clean",
            "fez_variant": "holdout",
        }
        for c in cases
    ]


def fit_temperature(cases, predictions):
    from kev.metrics import fit_temperature as kev_fit

    temperature = kev_fit(_fit_rows(cases, predictions), aggregation="macro", points=81)
    if not 0.25 <= temperature <= 4:
        raise ValueError("temperature fitter returned an invalid value")
    return temperature


def apply_temperature(cases, predictions, temperature):
    from kev.metrics import probabilities_at_temperature

    by_id = {p["id"]: p for p in predictions}
    return [
        {
            **by_id[c["id"]],
            "probabilities": dict(
                zip(row["keys"], probabilities_at_temperature(row, temperature).tolist())
            ),
        }
        for c, row in zip(cases, _fit_rows(cases, predictions), strict=True)
    ]


def case_losses(cases, predictions):
    by_id = {p["id"]: p for p in predictions}
    return {
        c["id"]: sum(
            (by_id[c["id"]]["probabilities"][k] - (k == c["label"])) ** 2
            for k in zils.options(c["question"])
        )
        for c in cases
    }


def paired_improvement(cases, incumbent, candidate, *, resamples, seed=0):
    """Equal-family mean of (incumbent − candidate) Brier, with a family-stratified bootstrap 95% lower bound."""
    inc, cand = case_losses(cases, incumbent), case_losses(cases, candidate)
    families = defaultdict(list)
    for c in cases:
        families[c["family"]].append(inc[c["id"]] - cand[c["id"]])
    point = statistics.mean(statistics.mean(v) for v in families.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        draws.append(
            statistics.mean(statistics.mean(rng.choices(v, k=len(v))) for v in families.values())
        )
    draws.sort()
    return {
        "improvement": point,
        "lower_95": draws[int(0.025 * resamples)],
        "upper_95": draws[min(resamples - 1, int(0.975 * resamples))],
    }


def decide(cases, incumbent_uid, calibration_rows, test_rows, policy=None):
    """One draw: calibrate every model on calibration, compare each candidate with the incumbent on test."""
    policy = {**DEFAULT_POLICY, **(policy or {})}
    if incumbent_uid not in calibration_rows or incumbent_uid not in test_rows:
        raise ValueError("incumbent must be evaluated on both splits")

    def calibrated(uid):
        temperature = fit_temperature(cases["calibration"], calibration_rows[uid]["predictions"])
        return temperature, apply_temperature(
            cases["test"], test_rows[uid]["predictions"], temperature
        )

    incumbent_temperature, incumbent = calibrated(incumbent_uid)
    baseline = zils.score(cases["test"], incumbent)
    candidates = []
    for uid in sorted(test_rows):
        if uid == incumbent_uid or uid not in calibration_rows:
            continue
        temperature, predictions = calibrated(uid)
        scores = zils.score(cases["test"], predictions)
        paired = paired_improvement(
            cases["test"], incumbent, predictions, resamples=policy["bootstrap_resamples"], seed=uid
        )
        eligible = (
            scores["skill"] > 0
            and scores["accuracy"] >= baseline["accuracy"] - policy["max_accuracy_drop"]
            and paired["improvement"] >= policy["min_brier_improvement"]
            and paired["lower_95"] > 0
        )
        candidates.append(
            {
                "uid": uid,
                "sha256": test_rows[uid]["sha256"],
                "temperature": temperature,
                "eligible": eligible,
                **paired,
                **{k: scores[k] for k in ("brier", "accuracy", "skill", "confident_errors")},
            }
        )
    eligible = [c for c in candidates if c["eligible"]]
    winner = (
        min(eligible, key=lambda c: (c["brier"], -c["accuracy"], c["uid"])) if eligible else None
    )
    return {
        "incumbent": {
            "uid": incumbent_uid,
            "sha256": test_rows[incumbent_uid]["sha256"],
            "temperature": incumbent_temperature,
            **{k: baseline[k] for k in ("brier", "accuracy", "skill")},
        },
        "policy": policy,
        "candidates": candidates,
        # Winner takes all (owner decision, 18 October 2026): eligible runners-up earn nothing.
        "weights": {winner["uid"]: 1.0} if winner else {},
        "pending_winner": None
        if winner is None
        else {"uid": winner["uid"], "sha256": winner["sha256"]},
        "dataset_sha256": dataset_hash(cases["test"]),
    }


def draw(root):
    """The identity of one sealed draw: its manifest hash and the test split's dataset hash."""
    _, cases = audit(root)
    return {
        "manifest_sha256": benchmark.file_hash(Path(root) / "manifest.json"),
        "dataset_sha256": dataset_hash(cases["test"]),
    }


def validate_track(track):
    """A fleet config's base-tier track: share, incumbent, allowed draws and the decision policy."""
    if not isinstance(track, dict) or set(track) - {
        "share",
        "incumbent_sha256",
        "incumbent_uid",
        "draws",
        "policy",
    }:
        raise ValueError(
            "base_tier accepts share, incumbent_sha256, incumbent_uid, draws and policy"
        )
    share = track.get("share")
    if type(share) not in (int, float) or not 0 <= share <= 1:
        raise ValueError("base_tier.share must be a number in [0, 1]")
    if not re.fullmatch("[a-f0-9]{64}", str(track.get("incumbent_sha256"))):
        raise ValueError("base_tier.incumbent_sha256 must be the incumbent checkpoint hash")
    king = track.get("incumbent_uid")
    if king is not None and (type(king) is not int or not 0 <= king <= 65535):
        raise ValueError("base_tier.incumbent_uid must be the roster uid that owns the incumbent")
    draws = track.get("draws")
    if not isinstance(draws, list) or not draws:
        raise ValueError("base_tier.draws must list at least one sealed draw")
    for item in draws:
        if not isinstance(item, dict) or set(item) != {"manifest_sha256", "dataset_sha256"}:
            raise ValueError("each draw needs manifest_sha256 and dataset_sha256")
        if any(not re.fullmatch("[a-f0-9]{64}", str(v)) for v in item.values()):
            raise ValueError("draw hashes must be sha256 hex")
    policy = {**DEFAULT_POLICY, **track.get("policy", {})}
    if set(policy) != set(DEFAULT_POLICY) or any(
        type(v) not in (int, float) or v < 0 for v in policy.values()
    ):
        raise ValueError("base_tier.policy may only override the base_round policy keys")
    return {**track, "policy": policy}


def king_weights(track, members):
    """King of the hill: with no eligible challenger the incumbent's miner keeps the whole base share.
    The vendor base has no owner (`incumbent_uid` unset) and pays nobody."""
    king = validate_track(track).get("incumbent_uid")
    if king is None:
        return {}
    if str(king) not in members:
        raise ValueError("base_tier.incumbent_uid is outside this registered fleet")
    return {king: 1.0}


def track_weights(track, decision, members):
    """Base-tier weights a decision earns under a track: the pending winner takes the whole share,
    otherwise the incumbent's miner keeps it (king of the hill)."""
    track = validate_track(track)
    for key in (
        "manifest_sha256",
        "dataset_sha256",
        "policy",
        "incumbent",
        "candidates",
        "pending_winner",
    ):
        if key not in decision:
            raise ValueError("base-tier decision is missing " + key)
    if {k: decision[k] for k in ("manifest_sha256", "dataset_sha256")} not in track["draws"]:
        raise ValueError("base-tier decision is not for a draw of the configured track")
    if decision["policy"] != track["policy"]:
        raise ValueError("base-tier decision used a different policy than the track")
    if decision["incumbent"]["sha256"] != track["incumbent_sha256"]:
        raise ValueError("base-tier decision scored against a different incumbent")
    if str(decision["incumbent"]["uid"]) in members:
        raise ValueError("the incumbent row must use a uid outside the roster")
    eligible = [c for c in decision["candidates"] if c["eligible"]]
    hashes = [c["sha256"] for c in eligible]
    if len(set(hashes)) != len(hashes) or track["incumbent_sha256"] in hashes:
        raise ValueError("eligible base candidates must be distinct and differ from the incumbent")
    if any(str(c["uid"]) not in members for c in eligible):
        raise ValueError("base-tier decision pays a miner outside this registered fleet")
    pending = decision["pending_winner"]
    if pending is None:
        weights = {}
    elif any(c["uid"] == pending["uid"] and c["sha256"] == pending["sha256"] for c in eligible):
        weights = {pending["uid"]: 1.0}
    else:
        raise ValueError("base-tier pending winner is not an eligible candidate")
    if {str(k): v for k, v in weights.items()} != {
        str(k): v for k, v in decision.get("weights", weights).items()
    }:
        raise ValueError("base-tier decision weights differ from its pending winner")
    return weights or king_weights(track, members)


def confirm(first, second):
    """Promotion needs the same checkpoint eligible on two disjoint draws; return the confirmed winner or None."""
    if first["dataset_sha256"] == second["dataset_sha256"]:
        raise ValueError("confirmation requires a disjoint draw")
    pending = first["pending_winner"]
    if pending is None:
        return None
    again = [c for c in second["candidates"] if c["sha256"] == pending["sha256"] and c["eligible"]]
    return pending if again else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True, help="sealed round directory with manifest.json")
    parser.add_argument(
        "--describe-draw",
        action="store_true",
        help="audit the round and print its draw entry for a fleet config's base_tier.draws",
    )
    parser.add_argument("--calibration-report")
    parser.add_argument("--test-report")
    parser.add_argument("--incumbent-uid", type=int)
    parser.add_argument(
        "--min-brier-improvement", type=float, default=DEFAULT_POLICY["min_brier_improvement"]
    )
    parser.add_argument(
        "--confirm", help="an earlier decision.json; print the confirmed winner instead"
    )
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.describe_draw:
        print(json.dumps(draw(args.round)))
        return
    if (
        not (args.calibration_report and args.test_report and args.out)
        or args.incumbent_uid is None
    ):
        parser.error("--calibration-report, --test-report, --incumbent-uid and --out are required")
    manifest, cases = audit(args.round)
    calibration = rows_for(
        json.loads(Path(args.calibration_report).read_text()), cases["calibration"]
    )
    test = rows_for(json.loads(Path(args.test_report).read_text()), cases["test"])
    decision = decide(
        cases,
        args.incumbent_uid,
        calibration,
        test,
        {"min_brier_improvement": args.min_brier_improvement},
    )
    decision["manifest_sha256"] = benchmark.file_hash(Path(args.round) / "manifest.json")
    if args.confirm:
        decision["confirmed_winner"] = confirm(json.loads(Path(args.confirm).read_text()), decision)
    with Path(args.out).open("x") as output:
        json.dump(decision, output, indent=2, allow_nan=False)
        output.write("\n")
    print(
        json.dumps(
            {
                k: decision[k]
                for k in (
                    "pending_winner",
                    "weights",
                    *(["confirmed_winner"] if args.confirm else []),
                )
            }
        )
    )


if __name__ == "__main__":
    main()
