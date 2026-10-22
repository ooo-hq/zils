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
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import zils

from . import benchmark

VERSION = "zils-base-holdout/v1"
SPLITS = ("calibration", "test")
# Kings paid from the base share, newest first (Teutonic's MAIN_SHARES with no specialists).
KINGS_PAID = 5
LEGACY_POLICY = {
    "min_brier_improvement": 0.01,
    "bootstrap_resamples": 2000,
    "max_accuracy_drop": 0.0,
}


DEFAULT_POLICY = {
    "score_version": 1,
    "min_score_improvement": 1.0,
    "max_brier_increase": 0.005,
    "bootstrap_resamples": 2000,
}


def audit(root):
    """Verify the sealed round files against the public manifest; return cases per split."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("version") == "zils-base-holdout/v2":
        from . import base_exam

        return base_exam.audit(root)
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


def archival_decide(cases, incumbent_uid, calibration_rows, test_rows, policy=None):
    """One draw: calibrate every model on calibration, compare each candidate with the incumbent on test."""
    policy = {**LEGACY_POLICY, **(policy or {})}
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
        # The exam winner is only a challenger: it earns nothing until promoted (king of the hill,
        # owner decision, 19 October 2026). Eligible runners-up are not challengers at all.
        "pending_winner": None
        if winner is None
        else {"uid": winner["uid"], "sha256": winner["sha256"]},
        "dataset_sha256": dataset_hash(cases["test"]),
    }


def validate_policy(policy=None):
    if not isinstance(policy or {}, dict):
        raise ValueError("base policy must be an object")
    result = {**DEFAULT_POLICY, **(policy or {})}
    if (
        set(result) != set(DEFAULT_POLICY)
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in result.values())
        or type(result["score_version"]) is not int
        or type(result["bootstrap_resamples"]) is not int
        or result != DEFAULT_POLICY
    ):
        raise ValueError("base policy must equal the fixed score-version-1 production policy")
    return result


def decide(
    cases, incumbent_uid, calibration_rows, test_rows, policy=None, *, draw_identity, commitments
):
    """Calibrate on calibration only; elect on fixed decision quality with a Brier guard."""
    from . import base_exam, base_metrics

    policy = validate_policy(policy)
    if set(draw_identity) != {"manifest_sha256", "dataset_sha256"} or draw_identity[
        "dataset_sha256"
    ] != dataset_hash(cases["test"]):
        raise ValueError("decision draw differs from its cases")
    seed = int(base_exam._hash(draw_identity["manifest_sha256"]), 16)
    for rows in cases.values():
        base_metrics.baselines(rows)

    def calibrated(uid):
        cal, test = calibration_rows[uid], test_rows[uid]
        if cal["sha256"] != test["sha256"]:
            raise ValueError("split checkpoint hashes differ")
        zils.score(cases["calibration"], cal["predictions"])
        zils.score(cases["test"], test["predictions"])
        temperature = fit_temperature(cases["calibration"], cal["predictions"])
        predictions = apply_temperature(cases["test"], test["predictions"], temperature)
        shared = zils.score(cases["test"], predictions)
        return predictions, {
            "uid": uid,
            "sha256": test["sha256"],
            "temperature": temperature,
            **{k: shared[k] for k in ("brier", "accuracy", "skill", "confident_errors")},
            **base_metrics.score(cases["test"], predictions),
            "evaluation": {
                s: r.get("evaluation") for s, r in (("calibration", cal), ("test", test))
            },
        }

    if incumbent_uid not in calibration_rows or incumbent_uid not in test_rows:
        raise ValueError("incumbent must be evaluated on both splits")
    incumbent_predictions, incumbent = calibrated(incumbent_uid)
    candidates = []
    for uid in sorted(set(calibration_rows) | set(test_rows) | set(commitments)):
        if uid == incumbent_uid:
            continue
        claim = commitments.get(uid, {})
        candidate = {
            "uid": uid,
            "sha256": claim.get("sha256"),
            "hotkey": claim.get("hotkey"),
            "commitment_block": claim.get("block"),
            "eligible": False,
        }
        try:
            base_exam._hash(claim.get("sha256"))
            if (
                type(claim.get("block")) is not int
                or claim["block"] < 0
                or not isinstance(claim.get("hotkey"), str)
                or not claim["hotkey"]
            ):
                raise ValueError("missing immutable commitment ownership")
            predictions, values = calibrated(uid)
            if values["sha256"] != claim["sha256"] or values["sha256"] == incumbent["sha256"]:
                raise ValueError("candidate must match commitment and differ from incumbent")
            paired = base_metrics.paired_improvement(
                cases["test"],
                incumbent_predictions,
                predictions,
                resamples=policy["bootstrap_resamples"],
                seed=seed,
            )
            candidate.update(values, **paired)
            candidate["eligible"] = (
                paired["improvement"] >= policy["min_score_improvement"]
                and paired["lower_95"] > 0
                and values["brier"] <= incumbent["brier"] + policy["max_brier_increase"] + 1e-12
            )
        except (ValueError, KeyError, TypeError) as error:
            candidate["error"] = type(error).__name__ + ": invalid candidate evidence"
        candidates.append(candidate)
    eligible = [c for c in candidates if c["eligible"]]
    winner = (
        min(eligible, key=lambda c: (-c["score"], c["commitment_block"], c["sha256"]))
        if eligible
        else None
    )
    return {
        **draw_identity,
        "metric_version": base_metrics.VERSION,
        "policy": policy,
        "incumbent": incumbent,
        "candidates": candidates,
        "pending_winner": {k: winner[k] for k in ("uid", "sha256")} if winner else None,
    }


def public_decision(decision):
    """Only aggregates and evidence digests leave the private validator store."""
    scalar = {
        k: decision[k]
        for k in (
            "metric_version",
            "manifest_sha256",
            "dataset_sha256",
            "pool_id",
            "model",
            "policy",
            "pending_winner",
            "confirmed_winner",
            "round_id",
            "evaluated_at",
        )
        if k in decision
    }
    for field in ("incumbent", "candidates"):
        rows = [decision[field]] if field == "incumbent" else decision[field]
        public = [
            {
                k: r[k]
                for k in (
                    "uid",
                    "sha256",
                    "hotkey",
                    "commitment_block",
                    "eligible",
                    "score",
                    "brier",
                    "accuracy",
                    "skill",
                    "temperature",
                    "improvement",
                    "lower_95",
                    "upper_95",
                )
                if k in r
            }
            for r in rows
        ]
        scalar[field] = public[0] if field == "incumbent" else public
    scalar["evidence_sha256"] = dataset_hash(decision)
    return scalar


def verify_decision(decision, *, active):
    """Reopen private raw reports; never authorize a crown from claimed aggregates."""
    from . import base_exam, base_metrics

    if (
        decision.get("metric_version") != base_metrics.VERSION
        or any(decision.get(k) != active[k] for k in ("pool_id", "model", "policy"))
        or decision["incumbent"]["sha256"] != active["incumbent_sha256"]
    ):
        raise ValueError("decision differs from active competition")
    identity = {k: decision[k] for k in ("manifest_sha256", "dataset_sha256")}
    if identity not in active["draws"]:
        raise ValueError("decision draw is outside active competition")
    root = Path(active["draw_roots"][identity["manifest_sha256"]])
    if draw(root) != identity:
        raise ValueError("active draw identity changed")
    history = active.get("history")
    cases = base_exam.audit(root)[1]
    rows = {s: {} for s in SPLITS}
    claims = {}
    for candidate in [decision["incumbent"], *decision["candidates"]]:
        if candidate is not decision["incumbent"]:
            claims[candidate["uid"]] = {
                "sha256": candidate["sha256"],
                "hotkey": candidate["hotkey"],
                "block": candidate["commitment_block"],
            }
        if "evaluation" not in candidate:
            continue
        for split in SPLITS:
            evidence = candidate["evaluation"][split]
            if not isinstance(evidence, dict):
                raise ValueError("missing recorded evaluation timing")
            start, end = (base_exam._time(evidence[k]) for k in ("started_at", "completed_at"))
            if start > end:
                raise ValueError("evaluation timing is reversed")
            base_exam.audit(root, history=history, as_of=end)
            path = Path(evidence["path"])
            if benchmark.file_hash(base_exam._regular(path)) != evidence["sha256"]:
                raise ValueError("evaluation report hash changed")
            stored = json.loads(path.read_text())
            binding = {
                "manifest_sha256": identity["manifest_sha256"],
                "checkpoint_sha256": candidate["sha256"],
                "split": split,
                "started_at": start,
                "completed_at": end,
            }
            if (
                any(stored.get(k) != v for k, v in binding.items())
                or len(stored.get("miners", [])) != 1
            ):
                raise ValueError("evaluation report binding changed")
            row = next(iter(rows_for(stored, cases[split]).values()), None)
            if row is None or row["sha256"] != candidate["sha256"]:
                raise ValueError("evaluation report has no matching checkpoint")
            rows[split][candidate["uid"]] = {**row, "uid": candidate["uid"], "evaluation": evidence}
    actual = decide(
        cases,
        decision["incumbent"]["uid"],
        rows["calibration"],
        rows["test"],
        active["policy"],
        draw_identity=identity,
        commitments=claims,
    )
    for key in ("incumbent", "candidates", "pending_winner"):
        if actual[key] != decision[key]:
            raise ValueError("decision aggregates differ from raw evidence")
    return base_exam.audit(root), actual


def confirm_candidates(first, second, *, active):
    from . import base_exam

    audits = [verify_decision(d, active=active)[0] for d in (first, second)]
    base_exam.disjoint(audits)
    result = []
    for candidate in first["candidates"]:
        if not candidate["eligible"]:
            continue
        again = next(
            (
                c
                for c in second["candidates"]
                if c["sha256"] == candidate["sha256"]
                and c["hotkey"] == candidate["hotkey"]
                and c["commitment_block"] == candidate["commitment_block"]
                and c["eligible"]
            ),
            None,
        )
        if again is None:
            continue
        first_end = max(
            r["evaluation"][s]["completed_at"]
            for r in (first["incumbent"], candidate)
            for s in SPLITS
        )
        second_start = min(
            r["evaluation"][s]["started_at"] for r in (second["incumbent"], again) for s in SPLITS
        )
        if second_start >= first_end + 12 * 3600:
            result.append(again)
    return result


def draw(root):
    """The identity of one sealed draw: its manifest hash and the test split's dataset hash."""
    _, cases = audit(root)
    return {
        "manifest_sha256": benchmark.file_hash(Path(root) / "manifest.json"),
        "dataset_sha256": dataset_hash(cases["test"]),
    }


def validate_track(track, *, allow_empty_draws=False):
    """A fleet config's base-tier track: share, incumbent, allowed draws and the decision policy."""
    if not isinstance(track, dict) or set(track) - {
        "share",
        "incumbent_sha256",
        "kings",
        "burn_uid",
        "draws",
        "policy",
    }:
        raise ValueError(
            "base_tier accepts share, incumbent_sha256, kings, burn_uid, draws and policy"
        )
    share = track.get("share")
    if type(share) not in (int, float) or not 0 <= share <= 1:
        raise ValueError("base_tier.share must be a number in [0, 1]")
    if not re.fullmatch("[a-f0-9]{64}", str(track.get("incumbent_sha256"))):
        raise ValueError("base_tier.incumbent_sha256 must be the incumbent checkpoint hash")
    kings = track.get("kings", [])
    if not isinstance(kings, list) or any(type(k) is not str or not k for k in kings):
        raise ValueError("base_tier.kings must list king hotkeys, newest (the incumbent's) first")
    burn = track.get("burn_uid", 0)
    if type(burn) is not int or not 0 <= burn <= 65535:
        raise ValueError("base_tier.burn_uid must be a uid")
    draws = track.get("draws")
    if not isinstance(draws, list) or (not draws and not allow_empty_draws):
        raise ValueError("base_tier.draws must list at least one sealed draw")
    for item in draws:
        if not isinstance(item, dict) or set(item) != {"manifest_sha256", "dataset_sha256"}:
            raise ValueError("each draw needs manifest_sha256 and dataset_sha256")
        if any(not re.fullmatch("[a-f0-9]{64}", str(v)) for v in item.values()):
            raise ValueError("draw hashes must be sha256 hex")
    policy = validate_policy(track.get("policy"))
    return {**track, "kings": kings, "burn_uid": burn, "policy": policy}


def king_weights(track, members):
    """King history, as Teutonic pays its main competition: the last five distinct kings (by hotkey,
    newest first) split the base share equally. A king no longer registered drops out and the rest
    share its slice. The vendor base has no owner (empty history) and pays nobody."""
    recent = list(dict.fromkeys(validate_track(track)["kings"]))[:KINGS_PAID]
    uid_by_hotkey = {item["hotkey"]: int(uid) for uid, item in members.items()}
    paid = [uid_by_hotkey[k] for k in recent if k in uid_by_hotkey]
    return {uid: 1 / len(paid) for uid in paid}


def track_weights(track, decision, members):
    """Base-tier weights under a track, after checking the decision belongs to it. Only kings are
    paid: a pending or confirmed winner earns nothing until the owner's promotion PR crowns it by
    setting `incumbent_sha256` and prepending its hotkey to `kings` (owner decision, 19 October 2026)."""
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
        raise ValueError("base-tier decision names a miner outside this registered fleet")
    pending = decision["pending_winner"]
    if pending is not None and not any(
        c["uid"] == pending["uid"] and c["sha256"] == pending["sha256"] for c in eligible
    ):
        raise ValueError("base-tier pending winner is not an eligible candidate")
    return king_weights(track, members)


def confirm(first, second):
    """Promotion needs the same checkpoint eligible on two disjoint draws; return the confirmed winner or None."""
    if first["dataset_sha256"] == second["dataset_sha256"]:
        raise ValueError("confirmation requires a disjoint draw")
    pending = first["pending_winner"]
    if pending is None:
        return None
    again = [c for c in second["candidates"] if c["sha256"] == pending["sha256"] and c["eligible"]]
    return pending if again else None


def confirm_promotion(first, second):
    """New promotions require recorded evaluation times and the same competition contract."""
    times = [report.get("evaluated_at") for report in (first, second)]
    if any(type(t) not in (int, float) or not math.isfinite(t) for t in times):
        raise ValueError("promotion requires recorded evaluation times")
    if times[1] - times[0] < 12 * 3600:
        raise ValueError("confirmation must be evaluated at least 12 hours later")
    if any(first.get(k) != second.get(k) for k in ("pool_id", "model", "policy")) or (
        first["incumbent"]["sha256"] != second["incumbent"]["sha256"]
    ):
        raise ValueError("confirmation differs from the first competition contract")
    return confirm(first, second)


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
    parser.add_argument("--config")
    parser.add_argument("--pool")
    parser.add_argument("--chain-snapshot")
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
    if not (args.config and args.pool and args.chain_snapshot):
        parser.error("--config, --pool and --chain-snapshot are required")
    from . import base_release, commitments

    config_path = Path(args.config)
    active = base_release.active_context(
        json.loads(config_path.read_text()), args.pool, directory=config_path.parent
    )
    cases = audit(args.round)[1]
    snapshot = json.loads(Path(args.chain_snapshot).read_text())
    claims = commitments.candidates(snapshot, active["incumbent_sha256"])
    claims = {
        u: {**c, "hotkey": snapshot["members"][str(u)]}
        for u, c in claims.items()
        if "error" not in c
    }
    reports = {
        s: rows_for(json.loads(Path(p).read_text()), cases[s])
        for s, p in (("calibration", args.calibration_report), ("test", args.test_report))
    }
    decision = decide(
        cases,
        args.incumbent_uid,
        reports["calibration"],
        reports["test"],
        active["policy"],
        draw_identity=draw(args.round),
        commitments=claims,
    )
    decision.update(pool_id=args.pool, model=active["model"])
    verify_decision(decision, active=active)
    if args.confirm:
        ready = confirm_candidates(
            json.loads(Path(args.confirm).read_text()), decision, active=active
        )
        decision["confirmed_winner"] = (
            {k: ready[0][k] for k in ("uid", "sha256")} if ready else None
        )
    with Path(args.out).open("x") as output:
        json.dump(decision, output, indent=2, allow_nan=False)
        output.write("\n")
    print(
        json.dumps(
            {
                k: decision[k]
                for k in (
                    "pending_winner",
                    *(["confirmed_winner"] if args.confirm else []),
                )
            }
        )
    )


if __name__ == "__main__":
    main()
