"""Offline exam qualification from pinned native probability exports; never trains or crowns."""

import argparse
import itertools
import json
import math
import statistics
from datetime import date
from pathlib import Path

from . import base_exam, base_metrics, base_round, benchmark, models

VERSION = "zils-base-pilot/v1"


def ranks(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        for i, _ in ordered[start:end]:
            result[i] = (start + 1 + end) / 2
        start = end
    return result


def spearman(first, second):
    x, y = ranks(first), ranks(second)
    if len(x) < 2 or len(x) != len(y) or len(set(x)) == 1 or len(set(y)) == 1:
        return 0.0
    return statistics.correlation(x, y)


def ranking(external, measured):
    span = max(external) - min(external)
    rho = spearman(external, measured)
    inversions = [
        [i, j]
        for i, j in itertools.combinations(range(len(external)), 2)
        if abs(external[i] - external[j]) >= 5
        and (external[i] - external[j]) * (measured[i] - measured[j]) < 0
    ]
    return {
        "passed": len(external) >= 5 and span >= 20 and rho >= 0.9 and not inversions,
        "spearman": rho,
        "external_score_span": span,
        "inversions": inversions,
    }


# Decision Index references only answer Choice and yes/no; Score stays in elections between pool candidates.
EXTERNAL_TYPES = ("choice", "noul")


def external_score(cases, predictions):
    """Decision score over the question types the external references support."""
    kept = [c for c in cases if c["question"]["type"] in EXTERNAL_TYPES]
    ids = {c["id"] for c in kept}
    return base_metrics.score(kept, [p for p in predictions if p["id"] in ids])["score"]


def runtime_contract(model):
    if model not in models.d3_family():
        raise ValueError("base pilot requires a text pool runtime")
    pins = models.pins(model)
    files = {
        name: digest
        for name, digest in pins["files"].items()
        if not name.startswith("model-")
        and name
        not in {
            "model.safetensors",
            "model.safetensors.index.json",
            "MODEL_MANIFEST.json",
            "NOTICE",
            "LICENSE",
            "README.md",
        }
    }
    root = Path(__file__).resolve().parents[1]
    return {
        "pool_id": models.pool_id(model),
        "files": files,
        "runtime": {k: v for k, v in pins.items() if k not in ("files", "revision", "vendor")},
        "local_code": {
            name: benchmark.file_hash(root / name)
            for name in ("zils/d3.py", "zils/core.py", "zils/runtime.py", "requirements/d3.txt")
        },
        "limits": {
            k: v
            for k, v in models.spec(model).items()
            if k in ("max_input_tokens", "max_output_tokens")
        },
    }


def design(bundle):
    return {
        **{
            k: bundle[k]
            for k in (
                "pool_id",
                "policy",
                "blueprint",
                "runtime_identity",
                "roster",
                "pairs",
                "within_pool_pair",
                "draws",
                "capacity",
                "history",
            )
        },
        "development_reports": bundle["reports"]["development"],
    }


def _read(evidence, root):
    path = Path(root) / evidence["path"]
    if benchmark.file_hash(base_exam._regular(path)) != base_exam._hash(evidence["sha256"]):
        raise ValueError("pilot input differs from its pinned hash")
    return json.loads(path.read_text())


def _finite(value, minimum, maximum):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError("pilot measurement is outside finite bounds")
    return value


def check(bundle, *, root):
    root = Path(root).resolve()
    reasons, phases = [], {}
    result = {
        "version": "zils-base-qualification/v1",
        "pool_id": bundle.get("pool_id"),
        "policy": bundle.get("policy"),
        "blueprint": bundle.get("blueprint"),
        "runtime_identity": bundle.get("runtime_identity"),
        "bundle": bundle,
        "root": str(root),
    }
    try:
        if (
            bundle["version"] != VERSION
            or bundle["phase"] != "validation"
            or bundle["pool_id"] not in models.D3_POOL_MODELS
        ):
            raise ValueError("qualification requires independent validation for a text pool")
        base_round.validate_policy(bundle["policy"])
        if bundle["design_sha256"] != base_round.dataset_hash(design(bundle)):
            raise ValueError("pilot roster/design changed after freezing")
        frozen_at = base_exam._time(bundle["frozen_at"])
        roster = bundle["roster"]
        refs = [name for name, entry in roster.items() if entry["kind"] == "reference"]
        if len(refs) < 5 or len({roster[name]["model_sha256"] for name in refs}) < 5:
            raise ValueError("pilot requires at least five distinct pinned external references")
        editions = {base_round.dataset_hash(roster[n]["edition"]) for n in refs}
        if len(editions) != 1:
            raise ValueError("external Full scores must use one dated leaderboard edition")
        edition = roster[refs[0]]["edition"]
        date.fromisoformat(edition["date"])
        base_exam._hash(edition["sha256"])
        if not edition["url"].startswith("https://"):
            raise ValueError("leaderboard snapshot needs a public source URL")
        external = [_finite(roster[n]["full_score"], 0, 100) for n in refs]
        pairs = bundle["pairs"]
        if (
            not pairs
            or any(
                not isinstance(p, list)
                or len(p) != 2
                or p[0] == p[1]
                or any(n not in roster for n in p)
                for p in pairs
            )
            or len({tuple(sorted(p)) for p in pairs}) != len(pairs)
        ):
            raise ValueError("pilot requires distinct predeclared comparison pairs")
        within = bundle["within_pool_pair"]
        if (
            within not in pairs
            or any(
                roster[n].get("kind") != "pool"
                or roster[n].get("pool_id") != bundle["pool_id"]
                or roster[n]["runtime_sha256"]
                != base_round.dataset_hash(bundle["runtime_identity"])
                for n in within
            )
            or roster[within[0]]["model_sha256"] == roster[within[1]]["model_sha256"]
            or roster[within[0]]["settings_sha256"] != roster[within[1]]["settings_sha256"]
        ):
            raise ValueError(
                "predeclared within-pool pair requires distinct weights and matching execution contract"
            )
        if set(bundle["capacity"]) != set(roster):
            raise ValueError("every native runtime requires measured capacity evidence")
        capacities = {}
        for name, entry in roster.items():
            for key in ("model_sha256", "runtime_sha256", "settings_sha256"):
                base_exam._hash(entry[key])
            capacity = _read(bundle["capacity"][name], root)
            if (
                any(
                    capacity.get(k) != entry[k]
                    for k in ("model_sha256", "runtime_sha256", "settings_sha256")
                )
                or capacity.get("probability_export_reviewed") is not True
                or not isinstance(capacity.get("device"), str)
                or not capacity["device"]
            ):
                raise ValueError("runtime/capacity/probability conversion evidence is incomplete")
            _finite(
                capacity["peak_gpu_mib"],
                0.000001,
                _finite(capacity["available_gpu_mib"], 0.000001, 1e9),
            )
            capacities[name] = base_exam._time(capacity["completed_at"])
        history = _read(bundle["history"], root)
        audits = []
        for phase in ("development", "validation"):
            draw = bundle["draws"][phase]
            drawroot = root / draw["path"]
            identity = base_round.draw(drawroot)
            if identity != {k: draw[k] for k in identity}:
                raise ValueError("pilot draw identity changed")
            manifest, cases = base_exam.audit(drawroot)
            if (
                manifest["pool_id"] != bundle["pool_id"]
                or base_exam.blueprint(cases) != bundle["blueprint"]
            ):
                raise ValueError("pilot draw differs from pool/frozen blueprint")
            audits.append((manifest, cases))
            if set(bundle["reports"][phase]) != set(roster):
                raise ValueError("pilot predictions must cover the frozen roster exactly")
            predictions, scores = {}, {}
            for name, entry in roster.items():
                exported = {}
                if set(bundle["reports"][phase][name]) != set(base_round.SPLITS):
                    raise ValueError("pilot requires calibration and test exports")
                for split in base_round.SPLITS:
                    report = _read(bundle["reports"][phase][name][split], root)
                    if any(
                        report.get(k) != entry[k]
                        for k in ("model_sha256", "runtime_sha256", "settings_sha256")
                    ) or report["dataset_sha256"] != base_round.dataset_hash(cases[split]):
                        raise ValueError("native prediction export binding changed")
                    start, end = (
                        base_exam._time(report[k]) for k in ("started_at", "completed_at")
                    )
                    if (
                        not capacities[name] <= start <= end
                        or phase == "validation"
                        and start <= frozen_at
                        or phase == "development"
                        and end > frozen_at
                    ):
                        raise ValueError(
                            "validation must start after design freeze and completed development"
                        )
                    base_exam.audit(drawroot, history=history, as_of=end)
                    # Validate probabilities before fitting; labels stay in the sealed draw.
                    import zils

                    zils.score(cases[split], report["predictions"])
                    exported[split] = report["predictions"]
                temperature = base_round.fit_temperature(
                    cases["calibration"], exported["calibration"]
                )
                predictions[name] = base_round.apply_temperature(
                    cases["test"], exported["test"], temperature
                )
                scores[name] = external_score(cases["test"], predictions[name])
            rank = ranking(external, [scores[n] for n in refs])
            if not rank["passed"]:
                reasons.append(phase + ": external ranking/span requirement failed")
            comparisons = []
            for left, right in itertools.combinations(roster, 2):
                paired = base_metrics.paired_improvement(
                    cases["test"],
                    predictions[left],
                    predictions[right],
                    resamples=bundle["policy"]["bootstrap_resamples"],
                    seed=int(identity["manifest_sha256"], 16),
                )
                declared = [left, right] in pairs or [right, left] in pairs
                required = {left, right} == set(within)
                excluded = paired["bootstrap_variance"] <= 0
                precise = (
                    max(
                        abs(paired["improvement"] - paired["lower_95"]),
                        abs(paired["upper_95"] - paired["improvement"]),
                    )
                    <= 0.5
                )
                comparisons.append(
                    {
                        "pair": [left, right],
                        **paired,
                        "predeclared": declared,
                        "excluded": excluded,
                        "precision_passed": precise and not excluded,
                    }
                )
                if required and excluded or declared and not excluded and not precise:
                    reasons.append(
                        phase + ": within-pool variance or paired precision requirement failed"
                    )
            phases[phase] = {"scores": scores, "ranking": rank, "comparisons": comparisons}
        base_exam.disjoint(audits)
    except (ValueError, KeyError, TypeError, OSError, ZeroDivisionError) as error:
        reasons.append(str(error))
    payload = {**result, "passed": not reasons, "reasons": reasons, "phases": phases}
    return {**payload, "sha256": base_round.dataset_hash(payload)}


def verify_qualification(report, *, pool_id, policy, blueprint, runtime_identity):
    if report.get("pool_id") != pool_id:
        raise ValueError("pilot qualification belongs to a different pool")
    if report.get("passed") is not True or any(
        report.get(k) != v
        for k, v in {
            "policy": policy,
            "blueprint": blueprint,
            "runtime_identity": runtime_identity,
        }.items()
    ):
        raise ValueError("pilot qualification is failed or stale")
    if check(report["bundle"], root=Path(report["root"])) != report:
        raise ValueError("pilot qualification differs from pinned raw evidence")


def admit(config, directory, draws):
    if not config.get("pool_id") or not config.get("pilot_report") or not draws:
        raise ValueError("pool inference requires its own passing pilot_report")
    model = models.validate_spec(config["model"])
    if models.pool_id(model) != config["pool_id"]:
        raise ValueError("pilot pool differs from active model")
    if any(base_exam.audit(path)[0]["pool_id"] != config["pool_id"] for _, path in draws):
        raise ValueError("pilot draw belongs to another pool")
    blueprints = [base_exam.blueprint(base_exam.audit(path)[1]) for _, path in draws]
    if any(value != blueprints[0] for value in blueprints):
        raise ValueError("pool draws differ from frozen pilot blueprint")
    report = json.loads(base_exam._regular(Path(directory) / config["pilot_report"]).read_text())
    verify_qualification(
        report,
        pool_id=config["pool_id"],
        policy=base_round.validate_policy(config["base_tier"].get("policy")),
        blueprint=blueprints[0],
        runtime_identity=runtime_contract(models.validate_spec(config["model"])),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = check(json.loads(args.bundle.read_text()), root=args.bundle.resolve().parent)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("passed", "reasons", "sha256")}))


if __name__ == "__main__":
    main()
