"""Opt-in approved-pool scheduler and private operator controls. Preview never mutates."""

import argparse
import json
import re
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from . import miner_grading as grading, models
from .cloud import Supabase


def validate_config(config):
    if not isinstance(config, dict) or set(config) - {
        "mode",
        "policy_version",
        "pool",
        "contexts",
        "qualification_slots",
    }:
        raise ValueError("Invalid routing configuration fields")
    if config.get("mode") == "fixed" and set(config) == {"mode"}:
        return dict(config)
    if config.get("mode") != "graded" or config.get("policy_version") != grading.POLICY:
        raise ValueError("Unknown miner routing policy")
    pool, contexts = config.get("pool"), config.get("contexts")
    if (
        not isinstance(pool, list)
        or not 1 <= len(pool) <= 256
        or any(not isinstance(k, str) or not re.fullmatch("[a-zA-Z0-9_-]{1,128}", k) for k in pool)
        or len(set(pool)) != len(pool)
    ):
        raise ValueError("Provide 1–256 distinct approved hotkeys")
    if (
        not isinstance(contexts, dict)
        or not contexts
        or any(not grading.valid_context(c) or band != c["band"] for band, c in contexts.items())
    ):
        raise ValueError("Provide comparable benchmark contexts for each enabled workload band")
    for c in contexts.values():
        if any(c[k] != v for k, v in models.profile_identity(models.JEVK5).items()):
            raise ValueError("Context differs from the pinned model")
    slots = config.get("qualification_slots", 1)
    if type(slots) is not int or slots not in (0, 1):
        raise ValueError("qualification_slots must be 0 or 1")
    return {**deepcopy(config), "qualification_slots": slots}


def report_fingerprint(report):
    return grading.digest({k: v for k, v in report.items() if k != "id"})


def benchmark_binding(job):
    manifest = job["manifest"]
    if models.job_model(job) != models.JEVK5 or manifest.get("selection", {}).get("previous"):
        raise ValueError("Qualification requires the pinned base, not a customer predecessor")
    return grading.digest(
        {k: manifest[k] for k in ("model", "files", "counts", "workload")}
        | {"initial_sha256": job["initial_sha256"]}
    )


class GradedScheduler:
    def __init__(self, store, config):
        self.store, self.config = store, validate_config(config)
        if self.config["mode"] != "graded":
            raise ValueError("GradedScheduler requires graded routing")

    def preview(self, job, now):
        state = self.store.rpc("zils_grading_snapshot", {"p_job": job["id"]})
        if state["config"] != self.config:
            raise ValueError("Installed database policy differs from the scheduler configuration")
        decision = grading.rank_candidates(state["job"], state["workers"], now)
        return {**decision, "state_sha256": state["state_sha256"]}

    def qualification_candidate(self, job, now):
        result = self.preview(job, now)
        if result["purpose"] != "qualification":
            raise ValueError("Job has no trusted qualification authorization")
        return result

    def assign(self, job, now, *, qualification=False):
        for _ in range(2):
            decision = self.preview(job, now)
            if (decision["purpose"] == "qualification") != qualification:
                return {"status": "waiting", "reason": "separate_qualification_lane"}
            if not decision["selected_hotkey"]:
                return {"status": "waiting", "decision": decision}
            result = self.store.rpc(
                "zils_reserve_graded_job",
                {
                    "p_job": job["id"],
                    "p_hotkey": decision["selected_hotkey"],
                    "p_decision": decision,
                },
            )
            if result["status"] != "retry":
                return result
        return result

    def tick_qualification(self, now):
        self.store.rpc("zils_reap_graded", {})
        if self.config["qualification_slots"]:
            job = self.store.rpc("zils_next_qualification_job", {})
            if job:
                return self.assign(job, now, qualification=True)
        return {"status": "waiting"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    preview = commands.add_parser("preview")
    preview.add_argument("--config", type=Path, required=True)
    preview.add_argument("--job", required=True)
    configure = commands.add_parser("configure")
    configure.add_argument("--config", type=Path, required=True)
    bind = commands.add_parser("bind-resource")
    bind.add_argument("--hotkey", required=True)
    bind.add_argument("--resource", required=True)
    import_report = commands.add_parser("import-qualification")
    import_report.add_argument("--hotkey", required=True)
    import_report.add_argument("--report", type=Path, required=True)
    import_report.add_argument("--verified-by", required=True)
    qualification = commands.add_parser("authorize-qualification")
    qualification.add_argument("--job", required=True)
    qualification.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Private benchmark descriptor with binding, context, min_accuracy and quality_floor",
    )
    args = parser.parse_args()
    store = Supabase()
    if args.action in ("preview", "configure"):
        raw = json.loads(args.config.read_text())
        config = validate_config(raw.get("routing", raw))
        if args.action == "preview":
            result = GradedScheduler(store, config).preview(
                {"id": str(uuid.UUID(args.job))}, datetime.now(UTC)
            )
        else:
            result = store.rpc("zils_configure_routing", {"p_config": config})
    elif args.action == "bind-resource":
        result = store.rpc(
            "zils_bind_resource",
            {"p_hotkey": args.hotkey, "p_resource": str(uuid.UUID(args.resource))},
        )
    elif args.action == "import-qualification":
        report = json.loads(args.report.read_text())
        report.setdefault("id", str(uuid.uuid4()))
        if not grading.usable_reports([report], report.get("context"), datetime.now(UTC)):
            raise ValueError("Report is expired, invalid or below its declared qualification floor")
        result = store.rpc(
            "zils_import_qualification",
            {
                "p_hotkey": args.hotkey,
                "p_report": report,
                "p_verified_by": args.verified_by,
                "p_evidence_sha256": report_fingerprint(report),
            },
        )
    else:
        jid = str(uuid.UUID(args.job))
        rows = store.rows("fez_training_jobs", f"id=eq.{jid}")
        descriptor = json.loads(args.benchmark.read_text())
        if not rows or benchmark_binding(rows[0]) != descriptor.get("benchmark_sha256"):
            raise ValueError("Qualification benchmark differs from the frozen inputs/base")
        if (
            not grading.valid_context(descriptor.get("context"))
            or descriptor["context"]["benchmark_sha256"] != descriptor["benchmark_sha256"]
        ):
            raise ValueError("Benchmark context differs from its dataset binding")
        if not grading.finite(descriptor.get("min_accuracy"), maximum=1) or not grading.finite(
            descriptor.get("quality_floor"), minimum=-2, maximum=2
        ):
            raise ValueError("Benchmark requires finite acceptance floors")
        result = store.rpc(
            "zils_authorize_qualification", {"p_job": jid, "p_benchmark": descriptor}
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
