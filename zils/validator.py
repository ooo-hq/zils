"""Collect checkpoints, calibrate and score them, and optionally publish chain weights."""

import collections
import http.client
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import zils

from . import models, protocol as wire
from .runtime import CapacityUnavailable, digest, locked, run_child, signed, signing_key, verified

# The incumbent base is scored as a row outside every roster; the publisher refuses a roster uid there.
INCUMBENT_UID = 65535
# Per-submission evaluation budget: long inputs measured ~1 s/case (p95 2 s) on a 4090, so 3 s/case
# leaves headroom; never below the old fixed 600 s.
EVAL_SECONDS_PER_CASE = 3
EVAL_MIN_SECONDS = 600


def evaluation_timeout(cases):
    """Seconds one submission may take on `cases` (a JSONL file, or a JSON list for image requests)."""
    text = Path(cases).read_text()
    count = (
        len(json.loads(text))
        if text.lstrip().startswith("[")
        else sum(1 for line in text.splitlines() if line.strip())
    )
    return max(EVAL_MIN_SECONDS, EVAL_SECONDS_PER_CASE * count)


def run_evaluate(
    args,
    attempt,
    name,
    submissions,
    cases,
    *,
    base_revision,
    images=None,
    wait_for_capacity=False,
    capacity_model=None,
    lease=None,
):
    """One sandboxed `zils evaluate` of `submissions` on `cases`, deferred while GPU capacity is unavailable.
    A `lease` (base_queue.ComputeLease) is acquired before the first evaluation and kept by its owner."""
    manifest = attempt / f"{name}-submissions.json"
    wire.write_json(manifest, submissions)
    report = attempt / f"{name}.json"
    per_submission = evaluation_timeout(cases)
    command = [
        sys.executable,
        "-m",
        "zils",
        "evaluate",
        "--submissions",
        str(manifest),
        "--cases",
        str(cases),
        "--base-revision",
        base_revision,
        "--runner-python",
        args.runtime_python,
        "--device",
        args.device,
        "--timeout",
        str(per_submission),
        "--report",
        str(report),
    ]
    if images is not None:
        command.extend(["--images", str(images)])
    if lease is not None:
        lease.acquire()
    waiting = False
    while True:
        try:
            run_child(
                command,
                attempt / f"{name}.log",
                args.device,
                # Checkpoint staging also counts against the child, so each submission gets 120 s extra.
                timeout=120 + (per_submission + 120) * len(submissions),
                model=capacity_model,
            )
            break
        except CapacityUnavailable:
            if not wait_for_capacity:
                raise
            if not waiting:
                print("validator: waiting for evaluation capacity", flush=True)
                waiting = True
            time.sleep(max(1, args.poll))
    return json.loads(report.read_text())


def evaluate_round(
    config, directory, work, registry, args, *, fetch_checkpoint=None, wait_for_capacity=False
):
    from . import benchmark, calibrate, jobs

    if shutil.disk_usage(work).free < 2 * 1024**3:
        raise RuntimeError("less than 2 GiB free; archive completed rounds before evaluating again")
    data = directory / "benchmark"
    if digest(data / "manifest.json") != config["benchmark_sha256"]:
        raise ValueError("benchmark manifest changed")
    benchmark.audit(data)
    manifest = json.loads((data / "manifest.json").read_text())
    model = models.validate_spec(manifest["model"]) if "model" in manifest else models.KEV
    frozen_images = "assets" in manifest  # verified <sha256>.png cache
    if config["base_revision"] != models.spec(model)["base_revision"]:
        raise ValueError("Evaluation configuration differs from the frozen model profile")
    if config.get("job_sha256") != (
        digest(data / "manifest.json") if "job_id" in manifest else None
    ) or config.get("job_id") != manifest.get("job_id"):
        raise ValueError("customer job differs from the installed configuration")
    entries, failed = [], {}
    attempt = work / ("evaluation-" + uuid.uuid4().hex)
    attempt.mkdir(mode=0o700)
    for uid, message in sorted(registry.items()):
        claim = message["claim"]
        path = attempt / f"raw-{uid}"
        try:
            options = {"expected_endpoint": claim["endpoint"], "round_scoped": True}
            if fetch_checkpoint is None:
                options["model"] = model
                if admission_settings(config)[1]:
                    options["public"] = True
            (fetch_checkpoint or wire.fetch_checkpoint)(claim, path, **options)
            if models.checkpoint_model(path) != model:
                raise ValueError("Submission differs from the job model profile")
            entries.append(zils.submission(path, uid))
        except (ValueError, OSError, http.client.HTTPException) as error:
            failed[uid] = str(error)

    def evaluate(submissions, cases, name, base_revision=None):
        return run_evaluate(
            args,
            attempt,
            name,
            submissions,
            cases,
            base_revision=base_revision or config["base_revision"],
            images=directory / "images" if frozen_images else None,
            wait_for_capacity=wait_for_capacity,
        )

    baseline = None
    baseline_model = model
    selection = manifest.get("selection")
    previous = selection["previous"] if selection else None
    if "job_id" in config:
        # Calibrate the reference on the same population as each candidate. A
        # published reference may already carry a temperature from another task.
        source = directory / ("comparison" if previous else "reference")
        entry = zils.submission(source, 0)
        baseline_reference_sha256 = previous["sha256"] if previous else config["initial_sha256"]
        if entry["sha256"] != baseline_reference_sha256:
            raise ValueError("job baseline checkpoint changed")
        if previous:
            # Score the exact version the customer can call, including its serving temperature.
            fitted_baseline = source
            # An upgrade may change base: the frozen release runs on the base it was trained on.
            baseline_model = models.checkpoint_model(source)
            if baseline_model != model and selection.get("base_change") is not True:
                raise ValueError("cross-profile upgrade requires base_change")
        else:
            raw = attempt / "baseline-raw"
            zils.stage(entry, raw)
            models.set_temperature(raw, 1.0)
            calibration = evaluate(
                [zils.submission(raw, 0)], data / "calibration.jsonl", "baseline-calibration"
            )
            if calibration["miners"][0]["status"] != "evaluated":
                raise RuntimeError("baseline calibration failed; no customer model can be accepted")
            fitted_baseline = attempt / "baseline-calibrated"
            calibrate.fit(data, calibration, 0, raw, fitted_baseline)
        baseline_report = evaluate(
            [zils.submission(fitted_baseline, 0)],
            data / "test.jsonl",
            "baseline-test",
            base_revision=models.spec(baseline_model)["base_revision"],
        )
        jobs.verify_report(data, baseline_report, "test")
        baseline = baseline_report["miners"][0]
        if baseline["status"] != "evaluated":
            raise RuntimeError(
                "baseline evaluation failed; no customer model can be accepted: "
                + baseline.get("error", "no error recorded")
            )

    fitted = []
    if entries:
        calibration = evaluate(entries, data / "calibration.jsonl", "calibration")
        for entry, row in zip(entries, calibration["miners"]):
            if row["status"] != "evaluated":
                failed[entry["uid"]] = row.get("error", "calibration inference failed")
                continue
            target = attempt / f"calibrated-{entry['uid']}"
            try:
                calibrate.fit(data, calibration, entry["uid"], entry["checkpoint"], target)
                fitted.append(zils.submission(target, entry["uid"]))
            except (ValueError, OSError) as error:
                failed[entry["uid"]] = str(error)
    report = (
        evaluate(fitted, data / "test.jsonl", "evaluation")
        if fitted
        else {"miners": [], "weights": {}}
    )
    present = {m["uid"] for m in report["miners"]}
    for uid in sorted(int(k) for k in config["members"]):
        if uid not in present:
            report["miners"].append(
                {
                    "uid": uid,
                    "status": "rejected" if uid in failed else "missing",
                    "skill": 0.0,
                    "error": failed.get(uid, "no submission before deadline"),
                }
            )
    report.update(
        model=models.spec(model),
        mode="private-lan-development",
        chain_write=False,
        benchmark_use="repeated-local-development",
        weights=zils.weight_vector(report["miners"]),
        submitted={uid: m["claim"]["sha256"] for uid, m in registry.items()},
    )
    if baseline is not None:
        delivery = jobs.select(baseline, report["miners"], manifest["acceptance"])
        if delivery["status"] == "accepted":
            winner = next(entry for entry in fitted if entry["uid"] == delivery["uid"])
            release = attempt / "accepted-model"
            zils.stage(winner, release)
            delivery["checkpoint"] = str(release.relative_to(work))
            wire.write_json(
                release / "release.json",
                {
                    "job_id": config["job_id"],
                    "job_sha256": config["job_sha256"],
                    "round_id": work.name,
                    "model": models.spec(models.checkpoint_model(release)),
                    "base": models.spec(models.checkpoint_model(release))["base"],
                    "base_revision": config["base_revision"],
                    "initial_sha256": config["initial_sha256"],
                    "submitted_sha256": registry[delivery["uid"]]["claim"]["sha256"],
                    "baseline_brier": baseline["brier"],
                    **(
                        {
                            "selection": selection,
                            "baseline_reference_sha256": baseline_reference_sha256,
                        }
                        if selection
                        else {}
                    ),
                    **delivery,
                },
            )
        report.update(
            job_id=config["job_id"],
            job_sha256=config["job_sha256"],
            benchmark_use="customer-held-out-reused-across-rounds",
            baseline=baseline,
            baseline_model=baseline_model,
            delivery=delivery,
        )
        if selection:
            report.update(selection=selection, baseline_reference_sha256=baseline_reference_sha256)
    if "chain" in config:
        report.update(
            mode="testnet-closed-development",
            chain=config["chain"],
            identities={uid: item["hotkey"] for uid, item in config["members"].items()},
        )
    wire.write_json(work / "report.json", report)
    return report


def attach_base_tier(config, directory, work, report):
    """Embed the epoch's base-tier decision, frozen with the round the first time it is published."""
    if "base_tier" not in config:
        return report
    frozen = work / "base-tier.json"
    if not frozen.exists():
        current = directory / "base-tier" / "decision.json"
        wire.write_json(frozen, json.loads(current.read_text()) if current.exists() else None)
    return {**report, "base_tier": json.loads(frozen.read_text())}


def attach_base_pools(config, directory, work, report):
    if "base_pools" not in config:
        return report
    from . import model_pools

    normalized = model_pools.validate(config["base_pools"])
    frozen = work / "base-pools.json"
    if not frozen.exists():
        decisions = {}
        if "base_validation_queue" in config:
            from . import base_queue

            decisions = base_queue.decisions(config, directory)
        else:
            for pool, entry in normalized["pools"].items():
                current = directory / "base-pools" / pool / "base-tier" / "decision.json"
                if entry["enabled"] and current.exists():
                    decisions[pool] = json.loads(current.read_text())
        wire.write_json(frozen, {"config": normalized, "decisions": decisions})
    saved = json.loads(frozen.read_text())
    if saved["config"] != normalized:
        raise ValueError("pending pool report differs from the installed configuration")
    return {**report, "base_pools": saved}


def pool_rounds(config, directory):
    """Keep operator paths outside the public, frozen payout contract."""
    if "base_pools" not in config:
        return {}
    from . import model_pools

    normalized = model_pools.validate(config["base_pools"])
    enabled = {p: e for p, e in normalized["pools"].items() if e["enabled"]}
    paths = config.get("base_pool_rounds", {})
    if not isinstance(paths, dict) or set(paths) != set(enabled):
        raise ValueError("base_pool_rounds must name each enabled pool exactly once")
    rounds = {}
    for pool, entry in enabled.items():
        if not isinstance(paths[pool], dict) or set(paths[pool]) != {
            "base_round_dir",
            "base_reference",
            "family_history",
            "suite_manifest",
            "pilot_report",
        }:
            raise ValueError(
                "each pool requires base_round_dir, base_reference, family_history, suite_manifest and pilot_report"
            )
        settings = {
            **{k: v for k, v in config.items() if k not in ("base_pools", "base_pool_rounds")},
            "model": entry["model"],
            "base_tier": model_pools.track(entry),
            **{k: str((Path(directory) / v).resolve()) for k, v in paths[pool].items()},
            "pool_id": pool,
        }
        root = directory / "base-pools" / pool
        audited = base_draws(settings, root)
        from . import base_pilot

        base_pilot.admit(settings, root, audited)
        rounds[pool] = (settings, root, audited)
    return rounds


def base_pools_epoch(config, directory, work, snapshot, args, rounds):
    from . import commitments, feature_commitments

    if "base_validation_queue" in config:
        from . import base_queue

        summary = base_queue.enqueue(config, directory, snapshot)
        if not (work / "pool-intake.json").exists():
            wire.write_json(work / "pool-intake.json", {"queue": summary})
        return

    snapshot = {
        **snapshot,
        "commitments": {
            uid: item
            for uid, item in snapshot.get("commitments", {}).items()
            if not str(item.get("data", "")).startswith(feature_commitments.PREFIX + ":")
        },
    }
    routed, rejected = commitments.route_pools(
        commitments.candidates(snapshot, None), config["base_pools"]
    )
    errors = {}
    for pool, (pool_config, root, draws) in rounds.items():
        pool_work = work / pool
        pool_work.mkdir(mode=0o700, exist_ok=True)
        try:
            base_tier_epoch(
                pool_config,
                root,
                pool_work,
                {u: {**c, "hotkey": snapshot["members"][str(u)]} for u, c in routed[pool].items()},
                args,
                draws,
                wait_for_capacity=True,
            )
        except (ValueError, OSError, RuntimeError, http.client.HTTPException) as error:
            (root / "base-tier" / "decision.json").unlink(missing_ok=True)
            errors[pool] = str(error)[:400]
    if not (work / "pool-intake.json").exists():
        wire.write_json(work / "pool-intake.json", {"rejected": rejected, "errors": errors})


def attach_adapter_evidence(config, work, report, now=None):
    """Embed the platform's signed queue evidence, frozen with the round the first time it is published.

    A failed fetch raises, so the pending round retries after a restart instead of burning the share.
    """
    if "adapter_evidence" not in config:
        return report
    frozen = work / "adapter-evidence.json"
    if not frozen.exists():
        from . import queue_evidence

        track = queue_evidence.validate_track(config["adapter_evidence"])
        until = queue_evidence.window_end(now)
        wire.write_json(frozen, queue_evidence.fetch(track, until))
    return {**report, "adapter_evidence": json.loads(frozen.read_text())}


def publication_report(config, directory, work):
    """Reconstruct only already-frozen evidence for a manual publication retry."""
    from . import feature_round

    for key, filename in (
        ("base_tier", "base-tier.json"),
        ("base_pools", "base-pools.json"),
        ("feature_tracks", "feature-tracks.json"),
        ("adapter_evidence", "adapter-evidence.json"),
    ):
        if key in config and not (work / filename).is_file():
            raise ValueError(f"publication requires frozen {filename}")
    report = json.loads((work / "report.json").read_text())
    report = attach_base_tier(config, directory, work, report)
    report = attach_base_pools(config, directory, work, report)
    report = feature_round.attach(config, directory, work, report)
    return attach_adapter_evidence(config, work, report)


def base_draws(config, directory):
    """The sealed draws this validator scores itself, audited now and ordered as the track lists them
    (the first decides, the last confirms). None when the config has no `base_round_dir`."""
    if "base_round_dir" not in config:
        return None
    from . import base_round

    if "base_tier" not in config or "base_reference" not in config:
        raise ValueError("base_round_dir requires a base_tier track and base_reference")
    if "chain" not in config:
        raise ValueError("the base-tier intake reads candidates from chain commitments; add chain")
    if not config.get("pool_id") or not config.get("family_history"):
        raise ValueError(
            "new base elections require pool configuration and family_history; migrate legacy config"
        )
    track = base_round.validate_track(config["base_tier"])
    model = models.validate_spec(config["model"]) if "model" in config else models.KEV
    if model not in models.d3_family():
        raise ValueError("the base-tier intake scores d3-family candidates only")
    found = {}
    root = Path(directory) / config["base_round_dir"]
    for path in sorted(p for p in root.iterdir() if (p / "manifest.json").is_file()):
        found[frozenset(base_round.draw(path).items())] = path
    draws = []
    for item in track["draws"]:
        if frozenset(item.items()) not in found:
            raise ValueError(f"base_round_dir has no sealed draw {item['manifest_sha256'][:12]}")
        draws.append((item, found[frozenset(item.items())]))
    reference = Path(directory) / config["base_reference"]
    if (
        models.checkpoint_model(reference) != model
        or zils.checkpoint_hash(reference) != track["incumbent_sha256"]
    ):
        raise ValueError("base_reference is not the track's incumbent checkpoint")
    from . import base_exam

    base_exam.disjoint([base_exam.audit(path) for _, path in draws])
    return draws


def base_tier_epoch(
    config, directory, work, registry, args, draws, *, wait_for_capacity=False, lease=None
):
    """Evaluate immutable checkpoints; preserve original timing and private evidence on reuse."""
    from . import base_exam, base_round, benchmark, commitments

    if not config.get("pool_id") or not config.get("family_history"):
        raise ValueError(
            "new base elections require pool configuration and family_history; migrate the legacy config"
        )
    from . import base_pilot

    base_pilot.admit(config, directory, draws)
    track = base_round.validate_track(config["base_tier"])
    model = models.validate_spec(config["model"])
    history = json.loads((Path(directory) / config["family_history"]).read_text())
    active = {
        "pool_id": config["pool_id"],
        "model": models.spec(model),
        **track,
        "draw_roots": {item["manifest_sha256"]: str(path) for item, path in draws},
        "history": history,
    }
    audited = [base_exam.audit(path) for _, path in draws]
    base_exam.disjoint(audited)
    if any(m["pool_id"] != config["pool_id"] for m, _ in audited):
        raise ValueError("exam belongs to another pool")
    validate_candidate = models.text_runtime(model).validate_candidate
    store = Path(directory) / "base-tier" / "store"
    sidecar = store.parent / "decision.json"
    sidecar.unlink(missing_ok=True)
    store.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempt = work / ("base-tier-" + uuid.uuid4().hex)
    attempt.mkdir(mode=0o700)
    reference = (Path(directory) / config["base_reference"]).resolve()
    if (
        zils.checkpoint_hash(reference) != track["incumbent_sha256"]
        or models.checkpoint_model(reference) != model
    ):
        raise ValueError("base_reference is not the active incumbent")
    entries = {INCUMBENT_UID: (track["incumbent_sha256"], reference)}
    rejected, claims = {}, {}
    for uid, original in sorted(registry.items()):
        claim = dict(original)
        if "error" in claim:
            rejected[uid] = claim["error"][:400]
            continue
        if claim["sha256"] == track["incumbent_sha256"]:
            rejected[uid] = "candidate is the incumbent"
            continue
        if not claim.get("hotkey"):
            rejected[uid] = "commitment has no frozen hotkey"
            continue
        claims[uid] = claim
        target = store / claim["sha256"] / "checkpoint"
        if not target.exists():
            fetched = target.with_name("fetch-" + uuid.uuid4().hex)
            fetched.parent.mkdir(mode=0o700, exist_ok=True)
            try:
                commitments.fetch(claim, fetched, model=model)
                validate_candidate(fetched, model)
                fetched.rename(target)
            except (ValueError, OSError, http.client.HTTPException) as error:
                shutil.rmtree(fetched, ignore_errors=True)
                if "base_validation_queue" in config and not isinstance(error, ValueError):
                    raise
                rejected[uid] = str(error)[:400]
                continue
            finally:
                if fetched.exists():
                    shutil.rmtree(fetched, ignore_errors=True)
        if zils.checkpoint_hash(target) != claim["sha256"]:
            raise ValueError("cached checkpoint changed")
        entries[uid] = (claim["sha256"], target)
        claims[uid] = claim
    decisions = []
    for item, path in draws:
        _, cases = base_exam.audit(path)
        key = item["manifest_sha256"]
        if base_round.draw(path) != item:
            raise ValueError("draw identity changed")
        selected = dict(entries)
        confirmation_after = None
        if decisions:
            first = decisions[0]
            inc_end = max(v["completed_at"] for v in first["incumbent"]["evaluation"].values())
            candidates = {c["sha256"]: c for c in first["candidates"] if "evaluation" in c}
            selected = {
                u: pair
                for u, pair in entries.items()
                if u == INCUMBENT_UID
                or pair[0] in candidates
                and time.time()
                >= max(
                    inc_end,
                    *(v["completed_at"] for v in candidates[pair[0]]["evaluation"].values()),
                )
                + 12 * 3600
            }
            if time.time() < inc_end + 12 * 3600 or len(selected) == 1 and len(entries) > 1:
                break
            confirmation_after = (
                max(
                    [inc_end]
                    + [
                        v["completed_at"]
                        for uid, (sha, _) in selected.items()
                        if uid != INCUMBENT_UID
                        for v in candidates[sha]["evaluation"].values()
                    ]
                )
                + 12 * 3600
            )

        def scores(sha256, split):
            cache_key = key
            if sha256 == track["incumbent_sha256"] and confirmation_after is not None:
                # Keep earlier reports immutable while later entrants get timely comparisons.
                cache_key += "-after-" + base_round.dataset_hash(confirmation_after)
            return store / sha256 / f"{cache_key}-{split}.json"

        rows = {s: {} for s in base_round.SPLITS}
        for split in base_round.SPLITS:
            pending = [
                {"uid": uid, "checkpoint": str(checkpoint), "sha256": sha}
                for uid, (sha, checkpoint) in sorted(selected.items())
                if not scores(sha, split).exists()
            ]
            if pending:
                base_exam.audit(path, history=history, as_of=time.time())
                started = time.time()
                report = run_evaluate(
                    args,
                    attempt,
                    f"{key[:12]}-{split}",
                    pending,
                    path / f"{split}.jsonl",
                    base_revision=models.spec(model)["base_revision"],
                    wait_for_capacity=wait_for_capacity,
                    capacity_model=model if "base_validation_queue" in config else None,
                    lease=lease,
                )
                completed = time.time()
                # Re-read the current history in case publication happened during inference.
                history = json.loads((Path(directory) / config["family_history"]).read_text())
                active["history"] = history
                base_exam.audit(path, history=history, as_of=completed)
                returned = {row["uid"]: row for row in report["miners"]}
                for submission in pending:
                    uid, sha = submission["uid"], submission["sha256"]
                    row = returned.get(uid)
                    try:
                        if (
                            not row
                            or row["status"] != "evaluated"
                            or row["sha256"] != sha
                            or report["dataset_sha256"] != base_round.dataset_hash(cases[split])
                        ):
                            raise ValueError("missing or invalid evaluation")
                        zils.score(cases[split], row["predictions"])
                    except (ValueError, KeyError, TypeError) as error:
                        if uid == INCUMBENT_UID:
                            raise RuntimeError("incumbent evaluation failed") from error
                        rejected[uid] = "invalid candidate evaluation"
                        continue
                    target = scores(sha, split)
                    target.parent.mkdir(mode=0o700, exist_ok=True)
                    wire.write_json(
                        target,
                        {
                            "dataset_sha256": report["dataset_sha256"],
                            "miners": [row],
                            "manifest_sha256": key,
                            "checkpoint_sha256": sha,
                            "split": split,
                            "started_at": started,
                            "completed_at": completed,
                        },
                    )
            for uid, (sha, _) in selected.items():
                target = scores(sha, split)
                if not target.exists():
                    continue
                stored = json.loads(target.read_text())
                for field, expected in (
                    ("manifest_sha256", key),
                    ("checkpoint_sha256", sha),
                    ("split", split),
                ):
                    if stored.get(field) != expected:
                        raise ValueError(
                            "cached evaluation binding changed; old caches cannot qualify"
                        )
                start, end = (
                    base_exam._time(stored.get(k)) for k in ("started_at", "completed_at")
                )
                if start > end:
                    raise ValueError("cached evaluation time reversed")
                base_exam.audit(path, history=history, as_of=end)
                found = base_round.rows_for(stored, cases[split])
                if len(found) != 1:
                    raise ValueError("cached evaluation must contain one checkpoint")
                row = next(iter(found.values()))
                if row["sha256"] != sha:
                    raise ValueError("cached evaluation checkpoint changed")
                rows[split][uid] = {
                    **row,
                    "uid": uid,
                    "evaluation": {
                        "path": str(target.resolve()),
                        "sha256": benchmark.file_hash(target),
                        "started_at": start,
                        "completed_at": end,
                    },
                }
        decision = base_round.decide(
            cases,
            INCUMBENT_UID,
            rows["calibration"],
            rows["test"],
            track["policy"],
            draw_identity=item,
            commitments={u: c for u, c in claims.items() if u in selected or u not in entries},
        )
        decision.update(
            pool_id=config["pool_id"],
            model=models.spec(model),
            round_id=work.name,
            rejected={str(u): e for u, e in rejected.items()},
            evaluated_at=max(
                v["completed_at"]
                for r in rows.values()
                for row in r.values()
                for v in [row["evaluation"]]
            ),
        )
        decisions.append(decision)
    if len(decisions) > 1:
        ready = base_round.confirm_candidates(decisions[0], decisions[-1], active=active)
        decisions[-1]["confirmed_candidates"] = ready
        decisions[-1]["confirmed_winner"] = (
            {k: ready[0][k] for k in ("uid", "sha256")} if ready else None
        )
    for decision in decisions:
        wire.write_json(attempt / f"decision-{decision['manifest_sha256'][:12]}.json", decision)
        record = store.parent / "decisions" / (base_round.dataset_hash(decision) + ".json")
        record.parent.mkdir(mode=0o700, exist_ok=True)
        if not record.exists():
            wire.write_json(record, decision)
    wire.write_json(attempt / "decision.json", base_round.public_decision(decisions[-1]))
    os.replace(attempt / "decision.json", sidecar)
    return decisions[-1]


def accept(config, current, members, registry, rounds, message, source, public_endpoints):
    """Register one signed adapter announcement in the collecting round. Base candidates are not
    announced here: they are chain commitments read at the round's preflight block."""
    if current.get("status") != "collecting":
        return 409, {"error": "round is not collecting"}
    claim = message.get("claim")
    if not isinstance(claim, dict):
        raise ValueError("claim must be an object")
    uid = claim.get("uid")
    if type(uid) is not int or uid not in members:
        raise ValueError("unknown miner")
    if not public_endpoints and "port" not in config.get("members", {}).get(str(uid), {}):
        return 403, {"error": "this validator pays adapter work from queue evidence"}
    if claim.get("round_id") != current["round_id"]:
        return 409, {"error": "round has changed; request the current round"}
    # Pin downloads to the authenticated caller's TCP source address. The artifact port
    # is the operator-assigned one for static members, or the signed claim's for public miners.
    port = (
        wire.endpoint_port(claim.get("endpoint"))[1]
        if public_endpoints
        else config["members"][str(uid)]["port"]
    )
    updated = dict(registry)
    wire.register(
        message,
        current["round_id"],
        members,
        updated,
        endpoints={uid: f"http://{source}:{port}"},
        job_sha256=config.get("job_sha256"),
        public=public_endpoints,
    )
    path = rounds / current["round_id"] / "submissions" / f"{uid}.json"
    path.parent.mkdir(mode=0o700, exist_ok=True)
    if not path.exists():
        wire.write_json(path, message)
    registry.clear()
    registry.update(updated)
    return 200, {"status": "accepted", "round_id": current["round_id"]}


def round_config(config, admission, members):
    """The roster every scorer and publisher reads: the static one, or this round's chain snapshot."""
    if admission != "chain":
        return config
    return {**config, "members": {str(uid): {"hotkey": hotkey} for uid, hotkey in members.items()}}


def admission_settings(config):
    """Static member allowlist with private transport, or chain-derived members. Chain members
    announce adapter artifacts to a public endpoint, unless adapter weights come from queue
    evidence: then the validator takes no announcements and needs no inbound address."""
    mode = config.get("admission", "members")
    public = bool(config.get("public_endpoints"))
    if mode not in ("members", "chain"):
        raise ValueError("admission must be 'members' or 'chain'")
    if mode == "chain" and "chain" not in config:
        raise ValueError("chain admission requires a chain configuration")
    if mode == "chain" and not public and "adapter_evidence" not in config:
        raise ValueError("chain admission requires public_endpoints or adapter_evidence")
    if mode == "members" and public and any("port" not in m for m in config["members"].values()):
        raise ValueError("members need artifact ports")
    return mode, public


def validator(config, directory, args):
    from . import feature_round, feature_tracks

    if "base_validation_queue" in config:
        from . import base_queue

        base_queue.policy(config)
    key = signing_key(config)
    admission, public_endpoints = admission_settings(config)
    members = (
        {int(uid): item["hotkey"] for uid, item in config["members"].items()}
        if admission == "members"
        else {}
    )
    condition = threading.Condition()
    current, registry, acks = {}, {}, set()
    draws = base_draws(config, directory)
    pools = pool_rounds(config, directory)
    features = feature_round.rounds(config, directory)
    recent = collections.defaultdict(collections.deque)

    def throttled(address):
        now, queue = time.time(), recent[address]
        while queue and now - queue[0] > 60:
            queue.popleft()
        if len(queue) >= wire.SUBMITS_PER_MINUTE:
            return True
        queue.append(now)
        return False

    rounds = directory / "state/rounds"
    rounds.mkdir(mode=0o700, parents=True, exist_ok=True)

    class API(wire.Handler):
        def send(self, code, payload):
            self.reply(code, signed(payload, key))

        def do_GET(self):
            with condition:
                if self.path == "/round":
                    self.send(200, current or {"kind": "idle"})
                    return
                match = re.fullmatch(r"/results/([a-f0-9]{32})", self.path)
                if match:
                    path = rounds / match[1] / "result.json"
                    if path.exists():
                        self.reply(200, json.loads(path.read_text()))
                        return
                self.send(404, {"error": "result not ready"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if not 0 < length <= wire.MAX_ANNOUNCEMENT or self.headers.get("Transfer-Encoding"):
                    raise ValueError("invalid message size")
                message = json.loads(self.rfile.read(length))
                if not isinstance(message, dict):
                    raise ValueError("message must be an object")
                with condition:
                    if self.path == "/submit":
                        if public_endpoints and throttled(self.client_address[0]):
                            self.send(429, {"error": "too many submissions from this address"})
                            return
                        code, payload = accept(
                            config,
                            current,
                            members,
                            registry,
                            rounds,
                            message,
                            self.client_address[0],
                            public_endpoints,
                        )
                        self.send(code, payload)
                        if code != 200:
                            return
                    elif self.path == "/ack":
                        if not isinstance(message.get("payload"), dict):
                            raise ValueError("payload must be an object")
                        uid = message["payload"].get("uid")
                        if type(uid) is not int or uid not in members:
                            raise ValueError("unknown miner")
                        payload = verified(message, members[uid])
                        if (
                            payload
                            != {"kind": "ack", "uid": uid, "round_id": current.get("round_id")}
                            or current.get("status") != "complete"
                        ):
                            self.send(409, {"error": "ack does not match completed round"})
                            return
                        acks.add(uid)
                        self.send(200, {"status": "acknowledged"})
                    else:
                        self.send(404, {"error": "unknown endpoint"})
                        return
                    condition.notify_all()
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.send(400, {"error": str(error)[:200]})

    address = urlsplit(config["validator"])
    with wire.local_server(API, address.hostname, address.port):
        print(
            f"validator: listening at {config['validator']} for {len(members)} miners", flush=True
        )
        completed = 0
        while not args.rounds or completed < args.rounds:
            chain_state = None
            if "chain" in config:
                from . import testnet

                with testnet.connect(config) as sub:
                    chain_state = testnet.preflight(config, sub)
            pending = sorted(
                (
                    p
                    for p in rounds.iterdir()
                    if p.is_dir() and (p / "job.json").exists() and not (p / "result.json").exists()
                ),
                key=lambda p: p.stat().st_mtime,
            )
            if pending:
                work = pending[-1]
                job = json.loads((work / "job.json").read_text())
            else:
                rid = uuid.uuid4().hex
                work = rounds / rid
                work.mkdir(mode=0o700)
                (work / "submissions").mkdir(mode=0o700)
                job = {
                    "kind": "round",
                    "round_id": rid,
                    "status": "collecting",
                    "deadline": time.time() + config["round_timeout"],
                    **{
                        k: config[k] for k in ("base_revision", "initial_sha256", "training_sha256")
                    },
                }
                if chain_state is not None:
                    job["chain"] = config["chain"]
                if "job_id" in config:
                    job.update(job_id=config["job_id"], job_sha256=config["job_sha256"])
                for name in ("model", "min_free_mib", "max_seconds"):
                    if name in config:
                        job[name] = config[name]
                if draws is not None:
                    job["base_tier"] = {
                        "incumbent_sha256": config["base_tier"]["incumbent_sha256"],
                        "draws": [item for item, _ in draws],
                    }
                if "base_pools" in config:
                    from . import model_pools

                    job["base_pools"] = model_pools.validate(config["base_pools"])
                if "feature_tracks" in config:
                    job["feature_tracks"] = feature_tracks.validate(config["feature_tracks"])
                    job["feature_scope"] = feature_tracks.SCOPE
                wire.write_json(work / "job.json", job)
            if "base_pools" in config:
                from . import model_pools

                if job.get("base_pools") != model_pools.validate(config["base_pools"]):
                    raise ValueError("pending round differs from installed pool configuration")
            expected_features = (
                feature_tracks.validate(config["feature_tracks"])
                if "feature_tracks" in config
                else None
            )
            if job.get("feature_tracks") != expected_features:
                raise ValueError("pending round differs from installed feature configuration")
            for name in (
                "base_revision",
                "initial_sha256",
                "training_sha256",
                "job_id",
                "job_sha256",
                "model",
                "min_free_mib",
                "max_seconds",
            ):
                if job.get(name) != config.get(name):
                    raise ValueError("pending round differs from the installed job configuration")
            if chain_state is not None and not (work / "chain-snapshot.json").exists():
                wire.write_json(work / "chain-snapshot.json", chain_state)
            if admission == "chain":
                # Membership is the chain snapshot frozen with this round, never a live lookup mid-round.
                snapshot = json.loads((work / "chain-snapshot.json").read_text())
                with condition:
                    members.clear()
                    members.update(
                        {int(uid): hotkey for uid, hotkey in snapshot["members"].items()}
                    )
            settings_for_round = round_config(config, admission, members)
            with condition:
                registry.clear()
                acks.clear()
                current.clear()
                current.update(job)
                for path in (work / "submissions").glob("*.json"):
                    message = json.loads(path.read_text())
                    c = message["claim"]
                    wire.register(
                        message,
                        job["round_id"],
                        members,
                        registry,
                        endpoints={c["uid"]: c["endpoint"]},
                        job_sha256=config.get("job_sha256"),
                        public=public_endpoints,
                    )
                while len(registry) < len(members) and time.time() < job["deadline"]:
                    condition.wait(min(1, max(0.01, job["deadline"] - time.time())))
                current["status"] = "evaluating"
            candidates = {}
            if draws is not None:
                from . import commitments

                candidates = commitments.candidates(
                    json.loads((work / "chain-snapshot.json").read_text()),
                    config["base_tier"]["incumbent_sha256"],
                )
            print(
                f"validator: evaluating round {job['round_id']} ({len(registry)}/{len(members)} "
                f"submissions, {len(candidates)} base commitments)",
                flush=True,
            )
            report = (
                json.loads((work / "report.json").read_text())
                if (work / "report.json").exists()
                else evaluate_round(
                    settings_for_round, directory, work, registry, args, wait_for_capacity=True
                )
            )
            if draws is not None and not (work / "base-tier.json").exists():
                base_tier_epoch(
                    settings_for_round,
                    directory,
                    work,
                    candidates,
                    args,
                    draws,
                    wait_for_capacity=True,
                )
            report = attach_base_tier(settings_for_round, directory, work, report)
            if pools and not (work / "base-pools.json").exists():
                base_pools_epoch(
                    settings_for_round,
                    directory,
                    work,
                    json.loads((work / "chain-snapshot.json").read_text()),
                    args,
                    pools,
                )
            report = attach_base_pools(settings_for_round, directory, work, report)
            if features and not (work / "feature-tracks.json").exists():
                feature_round.epoch(
                    settings_for_round,
                    directory,
                    work,
                    json.loads((work / "chain-snapshot.json").read_text()),
                    args,
                    features,
                )
            report = feature_round.attach(settings_for_round, directory, work, report)
            report = attach_adapter_evidence(settings_for_round, work, report)
            fields = (
                "uid",
                "status",
                "skill",
                "accuracy",
                "brier",
                "confident_errors",
                "median_ms",
                "p95_ms",
            )
            public = {
                "model": report.get("model", models.spec(models.KEV)),
                "kind": "result",
                "round_id": job["round_id"],
                "weights": {str(k): v for k, v in report["weights"].items()},
                "miners": [{k: m[k] for k in fields if k in m} for m in report["miners"]],
                "chain_write": False,
            }
            if report.get("base_tier") is not None:
                # Aggregate base-tier metrics for miners; per-case predictions never leave the store.
                public["base_tier"] = report["base_tier"]
            if "base_pools" in report:
                public["base_pools"] = report["base_pools"]
            if "feature_tracks" in report:
                public["feature_tracks"] = report["feature_tracks"]
            if chain_state is not None:
                from bittensor.wallet import Wallet

                with locked(work / "chain.lock"), testnet.connect(config) as sub:
                    outcome = testnet.publish_round(
                        settings_for_round,
                        work,
                        report,
                        sub,
                        Wallet(**config["wallet"]),
                        publish=args.publish_weights,
                    )
                public.update(chain=outcome, chain_write=outcome["chain_write"])
                if "tracks" in outcome or "adapter_source" in outcome:
                    # Miners see the published vector, not the frozen-job round alone.
                    public["weights"] = outcome["requested_weights"]
                print(f"validator: chain weights {outcome['status']}", flush=True)
            wire.write_json(work / "result.json", signed(public, key))
            with condition:
                current["status"] = "complete"
                deadline = time.monotonic() + config["result_grace"]
                while not set(registry) <= acks and time.monotonic() < deadline:
                    condition.wait(min(1, max(0.01, deadline - time.monotonic())))
            completed += 1
            print(
                f"validator: completed round {completed}; proposed weights {public['weights']}",
                flush=True,
            )
            if not args.rounds or completed < args.rounds:
                time.sleep(config["round_pause"])
