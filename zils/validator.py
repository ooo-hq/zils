"""Collect checkpoints, calibrate and score them, and optionally publish testnet weights."""

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

from . import models, protocol as wire, settings
from .runtime import CapacityUnavailable, digest, locked, run_child, signed, signing_key, verified

# The incumbent base is scored as a row outside every roster; the publisher refuses a roster uid there.
INCUMBENT_UID = 65535
SUBMISSION_DIRS = {"adapter": "submissions", "candidate": "base-submissions"}


def run_evaluate(
    args,
    attempt,
    name,
    submissions,
    cases,
    *,
    base_revision,
    model,
    images=None,
    minimum_mib=0,
    wait_for_capacity=False,
):
    """One sandboxed `zils evaluate` of `submissions` on `cases`, deferred while GPU capacity is unavailable."""
    manifest = attempt / f"{name}-submissions.json"
    wire.write_json(manifest, submissions)
    report = attempt / f"{name}.json"
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
        settings.required("ZILS_IMAGE_RUNTIME_PYTHON")
        if model == models.IMAJEV
        else args.runtime_python,
        "--device",
        args.device,
        "--report",
        str(report),
    ]
    if images is not None:
        command.extend(["--images", str(images)])
    waiting = False
    while True:
        try:
            run_child(
                command,
                attempt / f"{name}.log",
                args.device,
                timeout=120 + 600 * len(submissions),
                **({"model": model, "minimum_mib": minimum_mib} if model == models.IMAJEV else {}),
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
    frozen_images = model == models.IMAJEV or "assets" in manifest  # verified <sha256>.png cache
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
            model=model,
            images=directory / "images" if frozen_images else None,
            minimum_mib=config.get("min_free_mib", 0),
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
        delivery = jobs.select(baseline, report["miners"], manifest["acceptance"], model=model)
        if delivery["status"] == "accepted":
            winner = next(entry for entry in fitted if entry["uid"] == delivery["uid"])
            release = attempt / "accepted-model"
            zils.stage(winner, release)
            delivery["checkpoint"] = str(release.relative_to(work))
            evidence = {}
            if model == models.IMAJEV:
                from .image_metrics import public_metrics

                selected = next(row for row in report["miners"] if row["uid"] == delivery["uid"])
                evidence = {
                    "image_metrics": {
                        "baseline": public_metrics(baseline),
                        "candidate": public_metrics(selected),
                    }
                }
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
                    **evidence,
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


def base_draws(config, directory):
    """The sealed draws this validator scores itself, audited now and ordered as the track lists them
    (the first decides, the last confirms). None when the config has no `base_round_dir`."""
    if "base_round_dir" not in config:
        return None
    from . import base_round

    if "base_tier" not in config or "base_reference" not in config:
        raise ValueError("base_round_dir requires a base_tier track and base_reference")
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
    return draws


def base_tier_epoch(
    config,
    directory,
    work,
    registry,
    args,
    draws,
    *,
    public_endpoints=False,
    wait_for_capacity=False,
):
    """Score this round's base candidates on the sealed draws and write the decision sidecar.

    Every distinct checkpoint hash is downloaded once into a content-addressed store and scored once
    per draw and split; a hash announced again in a later round costs no download and no GPU time
    (the cross-round cap). The decision covers the candidates announced this round. The old sidecar
    is removed first, so an epoch that fails pays nobody. Per-case predictions stay in the store."""
    from . import base_round

    track = base_round.validate_track(config["base_tier"])
    model = models.validate_spec(config["model"])
    validate_candidate = models.text_runtime(model).validate_candidate
    store = Path(directory) / "base-tier" / "store"
    sidecar = Path(directory) / "base-tier" / "decision.json"
    sidecar.unlink(missing_ok=True)
    store.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempt = work / ("base-tier-" + uuid.uuid4().hex)
    attempt.mkdir(mode=0o700)
    reference = (Path(directory) / config["base_reference"]).resolve()
    entries = {INCUMBENT_UID: (track["incumbent_sha256"], reference)}
    rejected = {}
    for uid, message in sorted(registry.items()):
        claim = message["claim"]
        target = store / claim["sha256"] / "checkpoint"
        if not target.exists():
            fetched = target.with_name("fetch-" + uuid.uuid4().hex)
            fetched.parent.mkdir(mode=0o700, exist_ok=True)
            try:
                wire.fetch_checkpoint(
                    claim,
                    fetched,
                    expected_endpoint=claim["endpoint"],
                    round_scoped=True,
                    model=model,
                    kind="candidate",
                    public=public_endpoints,
                )
                validate_candidate(fetched, model)
                fetched.rename(target)
            except (ValueError, OSError, http.client.HTTPException) as error:
                shutil.rmtree(fetched, ignore_errors=True)
                rejected[uid] = str(error)[:400]
                continue
        entries[uid] = (claim["sha256"], target)
    decisions = []
    for item, path in draws:
        _, cases = base_round.audit(path)
        key = item["manifest_sha256"]

        def scores(sha256, split):
            return store / sha256 / f"{key}-{split}.json"

        for split in base_round.SPLITS:
            pending = [
                {"uid": uid, "checkpoint": str(checkpoint), "sha256": sha256}
                for uid, (sha256, checkpoint) in sorted(entries.items())
                if not scores(sha256, split).exists()
            ]
            if not pending:
                continue
            report = run_evaluate(
                args,
                attempt,
                f"{key[:12]}-{split}",
                pending,
                path / f"{split}.jsonl",
                base_revision=models.spec(model)["base_revision"],
                model=model,
                wait_for_capacity=wait_for_capacity,
            )
            for row in report["miners"]:
                if row["uid"] == INCUMBENT_UID and row["status"] != "evaluated":
                    raise RuntimeError(
                        "incumbent evaluation failed; no base candidate can be scored: "
                        + row.get("error", "no error recorded")
                    )
                target = scores(entries[row["uid"]][0], split)
                target.parent.mkdir(mode=0o700, exist_ok=True)
                wire.write_json(
                    target, {"dataset_sha256": report["dataset_sha256"], "miners": [row]}
                )
        rows = {}
        for split in base_round.SPLITS:
            rows[split] = {}
            for uid, (sha256, _) in entries.items():
                stored = json.loads(scores(sha256, split).read_text())
                for row in base_round.rows_for(stored, cases[split]).values():
                    rows[split][uid] = {**row, "uid": uid}
                if uid not in rows[split]:
                    rejected[uid] = stored["miners"][0].get("error", "evaluation failed")[:400]
        decision = base_round.decide(
            cases, INCUMBENT_UID, rows["calibration"], rows["test"], track["policy"]
        )
        decision.update(
            manifest_sha256=key,
            round_id=work.name,
            rejected={str(uid): error for uid, error in sorted(rejected.items())},
        )
        decisions.append(decision)
    if len(decisions) > 1:
        decisions[-1]["confirmed_winner"] = base_round.confirm(decisions[0], decisions[-1])
    for decision in decisions:
        wire.write_json(attempt / f"decision-{decision['manifest_sha256'][:12]}.json", decision)
    wire.write_json(attempt / "decision.json", decisions[-1])
    os.replace(attempt / "decision.json", sidecar)
    return decisions[-1]


def accept(config, current, members, registries, rounds, message, source, public_endpoints):
    """Route one signed announcement into the collecting round: an adapter by default, a base
    candidate when the claim says `"kind": "candidate"` and this validator scores candidates."""
    if current.get("status") != "collecting":
        return 409, {"error": "round is not collecting"}
    claim = message.get("claim")
    if not isinstance(claim, dict):
        raise ValueError("claim must be an object")
    uid = claim.get("uid")
    if type(uid) is not int or uid not in members:
        raise ValueError("unknown miner")
    if claim.get("round_id") != current["round_id"]:
        return 409, {"error": "round has changed; request the current round"}
    kind = "candidate" if claim.get("kind") == "candidate" else "adapter"
    if kind == "candidate":
        if "base_tier" not in current:
            return 409, {"error": "this validator does not collect base candidates"}
        if claim.get("sha256") == current["base_tier"]["incumbent_sha256"]:
            raise ValueError("the incumbent base is not a candidate")
    # Pin downloads to the authenticated caller's TCP source address. The artifact port
    # is the operator-assigned one for static members, or the signed claim's for public miners.
    port = (
        wire.endpoint_port(claim.get("endpoint"))[1]
        if public_endpoints
        else config["members"][str(uid)]["port"]
    )
    registry = registries[kind]
    updated = dict(registry)
    wire.register(
        message,
        current["round_id"],
        members,
        updated,
        endpoints={uid: f"http://{source}:{port}"},
        job_sha256=config.get("job_sha256") if kind == "adapter" else None,
        public=public_endpoints,
        kind=kind,
    )
    path = rounds / current["round_id"] / SUBMISSION_DIRS[kind] / f"{uid}.json"
    path.parent.mkdir(mode=0o700, exist_ok=True)
    if not path.exists():
        wire.write_json(path, message)
    registry.clear()
    registry.update(updated)
    return 200, {"status": "accepted", "round_id": current["round_id"], "kind": kind}


def round_config(config, admission, members):
    """The roster every scorer and publisher reads: the static one, or this round's chain snapshot."""
    if admission != "chain":
        return config
    return {**config, "members": {str(uid): {"hotkey": hotkey} for uid, hotkey in members.items()}}


def admission_settings(config):
    """Static member allowlist with private transport, or chain-derived members with public endpoints."""
    mode = config.get("admission", "members")
    public = bool(config.get("public_endpoints"))
    if mode not in ("members", "chain"):
        raise ValueError("admission must be 'members' or 'chain'")
    if mode == "chain" and ("chain" not in config or not public):
        raise ValueError("chain admission requires a testnet configuration and public_endpoints")
    if mode == "members" and public and any("port" not in m for m in config["members"].values()):
        raise ValueError("members need artifact ports")
    return mode, public


def validator(config, directory, args):
    key = signing_key(config)
    admission, public_endpoints = admission_settings(config)
    members = (
        {int(uid): item["hotkey"] for uid, item in config["members"].items()}
        if admission == "members"
        else {}
    )
    condition = threading.Condition()
    current, registry, acks = {}, {}, set()
    registries = {"adapter": registry, "candidate": {}}
    draws = base_draws(config, directory)
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
                            registries,
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
                for folder in SUBMISSION_DIRS.values():
                    (work / folder).mkdir(mode=0o700)
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
                wire.write_json(work / "job.json", job)
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
                for kind in registries:
                    registries[kind].clear()
                acks.clear()
                current.clear()
                current.update(job)
                for kind, folder in SUBMISSION_DIRS.items():
                    for path in (work / folder).glob("*.json"):
                        message = json.loads(path.read_text())
                        c = message["claim"]
                        wire.register(
                            message,
                            job["round_id"],
                            members,
                            registries[kind],
                            endpoints={c["uid"]: c["endpoint"]},
                            job_sha256=config.get("job_sha256") if kind == "adapter" else None,
                            public=public_endpoints,
                            kind=kind,
                        )
                while len(registry) < len(members) and time.time() < job["deadline"]:
                    condition.wait(min(1, max(0.01, job["deadline"] - time.time())))
                current["status"] = "evaluating"
            print(
                f"validator: evaluating round {job['round_id']} ({len(registry)}/{len(members)} "
                f"submissions, {len(registries['candidate'])} base candidates)",
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
                    registries["candidate"],
                    args,
                    draws,
                    public_endpoints=public_endpoints,
                    wait_for_capacity=True,
                )
            report = attach_base_tier(settings_for_round, directory, work, report)
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
                if "tracks" in outcome:
                    # Miners see the blended vector, not the adapter round alone.
                    public["weights"] = outcome["requested_weights"]
                print(f"validator: testnet weights {outcome['status']}", flush=True)
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
