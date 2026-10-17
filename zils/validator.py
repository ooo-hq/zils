"""Collect checkpoints, calibrate and score them, and optionally publish testnet weights."""

import collections
import http.client
import json
import re
import shutil
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit

import zils

from . import models, protocol as wire, settings
from .runtime import CapacityUnavailable, digest, locked, run_child, signed, signing_key, verified


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
            (fetch_checkpoint or wire.fetch_checkpoint)(claim, path, **options)
            if models.checkpoint_model(path) != model:
                raise ValueError("Submission differs from the job model profile")
            entries.append(zils.submission(path, uid))
        except (ValueError, OSError, http.client.HTTPException) as error:
            failed[uid] = str(error)

    def evaluate(submissions, cases, name, base_revision=None):
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
            base_revision or config["base_revision"],
            "--runner-python",
            settings.required("ZILS_IMAGE_RUNTIME_PYTHON")
            if model == models.IMAJEV
            else args.runtime_python,
            "--device",
            args.device,
            "--report",
            str(report),
        ]
        if frozen_images:
            command.extend(["--images", str(directory / "images")])
        waiting = False
        while True:
            try:
                run_child(
                    command,
                    attempt / f"{name}.log",
                    args.device,
                    timeout=120 + 600 * len(submissions),
                    **(
                        {"model": model, "minimum_mib": config.get("min_free_mib", 0)}
                        if model == models.IMAJEV
                        else {}
                    ),
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
                        if current.get("status") != "collecting":
                            self.send(409, {"error": "round is not collecting"})
                            return
                        claim = message.get("claim")
                        if not isinstance(claim, dict):
                            raise ValueError("claim must be an object")
                        uid = claim.get("uid")
                        if type(uid) is not int or uid not in members:
                            raise ValueError("unknown miner")
                        if claim.get("round_id") != current["round_id"]:
                            self.send(
                                409, {"error": "round has changed; request the current round"}
                            )
                            return
                        # Pin downloads to the authenticated caller's TCP source address. The artifact port
                        # is the operator-assigned one for static members, or the signed claim's for public miners.
                        port = (
                            wire.endpoint_port(claim.get("endpoint"))[1]
                            if public_endpoints
                            else config["members"][str(uid)]["port"]
                        )
                        endpoint = f"http://{self.client_address[0]}:{port}"
                        updated = dict(registry)
                        wire.register(
                            message,
                            current["round_id"],
                            members,
                            updated,
                            endpoints={uid: endpoint},
                            job_sha256=config.get("job_sha256"),
                            public=public_endpoints,
                        )
                        path = rounds / current["round_id"] / "submissions" / f"{uid}.json"
                        if not path.exists():
                            wire.write_json(path, message)
                        registry.clear()
                        registry.update(updated)
                        self.send(200, {"status": "accepted", "round_id": current["round_id"]})
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
            print(
                f"validator: evaluating round {job['round_id']} ({len(registry)}/{len(members)} submissions)",
                flush=True,
            )
            report = (
                json.loads((work / "report.json").read_text())
                if (work / "report.json").exists()
                else evaluate_round(config, directory, work, registry, args, wait_for_capacity=True)
            )
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
            if chain_state is not None:
                from bittensor.wallet import Wallet

                with locked(work / "chain.lock"), testnet.connect(config) as sub:
                    outcome = testnet.publish_round(
                        config,
                        work,
                        report,
                        sub,
                        Wallet(**config["wallet"]),
                        publish=args.publish_weights,
                    )
                public.update(chain=outcome, chain_write=outcome["chain_write"])
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
