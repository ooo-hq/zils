"""Private evidence is rechecked through readiness, frozen windows and staging."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_base_exam import inputs, sealed
from tests.test_base_metrics import predictions
from tests.test_regression_suite import election_fixture
from zils import base_metrics, base_release, base_round, benchmark, commitments, models

PAIRED = base_metrics.paired_improvement


def fast_pair(*args, **kwargs):
    return PAIRED(*args, **{**kwargs, "resamples": 30})


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path.resolve()), "sha256": benchmark.file_hash(path)}


def decision(root, name, at, *, hashes=None, owners=None, regressed=False):
    hashes = hashes or {0: "a" * 64, 1: "b" * 64, 2: "c" * 64}
    owners = owners or {1: "miner-one", 2: "miner-two"}
    drawroot = root / name
    _, cases = sealed(drawroot, name)
    identity = base_round.draw(drawroot)
    claims = {
        u: {"sha256": h, "block": 20 + u, "hotkey": owners[u]} for u, h in hashes.items() if u
    }
    rows = {s: {} for s in cases}
    for s, data in cases.items():
        for uid, sha in hashes.items():

            def choose(c, i):
                import zils

                keys = zils.options(c["question"])
                wrong = i % (3 if uid == 0 or regressed else 8) == 0 and (uid != 2 or regressed)
                return next(k for k in keys if k != c["label"]) if wrong else c["label"]

            row = {
                "uid": uid,
                "sha256": sha,
                "status": "evaluated",
                "predictions": predictions(data, choose, confidence=0.8),
            }
            report = {
                "miners": [row],
                "dataset_sha256": base_round.dataset_hash(data),
                "manifest_sha256": identity["manifest_sha256"],
                "checkpoint_sha256": sha,
                "split": s,
                "started_at": at,
                "completed_at": at + 1,
            }
            evidence = save(root / "reports" / f"{name}-{uid}-{s}.json", report)
            rows[s][uid] = {
                **row,
                "evaluation": {**evidence, "started_at": at, "completed_at": at + 1},
            }
    result = base_round.decide(
        cases, 0, rows["calibration"], rows["test"], draw_identity=identity, commitments=claims
    )
    result.update(pool_id="d3-mini", model=models.spec(models.D3))
    snapshot = {
        "members": {str(u): owners[u] for u in owners},
        "commitments": {
            str(u): {
                "block": c["block"],
                "data": commitments.format_commitment(
                    c["sha256"], "d" * 40, f"miner/candidate-{u}"
                ),
            }
            for u, c in claims.items()
        },
    }
    return (
        result,
        save(root / f"{name}-decision.json", result),
        save(root / f"{name}-snapshot.json", snapshot),
    )


class BaseReleaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pair = patch.object(base_metrics, "paired_improvement", side_effect=fast_pair)
        self.pair.start()
        self.addCleanup(self.pair.stop)
        self.temperature = patch.object(base_round, "fit_temperature", return_value=1.0)
        self.temperature.start()
        self.addCleanup(self.temperature.stop)
        self.first, f, s1 = decision(self.root, "first", 100)
        self.second, s, s2 = decision(self.root, "second", 43302)
        self.suite, _ = election_fixture()
        self.active = {
            "pool_id": "d3-mini",
            "model": models.spec(models.D3),
            "policy": dict(base_round.DEFAULT_POLICY),
            "incumbent_sha256": "a" * 64,
            "kings": ["old"],
            "draws": [base_round.draw(self.root / n) for n in ("first", "second")],
            "draw_roots": {
                base_round.draw(self.root / n)["manifest_sha256"]: str(self.root / n)
                for n in ("first", "second")
            },
            "history": inputs()[2],
            "suite_manifest": self.suite,
            "suite_sha256": base_round.dataset_hash(self.suite),
            "pilot_sha256": "e" * 64,
            "blueprint_sha256": "f" * 64,
            "ready_dir": str(self.root / "ready"),
            "windows_dir": str(self.root / "windows"),
        }
        self.ready = []
        for uid, sha in ((1, "b" * 64), (2, "c" * 64)):
            _, suite = election_fixture(candidate=sha)
            evidence = {
                "first": f,
                "second": s,
                "snapshot_first": s1,
                "snapshot_second": s2,
                "suite": save(self.root / f"suite-{uid}.json", suite),
            }
            receipt = base_release.qualify(
                self.first,
                self.second,
                suite,
                candidate_sha256=sha,
                active=self.active,
                expected_suite=self.suite,
                evidence=evidence,
            )
            save(self.root / "ready" / (receipt["sha256"] + ".json"), receipt)
            self.ready.append(receipt)
        self.schedule = {
            "version": "zils-base-windows/v1",
            "anchor_utc": "1970-01-01T00:00:00Z",
            "interval_days": 14,
        }
        self.at = 14 * 86400
        self.channels = {"stable": "old", "latest": "prior"}

    def freeze(self):
        return base_release.freeze(
            self.ready,
            active=self.active,
            schedule=self.schedule,
            window_id="d3-mini:1",
            frozen_at=self.at,
            opening_channels=self.channels,
        )

    def closing(self):
        result, _, snapshot = decision(self.root, "closing", self.at + 10)
        return {"decision": result, "draw_root": str(self.root / "closing"), "snapshot": snapshot}

    def test_all_ready_candidates_survive_restart_and_common_closing(self):
        self.assertEqual({r["candidate"]["hotkey"] for r in self.ready}, {"miner-one", "miner-two"})
        frozen = self.freeze()
        self.assertEqual(frozen, self.freeze())
        closing = self.closing()
        members = json.loads(Path(closing["snapshot"]["path"]).read_text())["members"]
        result = base_release.close(
            frozen, closing, active=self.active, members=members, now=self.at + 100
        )
        self.assertEqual(result["winner"]["sha256"], "c" * 64)
        self.assertFalse(result["uploaded"])
        self.assertEqual(
            result,
            base_release.close(
                frozen, closing, active=self.active, members=members, now=self.at + 100
            ),
        )
        with self.assertRaisesRegex(ValueError, "inventory"):
            base_release.freeze(
                self.ready[:1],
                active=self.active,
                schedule=self.schedule,
                window_id="d3-mini:1",
                frozen_at=self.at,
                opening_channels=self.channels,
            )

    def test_stale_bindings_mutated_evidence_and_missing_roster_fail(self):
        with self.assertRaises(ValueError):
            base_release.freeze(
                self.ready,
                active={**self.active, "pilot_sha256": "0" * 64},
                schedule=self.schedule,
                window_id="d3-mini:1",
                frozen_at=self.at,
                opening_channels=self.channels,
            )
        frozen = self.freeze()
        closing = self.closing()
        members = json.loads(Path(closing["snapshot"]["path"]).read_text())["members"]
        bad = copy.deepcopy(closing)
        bad["decision"]["candidates"].pop()
        with self.assertRaises(ValueError):
            base_release.close(frozen, bad, active=self.active, members=members, now=self.at + 100)
        with self.assertRaises(ValueError):
            base_release.close(
                frozen,
                closing,
                active=self.active,
                members={"1": {"hotkey": "impostor"}},
                now=self.at + 100,
            )
        Path(self.first["incumbent"]["evaluation"]["test"]["path"]).write_text("{}")
        with self.assertRaises(ValueError):
            self.freeze()

    def test_window_retries_keep_recorded_times_and_reject_conflicting_inputs(self):
        frozen = self.freeze()
        self.assertEqual(
            frozen,
            base_release.freeze(
                self.ready,
                active=self.active,
                schedule=self.schedule,
                window_id="d3-mini:1",
                frozen_at=self.at + 1,
                opening_channels=self.channels,
            ),
        )
        closing = self.closing()
        members = base_release.evidence_read(closing["snapshot"])["members"]
        closed = base_release.close(
            frozen,
            closing,
            active=self.active,
            members=members,
            now=self.at + 100,
        )
        self.assertEqual(
            closed,
            base_release.close(
                frozen,
                closing,
                active=self.active,
                members=members,
                now=self.at + 101,
            ),
        )
        changed = copy.deepcopy(closing)
        changed["decision"]["round_id"] = "conflicting-attempt"
        with self.assertRaises(ValueError):
            base_release.close(
                frozen,
                changed,
                active=self.active,
                members=members,
                now=self.at + 102,
            )

    def test_cli_recovers_after_internal_receipt_before_output(self):
        import sys

        active = {**self.active, "settings": {"base_tier": {}}}
        config = self.root / "config.json"
        config.write_text("{}")
        schedule = self.root / "schedule.json"
        save(schedule, self.schedule)
        channels = self.root / "channels.json"
        save(channels, self.channels)
        frozen_file = self.root / "windows/d3-mini-1-frozen.json"
        closing = self.closing()
        original_persist = base_release.persist
        for command, at, extra, internal in (
            (
                "freeze",
                self.at,
                [
                    "--ready-dir",
                    self.active["ready_dir"],
                    "--schedule",
                    str(schedule),
                    "--channels",
                    str(channels),
                ],
                frozen_file,
            ),
            (
                "close",
                self.at + 100,
                [
                    "--frozen",
                    str(frozen_file),
                    "--closing-round",
                    closing["draw_root"],
                    "--chain-snapshot",
                    closing["snapshot"]["path"],
                ],
                self.root / "windows/d3-mini-1-closed.json",
            ),
        ):
            out = self.root / f"{command}-out.json"
            argv = [
                "base_release",
                command,
                "--config",
                str(config),
                "--pool",
                "d3-mini",
                "--out",
                str(out),
                *extra,
            ]

            def interrupted(path, value):
                result = original_persist(path, value)
                if Path(path) == internal:
                    raise RuntimeError("interrupted before output")
                return result

            with (
                patch.object(base_release, "active_context", return_value=active),
                patch(
                    "zils.validator.base_tier_epoch", return_value=closing["decision"]
                ) as evaluate,
                patch.object(sys, "argv", argv),
            ):
                with (
                    patch("zils.base_release.time.time", return_value=at),
                    patch.object(base_release, "persist", side_effect=interrupted),
                ):
                    with self.assertRaisesRegex(RuntimeError, "interrupted"):
                        base_release.main()
                self.assertFalse(out.exists())
                with patch("zils.base_release.time.time", return_value=at + 1):
                    base_release.main()
                self.assertEqual(base_release.read(out), base_release.read(internal))
                self.assertEqual(evaluate.call_count, 0 if command == "freeze" else 1)
                if command == "close":
                    self.assertEqual(
                        evaluate.call_args.kwargs["closing_admission"], active["settings"]
                    )

    def test_requalification_and_uid_movement_keep_one_ready_checkpoint(self):
        original = self.ready[1]
        evidence = copy.deepcopy(original["evidence"])
        evidence["snapshot_second"] = save(
            self.root / "snapshot-copy.json",
            base_release.evidence_read(evidence["snapshot_second"]),
        )
        suite = base_release.evidence_read(evidence["suite"])
        duplicate = base_release.qualify(
            self.first,
            self.second,
            suite,
            candidate_sha256="c" * 64,
            active=self.active,
            expected_suite=self.suite,
            evidence=evidence,
        )
        save(self.root / "ready" / (duplicate["sha256"] + ".json"), duplicate)
        moved = copy.deepcopy(self.second)
        next(c for c in moved["candidates"] if c["uid"] == 2)["uid"] = 22
        moved["pending_winner"]["uid"] = 22
        snapshot = base_release.evidence_read(evidence["snapshot_second"])
        snapshot["members"]["22"] = snapshot["members"].pop("2")
        snapshot["commitments"]["22"] = snapshot["commitments"].pop("2")
        evidence["snapshot_second"] = save(self.root / "moved-snapshot.json", snapshot)
        evidence["second"] = save(self.root / "moved-decision.json", moved)
        receipt = base_release.qualify(
            self.first,
            moved,
            suite,
            candidate_sha256="c" * 64,
            active=self.active,
            expected_suite=self.suite,
            evidence=evidence,
        )
        save(self.root / "ready" / (receipt["sha256"] + ".json"), receipt)
        current = base_release.inventory(self.active)
        self.assertEqual(len(current), 2)
        frozen = base_release.freeze(
            current,
            active=self.active,
            schedule=self.schedule,
            window_id="d3-mini:1",
            frozen_at=self.at,
            opening_channels=self.channels,
        )
        self.assertEqual(len(frozen["ready"]), 2)
        self.assertEqual(len(list((self.root / "ready").iterdir())), 4)

    def test_channel_proposal_requires_a_persisted_window(self):
        with self.assertRaisesRegex(ValueError, "window"):
            base_release.channel_proposal(
                {"stable": "old", "latest": "new"},
                new_profile_id=None,
                migration=None,
                release_review=None,
            )

    def test_channel_proposal_advances_only_opening_latest(self):
        frozen = self.freeze()
        closing = self.closing()
        members = base_release.evidence_read(closing["snapshot"])["members"]
        closed = base_release.close(
            frozen, closing, active=self.active, members=members, now=self.at + 100
        )
        migration = {
            "profile_id": "prior",
            "legacy_regression": {"job": {"incumbent": 0.3, "candidate": 0.29}},
            "outcomes": {
                "job": {
                    str(s): {"migration": True, "strict_upgrade": True} for s in (553, 771, 991)
                }
            },
        }
        review = {
            "profile_id": "prior",
            "approved": True,
            "unresolved_problems": [],
            "migration_sha256": base_round.dataset_hash(migration),
        }
        proposal = base_release.channel_proposal(
            closed,
            active=self.active,
            new_profile_id="new",
            migration=migration,
            release_review=review,
        )
        self.assertEqual(proposal["channels"], {"stable": "prior", "latest": "new"})
        self.assertEqual(
            proposal,
            base_release.channel_proposal(
                closed,
                active=self.active,
                new_profile_id="new",
                migration=migration,
                release_review=review,
            ),
        )
        changed = copy.deepcopy(closed)
        changed["frozen"]["opening_channels"] = proposal["channels"]
        with self.assertRaises(ValueError):
            base_release.channel_proposal(
                changed,
                active=self.active,
                new_profile_id=None,
                migration=None,
                release_review=None,
            )
        with self.assertRaises(ValueError):
            base_release.channel_proposal(
                closed,
                active=self.active,
                new_profile_id="other",
                migration=migration,
                release_review=review,
            )
        same_window_migration = {**migration, "profile_id": "new"}
        same_window_review = {
            **review,
            "profile_id": "new",
            "migration_sha256": base_round.dataset_hash(same_window_migration),
        }
        with self.assertRaisesRegex(ValueError, "opening latest"):
            base_release.channel_proposal(
                closed,
                active=self.active,
                new_profile_id=None,
                migration=same_window_migration,
                release_review=same_window_review,
            )
        # Stable eligibility still rejects an incomplete or troubled prior release.
        for label in ("missing", "failed", "problem"):
            with self.subTest(label=label):
                active = {**self.active, "windows_dir": str(self.root / label)}
                base_release.persist(
                    base_release.window_path(active, "d3-mini:1", "frozen"), frozen
                )
                gate, approval = copy.deepcopy(migration), copy.deepcopy(review)
                if label == "missing":
                    gate, approval = None, None
                elif label == "failed":
                    gate["outcomes"]["job"]["553"]["migration"] = False
                    approval["migration_sha256"] = base_round.dataset_hash(gate)
                else:
                    approval["unresolved_problems"] = ["regression"]
                result = base_release.channel_proposal(
                    closed,
                    active=active,
                    new_profile_id="new",
                    migration=gate,
                    release_review=approval,
                )
                self.assertEqual(result["channels"], {"stable": "old", "latest": "new"})

    def test_no_survivor_skips_and_early_closing_fails(self):
        frozen = self.freeze()
        early, _, snapshot = decision(self.root, "early", self.at - 10)
        members = json.loads(Path(snapshot["path"]).read_text())["members"]
        with self.assertRaisesRegex(ValueError, "freeze"):
            base_release.close(
                frozen,
                {"decision": early, "draw_root": str(self.root / "early"), "snapshot": snapshot},
                active=self.active,
                members=members,
                now=self.at + 100,
            )
        failed, _, snapshot = decision(self.root, "failed", self.at + 10, regressed=True)
        result = base_release.close(
            frozen,
            {"decision": failed, "draw_root": str(self.root / "failed"), "snapshot": snapshot},
            active=self.active,
            members=members,
            now=self.at + 100,
        )
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(
            result["proposed_base_tier"], {"incumbent_sha256": "a" * 64, "kings": ["old"]}
        )
        migration = {
            "profile_id": "prior",
            "legacy_regression": {"job": {"incumbent": 0.3, "candidate": 0.29}},
            "outcomes": {
                "job": {
                    str(seed): {"migration": True, "strict_upgrade": True}
                    for seed in (553, 771, 991)
                }
            },
        }
        review = {
            "profile_id": "prior",
            "approved": True,
            "unresolved_problems": [],
            "migration_sha256": base_round.dataset_hash(migration),
        }
        proposal = base_release.channel_proposal(
            result,
            active=self.active,
            new_profile_id=None,
            migration=migration,
            release_review=review,
        )
        self.assertEqual(proposal["channels"], {"stable": "prior", "latest": "prior"})

    def test_two_miners_through_real_intake_and_release_with_failing_other_pool(self):
        import os
        import sys
        from types import SimpleNamespace

        import zils
        from tests.test_base_tier_intake import RUNNER, candidate, fake_hub, revision
        from zils import protocol, validator

        root = self.root / "fleet"
        root.mkdir()
        reference = root / "reference"
        reference.mkdir()
        models.write_metadata(reference, kind="base", model=models.D3)
        incumbent = zils.checkpoint_hash(reference)
        first = candidate(root / "candidate-one", 0.90)
        second = candidate(root / "candidate-two", 0.99)
        hashes = {1: zils.checkpoint_hash(first), 2: zils.checkpoint_hash(second)}
        repos = {
            "miner/one": {revision("one"): first},
            "miner/two": {revision("two"): second},
            "miner/copy": {revision("copy"): candidate(root / "candidate-copy", 0.0)},
        }
        snapshot = {
            "members": {"1": "miner-one", "2": "miner-two", "3": "copy", "4": "old"},
            "commitments": {
                str(u): {
                    "block": 20 + u,
                    "data": commitments.format_commitment(sha, revision(name), "miner/" + name),
                }
                for u, sha, name in (
                    (1, hashes[1], "one"),
                    (2, hashes[2], "two"),
                    (3, incumbent, "copy"),
                )
            },
        }
        for name in ("first", "second", "closing"):
            sealed(root / name, name)
        draws = [(base_round.draw(root / n), root / n) for n in ("first", "second")]
        save(root / "history.json", inputs()[2])
        save(root / "pilot.json", {"fixture": True})
        track = {
            "share": 1.0,
            "incumbent_sha256": incumbent,
            "kings": ["old"],
            "draws": [i for i, _ in draws],
            "policy": base_round.DEFAULT_POLICY,
        }
        settings = {
            "pool_id": "d3-mini",
            "model": models.spec(models.D3),
            "base_tier": track,
            "base_reference": str(reference),
            "family_history": str(root / "history.json"),
            "pilot_report": str(root / "pilot.json"),
        }
        config = {
            "base_pools": {
                "share": 0.7,
                "burn_uid": 0,
                "pools": {
                    "d3-mini": {
                        **track,
                        "share": 0.5,
                        "enabled": True,
                        "model": models.spec(models.D3),
                    },
                    "d3-lite": {
                        **track,
                        "share": 0.5,
                        "enabled": True,
                        "model": models.spec("d3-lite-0.8b-v1"),
                    },
                },
            }
        }
        rounds = {
            "d3-mini": (settings, root / "base-pools/d3-mini", draws),
            "d3-lite": (
                {**settings, "pool_id": "d3-lite", "model": models.spec("d3-lite-0.8b-v1")},
                root / "base-pools/d3-lite",
                draws,
            ),
        }
        runner = root / "runner"
        runner.write_text(f"#!{sys.executable}\n{RUNNER}")
        runner.chmod(0o700)
        args = SimpleNamespace(device="cpu", runtime_python=sys.executable, poll=0.01)
        requested = []
        with (
            protocol.local_server(fake_hub(repos, requested)) as hub,
            patch.dict(
                os.environ,
                {
                    "ZILS_D3_RUNTIME_PYTHON": str(runner),
                    "ZILS_TEST_RUNNER_LOG": str(root / "calls"),
                },
            ),
            patch("zils.commitments.HF_ENDPOINT", hub),
            patch("zils.d3.validate_candidate"),
            patch("zils.base_pilot.verify_qualification"),
        ):
            for name, at in (("epoch-first", 1000), ("epoch-second", 44202)):
                work = root / name
                work.mkdir()
                with patch("zils.validator.time.time", return_value=at):
                    validator.base_pools_epoch(config, root, work, snapshot, args, rounds)
            intake = json.loads((work / "pool-intake.json").read_text())
            errors = intake["errors"]
            self.assertIn("incumbent", intake["rejected"]["3"])
            self.assertIn("d3-lite", errors)
            self.assertNotIn("d3-mini", errors)
            attempt = next(work.joinpath("d3-mini").glob("base-tier-*"))
            decisions = {
                json.loads(p.read_text())["manifest_sha256"]: (
                    json.loads(p.read_text()),
                    base_release.bound(p),
                )
                for p in attempt.glob("decision-*.json")
            }
            d1, e1 = decisions[draws[0][0]["manifest_sha256"]]
            d2, e2 = decisions[draws[1][0]["manifest_sha256"]]
            self.assertNotIn(3, {row["uid"] for row in d2["candidates"]})
            suite, _ = election_fixture()
            active = {
                **self.active,
                **track,
                "incumbent_sha256": incumbent,
                "draw_roots": {i["manifest_sha256"]: str(p) for i, p in draws},
                "ready_dir": str(root / "ready"),
                "windows_dir": str(root / "windows"),
            }
            snap = save(root / "snapshot.json", snapshot)
            ready = []
            for uid in (1, 2):
                _, report = election_fixture(candidate=hashes[uid], incumbent=incumbent)
                evidence = {
                    "first": e1,
                    "second": e2,
                    "snapshot_first": snap,
                    "snapshot_second": snap,
                    "suite": save(root / f"suite-{uid}.json", report),
                }
                receipt = base_release.qualify(
                    d1,
                    d2,
                    report,
                    candidate_sha256=hashes[uid],
                    active=active,
                    expected_suite=suite,
                    evidence=evidence,
                )
                save(root / "ready" / (receipt["sha256"] + ".json"), receipt)
                ready.append(receipt)
            self.assertEqual({r["candidate"]["hotkey"] for r in ready}, {"miner-one", "miner-two"})
            frozen = base_release.freeze(
                ready,
                active=active,
                schedule=self.schedule,
                window_id="d3-mini:1",
                frozen_at=self.at,
                opening_channels=self.channels,
            )
            claims = {
                u: {**c, "hotkey": snapshot["members"][str(u)]}
                for u, c in commitments.candidates(snapshot, incumbent).items()
                if "error" not in c
            }
            work = root / "closing-work"
            work.mkdir()
            identity = base_round.draw(root / "closing")
            with patch("zils.validator.time.time", return_value=self.at + 10):
                closing = validator.base_tier_epoch(
                    settings,
                    rounds["d3-mini"][1],
                    work,
                    claims,
                    args,
                    [(identity, root / "closing")],
                )
            result = base_release.close(
                frozen,
                {"decision": closing, "draw_root": str(root / "closing"), "snapshot": snap},
                active=active,
                members=snapshot["members"],
                now=self.at + 100,
            )
            self.assertEqual(result["winner"]["sha256"], hashes[2])
            self.assertFalse(result["uploaded"])
            self.assertEqual(
                base_round.king_weights(
                    track, {u: {"hotkey": h} for u, h in snapshot["members"].items()}
                ),
                {4: 1.0},
            )
