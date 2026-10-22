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

    def freeze(self):
        return base_release.freeze(
            self.ready,
            active=self.active,
            schedule=self.schedule,
            window_id="d3-mini:1",
            frozen_at=self.at,
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
            )

    def test_stale_bindings_mutated_evidence_and_missing_roster_fail(self):
        with self.assertRaises(ValueError):
            base_release.freeze(
                self.ready,
                active={**self.active, "pilot_sha256": "0" * 64},
                schedule=self.schedule,
                window_id="d3-mini:1",
                frozen_at=self.at,
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

    def test_channel_proposal_advances_only_opening_latest(self):
        channels = {"stable": "old", "latest": "prior"}
        self.assertEqual(
            base_release.channel_proposal(
                channels, new_profile_id="new", migration=None, release_review=None
            ),
            {"stable": "old", "latest": "new"},
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
        self.assertEqual(
            base_release.channel_proposal(
                channels, new_profile_id="new", migration=migration, release_review=review
            ),
            {"stable": "prior", "latest": "new"},
        )
        review["unresolved_problems"] = ["regression"]
        self.assertEqual(
            base_release.channel_proposal(
                channels, new_profile_id=None, migration=migration, release_review=review
            ),
            channels,
        )

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
