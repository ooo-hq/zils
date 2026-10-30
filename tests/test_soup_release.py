"""A validator-built soup qualifies and closes a release window without becoming a king."""

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import zils
from tests.test_base_candidate import candidate
from tests.test_base_exam import inputs
from tests.test_base_release import decision, fast_pair, save
from tests.test_regression_suite import election_fixture
from zils import base_metrics, base_release, base_round, models, regression_suite

with tempfile.TemporaryDirectory() as _tmp:
    # The soup is a real candidate checkpoint so the close command can hash-verify it.
    SOUP = zils.checkpoint_hash(candidate(Path(_tmp) / "soup"))
HASHES = {0: "a" * 64, 1: "b" * 64, 2: SOUP}
OWNERS = {1: "miner-one", 2: "validator-soup"}
SCHEDULE = {
    "version": "zils-base-windows/v1",
    "anchor_utc": "1970-01-01T00:00:00Z",
    "interval_days": 14,
}
AT = 14 * 86400


def soup_decision(root, name, at, **kwargs):
    """The soup (uid 2) is evaluated like any row but has no chain commitment."""
    result, evidence, snapshot = decision(root, name, at, hashes=HASHES, owners=OWNERS, **kwargs)
    chain = base_release.evidence_read(snapshot)
    chain["members"].pop("2")
    chain["commitments"].pop("2")
    return result, evidence, save(Path(snapshot["path"]), chain)


def failing_suite(candidate):
    _, report = election_fixture(candidate=candidate)
    for seeds in report["runs"].values():
        for run in seeds.values():
            run["candidate"]["accuracy"] = 0.7
    verdict = regression_suite.election_verdict(report["runs"], expected_jobs=list(report["jobs"]))
    report["verdict"] = verdict
    report["jobs"] = {
        j: {s: value[s] for s in ("incumbent", "candidate")} for j, value in verdict["jobs"].items()
    }
    return report


class SoupReleaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for target in (
            patch.object(base_metrics, "paired_improvement", side_effect=fast_pair),
            patch.object(base_round, "fit_temperature", return_value=1.0),
        ):
            target.start()
            self.addCleanup(target.stop)
        self.first, self.f, self.s1 = soup_decision(self.root, "first", 100)
        self.second, self.s, self.s2 = soup_decision(self.root, "second", 43302)
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
        self.soup = {
            "soup_sha256": SOUP,
            "members": [
                {"body_sha256": "1" * 64, "hotkey": "miner-one"},
                {"body_sha256": "2" * 64, "hotkey": "miner-three"},
            ],
            "round_id": "d3-mini:soup:7",
        }

    def qualify_soup(self, first=None, second=None, report=None, soup=None, evidence=None):
        first, second = first or self.first, second or self.second
        report = report or election_fixture(candidate=SOUP)[1]
        evidence = evidence or {
            "first": save(self.root / "soup-first.json", first),
            "second": save(self.root / "soup-second.json", second),
            "suite": save(self.root / "soup-suite.json", report),
        }
        return base_release.qualify_soup(
            first,
            second,
            report,
            soup=soup or self.soup,
            active=self.active,
            expected_suite=self.suite,
            evidence=evidence,
        )

    def king(self):
        _, report = election_fixture(candidate="b" * 64)
        return base_release.qualify(
            self.first,
            self.second,
            report,
            candidate_sha256="b" * 64,
            active=self.active,
            expected_suite=self.suite,
            evidence={
                "first": self.f,
                "second": self.s,
                "snapshot_first": self.s1,
                "snapshot_second": self.s2,
                "suite": save(self.root / "king-suite.json", report),
            },
        )

    def freeze(self, ready):
        return base_release.freeze(
            ready,
            active=self.active,
            schedule=SCHEDULE,
            window_id="d3-mini:1",
            frozen_at=AT,
            opening_channels={"stable": "old", "latest": "prior"},
        )

    def test_soup_qualifies_on_two_decisions_and_suite(self):
        receipt = self.qualify_soup()
        self.assertEqual(receipt["candidate"]["sha256"], SOUP)
        self.assertEqual(receipt["candidate"]["hotkey"], base_release.SOUP_OWNER)
        self.assertEqual(
            receipt["ownership"],
            {"owner": base_release.SOUP_OWNER, **self.soup},
        )
        self.assertEqual(set(receipt["evidence"]), {"first", "second", "suite"})
        self.assertEqual(base_release.verify_ready(receipt, self.active), receipt)
        tampered = copy.deepcopy(receipt)
        tampered["ownership"]["members"].pop()
        with self.assertRaises(ValueError):
            base_release.verify_ready(base_release.addressed(tampered), self.active)

    def test_soup_fails_on_either_decision_suite_or_hash(self):
        bad, _, _ = soup_decision(self.root, "bad-first", 100, regressed=True)
        bad2, _, _ = soup_decision(self.root, "bad-second", 43302, regressed=True)
        for name, args in (
            ("first", {"first": bad}),
            ("second", {"second": bad2}),
            ("suite", {"report": failing_suite(SOUP)}),
            ("hash", {"soup": {**self.soup, "soup_sha256": "d" * 64}}),
            ("members", {"soup": {**self.soup, "members": []}}),
        ):
            with self.subTest(name=name):
                if name in ("first", "second"):
                    self.active["draws"].append(base_round.draw(self.root / ("bad-" + name)))
                    self.active["draw_roots"][self.active["draws"][-1]["manifest_sha256"]] = str(
                        self.root / ("bad-" + name)
                    )
                with self.assertRaises(ValueError):
                    self.qualify_soup(**args)

    def test_soup_row_must_be_validator_built_and_kings_keep_chain_owners(self):
        with self.assertRaisesRegex(ValueError, "validator-built"):
            self.qualify_soup(
                soup={**self.soup, "soup_sha256": "b" * 64},
                report=election_fixture(candidate="b" * 64)[1],
            )
        with self.assertRaisesRegex(ValueError, "ownership"):
            base_release.qualify(
                self.first,
                self.second,
                election_fixture(candidate=SOUP)[1],
                candidate_sha256=SOUP,
                active=self.active,
                expected_suite=self.suite,
                evidence={
                    "first": self.f,
                    "second": self.s,
                    "snapshot_first": self.s1,
                    "snapshot_second": self.s2,
                    "suite": save(self.root / "x.json", election_fixture(candidate=SOUP)[1]),
                },
            )

    def test_window_closes_on_soup_without_crowning_a_king(self):
        king, soup = self.king(), self.qualify_soup()
        for receipt in (king, soup):
            save(self.root / "ready" / (receipt["sha256"] + ".json"), receipt)
        ready = base_release.inventory(self.active)
        self.assertEqual(sorted(r["candidate"]["sha256"] for r in ready), sorted(["b" * 64, SOUP]))
        frozen = self.freeze(ready)
        result, _, snapshot = soup_decision(self.root, "closing", 14 * 86400 + 10)
        closing = {
            "decision": result,
            "draw_root": str(self.root / "closing"),
            "snapshot": snapshot,
        }
        members = json.loads(Path(snapshot["path"]).read_text())["members"]
        closed = base_release.close(
            frozen, closing, active=self.active, members=members, now=14 * 86400 + 100
        )
        self.assertEqual(closed["status"], "qualified")
        self.assertEqual(closed["winner"]["sha256"], SOUP)
        self.assertEqual(closed["winner"]["ownership"], soup["ownership"])
        self.assertEqual(closed["proposed_base_tier"], {"incumbent_sha256": SOUP, "kings": ["old"]})
        self.assertEqual(base_release.verify_window(closed, self.active), closed)

    def close_cli(self, frozen_file, closing, *extra):
        """Run `base_release close` with the evaluator mocked as the existing CLI test does."""
        active = {**self.active, "settings": {"base_tier": {}}}
        config = self.root / "config.json"
        config.write_text("{}")
        out = self.root / ("close-out-" + str(len(list(self.root.glob("close-out-*")))) + ".json")
        argv = [
            "base_release",
            "close",
            "--config",
            str(config),
            "--pool",
            "d3-mini",
            "--out",
            str(out),
            "--frozen",
            str(frozen_file),
            "--closing-round",
            closing["draw_root"],
            "--chain-snapshot",
            closing["snapshot"]["path"],
            *extra,
        ]
        with (
            patch.object(base_release, "active_context", return_value=active),
            patch("zils.validator.base_tier_epoch", return_value=closing["decision"]) as evaluate,
            patch("zils.d3.validate_candidate") as validate,
            patch("zils.base_release.time.time", return_value=AT + 100),
            patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            base_release.main()
        return base_release.read(out), evaluate, validate

    def soup_window(self):
        king, soup = self.king(), self.qualify_soup()
        for receipt in (king, soup):
            save(self.root / "ready" / (receipt["sha256"] + ".json"), receipt)
        self.freeze(base_release.inventory(self.active))
        result, _, snapshot = soup_decision(self.root, "closing", AT + 10)
        closing = {
            "decision": result,
            "draw_root": str(self.root / "closing"),
            "snapshot": snapshot,
        }
        return soup, self.root / "windows/d3-mini-1-frozen.json", closing

    def test_close_command_evaluates_the_soup_checkpoint_and_chain_kings(self):
        soup, frozen_file, closing = self.soup_window()
        checkpoint = candidate(self.root / "soup-checkpoint")
        with self.assertRaisesRegex(ValueError, "soup-checkpoint"):
            self.close_cli(frozen_file, closing)
        wrong = candidate(self.root / "wrong-checkpoint")
        (wrong / "model-00001-of-00002.safetensors").write_text("other weights")
        with self.assertRaisesRegex(ValueError, "soup-checkpoint"):
            self.close_cli(frozen_file, closing, "--soup-checkpoint", str(wrong))
        closed, evaluate, validate = self.close_cli(
            frozen_file, closing, "--soup-checkpoint", str(checkpoint)
        )
        validate.assert_called_once()
        registry = evaluate.call_args.args[3]
        row = soup["candidate"]
        self.assertEqual(
            registry[row["uid"]],
            {"sha256": SOUP, "hotkey": base_release.SOUP_OWNER, "block": row["commitment_block"]},
        )
        self.assertEqual(registry[1]["hotkey"], "miner-one")
        self.assertEqual(set(registry), {1, row["uid"]})
        cached = Path(self.active["ready_dir"]).parent / "base-tier/store" / SOUP / "checkpoint"
        self.assertEqual(zils.checkpoint_hash(cached), SOUP)
        self.assertEqual(closed["winner"]["ownership"], soup["ownership"])
        self.assertEqual(closed["proposed_base_tier"], {"incumbent_sha256": SOUP, "kings": ["old"]})

    def promote(self, winner, *extra):
        import runpy

        weights = candidate(self.root / "promote-candidate")
        report = self.root / "window.json"
        report.write_text("{}")
        decision = {"manifest_sha256": "9" * 64}
        verified = {"status": "qualified", "winner": winner, "closing": {"decision": decision}}
        argv = [
            "promote_base.py",
            "--candidate",
            str(weights),
            "--round",
            "4",
            "--staging",
            str(self.root / "staging"),
            "--config",
            str(report),
            "--pool",
            "d3-mini",
            "--window-report",
            str(report),
            *extra,
        ]
        out = io.StringIO()
        with (
            patch(
                "zils.base_release.active_context",
                return_value={"model": models.spec(models.D3), "kings": ["hk-old"]},
            ),
            patch("zils.base_release.verify_window", return_value=verified),
            patch("zils.d3.validate_candidate"),
            patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(out),
        ):
            runpy.run_path(
                str(Path(__file__).parents[1] / "scripts/promote_base.py"), run_name="__main__"
            )
        return json.loads(out.getvalue())

    def test_promotion_of_a_soup_sets_incumbent_and_records_members_without_a_king(self):
        ownership = {"owner": base_release.SOUP_OWNER, **self.soup}
        winner = {
            "uid": 9,
            "sha256": SOUP,
            "hotkey": base_release.SOUP_OWNER,
            "commitment_block": 0,
            "score": 1.0,
            "ownership": ownership,
        }
        with self.assertRaisesRegex(ValueError, "king-hotkey"):
            self.promote(winner, "--king-hotkey", base_release.SOUP_OWNER)
        result = self.promote(winner)
        self.assertEqual(
            result["proposed_base_tier"], {"incumbent_sha256": SOUP, "kings": ["hk-old"]}
        )
        self.assertEqual(result["release"], ownership)
        self.assertIsNone(result["crowned_uid"])


if __name__ == "__main__":
    unittest.main()
