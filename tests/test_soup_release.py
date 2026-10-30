"""A validator-built soup qualifies and closes a release window without becoming a king."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_base_exam import inputs
from tests.test_base_release import decision, fast_pair, save
from tests.test_regression_suite import election_fixture
from zils import base_metrics, base_release, base_round, models, regression_suite

SOUP = "c" * 64
OWNERS = {1: "miner-one", 2: "validator-soup"}


def soup_decision(root, name, at, **kwargs):
    """The soup (uid 2) is evaluated like any row but has no chain commitment."""
    result, evidence, snapshot = decision(root, name, at, owners=OWNERS, **kwargs)
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
        self.assertEqual([r["candidate"]["sha256"] for r in ready], ["b" * 64, SOUP])
        frozen = base_release.freeze(
            ready,
            active=self.active,
            schedule={
                "version": "zils-base-windows/v1",
                "anchor_utc": "1970-01-01T00:00:00Z",
                "interval_days": 14,
            },
            window_id="d3-mini:1",
            frozen_at=14 * 86400,
            opening_channels={"stable": "old", "latest": "prior"},
        )
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


if __name__ == "__main__":
    unittest.main()
