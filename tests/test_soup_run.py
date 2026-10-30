"""A full soup round on tiny synthetic bodies with a fake runner, its payout and its daily schedule."""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file

import zils
from tests import test_bodies, test_soup
from tests.test_base_metrics import CATEGORIES
from tests.test_soup_exam import bank as exam_bank
from zils import bodies, d3, model_pools, models, soup_exam, soup_grade, soup_pool, soup_round

STRENGTH = "language_model.layers.0.mlp.weight"
CHASSIS = {
    STRENGTH: torch.zeros(len(CATEGORIES), dtype=torch.bfloat16),
    "language_model.norm.weight": torch.zeros(2, dtype=torch.bfloat16),
    "visual.patch.weight": torch.zeros(2, dtype=torch.bfloat16),
}
HOTKEYS = {1: "5Trainer1", 2: "5Trainer2", 3: "5Copier", 4: "5Weak", 5: "5Scout", 6: "5Noisy"}
REVISION = "b" * 40
OFFSET = 8  # the vendor base's strength tensor; a body's strength is its change from the base


def strengths(*values, norm=1):
    return {
        STRENGTH: torch.tensor([OFFSET + v for v in values], dtype=torch.bfloat16),
        "language_model.norm.weight": torch.full((2,), norm, dtype=torch.bfloat16),
        "visual.patch.weight": torch.ones(2, dtype=torch.bfloat16),
    }


# Trainer 1 knows reasoning and language, trainer 2 the other three; their soup knows everything.
BODIES = {
    "t1": strengths(1, 1, 0, 0, 0),
    "t2": strengths(0, 0, 1, 1, 1),
    "house": strengths(0.25, 0.25, 0.25, 0.25, 0.25),
    "weak": strengths(0.0625, 0, 0, 0, 0.0625),
    # A light retrain of the house model (a scout) and trainer 1's layers with a new norm (a copy).
    "scout": strengths(0.25, 0.25, 0.25, 0.25, 0.3125),
    "t1copy": strengths(1, 1, 0, 0, 0, norm=2),
}


def fraction(identity):
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") / 2**64


class FakeRunner:
    """Answers a case correctly when its hash fraction is below min(1, 2 x the folder's strength for the
    case's category); otherwise answers confidently wrong. Soups average strengths, so they can win."""

    def __init__(self):
        self.graded = {}
        self.fail_after = None

    def __call__(self, folder):
        folder = Path(folder)
        tensors = {}
        for path in folder.glob("*.safetensors"):
            if path.name != "readout.safetensors":
                tensors.update(load_file(str(path)))
        strength = next(v for k, v in tensors.items() if k.endswith(STRENGTH)).float() - OFFSET
        strength = strength.tolist()

        def run(cases):
            rows = []
            for case in cases:
                if self.fail_after is not None:
                    if self.fail_after <= 0:
                        raise RuntimeError("simulated crash")
                    self.fail_after -= 1
                self.graded[str(folder)] = self.graded.get(str(folder), 0) + 1
                keys = zils.options(case["question"])
                p = min(1.0, 2 * strength[CATEGORIES.index(case["category"])])
                wrong = next(k for k in reversed(keys) if k != case["label"])
                chosen = case["label"] if fraction(case["id"]) < p else wrong
                probabilities = {k: 1.0 if k == chosen else 0.0 for k in keys}
                rows.append({"id": case["id"], "probabilities": probabilities, "elapsed_ms": 1})
            return rows

        return run


class Round:
    """Fixture: chassis, synthetic body repos, a fake chain snapshot and the round's injected parts."""

    def __init__(self, root):
        self.root = Path(root)
        self.chassis = test_soup.chassis(self.root / "chassis", dict(CHASSIS))
        self.base = test_soup.body(self.root / "qwen", strengths(0, 0, 0, 0, 0))
        self.repos, self.sha = {}, {}
        for name, tensors in BODIES.items():
            repo = test_bodies.checkpoint(
                self.root / "repos" / name, tensors, prefix="model.", owner=None
            )
            self.repos[name] = repo
            self.sha[name] = bodies.body_hash(repo, list(bodies.load_index(self.chassis)))
        self.commits = {1: ("t1", 100), 2: ("t2", 101), 3: ("t1", 102), 4: ("weak", 103)}
        self.fetched = []
        self.runner = FakeRunner()
        self.bank = exam_bank()
        self.state = self.root / "state"
        self.ledger = soup_exam.UseLedger(self.state / "uses.jsonl")

    def snapshot(self):
        return {
            "block": 200,
            "members": {str(uid): hotkey for uid, hotkey in HOTKEYS.items()},
            "commitments": {
                str(uid): {
                    "block": block,
                    "data": bodies.format_commitment(self.sha[name], REVISION, f"miner/{name}"),
                }
                for uid, (name, block) in self.commits.items()
            },
        }

    def fetch(self, claim, dest):
        name = claim["repo"].split("/")[1]
        self.fetched.append(self.sha[name])
        dest = Path(dest)
        dest.mkdir(parents=True)
        for path in self.repos[name].iterdir():
            dest.joinpath(path.name).write_bytes(path.read_bytes())
        if claim.get("hotkey"):
            (dest / "zils-owner.json").write_text(json.dumps({"hotkey": claim["hotkey"]}))
        return dest

    def config(self, **changes):
        return {
            "pool_id": "d3-mini",
            "model": models.D3,
            "house": [
                {"name": "house", "repo": "open/house", "revision": REVISION, "body_sha256": None}
            ],
            "sketch_seed": "5e" * 32,
            "subset_size": 300,
            "screen_size": 60,
            "max_pool": 20,
            **changes,
        }

    def run(self, round_id="d3-mini-2026-11-03", **kwargs):
        return soup_round.run(
            kwargs.pop("config", self.config()),
            self.state,
            round_id=round_id,
            snapshot=kwargs.pop("snapshot", self.snapshot()),
            block_hash="0x" + "ab" * 32,
            bank=self.bank,
            ledger=self.ledger,
            cache=soup_grade.PredictionCache(self.state / "cache"),
            runner_for=self.runner,
            fetch_body=kwargs.pop("fetch_body", self.fetch),
            chassis=self.chassis,
            base=kwargs.pop("base", self.base),
            **kwargs,
        )


class SoupRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.r = Round(self.tmp)
        pinned = hashlib.sha256((self.r.chassis / "readout.safetensors").read_bytes()).hexdigest()
        self.enterContext(patch.dict(d3.PINS["files"], {"readout.safetensors": pinned}))

    def public(self, round_id="d3-mini-2026-11-03"):
        return json.loads((self.r.state / "rounds" / round_id / "public.json").read_text())

    def by_body(self, record):
        return {m["body_sha256"]: m for m in record["members"]}

    def test_end_to_end_round_seats_the_two_trainers_and_crowns_the_best_solo(self):
        sha = self.r.sha
        record = self.r.run()
        self.assertEqual(set(self.by_body(record)), {sha["t1"], sha["t2"]})
        members = self.by_body(record)
        self.assertEqual(members[sha["t1"]]["hotkey"], HOTKEYS[1])
        self.assertEqual(members[sha["t2"]]["hotkey"], HOTKEYS[2])
        for member in members.values():
            self.assertEqual(member["rate"], 1.0)
            self.assertGreater(member["contribution"], 30)
            self.assertAlmostEqual(member["seat_weight"], 0.4)
        # Trainer 1's categories weigh 51.7% of the score, so it is the best solo body.
        self.assertGreater(record["solo"][sha["t1"]], record["solo"][sha["t2"]])
        self.assertEqual(record["crown"]["body_sha256"], sha["t1"])
        self.assertEqual(record["crown"]["hotkey"], HOTKEYS[1])
        self.assertAlmostEqual(record["crown"]["weight"], 0.2)
        self.assertEqual(record["category_scores"], {c: 100.0 for c in CATEGORIES})
        # The copier's commitment of trainer 1's hash is refused without a download.
        self.assertEqual(self.r.fetched.count(sha["t1"]), 1)
        self.assertIn(sha["house"], record["pool"])
        self.assertEqual(record["rates"][sha["house"]], 0.0)
        self.assertIn(sha["weak"], record["pool"])
        self.assertEqual(len(record["subset_ids"]), 300)
        roster = {str(uid): {"hotkey": h} for uid, h in HOTKEYS.items()}
        weights = soup_round.soup_weights(record, roster)
        self.assertAlmostEqual(weights[1], 0.6)
        self.assertAlmostEqual(weights[2], 0.4)
        # The final soup stays on disk; trial soups are deleted after grading.
        round_dir = self.r.state / "rounds" / "d3-mini-2026-11-03"
        self.assertTrue((round_dir / "soup" / "readout.safetensors").exists())
        self.assertEqual(list((self.r.state / "tmp").iterdir()), [])

    def test_every_graded_body_and_soup_is_a_valid_d3_candidate(self):
        # Production grading loads d3.DecisionModel, which needs candidate metadata and the chassis files.
        pins = {
            name: hashlib.sha256((self.r.chassis / name).read_bytes()).hexdigest()
            for name in models.D3_CONTRACT_FILES
        }
        self.enterContext(patch.dict(d3.PINS["files"], pins))
        self.enterContext(patch("zils.d3.pinned_base_path", return_value=self.r.chassis))
        graded, fake = [], self.r.runner

        def runner_for(folder):
            d3.validate_candidate(folder, models.D3)
            graded.append(Path(folder))
            return fake(folder)

        self.r.runner = runner_for
        record = self.r.run()
        self.assertEqual(len(record["members"]), 2)
        self.assertTrue(graded)
        self.assertFalse(any(p.parent == self.r.state / "bodies" for p in graded))
        self.assertEqual(list((self.r.state / "tmp").iterdir()), [])
        for sha in record["pool"]:  # the body folders themselves are kept for the next round
            self.assertTrue((self.r.state / "bodies" / sha).exists())

    def validating_runner(self, *, failing=None):
        """A runner_for that requires every folder to be a valid built d3 candidate (production
        loads d3.DecisionModel); `failing` names a body whose grading raises."""
        pins = {
            name: hashlib.sha256((self.r.chassis / name).read_bytes()).hexdigest()
            for name in models.D3_CONTRACT_FILES
        }
        self.enterContext(patch.dict(d3.PINS["files"], pins))
        self.enterContext(patch("zils.d3.pinned_base_path", return_value=self.r.chassis))
        graded, fake = [], self.r.runner

        def runner_for(folder):
            d3.validate_candidate(folder, models.D3)
            graded.append(Path(folder))
            if failing and Path(folder).name.startswith(self.r.sha[failing][:32]):

                def fail(cases):
                    raise ValueError("bad output")

                return fail
            return fake(folder)

        self.r.runner = runner_for
        return graded

    def test_a_one_member_soup_grades_its_baseline_through_a_built_chassis(self):
        graded = self.validating_runner()
        self.r.commits = {1: ("t1", 100)}
        record = self.r.run(config=self.r.config(house=[]))
        self.assertEqual([m["body_sha256"] for m in record["members"]], [self.r.sha["t1"]])
        self.assertGreater(record["members"][0]["contribution"], 0.3)
        self.assertNotIn(self.r.chassis, graded)
        self.assertTrue(any(p.name.startswith("chassis-") for p in graded))
        roster = {str(uid): {"hotkey": h} for uid, h in HOTKEYS.items()}
        weights = soup_round.soup_weights(record, roster)
        self.assertAlmostEqual(weights[1], 1.0)  # its seat (0.8) and the crown (0.2)
        self.assertEqual(list((self.r.state / "tmp").iterdir()), [])

    def test_the_chassis_check_after_a_failed_body_grades_a_built_chassis(self):
        graded = self.validating_runner(failing="weak")
        record = self.r.run()
        self.assertIn(self.r.sha["weak"], record["grade_failures"])
        self.assertNotIn(self.r.chassis, graded)
        self.assertTrue(record["members"])

    def test_seed_comes_from_block_hash_and_pool(self):
        record = self.r.run()
        expected = hashlib.sha256(("0x" + "ab" * 32 + ":d3-mini").encode()).digest()
        self.assertEqual(record["seed"], expected.hex())
        chosen = soup_exam.subset(self.r.bank, expected, 300)
        self.assertEqual(record["subset_ids"], [c["id"] for c in chosen])

    def test_public_record_and_decision_leak_no_items_labels_or_families(self):
        self.r.run()
        round_dir = self.r.state / "rounds" / "d3-mini-2026-11-03"
        public = json.loads((round_dir / "public.json").read_text())
        decision = json.loads((self.r.state / "decision.json").read_text())
        self.assertEqual(public, decision)
        text = (round_dir / "public.json").read_text()
        for case in self.r.bank:
            self.assertNotIn(case["id"], text)
        for family in {c["family"] for c in self.r.bank}:
            self.assertNotIn(family, text)
        self.assertNotIn("predictions", public)
        self.assertNotIn("family_scores", public)
        private = json.loads((round_dir / "private.json").read_text())
        from zils import soup_publish

        soup_publish.verify_public(public, private)

    def test_ledger_records_once_and_a_finished_round_is_not_rerun(self):
        first = self.r.run()
        lines = self.r.ledger.path.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["round"], "d3-mini-2026-11-03")
        graded = dict(self.r.runner.graded)
        again = self.r.run()
        self.assertEqual(again, first)
        self.assertEqual(self.r.runner.graded, graded)
        self.assertEqual(len(self.r.ledger.path.read_text().splitlines()), 1)

    def test_a_crashed_round_resumes_from_the_cache_without_double_recording(self):
        self.r.runner.fail_after = 1500  # past the solos, inside the soup trials
        with self.assertRaisesRegex(RuntimeError, "simulated crash"):
            self.r.run()
        self.assertFalse(self.r.ledger.path.exists())
        solo_items = {k: v for k, v in self.r.runner.graded.items() if "/tmp/soup-" not in k}
        self.assertTrue(solo_items)
        self.r.runner.fail_after = None
        record = self.r.run()
        self.assertEqual(len(self.r.ledger.path.read_text().splitlines()), 1)
        for folder, count in solo_items.items():
            self.assertEqual(self.r.runner.graded[folder], count, "a cached body was regraded")
        fresh = Round(self.tmp / "fresh")
        clean = fresh.run()
        self.assertEqual(
            {m["hotkey"]: m["seat_weight"] for m in record["members"]},
            {m["hotkey"]: m["seat_weight"] for m in clean["members"]},
        )

    def test_next_round_keeps_members_and_skips_known_bodies(self):
        self.r.run()
        fetched = list(self.r.fetched)
        record = self.r.run(round_id="d3-mini-2026-11-04")
        self.assertEqual(self.r.fetched, fetched)
        self.assertEqual(set(self.by_body(record)), {self.r.sha["t1"], self.r.sha["t2"]})
        self.assertEqual(len(self.r.ledger.path.read_text().splitlines()), 2)
        old = self.r.state / "rounds" / "d3-mini-2026-11-03" / "soup"
        self.assertFalse(old.exists())

    def test_copies_hook_rejects_before_screening(self):
        seen = []

        def copies(sha, folder, earlier, *, soup):
            seen.append((sha, set(earlier), soup))
            return "copy of an earlier body" if sha == self.r.sha["weak"] else None

        record = self.r.run(copies=copies)
        self.assertNotIn(self.r.sha["weak"], record["pool"])
        self.assertEqual(record["rejected"][self.r.sha["weak"]], "copy: copy of an earlier body")
        weak = next(s for s in seen if s[0] == self.r.sha["weak"])
        self.assertEqual(weak[1], {self.r.sha["t1"], self.r.sha["t2"]})  # never house bodies
        self.assertEqual(weak[2], ())
        self.assertFalse((self.r.state / "bodies" / self.r.sha["weak"]).exists())

    def test_twins_hook_compares_screen_predictions_with_pool_members(self):
        def twins(rows, other_rows):
            return rows == other_rows

        snapshot = self.r.snapshot()
        snapshot["commitments"]["4"]["block"] = 99  # weak commits first; nobody else is its twin
        record = self.r.run(twins=twins, snapshot=snapshot)
        self.assertIn(self.r.sha["weak"], record["pool"])
        self.assertEqual(record["rejected"], {})

    def twin_round(self):
        """uid 6 commits a body that predicts exactly like trainer 1's (a behavioural twin)."""
        self.r.commits[6] = ("t1copy", 104)

        def twins(rows, other_rows):
            return rows == other_rows

        def run(round_id, **kwargs):
            return self.r.run(round_id=round_id, twins=twins, copies=lambda *a, **k: None, **kwargs)

        return run

    def test_a_twin_is_refused_by_claim_and_never_banned(self):
        run, twin = self.twin_round(), self.r.sha["t1copy"]
        record = run("d3-mini-2026-11-03")
        self.assertNotIn(twin, record["rejected"])
        self.assertEqual(record["twins"][twin]["partner"], self.r.sha["t1"])
        state = json.loads((self.r.state / "pool.json").read_text())
        self.assertNotIn(twin, state["rejected"])
        self.assertNotIn(twin, state["bodies"])
        self.assertFalse((self.r.state / "bodies" / twin).exists())
        fetched = self.r.fetched.count(twin)
        again = run("d3-mini-2026-11-04")
        self.assertEqual(self.r.fetched.count(twin), fetched)  # unchanged claim, partner still here
        self.assertIn("twin", again["refused"]["6"])

    def test_a_twin_is_retried_when_re_committed(self):
        run, twin = self.twin_round(), self.r.sha["t1copy"]
        run("d3-mini-2026-11-03")
        fetched = self.r.fetched.count(twin)
        snapshot = self.r.snapshot()
        snapshot["commitments"]["6"]["data"] = bodies.format_commitment(
            twin, "c" * 40, "miner/t1copy"
        )
        run("d3-mini-2026-11-04", snapshot=snapshot)
        self.assertEqual(self.r.fetched.count(twin), fetched + 1)

    def test_a_twin_is_retried_when_its_partner_leaves_the_pool(self):
        run, twin = self.twin_round(), self.r.sha["t1copy"]
        run("d3-mini-2026-11-03")
        fetched = self.r.fetched.count(twin)
        state = json.loads((self.r.state / "pool.json").read_text())
        state["bodies"][self.r.sha["t1"]]["pruned"] = True
        state["members"] = [self.r.sha["t2"]]
        (self.r.state / "pool.json").write_text(json.dumps(state))
        record = run("d3-mini-2026-11-04")
        self.assertEqual(self.r.fetched.count(twin), fetched + 1)
        self.assertIn(twin, record["new"])

    def test_a_hotkeys_new_body_is_never_its_own_seated_bodys_twin(self):
        def twins(rows, other_rows):
            return rows == other_rows

        self.r.run(twins=twins)
        self.r.commits[1] = (
            "t1copy",
            150,
        )  # trainer 1's newer body predicts exactly like its old one
        record = self.r.run(round_id="d3-mini-2026-11-04", twins=twins, copies=lambda *a, **k: None)
        self.assertNotIn(self.r.sha["t1copy"], record["rejected"])
        self.assertIn(self.r.sha["t1copy"], record["pool"])

    def test_an_exhausted_bank_skips_the_round_and_keeps_the_last_decision(self):
        first = self.r.run()
        decision = (self.r.state / "decision.json").read_text()
        config = self.r.config(subset_size=len(self.r.bank) + 1)
        for day in ("04", "05"):
            self.assertIsNone(self.r.run(round_id=f"d3-mini-2026-11-{day}", config=config))
        self.assertEqual((self.r.state / "decision.json").read_text(), decision)
        rounds = [json.loads(line) for line in self.r.ledger.path.read_text().splitlines()]
        self.assertEqual(
            [r["round"] for r in rounds][1:], ["d3-mini-2026-11-04", "d3-mini-2026-11-05"]
        )
        self.assertEqual([r["ids"] for r in rounds][1:], [[], []])
        self.assertTrue(first["members"])

    def test_the_crown_must_beat_the_current_soup(self):
        first = self.r.run()
        self.assertIsNotNone(first["crown"])  # no soup yet: the house bodies are the bar
        again = self.r.run(round_id="d3-mini-2026-11-04")
        # The seated soup scores 100 on the new subset; no single body beats it by the margin.
        self.assertGreater(again["soup_bar"], max(again["solo"].values()))
        self.assertIsNone(again["crown"])

    def test_a_current_soup_that_cannot_grade_burns_the_crown_not_the_round(self):
        self.r.run()
        shutil.rmtree(self.r.state / "cache")  # regrade everything in the next round
        t1, t2 = self.r.sha["t1"], self.r.sha["t2"]
        current = "soup-" + hashlib.sha256("+".join(sorted((t1, t2))).encode()).hexdigest()[:32]
        fake = self.r.runner

        def runner_for(folder):
            name = Path(folder).name
            if name.startswith(t2[:32]) or name.startswith(current):

                def fail(cases):
                    raise ValueError("bad output")

                return fail
            return fake(folder)

        self.r.runner = runner_for
        record = self.r.run(round_id="d3-mini-2026-11-04")
        self.assertIsNone(record["soup_bar"])
        self.assertIsNone(record["crown"])
        self.assertTrue(record["members"])

    def test_rates_hook_sets_seat_and_crown_pay(self):
        def rates(sha, info, house):
            return 0.25 if sha == self.r.sha["t1"] else (0.0 if sha in house else 1.0)

        record = self.r.run(rates=rates)
        members = self.by_body(record)
        self.assertAlmostEqual(members[self.r.sha["t1"]]["seat_weight"], 0.1)
        self.assertAlmostEqual(members[self.r.sha["t2"]]["seat_weight"], 0.4)
        self.assertAlmostEqual(record["crown"]["weight"], 0.05)

    def test_a_commitment_failing_validation_fails_and_leaves_no_files(self):
        def fetch(claim, dest):
            dest = self.r.fetch(claim, dest)
            if claim["sha256"] == self.r.sha["weak"]:
                (dest / "zils-owner.json").write_text(json.dumps({"hotkey": "5Someone"}))
            return dest

        record = self.r.run(fetch_body=fetch)
        key = f"{HOTKEYS[4]}:miner/weak@{REVISION}"
        self.assertIn("zils-owner.json", record["failed"][key])
        self.assertIn("zils-owner.json", record["refused"]["4"])
        self.assertNotIn(self.r.sha["weak"], record["rejected"])
        self.assertFalse((self.r.state / "bodies" / self.r.sha["weak"]).exists())
        self.assertEqual([p.name for p in (self.r.state / "bodies").glob(".*")], [])

    def hostile(self, name, folder, block, sha=None):
        """Commit a hand-made body repo `folder` as `name`, claiming `sha` (default: a made-up hash)."""
        self.r.repos[name] = folder
        self.r.sha[name] = sha or hashlib.sha256(name.encode()).hexdigest()
        uid = max(self.r.commits) + 1
        self.r.commits[uid] = (name, block)
        self.enterContext(patch.dict(HOTKEYS, {uid: HOTKEYS.get(uid, f"5Hostile{uid}")}))
        return self.r.sha[name]

    def test_hostile_bodies_are_rejected_and_the_round_completes(self):
        nan = strengths(1, 1, 0, 0, 0)
        nan["visual.patch.weight"][0] = math.nan
        repos = self.tmp / "hostile"
        folder = test_bodies.checkpoint(repos / "nan", nan, prefix="model.", owner=None)
        names = list(bodies.load_index(self.r.chassis))
        nan_sha = self.hostile("nan", folder, 90, sha=bodies._digest(folder, names)[0])
        header, total = test_bodies.raw_header()
        header = {
            STRENGTH: {"dtype": "BF16", "shape": [5], "data_offsets": ["0", "10"]},
            "language_model.norm.weight": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [10.0, 14.0],
            },
            "visual.patch.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [14, 18]},
        }
        offsets = test_bodies.raw_body(repos / "offsets", json.dumps(header).encode(), bytes(18))
        (offsets / "README.md").write_text("---\nlicense: mit\n---\n")
        offsets_sha = self.hostile("offsets", offsets, 91)
        deep = test_bodies.raw_body(repos / "deep", b"[" * 200000 + b"]" * 200000)
        (deep / "README.md").write_text("---\nlicense: mit\n---\n")
        deep_sha = self.hostile("deep", deep, 92)
        record = self.r.run()
        self.assertIn("finite", record["rejected"][nan_sha])  # its own weights: the hash is banned
        # Unreadable files never verify a hash, so only the commitment fails.
        self.assertIn("malformed", record["refused"]["6"])
        self.assertIn("unreadable", record["refused"]["7"])
        self.assertNotIn(offsets_sha, record["rejected"])
        self.assertNotIn(deep_sha, record["rejected"])
        self.assertEqual(set(self.by_body(record)), {self.r.sha["t1"], self.r.sha["t2"]})

    def test_any_exception_from_one_body_rejects_that_body_only(self):
        def copies(sha, folder, earlier, *, soup):
            if sha == self.r.sha["weak"]:
                raise RuntimeError("sketch blew up")
            return None

        record = self.r.run(copies=copies)
        self.assertIn("RuntimeError", record["refused"]["4"])
        self.assertNotIn(self.r.sha["weak"], record["pool"])
        self.assertEqual(set(self.by_body(record)), {self.r.sha["t1"], self.r.sha["t2"]})

    def failing(self, error, *, chassis_error=None):
        """A runner whose `weak` body (and, with `chassis_error`, the chassis) raises."""
        fake = self.r.runner

        def runner_for(folder):
            run = fake(folder)
            broken = None
            if Path(folder).name.startswith(self.r.sha["weak"][:32]):
                broken = error
            elif chassis_error is not None and Path(folder).name.startswith("chassis-"):
                broken = chassis_error
            if broken is None:
                return run

            def fail(cases):
                raise broken

            return fail

        return runner_for

    def test_a_gpu_fault_fails_the_round_and_never_the_body(self):
        healthy = self.r.runner
        for error in (
            torch.cuda.OutOfMemoryError("CUDA out of memory"),
            RuntimeError("CUDA error: an illegal memory access"),
        ):
            with self.subTest(error=str(error)):
                self.r.runner = self.failing(error)
                with self.assertRaises(RuntimeError):
                    self.r.run()
        self.r.runner = healthy
        record = self.r.run()
        self.assertIn(self.r.sha["weak"], record["pool"])
        self.assertEqual(record["rejected"], {})

    def test_a_chassis_that_cannot_grade_fails_the_round(self):
        self.r.runner = self.failing(
            ValueError("bad output"), chassis_error=ValueError("driver gone")
        )
        with self.assertRaisesRegex(RuntimeError, "chassis"):
            self.r.run()
        self.assertTrue((self.r.state / "bodies" / self.r.sha["weak"]).exists())
        state = json.loads((self.r.state / "pool.json").read_text())
        self.assertNotIn(self.r.sha["weak"], state["rejected"])

    def test_a_failed_bodys_memory_is_released_before_the_chassis_check(self):
        import weakref

        fake, events, held = self.r.runner, [], []

        class Model:  # stands in for the failed body's loaded weights
            pass

        def runner_for(folder):
            if Path(folder).name.startswith("chassis-"):
                events.append(("chassis", held[0]() is None))
                return fake(folder)
            run = fake(folder)
            if not Path(folder).name.startswith(self.r.sha["weak"][:32]):
                return run
            model = Model()
            held.append(weakref.ref(model))

            def fail(cases):
                del cases
                raise ValueError(f"bad output from {model!r}")

            return fail

        self.r.runner = runner_for
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache", side_effect=lambda: events.append(("empty_cache",))),
        ):
            self.r.run()
        self.assertEqual(events[:2], [("empty_cache",), ("chassis", True)])

    def test_a_body_failing_grading_is_banned_only_after_three_rounds(self):
        weak = self.r.sha["weak"]
        self.r.runner = self.failing(ValueError("bad output"))
        for day, banned in (("03", False), ("04", False), ("05", True)):
            with self.subTest(day=day):
                record = self.r.run(round_id=f"d3-mini-2026-11-{day}")
                self.assertNotIn(weak, record["pool"])
                self.assertEqual(set(self.by_body(record)), {self.r.sha["t1"], self.r.sha["t2"]})
                state = json.loads((self.r.state / "pool.json").read_text())
                self.assertEqual(weak in state["rejected"], banned)
                self.assertEqual((self.r.state / "bodies" / weak).exists(), not banned)
        self.assertEqual(len(state["grade_failures"][weak]), 3)

    def test_a_twin_check_failure_is_a_retryable_strike(self):
        def twins(rows, other_rows):
            if twins.calls == 0:
                twins.calls += 1
                raise ValueError("bad rows")
            return False

        twins.calls = 0
        record = self.r.run(twins=twins)
        self.assertEqual(record["rejected"], {})
        self.assertEqual(len(record["grade_failures"]), 1)
        self.assertTrue(record["members"])
        failed = next(iter(record["grade_failures"]))
        again = self.r.run(round_id="d3-mini-2026-11-04", twins=lambda a, b: False)
        self.assertIn(failed, again["pool"])  # retried in the next round

    def test_each_body_is_built_once_and_graded_once_on_the_whole_subset(self):
        from zils import soup

        build, built, sizes = soup.build, [], []
        fake = self.r.runner

        def counting(out, chassis, members, *, model):
            built.append(Path(out).name)
            return build(out, chassis, members, model=model)

        def runner_for(folder):
            run = fake(folder)

            def sized(cases):
                sizes.append(len(cases))
                return run(cases)

            return sized

        self.r.runner = runner_for
        with patch("zils.soup.build", side_effect=counting):
            record = self.r.run()
        for sha in record["pool"]:
            self.assertEqual(sum(n.startswith(sha[:32]) for n in built), 1, sha)
            graded = [v for k, v in fake.graded.items() if Path(k).name.startswith(sha[:32])]
            self.assertEqual(graded, [300])  # the screen items are read from the full-subset rows
        self.assertLessEqual(max(sizes), soup_round.GRADE_CHUNK)

    def test_a_crash_mid_body_keeps_its_graded_chunks(self):
        self.r.runner.fail_after = 300 + 250  # the second body dies in its second chunk
        with self.assertRaises(RuntimeError):  # the chassis cannot grade either: a round fault
            self.r.run()
        self.r.runner.fail_after = None
        record = self.r.run()
        bodies_graded = sum(
            v
            for k, v in self.r.runner.graded.items()
            if "/tmp/soup-" not in k and "/tmp/final" not in k
        )
        self.assertLess(bodies_graded - 300 * len(record["pool"]), soup_round.GRADE_CHUNK)

    def grief_snapshot(self, victim, griefer_repo, block=95, uid=5):
        """The roster commits as usual, and `uid` first commits `victim`'s hash from `griefer_repo`."""
        snapshot = self.r.snapshot()
        snapshot["commitments"][str(uid)] = {
            "block": block,
            "data": bodies.format_commitment(self.r.sha[victim], REVISION, griefer_repo),
        }
        return snapshot

    def test_a_griefer_cannot_ban_another_committers_hash(self):
        sha = self.r.sha
        for label, repo in (("hash mismatch", "miner/weak"), ("no owner file", "miner/t1")):
            with self.subTest(label):
                r = Round(self.tmp / label.replace(" ", "-"))
                self.r = r

                def fetch(claim, dest, r=r, label=label):
                    dest = r.fetch(claim, dest)
                    if label == "no owner file" and claim["hotkey"] == HOTKEYS[5]:
                        (dest / "zils-owner.json").unlink()
                    return dest

                snapshot = self.grief_snapshot("t1", repo)
                record = r.run(snapshot=snapshot, fetch_body=fetch)
                self.assertNotIn(r.sha["t1"], record["rejected"])
                self.assertIn(r.sha["t1"], self.by_body(record))
                self.assertEqual(self.by_body(record)[r.sha["t1"]]["hotkey"], HOTKEYS[1])
                self.assertTrue((r.state / "bodies" / r.sha["t1"]).exists())
                key = f"{HOTKEYS[5]}:{repo}@{REVISION}"
                self.assertIn(key, record["failed"])
                fetched = len(r.fetched)
                again = r.run(round_id="d3-mini-2026-11-04", snapshot=snapshot, fetch_body=fetch)
                self.assertEqual(len(r.fetched), fetched)  # the failed claim is not fetched again
                self.assertIn("5", again["refused"])
                self.assertIn(r.sha["t1"], self.by_body(again))
        del sha

    def test_weight_failures_ban_the_hash(self):
        nan = strengths(1, 0, 0, 0, 0)
        nan["visual.patch.weight"][0] = math.nan
        folder = test_bodies.checkpoint(self.tmp / "nanrepo", nan, prefix="model.", owner=None)
        names = list(bodies.load_index(self.r.chassis))
        nan_sha = bodies._digest(folder, names)[0]
        self.r.repos["nan"], self.r.sha["nan"] = folder, nan_sha
        self.r.commits[5] = ("nan", 90)
        record = self.r.run()
        self.assertIn("finite", record["rejected"][nan_sha])
        self.r.commits[6] = ("nan", 150)
        self.r.commits.pop(5)
        again = self.r.run(round_id="d3-mini-2026-11-04")
        self.assertIn("rejected in an earlier round", again["refused"]["6"])

    def test_a_download_deadline_is_transient(self):
        def fetch(claim, dest):
            if claim["sha256"] == self.r.sha["weak"]:
                raise OSError("artifact download deadline exceeded")
            return self.r.fetch(claim, dest)

        record = self.r.run(fetch_body=fetch)
        self.assertIn("deadline", record["refused"]["4"])
        self.assertNotIn(self.r.sha["weak"], record["rejected"])
        self.assertEqual(record["failed"], {})
        self.assertNotIn(self.r.sha["weak"], record["new"])
        # Retried once the hotkey's back-off (none here) is over.
        retry = self.r.config(failed_claim_backoff_rounds=0)
        again = self.r.run(round_id="d3-mini-2026-11-04", config=retry)
        self.assertIn(self.r.sha["weak"], again["pool"])

    def test_an_incomplete_http_read_is_transient(self):
        import http.client

        def fetch(claim, dest):
            if claim["sha256"] == self.r.sha["weak"]:
                raise http.client.IncompleteRead(b"", 10)
            return self.r.fetch(claim, dest)

        record = self.r.run(fetch_body=fetch)
        self.assertIn("download failed", record["refused"]["4"])
        self.assertEqual(record["failed"], {})

    def test_finish_never_deletes_a_seated_or_earlier_admitted_body(self):
        first = self.r.run()
        seated = first["members"][0]["body_sha256"]
        round_dir = self.r.state / "rounds" / "d3-mini-2026-11-04"
        private = {**first, "round_id": "d3-mini-2026-11-04", "new": []}
        private["rejected"] = {seated: "simulated", self.r.sha["weak"]: "simulated"}
        round_dir.mkdir(parents=True)
        (round_dir / "private.json").write_text(json.dumps(private))
        soup_round._finish(self.r.state, round_dir, self.r.ledger)
        for sha in (seated, self.r.sha["weak"]):
            self.assertTrue((self.r.state / "bodies" / sha).exists())
        state = json.loads((self.r.state / "pool.json").read_text())
        self.assertIn(self.r.sha["weak"], state["bodies"])

    def test_new_bodies_per_round_are_capped_in_commitment_order(self):
        record = self.r.run(config=self.r.config(max_new_per_round=2))
        self.assertEqual(sorted(record["new"]), sorted([self.r.sha["t1"], self.r.sha["t2"]]))
        self.assertIn("cap", record["refused"]["4"])
        self.assertNotIn(self.r.sha["weak"], self.r.fetched)
        again = self.r.run(round_id="d3-mini-2026-11-04", config=self.r.config(max_new_per_round=2))
        self.assertEqual(again["new"], [self.r.sha["weak"]])

    def junk(self, *uids):
        """Snapshot where each of `uids` commits a made-up hash from trainer 1's repo, early."""
        snapshot = self.r.snapshot()
        for n, uid in enumerate(uids):
            snapshot["commitments"][str(uid)] = {
                "block": 10 + n,
                "data": bodies.format_commitment(f"{uid:064x}", REVISION, "miner/t1"),
            }
        return snapshot

    def test_dead_commitments_do_not_starve_intake_and_retire_after_three_rounds(self):
        dead = range(20, 32)  # twelve commitments whose repos never answer, ahead of everyone
        self.enterContext(patch.dict(HOTKEYS, {uid: f"5Dead{uid}" for uid in dead}))
        snapshot = self.r.snapshot()
        for n, uid in enumerate(dead):
            snapshot["commitments"][str(uid)] = {
                "block": 10 + n,
                "data": bodies.format_commitment(f"{uid:064x}", REVISION, f"dead/r{uid}"),
            }
        calls = []

        def fetch(claim, dest):
            if claim["repo"].startswith("dead/"):
                calls.append(claim["repo"])
                raise OSError("connection refused")
            return self.r.fetch(claim, dest)

        config = self.r.config(failed_claim_backoff_rounds=0)  # retry the dead claims every round
        for day in ("03", "04", "05"):
            record = self.r.run(
                round_id=f"d3-mini-2026-11-{day}",
                snapshot=snapshot,
                fetch_body=fetch,
                config=config,
            )
            if day == "03":
                self.assertEqual(
                    sorted(record["new"]),
                    sorted(self.r.sha[n] for n in ("t1", "t2", "weak")),
                )
        self.assertEqual(len(record["failed"]), 12)  # retired on the third failed round
        self.assertEqual(len(calls), 36)
        again = self.r.run(
            round_id="d3-mini-2026-11-06", snapshot=snapshot, fetch_body=fetch, config=config
        )
        self.assertEqual(len(calls), 36)
        self.assertIn("earlier round", again["refused"]["20"])
        state = json.loads((self.r.state / "pool.json").read_text())
        self.assertFalse(set(state["bodies"]) & {f"{uid:064x}" for uid in dead})  # no hash banned
        self.assertFalse(set(state["rejected"]) & {f"{uid:064x}" for uid in dead})

    def test_tiny_junk_files_cannot_keep_an_honest_body_out(self):
        junk = range(20, 32)  # twelve hotkeys committing a 1-byte safetensors file every day
        self.enterContext(patch.dict(HOTKEYS, {uid: f"5Junk{uid}" for uid in junk}))
        calls = []

        def fetch(claim, dest):
            if not claim["repo"].startswith("junk/"):
                return self.r.fetch(claim, dest)
            calls.append(claim["repo"])
            dest = Path(dest)
            dest.mkdir(parents=True)
            (dest / "model.safetensors").write_bytes(b"x")
            (dest / "README.md").write_text("---\nlicense: mit\n---\n")
            (dest / "zils-owner.json").write_text(json.dumps({"hotkey": claim["hotkey"]}))
            return dest

        def snapshot(day):
            value = self.r.snapshot()
            for n, uid in enumerate(junk):
                value["commitments"][str(uid)] = {
                    "block": 10 + n,  # ahead of everyone, and a fresh revision every day
                    "data": bodies.format_commitment(f"{uid:064x}", f"{day:040x}", f"junk/r{uid}"),
                }
            return value

        first = self.r.run(snapshot=snapshot(3), fetch_body=fetch)
        self.assertEqual(sorted(first["new"]), sorted(self.r.sha[n] for n in ("t1", "t2", "weak")))
        self.assertEqual(len(calls), 12)
        for day in (4, 5, 6):  # each junk hotkey sits out the next three rounds
            record = self.r.run(
                round_id=f"d3-mini-2026-11-0{day}", snapshot=snapshot(day), fetch_body=fetch
            )
            self.assertEqual(len(calls), 12)
            self.assertIn("backs off", record["refused"]["20"])
        self.r.run(round_id="d3-mini-2026-11-07", snapshot=snapshot(7), fetch_body=fetch)
        self.assertEqual(len(calls), 24)

    def test_failed_claim_backoff_is_a_pool_setting(self):
        checked = model_pools.validate(pools())["pools"]["d3-mini"]
        self.assertEqual(checked["failed_claim_backoff_rounds"], 3)
        model_pools.validate(pools(soup_entry(failed_claim_backoff_rounds=0)))
        for bad in (-1, 31, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model_pools.validate(pools(soup_entry(failed_claim_backoff_rounds=bad)))

    def test_failed_claims_count_toward_the_claims_cap(self):
        config = self.r.config(max_claims_per_round=2)
        snapshot = self.junk(5, 6)
        record = self.r.run(config=config, snapshot=snapshot)
        self.assertEqual(len(record["failed"]), 2)
        self.assertEqual(record["new"], [])
        for uid in ("1", "2", "4"):
            self.assertIn("claims cap", record["refused"][uid])
        again = self.r.run(round_id="d3-mini-2026-11-04", config=config, snapshot=snapshot)
        self.assertEqual(sorted(again["new"]), sorted([self.r.sha["t1"], self.r.sha["t2"]]))

    def test_download_bytes_per_round_are_capped(self):
        size = sum(p.stat().st_size for p in self.r.repos["t1"].iterdir())
        record = self.r.run(config=self.r.config(max_download_bytes_per_round=size))
        self.assertEqual(record["new"], [self.r.sha["t1"]])
        self.assertIn("download budget", record["refused"]["2"])

    def test_intake_progress_survives_a_crash(self):
        def dies_on_weak(claim, dest):
            if claim["sha256"] == self.r.sha["weak"]:
                raise SystemExit("killed at the round deadline")
            return self.r.fetch(claim, dest)

        with self.assertRaises(SystemExit):
            self.r.run(fetch_body=dies_on_weak)
        self.assertEqual(len(self.r.fetched), 3)  # the house body, trainer 1 and trainer 2
        record = self.r.run()
        expected = [self.r.sha[n] for n in ("house", "t1", "t2", "weak")]
        self.assertEqual(sorted(self.r.fetched), sorted(expected))  # nothing fetched twice
        self.assertEqual(len(record["new"]), 3)

    def test_default_intake_limits(self):
        self.assertEqual(soup_round.MAX_CLAIMS_PER_ROUND, 12)
        self.assertEqual(soup_round.MAX_DOWNLOAD_BYTES_PER_ROUND, 60 * 1024**3)
        checked = model_pools.validate(pools())["pools"]["d3-mini"]
        self.assertEqual(checked["max_claims_per_round"], 12)
        self.assertEqual(checked["max_download_bytes_per_round"], 60 * 1024**3)
        for bad in ({"max_claims_per_round": 0}, {"max_download_bytes_per_round": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model_pools.validate(pools(soup_entry(**bad)))

    def test_default_new_body_cap_is_six(self):
        self.assertEqual(soup_round.MAX_NEW_PER_ROUND, 6)

    def test_bodies_out_of_the_pool_for_three_rounds_are_pruned_and_old_soups_dropped(self):
        weak = self.r.sha["weak"]
        first = self.r.run()
        self.assertIn(weak, first["pool"])
        small = self.r.config(max_pool=3)  # the two members and the house body only
        used = []
        for day in (4, 5, 6):
            record = self.r.run(round_id=f"d3-mini-2026-11-0{day}", config=small)
            self.assertNotIn(weak, record["pool"])
            used.append(set(record["soup_keys"]))
            if day < 6:
                self.assertTrue((self.r.state / "bodies" / weak).exists())
        self.assertFalse((self.r.state / "bodies" / weak).exists())
        state = json.loads((self.r.state / "pool.json").read_text())
        self.assertIn(weak, state["bodies"])  # the hash is kept, so a re-commit is not re-admitted
        self.assertTrue(list((self.r.state / "sketches").glob(f"{weak}.npz")))
        for sha in (self.r.sha["t1"], self.r.sha["t2"], self.r.sha["house"]):
            self.assertTrue((self.r.state / "bodies" / sha).exists())
        soups = {p.stem for p in (self.r.state / "cache").glob("soup-*.jsonl")}
        self.assertEqual(soups, set().union(*used))
        self.assertFalse(set(first["soup_keys"]) - set().union(*used) & soups)
        later = self.r.run(round_id="d3-mini-2026-11-07")
        self.assertNotIn(weak, later["pool"])

    def test_the_sketch_seed_is_a_local_secret(self):
        config = self.r.config()
        del config["sketch_seed"]
        self.r.run(config=config)
        path = self.r.state / "sketch-seed"
        secret = path.read_text().strip()
        self.assertRegex(secret, "^[0-9a-f]{64}$")
        self.assertEqual(path.stat().st_mode & 0o077, 0)
        sketches = sorted(p.name for p in (self.r.state / "sketches").glob("*.npz"))
        self.r.run(round_id="d3-mini-2026-11-04", config=config)
        self.assertEqual(path.read_text().strip(), secret)  # generated once, then reused
        self.assertEqual(
            sorted(p.name for p in (self.r.state / "sketches").glob("*.npz")), sketches
        )
        for name in ("d3-mini-2026-11-03", "d3-mini-2026-11-04"):
            public = (self.r.state / "rounds" / name / "public.json").read_text()
            self.assertNotIn(secret, public)
            self.assertNotIn("sketch", public)
        self.assertNotIn(secret, (self.r.state / "decision.json").read_text())

    def test_sketches_are_keyed_by_body_hash_with_the_seed_tag_inside(self):
        import numpy as np

        record = self.r.run()
        tag = hashlib.sha256(bytes.fromhex("5e" * 32)).hexdigest()[:16]
        for sha in record["pool"]:
            with np.load(self.r.state / "sketches" / f"{sha}.npz") as stored:
                self.assertEqual(str(stored["tag"]), tag)

    def test_a_changed_sketch_seed_fails_loudly(self):
        self.r.run()
        changed = self.r.config(sketch_seed="7a" * 32)
        with self.assertRaisesRegex(ValueError, "sketch_seed .*must never change"):
            self.r.run(round_id="d3-mini-2026-11-04", config=changed)
        unset = self.r.config()
        del unset["sketch_seed"]  # a generated local seed differs from the configured one too
        with self.assertRaisesRegex(ValueError, "sketch_seed"):
            self.r.run(round_id="d3-mini-2026-11-04", config=unset)

    def test_a_configured_sketch_seed_never_reaches_the_public_record(self):
        self.r.run()
        for path in (
            self.r.state / "decision.json",
            self.r.state / "rounds" / "d3-mini-2026-11-03" / "public.json",
        ):
            self.assertNotIn("5e" * 32, path.read_text())

    def test_a_lightly_changed_house_re_upload_is_rejected_as_a_copy(self):
        BODIES["rehouse"] = strengths(0.25, 0.25, 0.25, 0.25, 0.25, norm=2)
        self.addCleanup(BODIES.pop, "rehouse")
        r = Round(self.tmp / "rehouse")
        r.commits[5] = ("rehouse", 90)
        record = r.run()
        self.assertEqual(record["rejected"], {r.sha["rehouse"]: "copy: house re-upload"})
        self.assertNotIn(r.sha["rehouse"], record["pool"])

    def test_default_checks_reject_a_layer_copy_and_admit_a_house_fine_tune_as_a_trainer(self):
        self.r.commits.update({5: ("scout", 50), 6: ("t1copy", 104)})
        record = self.r.run()
        sha = self.r.sha
        self.assertEqual(record["rejected"], {sha["t1copy"]: "copy: cosine"})
        self.assertFalse((self.r.state / "bodies" / sha["t1copy"]).exists())
        # A light fine-tune of a house model is allowed and paid as a trainer's.
        self.assertIn(sha["scout"], record["pool"])
        self.assertEqual(record["rates"][sha["scout"]], 1.0)
        self.assertEqual(record["rates"][sha["t1"]], 1.0)
        self.assertEqual(record["rates"][sha["house"]], 0.0)
        self.assertEqual(set(self.by_body(record)), {sha["t1"], sha["t2"]})
        # One fixed sketch seed per pool: each body is sketched once, whatever the round's seed.
        self.assertEqual(len(list((self.r.state / "sketches").glob("*.npz"))), 6)
        self.r.run(round_id="d3-mini-2026-11-04")
        self.assertEqual(len(list((self.r.state / "sketches").glob("*.npz"))), 6)

    def test_a_merge_with_another_hotkeys_body_is_a_copy_but_not_with_ones_own(self):
        BODIES["merge"] = strengths(0.625, 0.625, 0.125, 0.125, 0.125)  # (t1 + house) / 2
        self.addCleanup(BODIES.pop, "merge")
        r = Round(self.tmp / "merge")
        r.run()
        r.commits[6] = ("merge", 150)  # another hotkey merges trainer 1's body with the house body
        record = r.run(round_id="d3-mini-2026-11-04")
        self.assertIn("explained by earlier", record["refused"]["6"])
        own = Round(self.tmp / "own")
        own.run()
        own.commits[1] = ("merge", 150)  # trainer 1 merges its own body with the house body
        record = own.run(round_id="d3-mini-2026-11-04")
        self.assertEqual(record["rejected"], {})
        self.assertIn(own.sha["merge"], record["new"])

    def merge_round(self):
        BODIES["merge"] = strengths(0.625, 0.625, 0.125, 0.125, 0.125)  # (t1 + house) / 2
        self.addCleanup(BODIES.pop, "merge")
        r = Round(self.tmp / "explained")
        r.run()
        r.commits[6] = ("merge", 150)
        return r

    def test_explained_by_earlier_never_bans_the_hash(self):
        r = self.merge_round()
        record = r.run(round_id="d3-mini-2026-11-04")
        merge = r.sha["merge"]
        self.assertNotIn(merge, record["rejected"])
        self.assertIn("explained by earlier", record["refused"]["6"])
        state = json.loads((r.state / "pool.json").read_text())
        self.assertNotIn(merge, state["rejected"])
        # Unchanged claim and unchanged earlier bodies: not downloaded again.
        fetched = len(r.fetched)
        again = r.run(round_id="d3-mini-2026-11-05")
        self.assertEqual(len(r.fetched), fetched)
        self.assertIn("explained by earlier", again["refused"]["6"])
        # Another hotkey committing the same hash is still evaluated (and judged on its own).
        r.commits[4] = ("merge", 160)
        third = r.run(round_id="d3-mini-2026-11-06")
        self.assertEqual(len(r.fetched), fetched + 1)
        self.assertIn("explained by earlier", third["refused"]["4"])

    def test_an_explained_claim_is_retried_when_re_committed(self):
        r = self.merge_round()
        r.run(round_id="d3-mini-2026-11-04")
        fetched = len(r.fetched)
        snapshot = r.snapshot()
        snapshot["commitments"]["6"]["data"] = bodies.format_commitment(
            r.sha["merge"], "c" * 40, "miner/merge"
        )
        r.run(round_id="d3-mini-2026-11-05", snapshot=snapshot)
        self.assertEqual(len(r.fetched), fetched + 1)

    def test_an_explained_claim_is_retried_when_the_earlier_body_leaves_the_pool(self):
        r = self.merge_round()
        r.run(round_id="d3-mini-2026-11-04")
        fetched = len(r.fetched)
        state = json.loads((r.state / "pool.json").read_text())
        state["bodies"][r.sha["t1"]]["pruned"] = True  # trainer 1's body left the pool
        state["members"] = [r.sha["t2"]]
        (r.state / "pool.json").write_text(json.dumps(state))
        record = r.run(round_id="d3-mini-2026-11-05")
        self.assertEqual(len(r.fetched), fetched + 1)
        self.assertIn(r.sha["merge"], record["new"])

    def test_the_other_hotkey_share_bar_is_a_pool_setting(self):
        from zils import soup_copies

        r = self.merge_round()
        seen, is_copy = [], soup_copies.is_copy

        def spy(*args, **kwargs):
            seen.append(kwargs["others_max"])
            return is_copy(*args, **kwargs)

        with patch("zils.soup_copies.is_copy", side_effect=spy):
            r.run(round_id="d3-mini-2026-11-04", config=r.config(others_max=0.9))
        self.assertEqual(set(seen), {0.9})
        checked = model_pools.validate(pools())["pools"]["d3-mini"]
        self.assertEqual(checked["others_max"], 0.3)
        for good in (0.1, 0.9, 0.5):
            model_pools.validate(pools(soup_entry(others_max=good)))
        for bad in (0.05, 0.95, "0.3", True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model_pools.validate(pools(soup_entry(others_max=bad)))

    def test_an_identical_upload_of_the_base_is_rejected(self):
        BODIES["stock"] = strengths(0, 0, 0, 0, 0, norm=3)
        self.addCleanup(BODIES.pop, "stock")
        r = Round(self.tmp / "stock")
        r.commits[5] = ("stock", 90)
        record = r.run()
        self.assertEqual(record["rejected"], {r.sha["stock"]: "copy: no change from the base"})

    def test_house_registry_pins_and_null_hashes(self):
        house = self.r.config()["house"][0]
        record = self.r.run(config=self.r.config(house=[{**house, "body_sha256": None}]))
        self.assertIn(self.r.sha["house"], record["pool"])
        fresh = Round(self.tmp / "pinned")
        pinned = fresh.config(house=[{**house, "body_sha256": fresh.sha["house"]}])
        self.assertIn(fresh.sha["house"], fresh.run(config=pinned)["pool"])

    def test_a_bad_house_entry_is_skipped_and_recorded_not_fatal(self):
        house = self.r.config()["house"][0]
        broken = {
            "name": "broken",
            "repo": "open/broken",
            "revision": REVISION,
            "body_sha256": None,
        }
        wrong = {**house, "name": "wrong", "revision": "d" * 40, "body_sha256": "0" * 64}
        fetch = self.r.fetch

        def fetch_body(claim, dest):
            if claim["repo"] == "open/broken":
                raise ValueError("body tensor language_model.layers.0.mlp.weight is malformed")
            return fetch(claim, dest)

        config = self.r.config(house=[house, broken, wrong])
        record = self.r.run(config=config, fetch_body=fetch_body)
        self.assertIn(self.r.sha["house"], record["pool"])
        self.assertEqual(set(record["house_failures"]), {"broken", "wrong"})
        self.assertIn("malformed", record["house_failures"]["broken"])
        self.assertIn("pinned hash", record["house_failures"]["wrong"])
        self.assertTrue(record["members"])
        self.assertNotIn("house_failures", json.dumps(self.public()))

    def test_rate_constants_match_soup_rates(self):
        from zils import soup_rates

        self.assertEqual(
            (soup_rates.HOUSE_RATE, soup_rates.SCOUT_RATE, soup_rates.TRAINER_RATE),
            (soup_round.HOUSE_RATE, soup_round.SCOUT_RATE, soup_round.TRAINER_RATE),
        )

    def test_cache_keys_are_bank_and_profile_scoped(self):
        self.r.run()
        names = {p.stem for p in (self.r.state / "cache").iterdir()}
        bank_sha = soup_round.bank_sha256(self.r.bank)
        pin = d3.PINS["files"]["readout.safetensors"][:12]
        tag = f"{bank_sha[:16]}-{models.D3}-{pin}"
        self.assertIn(f"{self.r.sha['t1'][:32]}-{tag}", names)
        self.assertTrue(all(n.endswith(tag) for n in names))

    def test_a_profile_change_never_reuses_cached_rows(self):
        first = self.r.run()
        before = {p.stem for p in (self.r.state / "cache").iterdir()}
        test_soup.save_file(
            {"readout": test_soup.torch.ones(2, 3)}, str(self.r.chassis / "readout.safetensors")
        )
        pinned = hashlib.sha256((self.r.chassis / "readout.safetensors").read_bytes()).hexdigest()
        self.enterContext(patch.dict(d3.PINS["files"], {"readout.safetensors": pinned}))
        graded = sum(self.r.runner.graded.values())
        again = self.r.run(round_id="d3-mini-2026-11-04")
        after = {p.stem for p in (self.r.state / "cache").iterdir()} - before
        for sha in again["pool"]:
            self.assertTrue(any(n.startswith(sha[:32]) and n.endswith(pinned[:12]) for n in after))
        self.assertGreaterEqual(
            sum(self.r.runner.graded.values()) - graded, 300 * len(first["pool"])
        )


MEMBERS = {"1": {"hotkey": "a"}, "2": {"hotkey": "b"}, "3": {"hotkey": "c"}}
DRAW = {"manifest_sha256": "a" * 64, "dataset_sha256": "b" * 64}
HOUSE = [{"name": "house", "repo": "open/house", "revision": "e" * 40, "body_sha256": None}]


def soup_entry(**changes):
    return {
        "type": "soup",
        "model": models.D3,
        "enabled": True,
        "share": 0.4,
        "house": HOUSE,
        "schedule": {"deadline_utc": "00:00"},
        "sketch_seed": "5e" * 32,
        **changes,
    }


def pools(soup=None, king_share=0.6):
    king = {
        "incumbent_sha256": "c" * 64,
        "draws": [DRAW],
        "enabled": True,
        "model": "d3-lite-0.8b-v1",
        "share": king_share,
        "kings": ["a"],
    }
    return {
        "share": 0.7,
        "burn_uid": 0,
        "pools": {"d3-mini": soup or soup_entry(), "d3-lite": king},
    }


def decision(members, crown=None):
    return {
        "pool_id": "d3-mini",
        "members": [
            {"hotkey": h, "body_sha256": "f" * 64, "rate": 1.0, "contribution": 1, "seat_weight": w}
            for h, w in members
        ],
        "crown": crown,
    }


class SoupPoolWeightsTest(unittest.TestCase):
    def test_a_partial_soup_burns_its_unpaid_share(self):
        weights, record = model_pools.weights(pools(), {"d3-mini": decision([("c", 0.4)])}, MEMBERS)
        self.assertAlmostEqual(weights[3], 0.4 * 0.4)
        self.assertAlmostEqual(weights[0], 0.6 * 0.4)
        self.assertAlmostEqual(weights[1], 0.6)
        self.assertAlmostEqual(record["burned"], 0.6 * 0.4)
        alone = pools(soup_entry(share=1.0), king_share=0.0)
        weights, _ = model_pools.weights(alone, {"d3-mini": decision([("c", 0.4)])}, MEMBERS)
        self.assertAlmostEqual(weights[3], 0.4)
        self.assertAlmostEqual(weights[0], 0.6)

    def test_seat_and_crown_sum_per_hotkey_and_deregistered_hotkeys_burn(self):
        crown = {"hotkey": "b", "body_sha256": "f" * 64, "rate": 1.0, "weight": 0.2}
        paid = decision([("b", 0.4), ("gone", 0.4)], crown)
        weights, _ = model_pools.weights(pools(), {"d3-mini": paid}, MEMBERS)
        self.assertAlmostEqual(weights[2], 0.6 * 0.4)
        self.assertAlmostEqual(weights[0], 0.4 * 0.4)

    def test_no_decision_or_disabled_soup_pool_burns_everything(self):
        weights, record = model_pools.weights(pools(), {}, MEMBERS)
        self.assertEqual(weights, {0: 0.4, 1: 0.6})
        self.assertAlmostEqual(record["burned"], 0.4)
        off = pools(soup_entry(enabled=False))
        self.assertEqual(model_pools.weights(off, {}, MEMBERS)[0], {0: 0.4, 1: 0.6})
        with self.assertRaises(ValueError):
            model_pools.weights(off, {"d3-mini": decision([("c", 0.4)])}, MEMBERS)

    def test_decision_must_name_the_pool_and_pay_at_most_the_share(self):
        wrong = {**decision([("c", 0.4)]), "pool_id": "d3-lite"}
        with self.assertRaises(ValueError):
            model_pools.weights(pools(), {"d3-mini": wrong}, MEMBERS)
        with self.assertRaises(ValueError):
            model_pools.weights(pools(), {"d3-mini": decision([("a", 0.8), ("b", 0.4)])}, MEMBERS)

    def test_king_pools_are_unchanged(self):
        from tests.promoted_profile import promoted_profile
        from tests.test_pool_rewards import pools as king_pools

        with promoted_profile():
            self.assertEqual(model_pools.weights(king_pools(), {}, MEMBERS)[0], {1: 0.25, 2: 0.75})
            self.assertNotIn("type", model_pools.validate(king_pools())["pools"]["d3-mini"])

    def test_soup_entry_validation(self):
        checked = model_pools.validate(pools())["pools"]["d3-mini"]
        self.assertEqual(checked["type"], "soup")
        self.assertEqual(checked["model"], models.spec(models.D3))
        self.assertEqual(checked["house"], HOUSE)
        self.assertEqual(
            (checked["subset_size"], checked["screen_size"], checked["max_pool"]), (3000, 300, 20)
        )
        self.assertEqual(checked["max_new_per_round"], 6)
        self.assertNotIn("sketch_seed", checked)  # a local secret: never echoed into reports
        unseeded = soup_entry()
        del unseeded["sketch_seed"]
        self.assertNotIn("sketch_seed", model_pools.validate(pools(unseeded))["pools"]["d3-mini"])
        _, record = model_pools.weights(pools(), {}, MEMBERS)
        self.assertNotIn("5e" * 32, json.dumps(record))
        self.assertEqual(
            model_pools.validate(pools(soup_entry(house="soup-house.json")))["pools"]["d3-mini"][
                "house"
            ],
            "soup-house.json",
        )
        for bad in (
            {"type": "pot"},
            {"house": [{**HOUSE[0], "body_sha256": "x"}]},
            {"sketch_seed": "short"},
            {"sketch_seed": None},
            {"house": ""},
            {"house": 3},
            {"schedule": {"deadline_utc": "25:00"}},
            {"schedule": {}},
            {"incumbent_sha256": "c" * 64},
            {"max_pool": 21},
            {"max_pool": 1},
            {"max_new_per_round": 0},
            {"max_new_per_round": 21},
            {"subset_size": 0},
            {"share": 1.5},
            {"model": "d3-lite-0.8b-v1"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model_pools.validate(pools(soup_entry(**bad)))


def at(text):
    return datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp()


class FakeChain:
    """Block n has timestamp start + 12 n and hash 0x<n>."""

    def __init__(self, start):
        self.start, self.reads = start, 0

    def __call__(self, block):
        self.reads += 1
        return SimpleNamespace(
            number=block,
            hash=f"0x{block:064x}",
            timestamp=datetime.fromtimestamp(self.start + 12 * block, UTC),
        )


class ScheduleTest(unittest.TestCase):
    def test_due_round_is_the_latest_passed_deadline(self):
        schedule = {"deadline_utc": "00:00"}
        self.assertEqual(
            soup_pool.due(schedule, at("2026-11-03T05:00:00")),
            ("2026-11-03", at("2026-11-03T00:00:00")),
        )
        self.assertEqual(
            soup_pool.due({"deadline_utc": "06:30"}, at("2026-11-03T05:00:00")),
            ("2026-11-02", at("2026-11-02T06:30:00")),
        )

    def test_seed_block_is_the_first_block_after_the_deadline(self):
        start = at("2026-11-02T23:00:00")
        chain = FakeChain(start)
        deadline = at("2026-11-03T00:00:00")  # block 300 is exactly at the deadline
        block, block_hash = soup_pool.seed_block(chain, 1000, deadline)
        self.assertEqual(block, 301)
        self.assertEqual(block_hash, f"0x{301:064x}")
        self.assertLess(chain.reads, 40)
        with self.assertRaises(ValueError):
            soup_pool.seed_block(chain, 300, deadline)


class SoupEpochTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.config = {
            "base_pools": pools(),
            "base_pool_rounds": {
                "d3-mini": {"round_bank": "bank.jsonl", "base": "qwen"},
                "d3-lite": {
                    k: k
                    for k in (
                        "base_round_dir",
                        "base_reference",
                        "family_history",
                        "suite_manifest",
                        "pilot_report",
                    )
                },
            },
        }
        self.chain = FakeChain(at("2026-11-02T23:00:00"))
        self.snapshot = {
            "block": 1000,
            "members": {"1": "a", "2": "b"},
            "commitments": {
                "1": {"block": 250, "data": "zilsb2:" + "1" * 64 + ":" + "e" * 40 + ":m/one"},
                "2": {"block": 400, "data": "zilsb2:" + "2" * 64 + ":" + "e" * 40 + ":m/two"},
            },
        }
        self.calls = []
        self.queried = []
        # The chain's commitments by block: uid 1 committed at 250, then again after the deadline.
        self.history = {
            "1": [
                (250, "zilsb2:" + "1" * 64 + ":" + "e" * 40 + ":m/one"),
                (420, "zilsb2:" + "3" * 64 + ":" + "e" * 40 + ":m/three"),
            ],
            "2": [(400, "zilsb2:" + "2" * 64 + ":" + "e" * 40 + ":m/two")],
        }

    def commitments_at(self, block):
        self.queried.append(block)
        found = {}
        for uid, history in self.history.items():
            made = [(b, data) for b, data in history if b <= block]
            if made:
                found[uid] = {"block": made[-1][0], "data": made[-1][1]}
        return found

    def run_round(self, pool, request):
        self.calls.append((pool, request))
        decision_path = Path(request["state_dir"]) / "decision.json"
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(
            json.dumps(
                {"round_id": request["round_id"], "pool_id": pool, "members": [], "crown": None}
            )
        )
        soup_exam.UseLedger(Path(request["state_dir"]) / "uses.jsonl").record(
            request["round_id"], []
        )

    def epoch(self):
        return soup_pool.epoch(
            self.config,
            self.directory,
            self.snapshot,
            block_info=self.chain,
            commitments_at=self.commitments_at,
            run_round=self.run_round,
        )

    def test_commitments_are_read_at_the_seed_block(self):
        # The live snapshot only shows uid 1's newer commitment; the round sees the one at the seed.
        self.snapshot["commitments"]["1"] = {"block": 420, "data": self.history["1"][1][1]}
        self.epoch()
        request = self.calls[0][1]
        self.assertEqual(self.queried, [301])
        self.assertEqual(
            request["snapshot"]["commitments"],
            {"1": {"block": 250, "data": self.history["1"][0][1]}},
        )

    def test_one_round_per_day_seeded_after_the_deadline(self):
        summary = self.epoch()
        self.assertEqual(len(self.calls), 1)
        pool, request = self.calls[0]
        self.assertEqual(pool, "d3-mini")
        self.assertEqual(request["round_id"], "d3-mini-2026-11-03")
        self.assertEqual(request["block_hash"], f"0x{301:064x}")
        self.assertEqual(request["seed_block"], 301)
        # A commitment made after the deadline waits for tomorrow's round.
        self.assertEqual(set(request["snapshot"]["commitments"]), {"1"})
        self.assertEqual(request["round_bank"], str((self.directory / "bank.jsonl").resolve()))
        self.assertEqual(request["base"], str((self.directory / "qwen").resolve()))
        self.assertEqual(request["config"]["sketch_seed"], "5e" * 32)
        self.assertEqual(request["config"]["pool_id"], "d3-mini")
        self.assertEqual(request["config"]["house"], HOUSE)
        self.assertEqual(summary["d3-mini"]["round_id"], "d3-mini-2026-11-03")
        self.epoch()
        self.assertEqual(len(self.calls), 1)
        self.snapshot["block"] = 1000 + 7200
        self.epoch()
        self.assertEqual([r["round_id"] for _, r in self.calls][-1], "d3-mini-2026-11-04")
        self.assertEqual(soup_pool.decision(self.directory, "d3-mini")["pool_id"], "d3-mini")

    def test_a_failed_round_keeps_the_last_decision_and_reports_the_error(self):
        self.epoch()

        def broken(pool, request):
            raise RuntimeError("GPU fell over")

        self.snapshot["block"] = 1000 + 7200
        summary = soup_pool.epoch(
            self.config,
            self.directory,
            self.snapshot,
            block_info=self.chain,
            commitments_at=self.commitments_at,
            run_round=broken,
        )
        self.assertIn("GPU fell over", summary["d3-mini"]["error"])
        self.assertIsNotNone(soup_pool.decision(self.directory, "d3-mini"))

    def test_a_running_round_is_not_waited_on_or_restarted(self):
        starts = []

        def start(pool, request):
            starts.append(request["round_id"])
            return os.getpid()  # a live process: the round is still running

        def epoch():
            return soup_pool.epoch(
                self.config,
                self.directory,
                self.snapshot,
                block_info=self.chain,
                commitments_at=self.commitments_at,
                run_round=start,
            )

        self.assertTrue(epoch()["d3-mini"]["started"])
        summary = epoch()
        self.assertEqual(starts, ["d3-mini-2026-11-03"])
        self.assertTrue(summary["d3-mini"]["running"])
        self.assertFalse(summary["d3-mini"]["decision"])  # the previous decision (none) stands

    def test_a_reused_pid_is_not_mistaken_for_the_running_round(self):
        starts = []

        def start(pool, request):
            starts.append(request["round_id"])
            return os.getpid()

        def epoch():
            return soup_pool.epoch(
                self.config,
                self.directory,
                self.snapshot,
                block_info=self.chain,
                commitments_at=self.commitments_at,
                run_round=start,
            )

        epoch()
        job = json.loads((soup_pool.state_dir(self.directory, "d3-mini") / "job.json").read_text())
        self.assertEqual(job["pid_started"], soup_pool._started(os.getpid()))
        self.assertTrue(job["pid_started"])
        # The round's process died and another process now holds its pid.
        with patch("zils.soup_pool._started", return_value="another process"):
            summary = epoch()
        self.assertEqual(len(starts), 2)
        self.assertTrue(summary["d3-mini"]["started"])
        self.assertFalse(soup_pool._alive(os.getpid(), "another process"))
        self.assertTrue(soup_pool._alive(os.getpid(), soup_pool._started(os.getpid())))

    def test_a_dead_round_is_restarted_a_bounded_number_of_times(self):
        starts = []

        def start(pool, request):
            starts.append(request["round_id"])
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait()
            return child.pid  # exited without recording the round

        for _ in range(soup_pool.MAX_ATTEMPTS + 2):
            summary = soup_pool.epoch(
                self.config,
                self.directory,
                self.snapshot,
                block_info=self.chain,
                commitments_at=self.commitments_at,
                run_round=start,
            )
        self.assertEqual(len(starts), soup_pool.MAX_ATTEMPTS)
        self.assertIn("attempts", summary["d3-mini"]["error"])

    def test_timeouts_and_capacity_errors_never_escape_the_epoch(self):
        from zils import runtime

        for error in (
            subprocess.TimeoutExpired(["x"], 1),
            runtime.CapacityUnavailable(),
            KeyError("x"),
        ):

            def start(pool, request, error=error):
                raise error

            with self.subTest(error=type(error).__name__):
                summary = soup_pool.epoch(
                    self.config,
                    self.directory,
                    self.snapshot,
                    block_info=self.chain,
                    commitments_at=self.commitments_at,
                    run_round=start,
                )
                self.assertIn("error", summary["d3-mini"])

    def test_the_round_runner_bounds_capacity_waits_and_the_deadline(self):
        from zils import runtime

        path = self.directory / "request.json"
        path.write_text(
            json.dumps(
                {
                    "state_dir": str(self.directory / "state"),
                    "config": {"model": models.spec(models.D3)},
                    "runner": {"runtime_python": "python3", "device": "cuda", "poll": 1},
                }
            )
        )
        calls, sleeps = [], []

        def busy(command, log, device, timeout, *, model):
            calls.append((command, timeout, model))
            raise runtime.CapacityUnavailable()

        self.assertFalse(soup_pool.run_locked(path, run_child=busy, sleep=sleeps.append))
        self.assertEqual(len(calls), soup_pool.CAPACITY_TRIES)
        self.assertEqual(calls[0][0], ["python3", "-m", "zils.soup_pool", str(path)])
        self.assertEqual(calls[0][1], soup_pool.ROUND_SECONDS)
        self.assertEqual(calls[0][2], models.D3)

        def slow(command, log, device, timeout, *, model):
            raise subprocess.TimeoutExpired(command, timeout)

        self.assertFalse(soup_pool.run_locked(path, run_child=slow, sleep=sleeps.append))
        self.assertTrue(soup_pool.run_locked(path, run_child=lambda *a, **k: None))

    def test_background_round_starts_a_detached_runner_and_returns(self):
        args = SimpleNamespace(runtime_python="python3", device="cuda", poll=5)
        request = {"round_id": "d3-mini-2026-11-03", "state_dir": str(self.directory / "s")}
        with patch("zils.soup_pool.subprocess.Popen") as popen:
            popen.return_value.pid = 4321
            pid = soup_pool.background_round(args)("d3-mini", request)
        self.assertEqual(pid, 4321)
        command = popen.call_args.args[0]
        self.assertEqual(command[1:4], ["-m", "zils.soup_pool", "--run"])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        written = json.loads(Path(command[4]).read_text())
        self.assertEqual(
            written["runner"], {"runtime_python": "python3", "device": "cuda", "poll": 5}
        )

    def test_house_file_is_read_relative_to_the_validator_directory(self):
        (self.directory / "house.json").write_text(json.dumps(HOUSE))
        self.config["base_pools"] = pools(soup_entry(house="house.json"))
        self.epoch()
        self.assertEqual(self.calls[0][1]["config"]["house"], HOUSE)

    def test_validator_skips_soup_pools_for_king_rounds_and_attaches_their_decision(self):
        from zils import validator

        with patch("zils.validator.base_draws", return_value=[]), patch("zils.base_pilot.admit"):
            rounds = validator.pool_rounds(self.config, self.directory)
        self.assertEqual(set(rounds), {"d3-lite"})
        self.epoch()
        work = self.directory / "work"
        work.mkdir()
        with patch("zils.validator.time.time", return_value=at("2026-11-03T05:00:00")):
            report = validator.attach_base_pools(self.config, self.directory, work, {})
        self.assertEqual(report["base_pools"]["decisions"]["d3-mini"]["pool_id"], "d3-mini")
        weights, _ = model_pools.weights(
            self.config["base_pools"], report["base_pools"]["decisions"], MEMBERS
        )
        self.assertAlmostEqual(weights[0], 0.4)

    def test_a_decision_older_than_the_maximum_age_burns_the_pool(self):
        from zils import validator

        self.epoch()  # the round for 2026-11-03 (deadline 00:00 UTC)
        entry = soup_pool.soup_pools(self.config)["d3-mini"]
        self.assertEqual(entry["max_decision_age_days"], 3)
        fresh = soup_pool.fresh_decision(
            self.directory, "d3-mini", entry, now=at("2026-11-05T23:00:00")
        )
        self.assertEqual(fresh["round_id"], "d3-mini-2026-11-03")
        stale = soup_pool.fresh_decision(
            self.directory, "d3-mini", entry, now=at("2026-11-06T00:00:01")
        )
        self.assertIsNone(stale)
        work = self.directory / "work"
        work.mkdir()
        with patch("zils.validator.time.time", return_value=at("2026-11-07T00:00:00")):
            report = validator.attach_base_pools(self.config, self.directory, work, {})
        self.assertNotIn("d3-mini", report["base_pools"]["decisions"])
        weights, record = model_pools.weights(
            self.config["base_pools"], report["base_pools"]["decisions"], MEMBERS
        )
        self.assertAlmostEqual(weights[0], 0.4)  # the whole soup share burns
        self.config["base_pools"] = pools(soup_entry(max_decision_age_days=5))
        entry = soup_pool.soup_pools(self.config)["d3-mini"]
        self.assertIsNotNone(
            soup_pool.fresh_decision(
                self.directory, "d3-mini", entry, now=at("2026-11-07T00:00:00")
            )
        )
        for bad in (0, 31, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model_pools.validate(pools(soup_entry(max_decision_age_days=bad)))

    def test_zils1_candidates_cannot_enter_a_soup_pool(self):
        from zils import commitments

        claim = {"sha256": "1" * 64, "revision": "e" * 40, "repo": "m/one", "block": 1}
        with patch("zils.commitments.parent_model", return_value=models.D3):
            routed, rejected = commitments.route_pools({1: claim}, self.config["base_pools"])
        self.assertEqual(routed["d3-mini"], {})
        self.assertIn("soup", rejected[1])


if __name__ == "__main__":
    unittest.main()
