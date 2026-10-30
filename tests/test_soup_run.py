"""A full soup round on tiny synthetic bodies with a fake runner, its payout and its daily schedule."""

import hashlib
import json
import math
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

    def test_rates_hook_sets_seat_and_crown_pay(self):
        def rates(sha, info, house):
            return 0.25 if sha == self.r.sha["t1"] else (0.0 if sha in house else 1.0)

        record = self.r.run(rates=rates)
        members = self.by_body(record)
        self.assertAlmostEqual(members[self.r.sha["t1"]]["seat_weight"], 0.1)
        self.assertAlmostEqual(members[self.r.sha["t2"]]["seat_weight"], 0.4)
        self.assertAlmostEqual(record["crown"]["weight"], 0.05)

    def test_a_body_failing_validation_is_rejected_and_deleted(self):
        def fetch(claim, dest):
            dest = self.r.fetch(claim, dest)
            if claim["sha256"] == self.r.sha["weak"]:
                (dest / "zils-owner.json").write_text(json.dumps({"hotkey": "5Someone"}))
            return dest

        record = self.r.run(fetch_body=fetch)
        self.assertIn("zils-owner.json", record["rejected"][self.r.sha["weak"]])
        self.assertFalse((self.r.state / "bodies" / self.r.sha["weak"]).exists())

    def hostile(self, name, folder, block):
        """Commit a hand-made body repo `folder` as `name` (its claimed hash cannot be computed)."""
        self.r.repos[name] = folder
        self.r.sha[name] = hashlib.sha256(name.encode()).hexdigest()
        uid = max(self.r.commits) + 1
        self.r.commits[uid] = (name, block)
        self.enterContext(patch.dict(HOTKEYS, {uid: HOTKEYS.get(uid, f"5Hostile{uid}")}))
        return self.r.sha[name]

    def test_hostile_bodies_are_rejected_and_the_round_completes(self):
        nan = strengths(1, 1, 0, 0, 0)
        nan["visual.patch.weight"][0] = math.nan
        repos = self.tmp / "hostile"
        nan_sha = self.hostile(
            "nan", test_bodies.checkpoint(repos / "nan", nan, prefix="model.", owner=None), 90
        )
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
        self.assertIn("finite", record["rejected"][nan_sha])
        self.assertIn("malformed", record["rejected"][offsets_sha])
        self.assertIn("unreadable", record["rejected"][deep_sha])
        self.assertEqual(set(self.by_body(record)), {self.r.sha["t1"], self.r.sha["t2"]})

    def test_any_exception_from_one_body_rejects_that_body_only(self):
        def copies(sha, folder, earlier, *, soup):
            if sha == self.r.sha["weak"]:
                raise RuntimeError("sketch blew up")
            return None

        record = self.r.run(copies=copies)
        self.assertIn("RuntimeError", record["rejected"][self.r.sha["weak"]])
        self.assertEqual(set(self.by_body(record)), {self.r.sha["t1"], self.r.sha["t2"]})

    def test_a_body_that_fails_grading_is_rejected_and_the_round_completes(self):
        fake = self.r.runner

        def runner_for(folder):
            run = fake(folder)
            if Path(folder).name.startswith(self.r.sha["weak"][:32]):

                def broken(cases):
                    raise RuntimeError("cannot load this body")

                return broken
            return run

        self.r.runner = runner_for
        record = self.r.run()
        self.assertIn("cannot load", record["rejected"][self.r.sha["weak"]])
        self.assertNotIn(self.r.sha["weak"], record["pool"])
        self.assertEqual(set(self.by_body(record)), {self.r.sha["t1"], self.r.sha["t2"]})

    def test_a_twin_check_failure_rejects_only_the_new_body(self):
        def twins(rows, other_rows):
            if len(rows) and rows is not other_rows and twins.calls == 0:
                twins.calls += 1
                raise ValueError("bad rows")
            return False

        twins.calls = 0
        record = self.r.run(twins=twins)
        self.assertEqual(len(record["rejected"]), 1)
        self.assertIn("bad rows", next(iter(record["rejected"].values())))
        self.assertTrue(record["members"])

    def test_default_checks_reject_a_layer_copy_and_rate_a_house_retrain_as_a_scout(self):
        self.r.commits.update({5: ("scout", 50), 6: ("t1copy", 104)})
        record = self.r.run()
        sha = self.r.sha
        self.assertEqual(record["rejected"], {sha["t1copy"]: "copy: cosine"})
        self.assertFalse((self.r.state / "bodies" / sha["t1copy"]).exists())
        self.assertEqual(record["rates"][sha["scout"]], 0.25)
        self.assertEqual(record["rates"][sha["t1"]], 1.0)
        self.assertEqual(record["rates"][sha["house"]], 0.0)
        self.assertEqual(set(self.by_body(record)), {sha["t1"], sha["t2"]})
        # One fixed sketch seed per pool: each body is sketched once, whatever the round's seed.
        self.assertEqual(len(list((self.r.state / "sketches").glob("*.npy"))), 6)
        self.r.run(round_id="d3-mini-2026-11-04")
        self.assertEqual(len(list((self.r.state / "sketches").glob("*.npy"))), 6)

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
        wrong = Round(self.tmp / "wrong")
        with self.assertRaisesRegex(ValueError, "pinned hash"):
            wrong.run(config=wrong.config(house=[{**house, "body_sha256": "0" * 64}]))

    def test_rate_constants_match_soup_rates(self):
        from zils import soup_rates

        self.assertEqual(
            (soup_rates.HOUSE_RATE, soup_rates.SCOUT_RATE, soup_rates.TRAINER_RATE),
            (soup_round.HOUSE_RATE, soup_round.SCOUT_RATE, soup_round.TRAINER_RATE),
        )

    def test_cache_keys_are_bank_scoped(self):
        self.r.run()
        names = {p.stem for p in (self.r.state / "cache").iterdir()}
        bank_sha = soup_round.bank_sha256(self.r.bank)
        self.assertIn(f"{self.r.sha['t1'][:32]}-{bank_sha[:16]}", names)
        self.assertTrue(all(n.endswith(bank_sha[:16]) for n in names))


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

    def run_round(self, pool, request):
        self.calls.append((pool, request))
        decision_path = Path(request["state_dir"]) / "decision.json"
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(json.dumps({"pool_id": pool, "members": [], "crown": None}))
        soup_exam.UseLedger(Path(request["state_dir"]) / "uses.jsonl").record(
            request["round_id"], []
        )

    def epoch(self):
        return soup_pool.epoch(
            self.config,
            self.directory,
            self.snapshot,
            block_info=self.chain,
            run_round=self.run_round,
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
            self.config, self.directory, self.snapshot, block_info=self.chain, run_round=broken
        )
        self.assertIn("GPU fell over", summary["d3-mini"]["error"])
        self.assertIsNotNone(soup_pool.decision(self.directory, "d3-mini"))

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
        report = validator.attach_base_pools(self.config, self.directory, work, {})
        self.assertEqual(report["base_pools"]["decisions"]["d3-mini"]["pool_id"], "d3-mini")
        weights, _ = model_pools.weights(
            self.config["base_pools"], report["base_pools"]["decisions"], MEMBERS
        )
        self.assertAlmostEqual(weights[0], 0.4)

    def test_zils1_candidates_cannot_enter_a_soup_pool(self):
        from zils import commitments

        claim = {"sha256": "1" * 64, "revision": "e" * 40, "repo": "m/one", "block": 1}
        with patch("zils.commitments.parent_model", return_value=models.D3):
            routed, rejected = commitments.route_pools({1: claim}, self.config["base_pools"])
        self.assertEqual(routed["d3-mini"], {})
        self.assertIn("soup", rejected[1])


if __name__ == "__main__":
    unittest.main()
