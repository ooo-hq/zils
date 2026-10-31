"""Four soup pools at launch: the emission schedule, pool openings, per-pool bodies and grading hardware."""

import copy
import hashlib
import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import zils
from tests.test_soup_run import (
    MEMBERS,
    FakeChain,
    at,
    decision,
    pools,
    soup_entry,
)
from zils import bodies, model_pools, models, soup_pool, testnet, validator

DATA = Path(zils.__file__).with_name("data")
BLOCKS_PER_DAY = 7200
TWO_WEEKS = 14 * BLOCKS_PER_DAY
SHA = "1" * 64
REV = "e" * 40


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def soup(model, share, **changes):
    return soup_entry(model=model, share=share, **changes)


def four(flash=None, z3=None, *, lite=0.35, mini=0.30):
    """The owner's launch shape: lite and mini open, flash and z3 closed unless given a schedule."""
    return {
        "share": 1.0,
        "burn_uid": 0,
        "pools": {
            "d3-lite": soup("d3-lite-0.8b-v1", lite),
            "d3-mini": soup(models.D3, mini),
            "d3-flash": soup("d3-flash-8.4b-v1", 0.20, **(flash or {"opens_block": None})),
            "d3": soup("d3-ultra-26b-v1", 0.10, **(z3 or {"opens_block": None})),
        },
    }


FLASH = {
    "opens_block": 1000,
    "boost": [{"share": 0.30, "blocks": TWO_WEEKS}, {"share": 0.25, "blocks": TWO_WEEKS}],
}
Z3 = {"opens_block": 1000 + 2 * TWO_WEEKS, "boost": [{"share": 0.15, "blocks": TWO_WEEKS}]}


class NoScheduleIsUnchangedTest(unittest.TestCase):
    """Digests recorded from origin/main (822bc54) before the schedule existed."""

    def test_weights_records_and_validated_configs_are_byte_identical(self):
        from tests.promoted_profile import promoted_profile
        from tests.test_pool_rewards import MEMBERS as KING_MEMBERS, pools as king_pools

        crown = {"hotkey": "b", "body_sha256": "f" * 64, "rate": 1.0, "weight": 0.2}
        cases = {
            "c9106d5d638d05b562e41540ff8e428f5d8a0a6a33c414296bf31355eaf69a33": (
                {"d3-mini": decision([("c", 0.4)])}
            ),
            "082d29e42aee31c5dd85139d1ed6ae5f8273c9ab8f1a84d54cc0b7ccef3a6ad4": (
                {"d3-mini": decision([("b", 0.4), ("gone", 0.4)], crown)}
            ),
            "2bd539db82638516deac988a93917e3d8ecc207e0de296ec5b0c394a76f85ce0": {},
        }
        for expected, decisions in cases.items():
            for block in (None, 10**9):
                with self.subTest(expected=expected[:8], block=block):
                    weights, record = model_pools.weights(pools(), decisions, MEMBERS, block=block)
                    self.assertEqual(digest([sorted(weights.items()), record]), expected)
        self.assertEqual(
            digest(model_pools.validate(pools())),
            "04bfc26c250bb52d311a470f101127b94ed25e4a376aadd24bb9fcd01917d3d9",
        )
        with promoted_profile():
            weights, record = model_pools.weights(king_pools(), {}, KING_MEMBERS)
            self.assertEqual(
                digest([sorted(weights.items()), record]),
                "c9c61214855303f6edf05dcd2a3eaf5898889a21c1c9f26a67559b5aa25a655c",
            )
            self.assertEqual(
                digest(model_pools.validate(king_pools())),
                "977fba3a0b20187467378bbd1dc6462cce7d494e1ad3e94dea36b4a886bae555",
            )
        self.assertFalse(model_pools.scheduled(model_pools.validate(pools())))


class ScheduleValidationTest(unittest.TestCase):
    def test_shares_may_total_less_than_one_and_the_rest_burns(self):
        config = four()
        weights, record = model_pools.weights(config, {}, MEMBERS, block=0)
        self.assertEqual(weights, {0: 1.0})
        # Flash and z3 are closed: their 30% and the unassigned 5% burn.
        self.assertAlmostEqual(record["unassigned"], 0.35)
        open_all = model_pools.weights(four(FLASH, Z3), {}, MEMBERS, block=1000 + 3 * TWO_WEEKS)
        self.assertAlmostEqual(open_all[1]["unassigned"], 0.05)
        self.assertAlmostEqual(record["burned"], 1.0)

    def test_shares_over_one_are_refused_at_every_point_of_the_schedule(self):
        with self.assertRaisesRegex(ValueError, "exceed"):
            model_pools.validate(four(lite=0.45))  # 0.45 + 0.30 + 0.20 + 0.10 > 1
        model_pools.validate(four(FLASH, Z3))
        # z3 opening while flash is still boosted would pay 0.35 + 0.30 + 0.30 + 0.15.
        early = {**Z3, "opens_block": 1000 + TWO_WEEKS // 2}
        with self.assertRaisesRegex(ValueError, "block 51400"):
            model_pools.validate(four(FLASH, early))
        # A boost that by itself overfills the budget.
        with self.assertRaisesRegex(ValueError, "exceed"):
            model_pools.validate(four({"opens_block": 5, "boost": [{"share": 0.4, "blocks": 9}]}))

    def test_malformed_schedules_fail_closed(self):
        for bad in (
            {"opens_block": -1},
            {"opens_block": 1.5},
            {"opens_block": True},
            {"opens_block": "soon"},
            {"boost": [{"share": 0.3, "blocks": 10}]},  # a boost needs an opening block
            {"opens_block": 5, "boost": []},
            {"opens_block": 5, "boost": [{"share": 0.3}]},
            {"opens_block": 5, "boost": [{"share": 0.3, "blocks": 0}]},
            {"opens_block": 5, "boost": [{"share": 1.3, "blocks": 10}]},
            {"opens_block": 5, "boost": [{"share": float("nan"), "blocks": 10}]},
            {"opens_block": 5, "boost": [{"share": 0.3, "blocks": 10, "x": 1}]},
            {"opens_block": 5, "boost": [{"share": 0.21, "blocks": 1}] * 9},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model_pools.validate(four(bad))

    def test_only_soup_pools_take_a_schedule(self):
        from tests.promoted_profile import promoted_profile
        from tests.test_pool_rewards import pools as king_pools

        with promoted_profile():
            config = king_pools()
            config["pools"]["d3-lite"]["opens_block"] = 10
            with self.assertRaisesRegex(ValueError, "soup"):
                model_pools.validate(config)

    def test_a_validated_schedule_is_kept_for_the_frozen_report(self):
        checked = model_pools.validate(four(FLASH))["pools"]["d3-flash"]
        self.assertEqual(checked["opens_block"], 1000)
        self.assertEqual(checked["boost"], FLASH["boost"])
        self.assertNotIn("opens_block", model_pools.validate(four(FLASH))["pools"]["d3-mini"])
        self.assertTrue(model_pools.scheduled(model_pools.validate(four(FLASH))))


class ShareScheduleTest(unittest.TestCase):
    def shares(self, block):
        config = model_pools.validate(four(FLASH, Z3))
        return {p: round(s, 6) for p, s in model_pools.shares_at(config, block).items()}

    def test_owner_launch_schedule(self):
        closed = {"d3-lite": 0.35, "d3-mini": 0.30, "d3-flash": 0.0, "d3": 0.0}
        self.assertEqual(self.shares(0), closed)
        self.assertEqual(self.shares(999), closed)
        self.assertEqual(self.shares(1000)["d3-flash"], 0.30)
        self.assertEqual(self.shares(1000 + TWO_WEEKS - 1)["d3-flash"], 0.30)
        self.assertEqual(self.shares(1000 + TWO_WEEKS)["d3-flash"], 0.25)
        self.assertEqual(self.shares(1000 + 2 * TWO_WEEKS)["d3-flash"], 0.20)
        self.assertEqual(self.shares(1000 + 2 * TWO_WEEKS)["d3"], 0.15)
        self.assertEqual(
            self.shares(1000 + 3 * TWO_WEEKS),
            {"d3-lite": 0.35, "d3-mini": 0.30, "d3-flash": 0.20, "d3": 0.10},
        )

    def test_a_closed_pool_burns_and_its_round_cannot_pay(self):
        config = four(FLASH, Z3)
        paid = {"d3-mini": decision([("c", 0.4)])}
        weights, record = model_pools.weights(config, paid, MEMBERS, block=999)
        # Lite and mini are open with no lite decision: mini pays 0.4 of its 0.30 to uid 3.
        self.assertAlmostEqual(weights[3], 0.30 * 0.4)
        self.assertAlmostEqual(weights[0], 1 - 0.30 * 0.4)
        self.assertAlmostEqual(record["burned"], 1 - 0.30 * 0.4)
        self.assertFalse(record["pools"]["d3-flash"]["open"])
        self.assertEqual(record["pools"]["d3-flash"]["paid_share"], 0.0)
        self.assertEqual(record["block"], 999)
        flash = {"d3-flash": {**decision([("c", 0.4)]), "pool_id": "d3-flash"}}
        with self.assertRaisesRegex(ValueError, "not open"):
            model_pools.weights(config, flash, MEMBERS, block=999)
        weights, record = model_pools.weights(config, flash, MEMBERS, block=1000)
        self.assertAlmostEqual(weights[3], 0.30 * 0.4)  # the boosted share
        self.assertTrue(record["pools"]["d3-flash"]["open"])
        self.assertAlmostEqual(record["pools"]["d3-flash"]["paid_share"], 0.30)

    def test_a_scheduled_config_needs_the_block(self):
        with self.assertRaisesRegex(ValueError, "block"):
            model_pools.weights(four(FLASH), {}, MEMBERS)

    def test_every_validator_computes_the_same_vector_at_a_block(self):
        config = four(FLASH, Z3)
        paid = {
            "d3-mini": decision([("a", 0.4)]),
            "d3-flash": {**decision([("b", 0.8)]), "pool_id": "d3-flash"},
        }
        for block in (1000, 1000 + TWO_WEEKS, 1000 + 3 * TWO_WEEKS):
            first = model_pools.weights(copy.deepcopy(config), paid, MEMBERS, block=block)
            again = model_pools.weights(copy.deepcopy(config), paid, MEMBERS, block=block)
            self.assertEqual(digest(first), digest(again))
            self.assertAlmostEqual(sum(first[0].values()), 1.0)


class LaunchDataTest(unittest.TestCase):
    def test_house_lists_per_pool(self):
        from zils import soup_rates

        expected = {
            "soup-house.json": {"vllm-sr/d3-mini"},
            "soup-house-lite.json": {"vllm-sr/d3-lite", "internlm/Intern-Decision-0.8B"},
            "soup-house-flash.json": {"vllm-sr/d3-flash"},
            "soup-house-z3.json": {"vllm-sr/d3-ultra"},
        }
        for name, repos in expected.items():
            house = soup_rates.load_house(DATA / name)
            self.assertLessEqual(repos, {h["repo"] for h in house}, name)
        # The house chassis entries are the pinned chassis revisions.
        for name, model in (
            ("soup-house-lite.json", "d3-lite-0.8b-v1"),
            ("soup-house-flash.json", "d3-flash-8.4b-v1"),
            ("soup-house-z3.json", "d3-ultra-26b-v1"),
        ):
            house = soup_rates.load_house(DATA / name)
            chassis = next(h for h in house if h["repo"] == models.pins(model)["vendor"]["repo"])
            self.assertEqual(chassis["revision"], models.pins(model)["revision"])

    def test_launch_template_follows_the_owner_split(self):
        config = json.loads((DATA / "base-pools-launch.json").read_text())
        checked = model_pools.validate(config)["pools"]
        self.assertEqual(set(checked), {"d3-lite", "d3-mini", "d3-flash", "d3"})
        self.assertTrue(all(model_pools.is_soup(e) and e["enabled"] for e in checked.values()))
        self.assertEqual(
            {p: e["share"] for p, e in checked.items()},
            {"d3-lite": 0.35, "d3-mini": 0.30, "d3-flash": 0.20, "d3": 0.10},
        )
        self.assertIsNone(checked["d3-flash"]["opens_block"])
        self.assertIsNone(checked["d3"]["opens_block"])
        shares = model_pools.shares_at(model_pools.validate(config), 0)
        self.assertEqual(shares["d3-flash"] + shares["d3"], 0)
        # Download caps fit at least one body of the pool's own size.
        for pool, entry in checked.items():
            self.assertGreaterEqual(entry["max_download_bytes_per_round"], bodies.max_bytes(pool))


class BodyCommitmentTest(unittest.TestCase):
    def test_mini_keeps_zilsb2_and_other_pools_name_themselves(self):
        self.assertEqual(
            bodies.format_pool_commitment("d3-mini", SHA, REV, "m/one"),
            bodies.format_commitment(SHA, REV, "m/one"),
        )
        text = bodies.format_pool_commitment("d3-lite", SHA, REV, "m/one")
        self.assertEqual(text, f"zilsb3:d3-lite:{SHA}:{REV}:m/one")
        self.assertEqual(
            bodies.parse_pool_commitment(text),
            ("d3-lite", {"sha256": SHA, "revision": REV, "repo": "m/one"}),
        )
        self.assertEqual(bodies.parse_pool_commitment(f"zilsb2:{SHA}:{REV}:m/one")[0], "d3-mini")
        self.assertEqual(bodies.as_zilsb2(text), bodies.format_commitment(SHA, REV, "m/one"))
        longest = bodies.format_pool_commitment("d3-flash", SHA, REV, "a" * 96 + "/" + "b" * 96)
        self.assertLessEqual(len(longest), 3 * testnet.COMMITMENT_FIELD)
        for bad in (
            f"zilsb3:d3-mini:{SHA}:{REV}:m/one",  # mini's canonical form is zilsb2
            f"zilsb3:d3-nano:{SHA}:{REV}:m/one",
            f"zilsb3:d3-lite:{SHA}:{REV}",
            f"zilsb3:d3-lite:{SHA[:-1]}:{REV}:m/one",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bodies.parse_pool_commitment(bad)
        self.assertIsNone(bodies.body_pool(f"zils1:{SHA}:{REV}:m/one"))
        self.assertIsNone(bodies.body_pool(f"zilsb3:junk:{SHA}"))
        self.assertEqual(bodies.body_pool("zilsb2:junk"), "d3-mini")

    def test_parents_and_byte_caps_per_pool(self):
        self.assertEqual(
            bodies.parent("d3-mini"),
            ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
        )
        self.assertEqual(bodies.parent("d3-lite")[0], "Qwen/Qwen3.5-0.8B")
        self.assertEqual(bodies.parent("d3-flash")[0], "Qwen/Qwen3.5-9B")
        self.assertEqual(bodies.parent("d3")[0], "Qwen/Qwen3.8-27B")
        self.assertEqual(bodies.max_bytes("d3-mini"), bodies.MAX_BYTES)
        # A body of each pool's own parent size fits; the z3 vendor body is 52.2 GB.
        for pool, size in (("d3-lite", 1.75e9), ("d3-flash", 19.31e9), ("d3", 55.57e9)):
            self.assertGreater(bodies.max_bytes(pool), size * 1.1, pool)
        self.assertEqual(set(bodies.BODY_POOLS), {"d3-lite", "d3-mini", "d3-flash", "d3"})

    def test_validate_body_takes_a_byte_cap(self):
        from tests.test_bodies import HOTKEY, checkpoint, weights

        with tempfile.TemporaryDirectory() as tmp:
            chassis = checkpoint(Path(tmp) / "chassis", weights(), extras=False)
            body = checkpoint(Path(tmp) / "body", weights(), prefix="model.")
            bodies.validate_body(body, chassis, hotkey=HOTKEY)
            with self.assertRaisesRegex(ValueError, "64 byte limit"):
                bodies.validate_body(body, chassis, hotkey=HOTKEY, max_bytes=64)


class SoupRoundBodyCapTest(unittest.TestCase):
    def test_intake_uses_the_pools_body_cap(self):
        from tests.test_soup_run import Round
        from zils import d3

        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        r = Round(tmp)
        pinned = hashlib.sha256((r.chassis / "readout.safetensors").read_bytes()).hexdigest()
        self.enterContext(patch.dict(d3.PINS["files"], {"readout.safetensors": pinned}))
        asked = []
        with patch.object(bodies, "max_bytes", lambda pool: asked.append(pool) or 64):
            r.run()
        self.assertIn("d3-mini", asked)
        intake = json.loads((r.state / "rounds" / "d3-mini-2026-11-03" / "intake.json").read_text())
        self.assertFalse(intake["new"])
        self.assertTrue(all("byte limit" in v for v in intake["failed"].values()), intake["failed"])


class FetchBodyTest(unittest.TestCase):
    def test_fetch_body_uses_the_pools_cap_and_download_time(self):
        seen = {}

        def download(claim, dest, names, listed, budget, **kwargs):
            seen.update(budget=budget, **kwargs)

        listed = {"model.safetensors": {"size": 1, "sha256": None}}
        with (
            patch("zils.commitments.revision_files", return_value=listed),
            patch("zils.commitments.download_files", download),
        ):
            soup_pool.fetch_body({"repo": "a/b", "revision": REV}, "/nonexistent")
            self.assertEqual(seen, {"budget": bodies.MAX_BYTES})
            soup_pool.fetch_body({"repo": "a/b", "revision": REV}, "/x", pool="d3")
            self.assertEqual(seen["budget"], bodies.max_bytes("d3"))
            self.assertEqual(seen["seconds"], bodies.download_seconds("d3"))


class FitBankTest(unittest.TestCase):
    def test_items_over_the_pools_token_cap_are_dropped(self):
        bank = [{"id": f"c{i}", "state": {"text": "x" * i}, "question": "q?"} for i in range(5)]
        bank.append({"id": "refused", "state": {"text": "bad"}, "question": "q?"})

        class Encoder:
            def encode(self, state, question):
                if state["text"] == "bad":  # the runtime refuses it (too long, a placeholder)
                    raise ValueError("Decision exceeds the input limit")
                return list(range(len(state["text"]) * 2000)), None

        logs = []
        kept = soup_pool.fit_bank(bank, "d3-lite-0.8b-v1", encoder=Encoder(), log=logs.append)
        self.assertEqual([c["id"] for c in kept], ["c0", "c1", "c2"])
        self.assertIn("3 of 6", logs[0])
        self.assertIn("4096", logs[0])
        # Mini's 8,192 cap keeps 6,000 and 8,000 tokens.
        kept = soup_pool.fit_bank(bank, models.D3, encoder=Encoder(), log=logs.append)
        self.assertEqual([c["id"] for c in kept], ["c0", "c1", "c2", "c3", "c4"])
        self.assertEqual(soup_pool.fit_bank([], models.D3, log=logs.append), [])

    def test_main_fits_the_fixed_bank_before_drawing(self):
        from tests.test_soup_current import Fixture
        from zils import d3, soup_round

        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        f = Fixture(tmp)
        bank_path = tmp / "bank.jsonl"
        bank_path.write_text("".join(json.dumps(c) + "\n" for c in f.bank))
        config = {k: v for k, v in f.config().items() if k != "exam"}
        request = {
            "round_id": "d3-lite-2026-11-03",
            "block_hash": "0x" + "ab" * 32,
            "snapshot": {"members": {}, "commitments": {}},
            "config": {**config, "pool_id": "d3-lite", "model": models.spec("d3-lite-0.8b-v1")},
            "state_dir": str(f.state),
            "round_bank": str(bank_path),
            "base": str(tmp / "qwen"),
        }
        path = tmp / "request.json"
        path.write_text(json.dumps(request))
        seen, fitted = {}, []

        def fit(bank, model, **kwargs):
            fitted.append(model)
            return bank[:-1]

        with (
            patch.object(soup_round, "run", lambda config, root, **kwargs: seen.update(kwargs)),
            patch.object(d3, "pinned_base_path", lambda model: tmp / "chassis"),
            patch.object(soup_pool, "fit_bank", fit),
        ):
            soup_pool.main([str(path)])
        self.assertEqual(fitted, ["d3-lite-0.8b-v1"])
        self.assertEqual(seen["bank"], f.bank[:-1])
        self.assertEqual(seen["fetch_body"].keywords, {"pool": "d3-lite"})


class EpochFixture:
    def __init__(self, test, config):
        self.test = test
        self.directory = Path(test.enterContext(tempfile.TemporaryDirectory()))
        self.config = config
        self.chain = FakeChain(at("2026-11-02T23:00:00"))
        self.calls = []

    def commitments_at(self, block):
        return {
            "1": {"block": 250, "data": f"zilsb2:{'1' * 64}:{REV}:m/mini"},
            "2": {"block": 251, "data": f"zilsb3:d3-lite:{'2' * 64}:{REV}:m/lite"},
            "3": {"block": 252, "data": f"zilsb3:d3-flash:{'3' * 64}:{REV}:m/flash"},
            "4": {"block": 253, "data": f"zils1:{'4' * 64}:{REV}:m/king"},
        }

    def run_round(self, pool, request):
        self.calls.append((pool, request))

    def epoch(self, block=1000):
        snapshot = {"block": block, "members": {str(u): f"hk{u}" for u in range(1, 5)}}
        return soup_pool.epoch(
            self.config,
            self.directory,
            {**snapshot, "commitments": {}},
            block_info=self.chain,
            commitments_at=self.commitments_at,
            run_round=self.run_round,
        )


def rounds_config(base_pools, **extra):
    return {
        "base_pools": base_pools,
        "base_pool_rounds": {
            p: {"round_bank": "bank.jsonl", "base": f"qwen-{p}"} for p in base_pools["pools"]
        },
        **extra,
    }


class EpochPoolsTest(unittest.TestCase):
    def test_each_pool_sees_only_its_own_bodies_as_zilsb2(self):
        f = EpochFixture(self, rounds_config(four(FLASH)))
        f.epoch(block=1000)
        seen = {pool: request["snapshot"]["commitments"] for pool, request in f.calls}
        self.assertEqual(set(seen), {"d3-lite", "d3-mini", "d3-flash"})
        self.assertEqual(
            seen["d3-mini"], {"1": {"block": 250, "data": f"zilsb2:{'1' * 64}:{REV}:m/mini"}}
        )
        self.assertEqual(
            seen["d3-lite"], {"2": {"block": 251, "data": f"zilsb2:{'2' * 64}:{REV}:m/lite"}}
        )
        self.assertEqual(
            seen["d3-flash"], {"3": {"block": 252, "data": f"zilsb2:{'3' * 64}:{REV}:m/flash"}}
        )

    def test_a_closed_pool_runs_no_round_until_its_opening_block(self):
        f = EpochFixture(self, rounds_config(four(FLASH)))
        summary = f.epoch(block=999)
        self.assertEqual({p for p, _ in f.calls}, {"d3-lite", "d3-mini"})
        self.assertIn("1000", summary["d3-flash"]["closed"])
        self.assertIn("no opening block", summary["d3"]["closed"])
        f.epoch(block=1000)
        self.assertIn("d3-flash", {p for p, _ in f.calls})

    def test_pools_this_validator_cannot_grade_are_skipped_and_logged(self):
        config = rounds_config(four(FLASH, Z3), soup_grading={"gpu_mib": 24564})
        f = EpochFixture(self, config)
        summary = f.epoch(block=1000 + 3 * TWO_WEEKS)
        self.assertEqual({p for p, _ in f.calls}, {"d3-lite", "d3-mini", "d3-flash"})
        self.assertIn("57344 MiB", summary["d3"]["skipped"])
        listed = rounds_config(four(FLASH, Z3), soup_grading={"pools": ["d3-mini"]})
        f = EpochFixture(self, listed)
        summary = f.epoch(block=1000 + 3 * TWO_WEEKS)
        self.assertEqual({p for p, _ in f.calls}, {"d3-mini"})
        self.assertIn("not in soup_grading.pools", summary["d3-lite"]["skipped"])

    def test_grading_settings_fail_closed(self):
        for bad in (
            {"pools": ["d3-nano"]},
            {"pools": "d3-mini"},
            {"gpu_mib": 0},
            {"gpu_mib": 1.5},
            {"other": 1},
            [],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                soup_pool.grading({"soup_grading": bad})
        self.assertEqual(soup_pool.grading({}), {"pools": None, "gpu_mib": None})
        self.assertEqual(set(soup_pool.GRADING_MIB), {"d3-lite", "d3-mini", "d3-flash", "d3"})


class ValidatorPoolsTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.work = self.root / "work"
        self.work.mkdir()

    def write_decision(self, pool, round_id):
        path = soup_pool.state_dir(self.root, pool) / "decision.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**decision([("a", 0.4)]), "pool_id": pool, "round_id": round_id})
        )

    def test_freezes_the_snapshot_block_and_only_open_graded_decisions(self):
        config = rounds_config(four(FLASH), soup_grading={"pools": ["d3-mini", "d3-flash"]})
        today = datetime.now(UTC).date().isoformat()
        for pool in ("d3-lite", "d3-mini", "d3-flash"):
            self.write_decision(pool, f"{pool}-{today}")
        (self.work / "chain-snapshot.json").write_text(json.dumps({"block": 999, "members": {}}))
        report = validator.attach_base_pools(config, self.root, self.work, {})
        frozen = report["base_pools"]
        self.assertEqual(frozen["block"], 999)
        self.assertEqual(set(frozen["decisions"]), {"d3-mini"})  # lite not graded, flash closed
        weights, record = model_pools.weights(
            config["base_pools"], frozen["decisions"], MEMBERS, block=frozen["block"]
        )
        self.assertAlmostEqual(weights[1], 0.30 * 0.4)

    def test_no_schedule_freezes_exactly_config_and_decisions(self):
        config = rounds_config(pools())
        (self.work / "chain-snapshot.json").write_text(json.dumps({"block": 999, "members": {}}))
        report = validator.attach_base_pools(config, self.root, self.work, {})
        self.assertEqual(set(report["base_pools"]), {"config", "decisions"})

    def test_a_scheduled_config_without_a_snapshot_refuses(self):
        with self.assertRaisesRegex(ValueError, "chain snapshot"):
            validator.attach_base_pools(rounds_config(four(FLASH)), self.root, self.work, {})

    def test_pools_not_graded_here_need_no_round_paths(self):
        config = rounds_config(four(FLASH, Z3), soup_grading={"pools": ["d3-lite", "d3-mini"]})
        del config["base_pool_rounds"]["d3"], config["base_pool_rounds"]["d3-flash"]
        self.assertEqual(validator.pool_rounds(config, self.root), {})
        del config["soup_grading"]  # graded here again: its paths are required
        with self.assertRaisesRegex(ValueError, "base_pool_rounds"):
            validator.pool_rounds(config, self.root)


class PublishBlockTest(unittest.TestCase):
    def test_the_publisher_pays_the_schedule_at_the_frozen_block(self):
        config = {
            "chain": {"network": "test", "netuid": 579},
            "validator_hotkey": "validator",
            "members": MEMBERS,
            "base_pools": four(FLASH),
        }
        flash = {"d3-flash": {**decision([("c", 0.5)]), "pool_id": "d3-flash"}}
        report = {
            "chain": config["chain"],
            "identities": {"1": "a", "2": "b", "3": "c"},
            "miners": [],
            "base_pools": {
                "config": model_pools.validate(config["base_pools"]),
                "decisions": flash,
                "block": 1000,
            },
        }
        state = {
            "members": report["identities"],
            "owner_uid": 0,
            "validator_uid": 0,
            "block": 5,  # the publish block never changes the frozen round's schedule
            "required_version": 1,
            "retry_after_block": 5,
        }
        chain = SimpleNamespace(plan=lambda intent, wallet: SimpleNamespace(to_dict=dict))
        wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("zils.testnet.preflight", return_value=state),
        ):
            result = testnet.publish_round(config, tmp, report, chain, wallet)
        self.assertAlmostEqual(result["requested_weights"]["3"], 0.30 * 0.5)
        unblocked = {**report, "base_pools": {**report["base_pools"]}}
        del unblocked["base_pools"]["block"]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("zils.testnet.preflight", return_value=state),
            self.assertRaisesRegex(ValueError, "block"),
        ):
            testnet.publish_round(config, tmp, unblocked, chain, wallet)


if __name__ == "__main__":
    unittest.main()
