"""Chain RPC is faked; signing, scoring, disk state and SDK intents are real."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec("bittensor"), "install requirements/testnet.txt")
class TestnetTest(unittest.TestCase):
    def test_chain_admission_preserves_every_evaluated_identity_before_publication(self):
        from bittensor.result import ExtrinsicResult

        from zils import testnet as t

        config = {
            "chain": {"network": "test", "netuid": 553},
            "admission": "chain",
            "validator_hotkey": "validator",
            "members": {"1": {"hotkey": "miner-a"}, "2": {"hotkey": "miner-b"}},
        }
        report = {
            "chain": config["chain"],
            "identities": {"1": "miner-a", "2": "miner-b"},
            "miners": [
                {"uid": 1, "status": "evaluated", "skill": 0.75},
                {"uid": 2, "status": "rejected", "skill": 0.0},
            ],
        }
        wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
        # Even an unpaid frozen member must retain its identity. New registrations
        # are safe when every frozen UID still belongs to the evaluated hotkey.
        for keys, allowed in (
            ({0: "validator", 1: "replacement", 2: "miner-b"}, False),
            ({0: "validator", 2: "miner-b"}, False),
            ({0: "validator", 1: "miner-a"}, False),
            ({0: "validator", 1: "miner-a", 2: "miner-b", 3: "new-miner"}, True),
        ):
            state = {
                "NetworksAdded": True,
                "SubnetworkN": 4,
                "Uids": {hotkey: uid for uid, hotkey in keys.items()},
                "Keys": keys,
                "ValidatorPermit": [True, False, False, False],
                "SubnetOwnerHotkey": "validator",
                "CommitRevealWeightsEnabled": False,
                "MinAllowedWeights": 1,
                "MaxWeightsLimit": 65535,
                "WeightsSetRateLimit": 0,
                "WeightsVersionKey": 0,
                "LastUpdate": [0, 0, 0, 0],
            }

            def query(item, params, *, block):
                self.assertEqual((params[0], block), (553, 250))
                value = state[item[1]]
                return value.get(params[1]) if isinstance(value, dict) else value

            sub = SimpleNamespace(
                network="test",
                endpoint=t.ENDPOINT,
                block=250,
                query=query,
                execute=Mock(return_value=ExtrinsicResult(True)),
                plan=Mock(),
            )
            with self.subTest(keys=keys), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                receipt = t.publish_round(config, work, report, sub, wallet, publish=True)
                sub.plan.assert_not_called()
                if allowed:
                    self.assertEqual(receipt["status"], "included")
                    self.assertTrue(receipt["chain_write"])
                    self.assertEqual(receipt["requested_weights"], {"1": 1.0})
                    sub.execute.assert_called_once()
                    self.assertEqual(sub.execute.call_args.args[0].uids, [1])
                else:
                    self.assertEqual(receipt["status"], "registration_changed")
                    self.assertIs(receipt["chain_write"], False)
                    self.assertIs(receipt["weights_verified"], False)
                    self.assertEqual(receipt["requested_weights"], {})
                    sub.execute.assert_not_called()
                    self.assertFalse((work / "chain-attempt.json").exists())
                    self.assertFalse((work / "chain-receipt.json").exists())

    def test_registration_and_publication_fail_closed(self):
        from bittensor.result import ChainError, ErrorCode, ExtrinsicResult

        from zils import testnet as t

        config = {
            "chain": {"network": "test", "netuid": 553},
            "validator_hotkey": "validator",
            "members": {
                "1": {"hotkey": "miner-a", "port": 8901},
                "2": {"hotkey": "miner-b", "port": 8902},
            },
        }
        state = {
            "NetworksAdded": True,
            "Uids": {"validator": 0, "miner-a": 1, "miner-b": 2},
            "Keys": {0: "validator", 1: "miner-a", 2: "miner-b"},
            "ValidatorPermit": [True, False, False],
            "SubnetOwnerHotkey": "validator",
            "CommitRevealWeightsEnabled": True,
            "MinAllowedWeights": 1,
            "MaxWeightsLimit": 65535,
            "WeightsSetRateLimit": 100,
            "WeightsVersionKey": 0,
            "LastUpdate": [100, 0, 0],
        }

        class Chain:
            network = "test"
            endpoint = "wss://test.finney.opentensor.ai:443"
            block = 250
            calls = 0
            expected_netuid = 553

            def query(self, item, params, *, block=None):
                assert params[0] == self.expected_netuid
                self.last_block = block
                value = state[item[1]]
                return value.get(params[1]) if isinstance(value, dict) else value

            def execute(self, intent, wallet, **kwargs):
                assert intent.netuid == self.expected_netuid and intent.mechid == 0
                assert intent.uids == [1, 2] and intent.weights == [0.75, 0.25]
                assert kwargs["wait_for_finalization"] and kwargs["retries"] == 0
                self.calls += 1
                return ExtrinsicResult(
                    True, block_hash="0xabc", extrinsic_id="251-1", data={"reveal_round": 900}
                )

            def plan(self, intent, wallet):
                from bittensor.intents.plan import Plan

                return Plan(
                    op=intent.op,
                    summary="fixture",
                    signer="hotkey",
                    signer_address="validator",
                    fee=None,
                    effects=[],
                    warnings=[],
                )

        sub = Chain()
        self.assertEqual(t.preflight(config, sub)["validator_uid"], 0)
        self.assertEqual(sub.last_block, 250)
        for field, value in (
            ("network", "mainnet"),
            ("network", "local"),
            ("netuid", 0),
            ("netuid", True),
            ("netuid", "553"),
        ):
            bad = copy.deepcopy(config)
            bad["chain"][field] = value
            with self.assertRaises(ValueError):
                t.preflight(bad, sub)
        mainnet = copy.deepcopy(config)
        mainnet["chain"]["network"] = "finney"
        self.assertEqual(t.check_config(mainnet), 553)
        with self.assertRaisesRegex(ValueError, "pinned mainnet endpoint"):
            t.preflight(mainnet, sub)
        sub.network, sub.endpoint = "finney", t.ENDPOINTS["finney"]
        self.assertEqual(t.preflight(mainnet, sub)["network"], "finney")
        with self.assertRaisesRegex(ValueError, "pinned testnet endpoint"):
            t.preflight(config, sub)
        sub.network, sub.endpoint = "test", t.ENDPOINT
        fresh = copy.deepcopy(config)
        fresh["chain"]["netuid"] = 777
        sub.expected_netuid = 777
        self.assertEqual(t.preflight(fresh, sub)["netuid"], 777)
        sub.expected_netuid = 553
        state["Uids"]["miner-a"] = 4
        with self.assertRaisesRegex(ValueError, "registration"):
            t.preflight(config, sub)
        state["Uids"]["miner-a"] = 1
        state["Keys"][1] = "replacement"
        with self.assertRaisesRegex(ValueError, "registration"):
            t.preflight(config, sub)
        state["Keys"][1] = "miner-a"
        report = {
            "chain": config["chain"],
            "identities": {"1": "miner-a", "2": "miner-b"},
            "miners": [
                {"uid": 1, "status": "evaluated", "skill": 0.75},
                {"uid": 2, "status": "evaluated", "skill": 0.25},
            ],
        }
        wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
        from zils import models

        lite = models.D3_POOL_MODELS["d3-lite"]
        for model in (models.D3, lite):
            profiled = {**config, "model": models.spec(model)}
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "profile"):
                    t.publish_round(
                        config,
                        tmp,
                        {**report, "model": models.spec(model)},
                        sub,
                        wallet,
                        publish=True,
                    )
                with self.assertRaisesRegex(ValueError, "profile"):
                    t.publish_round(profiled, tmp, report, sub, wallet, publish=True)
                wrong = models.D3 if model == lite else lite
                with self.assertRaisesRegex(ValueError, "profile"):
                    t.publish_round(
                        profiled,
                        tmp,
                        {**report, "model": models.spec(wrong)},
                        sub,
                        wallet,
                        publish=True,
                    )
                self.assertEqual(sub.calls, 0)
                result = t.publish_round(
                    profiled,
                    tmp,
                    {**report, "model": models.spec(model)},
                    sub,
                    wallet,
                    publish=False,
                )
                self.assertEqual(result["status"], "dry_run")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            dry = t.publish_round(config, work, report, sub, wallet, publish=False)
            self.assertEqual(dry["status"], "dry_run")
            self.assertEqual(sub.calls, 0)
            self.assertFalse((work / "chain-attempt.json").exists())
            receipt = t.publish_round(config, work, report, sub, wallet, publish=True)
            self.assertEqual(receipt["status"], "committed")
            self.assertTrue(receipt["chain_write"])
            self.assertFalse(receipt["weights_verified"])
            self.assertEqual(
                t.publish_round(config, work, report, sub, wallet, publish=True), receipt
            )
            self.assertEqual(sub.calls, 1, "restart must not republish")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            with patch.object(sub, "execute", side_effect=TimeoutError("receipt unavailable")):
                receipt = t.publish_round(config, work, report, sub, wallet, publish=True)
            self.assertEqual(receipt["status"], "unknown")
            self.assertIsNone(receipt["chain_write"])
            self.assertEqual(
                t.publish_round(config, work, report, sub, wallet, publish=True), receipt
            )
            self.assertEqual(sub.calls, 1)
        # A lost subscription can be returned as a failed SDK result even after inclusion.
        lost = ChainError("Connection lost waiting for finalization", code=ErrorCode.UNKNOWN)
        for outcome in (ExtrinsicResult(False, error=lost), ExtrinsicResult(False), lost):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                options = (
                    {"side_effect": outcome}
                    if isinstance(outcome, Exception)
                    else {"return_value": outcome}
                )
                with patch.object(sub, "execute", **options) as execute:
                    receipt = t.publish_round(config, work, report, sub, wallet, publish=True)
                    self.assertEqual(receipt["status"], "unknown")
                    self.assertIsNone(receipt["chain_write"])
                    self.assertEqual(json.loads((work / "chain-receipt.json").read_text()), receipt)
                    self.assertEqual(
                        t.publish_round(config, work, report, sub, wallet, publish=True), receipt
                    )
                    self.assertEqual(
                        execute.call_count, 1, "uncertain outcomes must never be retried"
                    )
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            state["LastUpdate"][0] = 240
            self.assertEqual(
                t.publish_round(config, work, report, sub, wallet, publish=True)["status"],
                "rate_limited",
            )
            self.assertEqual(sub.calls, 1)
            state["LastUpdate"][0] = 100
            empty = {**report, "miners": [{"uid": 1, "status": "rejected", "skill": 1}]}
            self.assertEqual(
                t.publish_round(config, work, empty, sub, wallet, publish=True)["status"],
                "no_weights",
            )
            self.assertEqual(sub.calls, 1, "never invent weights for failed miners")
            bad = {**report, "miners": [{"uid": 8, "status": "evaluated", "skill": 0.5}]}
            with self.assertRaises(ValueError):
                t.publish_round(config, work, bad, sub, wallet, publish=True)
            old_local_report = {"miners": report["miners"]}
            with self.assertRaises(ValueError):
                t.publish_round(config, work, old_local_report, sub, wallet, publish=True)
            state["Uids"]["miner-b"] = None
            with self.assertRaises(ValueError):
                t.publish_round(config, work, report, sub, wallet, publish=True)
            state["Uids"]["miner-b"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                sub,
                "execute",
                return_value=ExtrinsicResult(
                    False,
                    message="rejected",
                    error=ChainError("rejected", code=ErrorCode.INVALID_ARGUMENT),
                ),
            ):
                receipt = t.publish_round(config, Path(tmp), report, sub, wallet, publish=True)
            self.assertEqual(receipt["status"], "failed")
            self.assertFalse(receipt["chain_write"])
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            with patch.object(sub, "execute", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    t.publish_round(config, work, report, sub, wallet, publish=True)
            self.assertFalse((work / "chain-receipt.json").exists())
            self.assertEqual(
                t.publish_round(config, work, report, sub, wallet, publish=True)["status"],
                "unknown",
            )
            self.assertEqual(sub.calls, 1)
        state["Weights"] = [(1, 49151), (2, 16384)]
        state["LastUpdate"][0] = 251
        sub.block = 252
        record = {
            **t.preflight(config, sub),
            "block": 250,
            "chain_write": True,
            "requested_weights": {"1": 0.75, "2": 0.25},
        }
        self.assertTrue(t.verify_receipt(config, record, sub)["weights_verified"])
        state["LastUpdate"][0] = 100
        self.assertFalse(
            t.verify_receipt(config, record, sub)["weights_verified"],
            "old weights must not prove this commit revealed",
        )
        sub.expected_netuid = 777
        with tempfile.TemporaryDirectory() as tmp:
            receipt = t.publish_round(
                fresh, Path(tmp), {**report, "chain": fresh["chain"]}, sub, wallet, publish=True
            )
            self.assertEqual(receipt["status"], "committed")
            self.assertEqual(receipt["netuid"], 777)

    def test_base_tier_track_blends_into_one_weight_vector(self):
        from zils import base_round, testnet as t, validator

        draw = {"manifest_sha256": "a" * 64, "dataset_sha256": "b" * 64}
        track = {"share": 0.7, "incumbent_sha256": "c" * 64, "draws": [draw]}
        config = {
            "chain": {"network": "test", "netuid": 579},
            "validator_hotkey": "validator",
            "members": {
                "1": {"hotkey": "miner-a", "port": 8901},
                "2": {"hotkey": "miner-b", "port": 8902},
            },
            "base_tier": track,
        }
        state = {
            "NetworksAdded": True,
            "Uids": {"validator": 0, "miner-a": 1, "miner-b": 2},
            "Keys": {0: "validator", 1: "miner-a", 2: "miner-b"},
            "ValidatorPermit": [True, False, False],
            "SubnetOwnerHotkey": "validator",
            "CommitRevealWeightsEnabled": False,
            "MinAllowedWeights": 1,
            "MaxWeightsLimit": 65535,
            "WeightsSetRateLimit": 100,
            "WeightsVersionKey": 0,
            "LastUpdate": [100, 0, 0],
            "CommitmentOf": {},
        }
        sent = []

        class Chain:
            network = "test"
            endpoint = "wss://test.finney.opentensor.ai:443"
            block = 250

            def query(self, item, params, *, block=None):
                assert params[0] == 579
                value = state[item[1]]
                return value.get(params[1]) if isinstance(value, dict) else value

            def plan(self, intent, wallet):
                from bittensor.intents.plan import Plan

                return Plan(
                    op=intent.op,
                    summary="fixture",
                    signer="hotkey",
                    signer_address="validator",
                    fee=None,
                    effects=[],
                    warnings=[],
                )

            def execute(self, intent, wallet, **kwargs):
                from bittensor.result import ExtrinsicResult

                sent.append(dict(zip(intent.uids, intent.weights)))
                return ExtrinsicResult(True, block_hash="0xabc", extrinsic_id="251-1", data={})

        sub = Chain()
        wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
        adapter_rows = [
            {"uid": 1, "status": "evaluated", "skill": 0.75},
            {"uid": 2, "status": "evaluated", "skill": 0.25},
        ]
        decision = {
            **draw,
            "policy": dict(base_round.DEFAULT_POLICY),
            "incumbent": {"uid": 65535, "sha256": "c" * 64},
            "candidates": [
                {"uid": 2, "sha256": "d" * 64, "eligible": True, "skill": 0.4},
                {"uid": 1, "sha256": "e" * 64, "eligible": False, "skill": 0.5},
            ],
            "pending_winner": {"uid": 2, "sha256": "d" * 64},
        }

        def report(miners, base_tier):
            return {
                "chain": config["chain"],
                "identities": {"1": "miner-a", "2": "miner-b"},
                "miners": miners,
                "base_tier": base_tier,
            }

        # King of the hill: uid 1 (hotkey miner-a) holds the crowned base; uid 2's eligible
        # challenger earns nothing before promotion. Both tracks: 0.7 * {1: 1} + 0.3 * adapter.
        crowned = {**config, "base_tier": {**track, "kings": ["miner-a"]}}
        with tempfile.TemporaryDirectory() as tmp:
            dry = t.publish_round(crowned, tmp, report(adapter_rows, decision), sub, wallet)
            self.assertEqual(dry["status"], "dry_run")
            self.assertAlmostEqual(dry["requested_weights"]["1"], 0.925)
            self.assertAlmostEqual(dry["requested_weights"]["2"], 0.075)
            self.assertEqual(dry["tracks"]["base"], {"1": 1.0})
            self.assertEqual(dry["tracks"]["adapter"], {"1": 0.75, "2": 0.25})
            self.assertEqual(dry["tracks"]["base_share"], 0.7)
            self.assertEqual((dry["tracks"]["burn_uid"], dry["tracks"]["burned"]), (0, 0))
            self.assertEqual(sent, [])
            receipt = t.publish_round(
                crowned, tmp, report(adapter_rows, decision), sub, wallet, publish=True
            )
            self.assertEqual(receipt["status"], "included")
            self.assertEqual(len(sent), 1)
            self.assertAlmostEqual(sent[0][1], 0.925)
            self.assertAlmostEqual(sent[0][2], 0.075)
            self.assertEqual(
                json.loads((Path(tmp) / "chain-receipt.json").read_text())["tracks"],
                receipt["tracks"],
            )
        # An empty track burns its share at burn_uid (default 0, here the subnet owner's uid) instead
        # of enlarging the other track; both empty burns the whole vector, so weights keep flowing.
        rejected = [{"uid": 1, "status": "rejected", "skill": 0.0}, {"uid": 2, "status": "missing"}]
        for decided in (decision, None):
            with self.subTest(decision=decided is not None), tempfile.TemporaryDirectory() as tmp:
                dry = t.publish_round(crowned, tmp, report(rejected, decided), sub, wallet)
                self.assertEqual(set(dry["requested_weights"]), {"0", "1"})
                self.assertAlmostEqual(dry["requested_weights"]["0"], 0.3)
                self.assertAlmostEqual(dry["requested_weights"]["1"], 0.7)
                # The unowned stock base pays no miner, whatever the exam found.
                dry = t.publish_round(config, tmp, report(adapter_rows, decided), sub, wallet)
                self.assertEqual(set(dry["requested_weights"]), {"0", "1", "2"})
                self.assertAlmostEqual(dry["requested_weights"]["0"], 0.7)
                self.assertAlmostEqual(dry["requested_weights"]["1"], 0.225)
                self.assertAlmostEqual(dry["requested_weights"]["2"], 0.075)
                self.assertEqual(dry["tracks"]["base"], {})
                self.assertAlmostEqual(dry["tracks"]["burned"], 0.7)
                dry = t.publish_round(config, tmp, report(rejected, decided), sub, wallet)
                self.assertEqual((dry["status"], dry["requested_weights"]), ("dry_run", {"0": 1.0}))
        # A burn uid that is not the subnet owner's would pay a validator self-weight Yuma masks,
        # or a miner; refused before anything is sent.
        with tempfile.TemporaryDirectory() as tmp:
            for burn in (1, 3):
                elsewhere = {**config, "base_tier": {**track, "burn_uid": burn}}
                with self.assertRaisesRegex(ValueError, "subnet owner"):
                    t.publish_round(elsewhere, tmp, report(rejected, None), sub, wallet)
        # Refusals: a decision the track does not cover, a decision without a track, a bad track.
        with tempfile.TemporaryDirectory() as tmp:
            for bad in (
                {**decision, "dataset_sha256": "f" * 64},
                {**decision, "incumbent": {"uid": 65535, "sha256": "f" * 64}},
                {
                    **decision,
                    "candidates": [{"uid": 9, "sha256": "d" * 64, "eligible": True, "skill": 0.4}],
                    "weights": {"9": 1.0},
                },
            ):
                with self.assertRaises(ValueError):
                    t.publish_round(config, tmp, report(adapter_rows, bad), sub, wallet)
            untracked = {k: v for k, v in config.items() if k != "base_tier"}
            with self.assertRaisesRegex(ValueError, "no base-tier track"):
                t.publish_round(untracked, tmp, report(adapter_rows, decision), sub, wallet)
            with self.assertRaisesRegex(ValueError, "share"):
                t.publish_round(
                    {**config, "base_tier": {**track, "share": 2}},
                    tmp,
                    report(adapter_rows, decision),
                    sub,
                    wallet,
                )
            self.assertEqual(len(sent), 1)
        # The validator freezes the current decision with the round; a restart republishes the same one.
        with tempfile.TemporaryDirectory() as tmp:
            directory, work = Path(tmp) / "validator", Path(tmp) / "round"
            (directory / "base-tier").mkdir(parents=True)
            work.mkdir()
            plain = {"miners": adapter_rows}
            self.assertEqual(validator.attach_base_tier(untracked, directory, work, plain), plain)
            self.assertFalse((work / "base-tier.json").exists())
            self.assertIsNone(
                validator.attach_base_tier(config, directory, work, plain)["base_tier"]
            )
            (directory / "base-tier" / "decision.json").write_text(json.dumps(decision))
            self.assertIsNone(
                validator.attach_base_tier(config, directory, work, plain)["base_tier"]
            )
            later = Path(tmp) / "later"
            later.mkdir()
            attached = validator.attach_base_tier(config, directory, later, plain)
            self.assertEqual(attached["base_tier"], decision)
            self.assertEqual(json.loads((later / "base-tier.json").read_text()), decision)
            self.assertNotIn("base_tier", plain)

    def test_wallet_signing_and_bundles_keep_keys_out(self):
        from bittensor.wallet import Wallet

        import zils
        from zils import benchmark, fleet, protocol as rehearsal, runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster = []
            for index in range(3):
                wallet = {"path": str(root / "wallets"), "name": "fixture", "hotkey": str(index)}
                key = (
                    Wallet(**wallet)
                    .regenerate_hotkey(seed=bytes([index + 1]) * 32, suppress=True)
                    .hotkey
                )
                roster.append({"uid": index + 8, "wallet": wallet, "hotkey": key.ss58_address})
            data = root / "data"
            benchmark.build(data, seed=42)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            for name in zils.ARTIFACT_FILES:
                (checkpoint / name).write_bytes(b"fixture")
            out = fleet.initialize(
                root / "fleet",
                data,
                checkpoint,
                "127.0.0.1",
                8900,
                [8901, 8902],
                {
                    "chain": {"network": "test", "netuid": 777},
                    "validator": roster[0],
                    "miners": roster[1:],
                },
            )
            cfg = json.loads((out / "validator/config.json").read_text())
            self.assertEqual(set(cfg["members"]), {"9", "10"})
            self.assertNotIn("seed", cfg)
            self.assertEqual(cfg["chain"], {"network": "test", "netuid": 777})
            for uid in (9, 10):
                cfg = json.loads((out / f"miner-{uid}/config.json").read_text())
                self.assertNotIn("seed", cfg)
                key = runtime.signing_key(cfg)
                message = runtime.signed({"hello": "Fez"}, key)
                self.assertEqual(runtime.verified(message, cfg["hotkey"]), {"hello": "Fez"})
                claim = {
                    "uid": uid,
                    "hotkey": cfg["hotkey"],
                    "round_id": "a" * 32,
                    "sha256": "b" * 64,
                    "endpoint": "http://127.0.0.1:8901",
                }
                rehearsal.register(
                    {"claim": claim, "signature": key.sign(rehearsal.canonical(claim)).hex()},
                    "a" * 32,
                    {uid: cfg["hotkey"]},
                    {},
                )
                import tarfile

                with tarfile.open(out / f"miner-{uid}.tar.gz") as archive:
                    self.assertFalse(
                        any(
                            "wallets/" in n or "/hotkeys/" in n or "test.jsonl" in n
                            for n in archive.getnames()
                        )
                    )
                cfg["hotkey"] = roster[0]["hotkey"]
                with self.assertRaises(ValueError):
                    runtime.signing_key(cfg)


if __name__ == "__main__":
    unittest.main()
