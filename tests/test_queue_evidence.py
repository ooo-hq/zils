"""The adapter track from signed hosted-queue evidence: rule, cap, refusals, freeze and the blend."""

import copy
import importlib.util
import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bittensor_wallet import Keypair

from zils import queue_evidence as q
from zils.runtime import signed

SIGNER = Keypair.create_from_uri("//Evidence")
A, B, C, V = (Keypair.create_from_uri(f"//{n}").ss58_address for n in ("A", "B", "C", "V"))
UNTIL = datetime(2026, 10, 19, tzinfo=UTC)
TRACK = {"url": "https://training.example/v1/network/adapter-evidence", "signer": SIGNER.ss58_address,
         "window_hours": 72, "revenue_cap": 0.5}  # fmt: skip
# Alpha at 0.003 TAO, 1 alpha per block to the subnet: 72 h of the miners' 41% at $200/TAO is $2,656.80.
CHAIN = {"alpha_price_tao": 0.003, "alpha_per_block": 1.0}
BUDGET = 0.41 * 1.0 * 72 * 300 * 0.003 * 200


def evidence(miners, *, key=SIGNER, until="2026-10-19T00:00:00Z", since="2026-10-16T00:00:00Z",
             tao_usd=200.0):  # fmt: skip
    payload = {"schema": q.SCHEMA, "since": since, "until": until, "tao_usd": tao_usd,
               "miners": copy.deepcopy(miners), "signer": key.ss58_address}  # fmt: skip
    return signed(payload, key)


def spend(hotkey, usd):
    return {"hotkey": hotkey, "spend_nanos": int(usd * 1e9)}


MINERS = [spend(A, 3), spend(B, 1), spend(C, 1)]  # C is unregistered
MEMBERS = {"4": {"hotkey": A}, "5": {"hotkey": B}}


class RuleTest(unittest.TestCase):
    def test_pay_follows_paid_spend_and_the_unregistered_slice_burns(self):
        payload = q.verify(evidence(MINERS), TRACK, UNTIL)
        weights = q.track_weights(payload, MEMBERS, 1.0, 0)
        self.assertEqual(set(weights), {0, 4, 5})
        self.assertAlmostEqual(weights[4], 3 / 5)
        self.assertAlmostEqual(weights[5], 1 / 5)
        self.assertAlmostEqual(weights[0], 1 / 5)

    def test_spend_above_the_revenue_cap_burns(self):
        payload = q.verify(evidence(MINERS), TRACK, UNTIL)
        # $5 of revenue at k = 0.5 buys $2.50 of a share worth half the budget.
        paid = q.paid_fraction(payload, TRACK, 0.5, CHAIN)
        self.assertAlmostEqual(paid, 0.5 * 5 / (0.5 * BUDGET))
        weights = q.track_weights(payload, MEMBERS, paid, 0)
        self.assertAlmostEqual(weights[4], paid * 3 / 5)
        self.assertAlmostEqual(weights[5], paid * 1 / 5)
        self.assertAlmostEqual(weights[0], 1 - paid * 4 / 5)
        rich = q.verify(evidence([spend(A, 100_000)]), TRACK, UNTIL)
        self.assertEqual(q.paid_fraction(rich, TRACK, 0.5, CHAIN), 1.0)
        self.assertEqual(q.track_weights(rich, MEMBERS, 1.0, 0), {4: 1.0})
        self.assertEqual(q.paid_fraction(payload, TRACK, 0.0, CHAIN), 0.0)

    def test_no_paid_spend_is_an_empty_track(self):
        empty = q.verify(evidence([]), TRACK, UNTIL)
        self.assertEqual(q.paid_fraction(empty, TRACK, 0.3, CHAIN), 0.0)
        self.assertEqual(q.track_weights(empty, MEMBERS, 0.0, 0), {})
        payload = q.verify(evidence(MINERS), TRACK, UNTIL)
        self.assertEqual(q.track_weights(payload, {}, 1.0, 0), {})

    def test_refusals(self):
        other = Keypair.create_from_uri("//Other")
        for message, pattern in (
            (evidence(MINERS, key=other), "signature|signer"),
            (
                evidence(MINERS, until="2026-10-18T23:00:00Z", since="2026-10-15T23:00:00Z"),
                "window",
            ),  # fmt: skip
            (evidence(MINERS, since="2026-10-18T00:00:00Z"), "window"),
            (evidence(MINERS, tao_usd=0), "price"),
            (evidence(MINERS, tao_usd="200"), "price"),
            (evidence([{**MINERS[0], "spend_nanos": 0}]), "row"),
            (evidence([{**MINERS[0], "spend_nanos": 1.5}]), "row"),
            (evidence([{**MINERS[0], "job_sha256": "1" * 64}]), "row"),
            (evidence([{**MINERS[0], "hotkey": "miner-a"}]), "row"),
            (evidence([MINERS[0], MINERS[0]]), "twice"),
        ):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                q.verify(message, TRACK, UNTIL)
        tampered = evidence(MINERS)
        tampered["payload"]["miners"][0]["spend_nanos"] = 10**15
        with self.assertRaisesRegex(ValueError, "signature"):
            q.verify(tampered, TRACK, UNTIL)
        v1 = signed({**evidence(MINERS)["payload"], "schema": "zils-adapter-evidence/v1"}, SIGNER)
        with self.assertRaisesRegex(ValueError, "schema"):
            q.verify(v1, TRACK, UNTIL)
        for bad in (
            {**TRACK, "url": "http://training.example/x"},
            {**TRACK, "signer": "not a key"},
            {**TRACK, "window_hours": 0},
            {**TRACK, "revenue_cap": 0},
            {**TRACK, "revenue_cap": 1.5},
            {k: v for k, v in TRACK.items() if k != "revenue_cap"},
        ):
            with self.subTest(track=bad), self.assertRaises(ValueError):
                q.validate_track(bad)

    def test_window_ends_on_the_hour(self):
        self.assertEqual(q.window_end(datetime(2026, 10, 19, 0, 59, 3, tzinfo=UTC)), UNTIL)


class FreezeTest(unittest.TestCase):
    def test_validator_freezes_one_fetch_per_round(self):
        from zils import validator

        config = {"adapter_evidence": TRACK}
        plain = {"miners": []}
        message = evidence(MINERS)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(q, "fetch", return_value=message) as fetch,
        ):
            work = Path(tmp)
            self.assertEqual(validator.attach_adapter_evidence({}, work, plain), plain)
            first = validator.attach_adapter_evidence(config, work, plain, UNTIL)
            again = validator.attach_adapter_evidence(
                config, work, plain, datetime(2026, 10, 20, tzinfo=UTC)
            )
            self.assertEqual(first["adapter_evidence"], message)
            self.assertEqual(again, first)
            fetch.assert_called_once_with(TRACK, UNTIL)
            self.assertNotIn("adapter_evidence", plain)


@unittest.skipUnless(importlib.util.find_spec("bittensor"), "install requirements/testnet.txt")
class PublishTest(unittest.TestCase):
    def test_queue_evidence_is_the_adapter_share(self):
        from zils import testnet as t

        config = {
            "chain": {"network": "test", "netuid": 579},
            "validator_hotkey": V,
            "members": MEMBERS,
            "base_tier": {
                "share": 0.7,
                "incumbent_sha256": "c" * 64,
                "draws": [{"manifest_sha256": "a" * 64, "dataset_sha256": "b" * 64}],
            },
            "adapter_evidence": TRACK,
        }
        state = {
            "NetworksAdded": True,
            "Uids": {V: 0, A: 4, B: 5},
            "Keys": {0: V, 4: A, 5: B},
            "ValidatorPermit": [True],
            "SubnetOwnerHotkey": V,
            "CommitRevealWeightsEnabled": False,
            "MinAllowedWeights": 1,
            "MaxWeightsLimit": 65535,
            "WeightsSetRateLimit": 100,
            "WeightsVersionKey": 0,
            "LastUpdate": [100],
            "CommitmentOf": {},
            "SubnetMovingPrice": {"bits": round(CHAIN["alpha_price_tao"] * 2**32)},
            "SubnetAlphaOutEmission": 10**9,
        }

        class Chain:
            network, endpoint, block = "test", t.ENDPOINT, 250

            def query(self, item, params, *, block=None):
                value = state[item[1]]
                if item[1] == "SubnetMovingPrice":
                    return value
                return value.get(params[1]) if isinstance(value, dict) else value

            def plan(self, intent, wallet):
                from bittensor.intents.plan import Plan

                return Plan(op=intent.op, summary="fixture", signer="hotkey", signer_address=V,
                            fee=None, effects=[], warnings=[])  # fmt: skip

        wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address=V))
        identities = {uid: m["hotkey"] for uid, m in MEMBERS.items()}
        frozen_rows = [
            {"uid": 4, "status": "evaluated", "skill": 1.0}
        ]  # ignored: queue replaces it

        def report(message):
            return {"chain": config["chain"], "identities": identities, "miners": frozen_rows,
                    **({"adapter_evidence": message} if message is not None else {})}  # fmt: skip

        payload = q.verify(evidence(MINERS), TRACK, UNTIL)
        paid = q.paid_fraction(payload, TRACK, 0.3, CHAIN)
        self.assertAlmostEqual(paid, 0.5 * 5 / (0.3 * BUDGET), places=4)
        with tempfile.TemporaryDirectory() as tmp:
            dry = t.publish_round(config, tmp, report(evidence(MINERS)), Chain(), wallet)
        self.assertEqual((dry["status"], dry["adapter_source"]), ("dry_run", "queue"))
        self.assertAlmostEqual(dry["alpha_price_tao"], CHAIN["alpha_price_tao"])
        self.assertEqual(dry["alpha_per_block"], 1.0)
        self.assertAlmostEqual(dry["adapter_paid"], paid, places=6)
        # No king yet: the base share burns at the owner uid; the adapter share pays queue workers
        # their capped slice of spend, and the rest of it burns too.
        weights = dry["requested_weights"]
        self.assertAlmostEqual(weights["4"], 0.3 * paid * 3 / 5, places=6)
        self.assertAlmostEqual(weights["5"], 0.3 * paid * 1 / 5, places=6)
        self.assertAlmostEqual(weights["0"], 1 - 0.3 * paid * 4 / 5, places=6)
        self.assertAlmostEqual(dry["tracks"]["burned"], 1 - 0.3 * paid * 4 / 5, places=6)
        with tempfile.TemporaryDirectory() as tmp:
            empty = t.publish_round(config, tmp, report(evidence([])), Chain(), wallet)
        self.assertEqual(empty["requested_weights"], {"0": 1.0})
        untracked = {k: v for k, v in config.items() if k != "adapter_evidence"}
        unburnable = {k: v for k, v in config.items() if k != "base_tier"}
        for cfg, message, pattern in (
            (config, None, "has none"),
            (untracked, evidence(MINERS), "takes no adapter evidence"),
            (config, evidence(MINERS, key=Keypair.create_from_uri("//Other")), "signature|signer"),
            (unburnable, evidence(MINERS), "burn"),
        ):
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, pattern):
                    t.publish_round(cfg, tmp, report(copy.deepcopy(message)), Chain(), wallet)
        json.dumps(dry)  # the receipt stays JSON


if __name__ == "__main__":
    unittest.main()
