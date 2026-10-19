"""The adapter track from signed hosted-queue evidence: rule, refusals, freeze and the 579 blend."""

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
         "window_hours": 72}  # fmt: skip


def row(hotkey, sha, *, tokens=1000, skill=0.5, outcome="valid", accepted=True, **extra):
    return {
        "hotkey": hotkey,
        "job_sha256": "1" * 64,
        "purpose": "customer",
        "internal": False,
        "owner_sha256": "2" * 64,
        "outcome": outcome,
        "total_tokens": tokens,
        "finished_at": "2026-10-18T12:00:00Z",
        "sha256": sha,
        "accepted": accepted,
        "skill": skill,
        **extra,
    }


def evidence(rows, *, key=SIGNER, until="2026-10-19T00:00:00Z", since="2026-10-16T00:00:00Z"):
    payload = {"schema": q.SCHEMA, "since": since, "until": until, "attempts": rows,
               "signer": key.ss58_address}  # fmt: skip
    return signed(payload, key)


ROWS = [
    row(A, "a" * 64, tokens=3000, skill=0.5),  # 1500
    row(B, "b" * 64, tokens=1000, skill=0.9),  # 900
    row(B, "c" * 64, tokens=1000, skill=0.9, accepted=False),  # trained, not delivered
    row(A, "d" * 64, skill=1.0, purpose="qualification"),  # qualification never earns
    row(B, "e" * 64, skill=1.0, internal=True),  # operator-owned job never earns
    row(A, None, skill=None, outcome="abandoned", accepted=False),  # A: 2 valid, 1 failure
    row(C, "f" * 64, skill=0.8),  # unregistered on 579
]
MEMBERS = {"4": {"hotkey": A}, "5": {"hotkey": B}}


class RuleTest(unittest.TestCase):
    def test_delivered_customer_work_times_reliability(self):
        payload = q.verify(evidence(ROWS), TRACK, UNTIL)
        stats = q.hotkey_scores(payload)
        self.assertEqual((stats[A]["valid"], stats[A]["failures"], stats[A]["earning"]), (2, 1, 1))
        self.assertAlmostEqual(stats[A]["reliability"], 3 / 5)
        self.assertAlmostEqual(stats[B]["reliability"], 4 / 5)
        a, b = 1500 * 3 / 5, 900 * 4 / 5
        weights = q.track_weights(payload, MEMBERS)
        self.assertEqual(set(weights), {4, 5})
        self.assertAlmostEqual(weights[4], a / (a + b))
        self.assertAlmostEqual(weights[5], b / (a + b))

    def test_nothing_delivered_is_an_empty_track(self):
        payload = q.verify(evidence(ROWS[2:6]), TRACK, UNTIL)
        self.assertEqual(q.track_weights(payload, MEMBERS), {})
        self.assertEqual(q.track_weights(q.verify(evidence(ROWS), TRACK, UNTIL), {}), {})

    def test_refusals(self):
        other = Keypair.create_from_uri("//Other")
        for message, pattern in (
            (evidence(ROWS, key=other), "signature|signer"),
            (evidence(ROWS, until="2026-10-18T23:00:00Z", since="2026-10-15T23:00:00Z"), "window"),
            (evidence(ROWS, since="2026-10-18T00:00:00Z"), "window"),
            (evidence([{**ROWS[0], "skill": 1.5}]), "row"),
            (evidence([{**ROWS[0], "total_tokens": 0}]), "row"),
            (evidence([{**ROWS[0], "extra": 1}]), "row"),
            (evidence([{**ROWS[0], "hotkey": "miner-a"}]), "row"),
        ):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                q.verify(message, TRACK, UNTIL)
        tampered = evidence(ROWS)
        tampered["payload"]["attempts"][0]["skill"] = 1.0
        with self.assertRaisesRegex(ValueError, "signature"):
            q.verify(tampered, TRACK, UNTIL)
        copied = q.verify(evidence([ROWS[0], {**ROWS[1], "sha256": "a" * 64}]), TRACK, UNTIL)
        with self.assertRaisesRegex(ValueError, "twice"):
            q.track_weights(copied, MEMBERS)
        for bad in (
            {**TRACK, "url": "http://training.example/x"},
            {**TRACK, "signer": "not a key"},
            {**TRACK, "window_hours": 0},
            {k: v for k, v in TRACK.items() if k != "window_hours"},
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
        message = evidence(ROWS)
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
        }

        class Chain:
            network, endpoint, block = "test", t.ENDPOINT, 250

            def query(self, item, params, *, block=None):
                value = state[item[1]]
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

        payload = q.verify(evidence(ROWS), TRACK, UNTIL)
        adapter = q.track_weights(payload, MEMBERS)
        with tempfile.TemporaryDirectory() as tmp:
            dry = t.publish_round(config, tmp, report(evidence(ROWS)), Chain(), wallet)
        self.assertEqual((dry["status"], dry["adapter_source"]), ("dry_run", "queue"))
        self.assertEqual(dry["tracks"]["adapter"], {str(k): v for k, v in adapter.items()})
        # No king yet: the base share burns at the owner uid, the adapter share pays queue workers.
        self.assertAlmostEqual(dry["requested_weights"]["0"], 0.7)
        self.assertAlmostEqual(dry["requested_weights"]["4"], 0.3 * adapter[4])
        self.assertAlmostEqual(dry["requested_weights"]["5"], 0.3 * adapter[5])
        with tempfile.TemporaryDirectory() as tmp:
            empty = t.publish_round(config, tmp, report(evidence(ROWS[2:6])), Chain(), wallet)
        self.assertEqual(empty["requested_weights"], {"0": 1.0})
        untracked = {k: v for k, v in config.items() if k != "adapter_evidence"}
        for cfg, message, pattern in (
            (config, None, "has none"),
            (untracked, evidence(ROWS), "takes no adapter evidence"),
            (config, evidence(ROWS, key=Keypair.create_from_uri("//Other")), "signature|signer"),
        ):
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, pattern):
                    t.publish_round(cfg, tmp, report(copy.deepcopy(message)), Chain(), wallet)
        json.dumps(dry)  # the receipt stays JSON


if __name__ == "__main__":
    unittest.main()
