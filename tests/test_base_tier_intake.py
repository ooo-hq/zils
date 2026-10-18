"""The validator's own base-tier round: fetch once per hash, score once per draw, decide, confirm."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import zils
from tests.test_base_round import make_cases, sealed
from zils import base_round, models, protocol as wire, validator

RUNNER = """import json, random, sys, os
from pathlib import Path
checkpoint = Path(sys.argv[sys.argv.index("--checkpoint") + 1])
meta = json.loads((checkpoint / "model.json").read_text())
accuracy = 0.55 if meta["kind"] == "base" else json.loads(
    (checkpoint / "model-00001-of-00002.safetensors").read_text())["accuracy"]
requests = json.load(sys.stdin)
rng = random.Random(f"{meta['kind']}-{accuracy}-{len(requests)}")
predictions = []
for row in requests:
    keys = sorted(row["question"]["criteria"])
    label = keys[int(row["state"]["text"].split()[-1]) % 4]
    chosen = label if rng.random() < accuracy else rng.choice([k for k in keys if k != label])
    p = {k: 0.1 for k in keys}
    p[chosen] = 0.7
    predictions.append({"id": row["id"], "elapsed_ms": 1.0, "probabilities": p})
with open(os.environ["ZILS_TEST_RUNNER_LOG"], "a") as log:
    log.write(json.dumps({"kind": meta["kind"], "accuracy": accuracy}) + "\\n")
print(json.dumps({"predictions": predictions, "runtime": {"fixture": True}}))
"""


def candidate(root, accuracy):
    root.mkdir()
    models.write_metadata(root, kind="candidate", model=models.D3)
    (root / "model-00001-of-00002.safetensors").write_text(json.dumps({"accuracy": accuracy}))
    (root / "model-00002-of-00002.safetensors").write_bytes(b"shard two")
    (root / "model.safetensors.index.json").write_text("{}")
    for name in models.D3_CONTRACT_FILES:
        (root / name).write_bytes(name.encode())
    return root


def artifact_server(source, requested, *, huge=False):
    class Files(wire.Handler):
        def do_GET(self):
            name = self.path.rsplit("/", 1)[-1]
            requested.append(name)
            path = source() / name
            if huge:
                self.send_response(200)
                self.send_header("Content-Length", str(models.MAX_CANDIDATE_BYTES + 1))
                self.end_headers()
                return
            if not path.is_file():
                self.reply(404, {"error": "unknown artifact"})
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Files


@unittest.skipUnless(
    importlib.util.find_spec("bittensor_wallet") and importlib.util.find_spec("kev"),
    "use .venv-kev",
)
class BaseTierIntakeTest(unittest.TestCase):
    def test_validator_collects_scores_and_confirms_base_candidates(self):
        from bittensor_wallet import Keypair

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vdir = root / "validator"
            (vdir / "draws").mkdir(parents=True)
            calibration = make_cases("cal", 60)
            sealed(vdir / "draws/first", calibration, make_cases("first", 60))
            sealed(vdir / "draws/second", calibration, make_cases("second", 60))
            draws = [base_round.draw(vdir / "draws/first"), base_round.draw(vdir / "draws/second")]
            (vdir / "reference").mkdir()
            models.write_metadata(vdir / "reference", kind="base", model=models.D3)
            incumbent = zils.checkpoint_hash(vdir / "reference")
            strong = candidate(root / "strong", 0.85)
            weak = candidate(root / "weak", 0.40)
            later = candidate(root / "later", 0.50)
            adapter = root / "adapter"
            adapter.mkdir()
            for name in models.TEXT_ADAPTER_FILES:
                (adapter / name).write_bytes(b"lora")
            models.write_metadata(adapter, model=models.D3)
            sources = {1: strong, 2: weak, 3: adapter, 4: strong}
            requested = {uid: [] for uid in sources}
            keys = {uid: Keypair.create_from_seed("0x" + f"{uid:02x}" * 32) for uid in sources}
            members = {uid: key.ss58_address for uid, key in keys.items()}
            runner = root / "runner-python"
            runner.write_text(f"#!{sys.executable}\n{RUNNER}")
            runner.chmod(0o700)
            log = root / "runner.log"
            log.touch()
            args = SimpleNamespace(device="cpu", runtime_python=sys.executable, poll=0.1)
            rounds = vdir / "state/rounds"
            validate = Mock()
            with (
                wire.local_server(artifact_server(lambda: sources[1], requested[1])) as s1,
                wire.local_server(artifact_server(lambda: sources[2], requested[2])) as s2,
                wire.local_server(artifact_server(lambda: sources[3], requested[3])) as s3,
                wire.local_server(
                    artifact_server(lambda: sources[4], requested[4], huge=True)
                ) as s4,
                patch.dict(
                    os.environ,
                    {"ZILS_D3_RUNTIME_PYTHON": str(runner), "ZILS_TEST_RUNNER_LOG": str(log)},
                ),
                patch("zils.d3.validate_candidate", validate),
            ):
                ports = {
                    uid: int(url.rsplit(":", 1)[1]) for uid, url in enumerate((s1, s2, s3, s4), 1)
                }
                config = {
                    "validator": "http://127.0.0.1:1",
                    "model": models.spec(models.D3),
                    "base_revision": models.spec(models.D3)["base_revision"],
                    "members": {
                        str(uid): {"hotkey": members[uid], "port": ports[uid]} for uid in sources
                    },
                    "base_tier": {
                        "share": 0.7,
                        "incumbent_sha256": incumbent,
                        "draws": draws,
                        "policy": {"bootstrap_resamples": 200},
                    },
                    "base_round_dir": "draws",
                    "base_reference": "reference",
                }
                audited = validator.base_draws(config, vdir)
                self.assertEqual([item for item, _ in audited], draws)
                self.assertEqual([p.name for _, p in audited], ["first", "second"])
                with self.assertRaisesRegex(ValueError, "no sealed draw"):
                    validator.base_draws(
                        {
                            **config,
                            "base_tier": {
                                **config["base_tier"],
                                "draws": [{**draws[0], "dataset_sha256": "0" * 64}],
                            },
                        },
                        vdir,
                    )
                with self.assertRaisesRegex(ValueError, "incumbent checkpoint"):
                    validator.base_draws(
                        {
                            **config,
                            "base_tier": {**config["base_tier"], "incumbent_sha256": "0" * 64},
                        },
                        vdir,
                    )
                self.assertIsNone(validator.base_draws({"members": {}}, vdir))

                def announce(current, registries, uid, source, **changes):
                    claim = {
                        "round_id": current["round_id"],
                        "uid": uid,
                        "hotkey": members[uid],
                        "sha256": zils.checkpoint_hash(source),
                        "endpoint": f"http://127.0.0.1:{ports[uid]}",
                        "kind": "candidate",
                        **changes,
                    }
                    message = {
                        "claim": claim,
                        "signature": keys[uid].sign(wire.canonical(claim)).hex(),
                    }
                    return validator.accept(
                        config, current, members, registries, rounds, message, "127.0.0.1", False
                    )

                def new_round(rid):
                    work = rounds / rid
                    (work / "base-submissions").mkdir(parents=True)
                    current = {
                        "round_id": rid,
                        "status": "collecting",
                        "base_tier": {"incumbent_sha256": incumbent, "draws": draws},
                    }
                    return work, current, {"adapter": {}, "candidate": {}}

                work, current, registries = new_round("a" * 32)
                for uid in (1, 2, 3):
                    self.assertEqual(announce(current, registries, uid, sources[uid])[0], 200)
                with self.assertRaisesRegex(ValueError, "duplicate checkpoint"):
                    announce(current, registries, 4, strong)
                self.assertEqual(announce(current, registries, 4, strong, sha256="f" * 64)[0], 200)
                self.assertEqual(set(registries["candidate"]), {1, 2, 3, 4})
                self.assertEqual(registries["adapter"], {})
                self.assertTrue((work / "base-submissions/1.json").exists())
                with self.assertRaisesRegex(ValueError, "incumbent"):
                    announce(current, registries, 1, vdir / "reference")
                with self.assertRaisesRegex(ValueError, "claim fields"):
                    announce(current, registries, 1, strong, kind="adapter")
                self.assertEqual(
                    announce(
                        {k: v for k, v in current.items() if k != "base_tier"},
                        registries,
                        1,
                        strong,
                    )[0],
                    409,
                )
                decision = validator.base_tier_epoch(
                    config, vdir, work, registries["candidate"], args, audited
                )
                self.assertEqual(decision["pending_winner"]["uid"], 1)
                self.assertEqual(decision["confirmed_winner"]["uid"], 1)
                self.assertEqual(decision["manifest_sha256"], draws[1]["manifest_sha256"])
                self.assertEqual(decision["incumbent"]["uid"], validator.INCUMBENT_UID)
                self.assertEqual(
                    {c["uid"]: c["eligible"] for c in decision["candidates"]}, {1: True, 2: False}
                )
                self.assertEqual(set(decision["rejected"]), {"3", "4"})
                self.assertIn("byte budget", decision["rejected"]["4"])
                self.assertFalse(any("predictions" in c for c in decision["candidates"]))
                # Fetch once per hash: every candidate file once for the two real candidates, the
                # non-candidate artifact rejected after its first missing file, the oversize one
                # refused on its first declared length.
                self.assertEqual(sorted(requested[1]), sorted(models.D3_CANDIDATE_FILES))
                self.assertEqual(sorted(requested[2]), sorted(models.D3_CANDIDATE_FILES))
                self.assertLess(len(requested[3]), len(models.D3_CANDIDATE_FILES))
                self.assertEqual(len(requested[4]), 1)
                self.assertEqual(validate.call_count, 2)
                store = vdir / "base-tier/store"
                self.assertTrue(
                    (store / zils.checkpoint_hash(strong) / "checkpoint/model.json").exists()
                )
                self.assertFalse(
                    (store / zils.checkpoint_hash(adapter)).exists()
                    and (store / zils.checkpoint_hash(adapter) / "checkpoint").exists()
                )
                runs = [json.loads(line) for line in log.read_text().splitlines()]
                # Two draws, two splits: the incumbent and each candidate once per split.
                self.assertEqual(len(runs), 12)
                self.assertEqual(sum(r["kind"] == "base" for r in runs), 4)
                sidecar = json.loads((vdir / "base-tier/decision.json").read_text())
                self.assertEqual(sidecar["round_id"], "a" * 32)
                frozen = validator.attach_base_tier(config, vdir, work, {"miners": []})
                self.assertEqual(frozen["base_tier"]["pending_winner"]["uid"], 1)
                self.assertTrue((work / "base-tier.json").exists())

                # Next round: the strong hash again (no fetch, no scoring), a new hash from uid 2.
                sources[2] = later
                work, current, registries = new_round("b" * 32)
                announce(current, registries, 1, strong)
                announce(current, registries, 2, later)
                before = {uid: len(requests) for uid, requests in requested.items()}
                decision = validator.base_tier_epoch(
                    config, vdir, work, registries["candidate"], args, audited
                )
                self.assertEqual(len(requested[1]), before[1])
                self.assertEqual(len(requested[2]) - before[2], len(models.D3_CANDIDATE_FILES))
                self.assertEqual(validate.call_count, 3)
                runs = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(len(runs), 16)
                self.assertEqual([r["accuracy"] for r in runs[12:]], [0.5] * 4)
                self.assertEqual(decision["round_id"], "b" * 32)
                self.assertEqual(decision["pending_winner"]["uid"], 1)
                self.assertEqual({c["uid"] for c in decision["candidates"]}, {1, 2})
                self.assertEqual(decision["rejected"], {})
                self.assertEqual(
                    json.loads((vdir / "base-tier/decision.json").read_text())["round_id"], "b" * 32
                )

    def test_candidate_claims_are_the_same_signed_claim_with_a_kind(self):
        from bittensor_wallet import Keypair

        key = Keypair.create_from_seed("0x" + "07" * 32)
        claim = {
            "round_id": "a" * 32,
            "uid": 1,
            "hotkey": key.ss58_address,
            "sha256": "b" * 64,
            "endpoint": "http://127.0.0.1:8901",
        }

        def signed(c):
            return {"claim": c, "signature": key.sign(wire.canonical(c)).hex()}

        members = {1: key.ss58_address}
        with self.assertRaisesRegex(ValueError, "claim fields"):
            wire.register(signed(claim), "a" * 32, members, {}, kind="candidate")
        with self.assertRaisesRegex(ValueError, "claim fields"):
            wire.register(signed({**claim, "kind": "candidate"}), "a" * 32, members, {})
        with self.assertRaisesRegex(ValueError, "not a base candidate"):
            wire.register(
                signed({**claim, "kind": "adapter"}), "a" * 32, members, {}, kind="candidate"
            )
        registry = {}
        wire.register(
            signed({**claim, "kind": "candidate"}),
            "a" * 32,
            members,
            registry,
            job_sha256="c" * 64,
            kind="candidate",
        )
        self.assertEqual(registry[1]["claim"]["kind"], "candidate")


if __name__ == "__main__":
    unittest.main()
