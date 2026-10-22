"""The validator's own base-tier round: read commitments at the preflight block, pull each new hash
from Hugging Face once, score once per draw, decide, confirm."""

import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import unquote, urlsplit

import zils
from tests.test_base_exam import inputs, sealed
from zils import base_round, commitments, models, protocol as wire, testnet, validator

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
    kind = row["question"]["type"]
    criteria = row["question"]["criteria"]
    keys = sorted(criteria) if kind == "choice" else (["false", "true"] if kind == "noul" else [str(i) for i in range(len(criteria))])
    index = int(row["state"]["text"].split("-")[-1])
    label = ({"choice": ["a", "b"], "noul": ["false", "true"], "score": ["0", "1", "3", "4"]}[kind])[index % (4 if kind == "score" else 2)]
    chosen = label if rng.random() < accuracy else rng.choice([k for k in keys if k != label])
    p = {k: 0.3 / (len(keys)-1) for k in keys}
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


def fake_hub(repos, requested, *, huge=()):
    """Hugging Face's two public routes over {repo: {revision: directory}}: revision info with file
    sizes and LFS digests, and resolve, which redirects once like the real CDN hop. A repo in `huge`
    lists its model.json at the whole candidate byte budget."""

    class Hub(wire.Handler):
        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            info = re.fullmatch(r"/api/models/([^/]+/[^/]+)/revision/([a-f0-9]{40})", path)
            if info:
                source = repos.get(info[1], {}).get(info[2])
                if source is None:
                    self.reply(404, {"error": "Revision Not Found"})
                    return
                siblings = []
                for item in sorted(source.iterdir()):
                    entry = {"rfilename": item.name, "size": item.stat().st_size}
                    if info[1] in huge and item.name == "model.json":
                        entry["size"] = models.MAX_CANDIDATE_BYTES
                    if item.suffix == ".safetensors":
                        entry["lfs"] = {"sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
                    siblings.append(entry)
                self.reply(200, {"id": info[1], "sha": info[2], "siblings": siblings})
                return
            resolve = re.fullmatch(r"/([^/]+/[^/]+)/resolve/([a-f0-9]{40})/(.+)", path)
            if resolve:
                self.send_response(302)
                self.send_header("Location", "/cdn" + path)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            cdn = re.fullmatch(r"/cdn/([^/]+/[^/]+)/resolve/([a-f0-9]{40})/(.+)", path)
            source = repos.get(cdn[1], {}).get(cdn[2]) if cdn else None
            if source is None or not (source / cdn[3]).is_file():
                self.reply(404, {"error": "Entry Not Found"})
                return
            requested.append((cdn[1], cdn[3]))
            data = (source / cdn[3]).read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Hub


def fake_chain(state):
    """The storage reads preflight makes, at one pinned block."""

    class Chain:
        network, endpoint, block = "test", testnet.ENDPOINT, 500

        def query(self, item, params, *, block=None):
            assert block == self.block
            value = state[item[1]]
            return value.get(params[1]) if isinstance(value, dict) else value

    return Chain()


def raw_record(block, text):
    data = text.encode()
    fields = [
        {f"Raw{len(data[i : i + 128])}": "0x" + data[i : i + 128].hex()}
        for i in range(0, len(data), 128)
    ]
    return {"block": block, "deposit": 0, "info": {"fields": fields}}


def revision(name):
    return hashlib.sha1(name.encode()).hexdigest()


@unittest.skipUnless(
    importlib.util.find_spec("bittensor_wallet") and importlib.util.find_spec("kev"),
    "use .venv-kev",
)
class BaseTierIntakeTest(unittest.TestCase):
    def test_commitments_are_pulled_once_scored_once_and_copies_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vdir = root / "validator"
            (vdir / "draws").mkdir(parents=True)
            sealed(vdir / "draws/first", "first")
            sealed(vdir / "draws/second", "second")
            (vdir / "pilot.json").write_text(json.dumps({"fixture": True}))
            (vdir / "history.json").write_text(json.dumps(inputs()[2]))
            draws = [base_round.draw(vdir / "draws/first"), base_round.draw(vdir / "draws/second")]
            (vdir / "reference").mkdir()
            models.write_metadata(vdir / "reference", kind="base", model=models.D3)
            incumbent = zils.checkpoint_hash(vdir / "reference")
            strong = candidate(root / "strong", 0.85)
            weak = candidate(root / "weak", 0.40)
            later = candidate(root / "later", 0.99)
            big = candidate(root / "big", 0.90)
            adapter = root / "adapter"
            adapter.mkdir()
            for name in models.TEXT_ADAPTER_FILES:
                (adapter / name).write_bytes(b"lora")
            models.write_metadata(adapter, model=models.D3)
            repos = {
                "alice/d3": {revision("strong"): strong, revision("later"): later},
                "bob/d3": {revision("weak"): weak},
                "carol/d3": {revision("adapter"): adapter},
                "dave/d3": {revision("big"): big},
            }
            requested = []

            def commit(source, repo, rev):
                return commitments.format_commitment(zils.checkpoint_hash(source), rev, repo)

            hotkeys = {uid: f"miner-{uid}" for uid in range(1, 9)}
            records = {
                # uid 1 commits first; uid 3 replays the same commitment two blocks later.
                hotkeys[1]: raw_record(400, commit(strong, "alice/d3", revision("strong"))),
                hotkeys[2]: raw_record(401, commit(weak, "bob/d3", revision("weak"))),
                hotkeys[3]: raw_record(402, commit(strong, "alice/d3", revision("strong"))),
                hotkeys[4]: raw_record(
                    403,
                    commit(weak, "bob/d3", revision("missing")).replace(
                        zils.checkpoint_hash(weak), "e" * 64
                    ),
                ),
                hotkeys[5]: raw_record(404, commit(big, "dave/d3", revision("big"))),
                hotkeys[6]: raw_record(405, "my model is on hugging face"),
                hotkeys[7]: raw_record(406, commit(adapter, "carol/d3", revision("adapter"))),
                hotkeys[8]: raw_record(
                    407, commitments.format_commitment(incumbent, revision("x"), "eve/d3")
                ),
            }
            state = {
                "NetworksAdded": True,
                "Uids": {"validator": 0, **{h: uid for uid, h in hotkeys.items()}},
                "Keys": {0: "validator", **hotkeys},
                "ValidatorPermit": [True],
                "SubnetOwnerHotkey": "validator",
                "CommitRevealWeightsEnabled": True,
                "MinAllowedWeights": 1,
                "MaxWeightsLimit": 65535,
                "WeightsSetRateLimit": 100,
                "WeightsVersionKey": 0,
                "LastUpdate": [100],
                "CommitmentOf": records,
            }
            runner = root / "runner-python"
            runner.write_text(f"#!{sys.executable}\n{RUNNER}")
            runner.chmod(0o700)
            log = root / "runner.log"
            log.touch()
            args = SimpleNamespace(device="cpu", runtime_python=sys.executable, poll=0.1)
            validate = Mock()
            config = {
                "validator": "http://127.0.0.1:1",
                "validator_hotkey": "validator",
                "chain": {"network": "test", "netuid": 579},
                "model": models.spec(models.D3),
                "base_revision": models.spec(models.D3)["base_revision"],
                "members": {str(uid): {"hotkey": h} for uid, h in hotkeys.items()},
                "base_tier": {
                    "share": 0.7,
                    "incumbent_sha256": incumbent,
                    "draws": draws,
                    "policy": dict(base_round.DEFAULT_POLICY),
                },
                "pool_id": "d3-mini",
                "family_history": "history.json",
                "pilot_report": "pilot.json",
                "base_round_dir": "draws",
                "base_reference": "reference",
            }
            with (
                wire.local_server(fake_hub(repos, requested, huge={"dave/d3"})) as hub,
                patch.dict(
                    os.environ,
                    {"ZILS_D3_RUNTIME_PYTHON": str(runner), "ZILS_TEST_RUNNER_LOG": str(log)},
                ),
                patch("zils.base_pilot.verify_qualification"),
                patch("zils.d3.validate_candidate", validate),
                patch("zils.commitments.HF_ENDPOINT", hub),
            ):
                audited = validator.base_draws(config, vdir)
                self.assertEqual([item for item, _ in audited], draws)
                with self.assertRaisesRegex(ValueError, "chain commitments"):
                    validator.base_draws({k: v for k, v in config.items() if k != "chain"}, vdir)
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
                self.assertIsNone(validator.base_draws({"members": {}}, vdir))

                clock = [1000]

                def epoch(rid):
                    snapshot = testnet.preflight(config, fake_chain(state))
                    work = vdir / "state/rounds" / rid
                    work.mkdir(parents=True)
                    wire.write_json(work / "chain-snapshot.json", snapshot)
                    registry = commitments.candidates(
                        json.loads((work / "chain-snapshot.json").read_text()), incumbent
                    )
                    owned = {u: {**c, "hotkey": hotkeys[u]} for u, c in registry.items()}
                    for _ in range(2):
                        with patch("zils.validator.time.time", return_value=clock[0]):
                            decision = validator.base_tier_epoch(
                                config, vdir, work, owned, args, audited
                            )
                        clock[0] += 43200
                    return registry, decision

                registry, decision = epoch("a" * 32)
                self.assertEqual(
                    registry[1],
                    {
                        "sha256": zils.checkpoint_hash(strong),
                        "revision": revision("strong"),
                        "repo": "alice/d3",
                        "block": 400,
                    },
                )
                self.assertIn("committed first by uid 1 at block 400", registry[3]["error"])
                self.assertIn("incumbent", registry[8]["error"])
                self.assertIn("not zils1", registry[6]["error"])
                self.assertEqual(decision["pending_winner"]["uid"], 1)
                self.assertEqual(decision["confirmed_winner"]["uid"], 1)
                self.assertEqual(decision["incumbent"]["uid"], validator.INCUMBENT_UID)
                self.assertEqual(
                    {c["uid"]: c["eligible"] for c in decision["candidates"]},
                    {1: True, 2: False, 4: False, 5: False, 7: False},
                )
                self.assertEqual(set(decision["rejected"]), {"3", "4", "5", "6", "7", "8"})
                self.assertIn("HTTP Error 404", decision["rejected"]["4"])
                self.assertIn("byte budget", decision["rejected"]["5"])
                self.assertIn("no model-00001", decision["rejected"]["7"])
                # Only the candidate files, once per hash; nothing from the refused commitments.
                self.assertEqual(
                    sorted(requested),
                    sorted(
                        [("alice/d3", n) for n in models.D3_CANDIDATE_FILES]
                        + [("bob/d3", n) for n in models.D3_CANDIDATE_FILES]
                    ),
                )
                self.assertEqual(validate.call_count, 2)
                store = vdir / "base-tier/store"
                self.assertTrue(
                    (store / zils.checkpoint_hash(strong) / "checkpoint/model.json").exists()
                )
                self.assertEqual(
                    [p.name for p in (store / zils.checkpoint_hash(big)).iterdir()], []
                )
                runs = [json.loads(line) for line in log.read_text().splitlines()]
                # Two draws, two splits: the incumbent and each candidate once per split.
                self.assertEqual(len(runs), 12)
                sidecar = json.loads((vdir / "base-tier/decision.json").read_text())
                self.assertEqual(sidecar["round_id"], "a" * 32)

                original_incumbent_reports = {
                    evidence["path"]: Path(evidence["path"]).read_bytes()
                    for evidence in decision["incumbent"]["evaluation"].values()
                }
                # Next round: uid 1 unchanged (no download, no scoring), uid 2 commits a new
                # revision, and uid 3's copy is still refused.
                records[hotkeys[2]] = raw_record(450, commit(later, "alice/d3", revision("later")))
                before = len(requested)
                registry, decision = epoch("b" * 32)
                self.assertIn(2, {c["uid"] for c in decision["confirmed_candidates"]})
                for path, contents in original_incumbent_reports.items():
                    self.assertEqual(Path(path).read_bytes(), contents)
                self.assertEqual(
                    requested[before:], [("alice/d3", n) for n in models.D3_CANDIDATE_FILES]
                )
                self.assertEqual(validate.call_count, 3)
                runs = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(len(runs), 18)
                self.assertEqual(sorted(r["accuracy"] for r in runs[12:]), [0.55] * 2 + [0.99] * 4)
                self.assertEqual(decision["round_id"], "b" * 32)
                self.assertEqual(decision["pending_winner"]["uid"], 2)
                self.assertEqual({c["uid"] for c in decision["candidates"]}, {1, 2, 4, 5, 7})
                self.assertIn("3", decision["rejected"])

                # A third identical round costs nothing at all.
                before = len(requested)
                epoch("c" * 32)
                self.assertEqual(len(requested), before)
                self.assertEqual(len(log.read_text().splitlines()), 18)

    def test_commitment_format_and_the_earliest_block_rule(self):
        sha, rev = "a" * 64, "b" * 40
        text = commitments.format_commitment(sha, rev, "alice/d3-candidate")
        self.assertEqual(text, f"zils1:{sha}:{rev}:alice/d3-candidate")
        self.assertEqual(
            commitments.parse_commitment(text),
            {"sha256": sha, "revision": rev, "repo": "alice/d3-candidate"},
        )
        longest = commitments.format_commitment(sha, rev, "a" * 96 + "/" + "b" * 96)
        self.assertLessEqual(len(longest.encode()), 3 * testnet.COMMITMENT_FIELD)
        for bad in (
            f"zils1:{sha}:main:alice/d3",
            f"zils1:{sha}:{rev[:12]}:alice/d3",
            f"zils1:{sha}:{rev}:alice",
            f"zils1:{sha}:{rev}:alice/../d3",
            f"zils1:{sha}:{rev}:alice/d3:extra",
            f"zils2:{sha}:{rev}:alice/d3",
            f"zils1:{sha.upper()}:{rev}:alice/d3",
        ):
            with self.assertRaises(ValueError, msg=bad):
                commitments.parse_commitment(bad)
        snapshot = {
            "commitments": {
                "1": {"block": 10, "data": text},
                "2": {"block": 9, "data": commitments.format_commitment(sha, rev, "copy/d3")},
                "3": {"block": 20, "data": commitments.format_commitment("c" * 64, rev, "x/y")},
                "4": {"block": 20, "data": commitments.format_commitment("c" * 64, rev, "z/y")},
                "5": {"block": 21, "data": commitments.format_commitment("c" * 64, rev, "w/y")},
            }
        }
        registry = commitments.candidates(snapshot, "f" * 64)
        self.assertEqual(registry[2]["repo"], "copy/d3")
        self.assertIn("first by uid 2 at block 9", registry[1]["error"])
        # A same-block tie belongs to nobody, and a later copy still loses to that block.
        self.assertIn("several hotkeys in block 20", registry[3]["error"])
        self.assertIn("several hotkeys in block 20", registry[4]["error"])
        self.assertIn("first by uid 3 at block 20", registry[5]["error"])
        self.assertEqual(testnet.commitment_text(raw_record(1, longest)), longest)
        self.assertIsNone(testnet.commitment_text({"info": {"fields": [{"Sha256": "0x00"}]}}))
        self.assertIsNone(testnet.commitment_text(None))


if __name__ == "__main__":
    unittest.main()
