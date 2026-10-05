"""Real localhost/signature tests; process test substitutes only model inference."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("bittensor_wallet"), "use .venv-kev for network tests"
)
class RehearsalTest(unittest.TestCase):
    def test_signed_discovery_and_download(self):
        from http.server import BaseHTTPRequestHandler

        from bittensor_wallet import Keypair

        import zils
        from zils import protocol as r

        keys = [Keypair.create_from_seed("0x" + (bytes([n]) * 32).hex()) for n in (1, 2)]
        members = {i + 1: k.ss58_address for i, k in enumerate(keys)}
        round_id = "a" * 32
        claim = {
            "round_id": round_id,
            "uid": 1,
            "hotkey": members[1],
            "sha256": "b" * 64,
            "endpoint": "http://127.0.0.1:12345",
        }

        def signed(c, key=keys[0]):
            return {"claim": c, "signature": key.sign(r.canonical(c)).hex()}

        registry = {}
        self.assertEqual(r.register(signed(claim), round_id, members, registry), claim)
        r.register(signed(claim), round_id, members, registry)  # retry is idempotent
        self.assertEqual(len(registry), 1)
        for changes in (
            {"uid": 2},
            {"uid": True},
            {"round_id": "c" * 32},
            {"endpoint": "http://example.com:80"},
            {"endpoint": "http://127.0.0.1:80@evil.test"},
            {"endpoint": "http://127.0.0.1:65536"},
            {"sha256": "invalid"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                r.register(signed({**claim, **changes}), round_id, members, {})
        tampered = signed(claim)
        tampered["claim"] = {**claim, "sha256": "c" * 64}
        with self.assertRaises(ValueError):
            r.register(tampered, round_id, members, {})
        with self.assertRaises(ValueError):
            r.register(signed({**claim, "sha256": "c" * 64}), round_id, members, registry)
        with self.assertRaises(ValueError):
            r.register(
                signed({**claim, "uid": 2, "hotkey": members[2]}, keys[1]),
                round_id,
                members,
                registry,
            )

        class Files(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                payload = b"test artifact"
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        with tempfile.TemporaryDirectory() as tmp, r.local_server(Files) as endpoint:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for name in zils.ARTIFACT_FILES:
                (source / name).write_bytes(b"test artifact")
            item = {**claim, "endpoint": endpoint, "sha256": zils.checkpoint_hash(source)}
            result = r.fetch_checkpoint(item, root / "download")
            self.assertEqual(result["bytes"], 3 * len(b"test artifact"))
            self.assertEqual(zils.checkpoint_hash(root / "download"), item["sha256"])
            with self.assertRaisesRegex(ValueError, "hash"):
                r.fetch_checkpoint({**item, "sha256": "c" * 64}, root / "corrupt")

    def test_two_miner_processes_one_validator(self):
        import zils

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = []
            for quality in ("good", "bad"):
                path = root / quality
                path.mkdir()
                checkpoints.append(str(path))
                for name in zils.ARTIFACT_FILES:
                    (path / name).write_text(quality)
            cases = root / "cases.jsonl"
            cases.write_text(
                json.dumps(
                    {
                        "id": "private",
                        "family": "fixture",
                        "state": "A private question.",
                        "question": {"type": "noul", "instructions": "Is it true?"},
                        "label": "true",
                    }
                )
                + "\n"
            )
            worker = root / "fixture-python"
            worker.write_text(
                f"#!{sys.executable}\n"
                + """import json, sys, subprocess
from pathlib import Path
checkpoint = Path(sys.argv[sys.argv.index('--checkpoint') + 1])
assert Path(sys.argv[0]).name == 'python', 'launcher resolved the interpreter symlink'
good = (checkpoint / 'head.pt').read_text() == 'good'
if good:
    marker = str(Path(sys.argv[0]).parent / 'orphan-marker')
    subprocess.Popen([sys.executable, '-c', f"import time; from pathlib import Path; time.sleep(1.5); Path({marker!r}).write_text('orphan survived')"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
requests = json.load(sys.stdin)
assert all(set(row) == {'id', 'state', 'question'} for row in requests)
predictions = [{'id': row['id'], 'probabilities': {'false': .1 if good else .9, 'true': .9 if good else .1}, 'elapsed_ms': 1} for row in requests]
print(json.dumps({'predictions': predictions, 'runtime': {'fixture': True}}))
"""
            )
            worker.chmod(0o700)
            environment = root / "fixture-env"
            environment.mkdir()
            interpreter = environment / "python"
            interpreter.symlink_to(worker)
            output = root / "run"
            command = [
                sys.executable,
                "-m",
                "scripts.rehearsal",
                "run",
                "--checkpoints",
                *checkpoints,
                "--cases",
                str(cases),
                "--runner-python",
                str(interpreter),
                "--out",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "validator/report.json").read_text())
            self.assertEqual(report["mode"], "local-network-rehearsal")
            self.assertEqual(report["weights"], {"1": 1.0})
            self.assertEqual(report["identity_source"], "local-allowlist")
            self.assertFalse(report["chain_write"])
            self.assertEqual(len(report["participants"]), 2)
            self.assertTrue(all(p["signature_verified"] for p in report["participants"]))
            self.assertTrue(all(m["status"] == "evaluated" for m in report["miners"]))
            processes = json.loads((output / "processes.json").read_text())
            self.assertEqual({p["role"] for p in processes}, {"miner-1", "miner-2", "validator"})
            self.assertEqual(len({p["pid"] for p in processes}), 3)
            downloaded = json.loads((output / "validator/submissions.json").read_text())
            self.assertTrue(
                all(str(output / "validator/downloads") in p["checkpoint"] for p in downloaded)
            )
            for uid, source in enumerate(checkpoints, 1):
                self.assertEqual(
                    zils.checkpoint_hash(source),
                    zils.checkpoint_hash(output / f"validator/downloads/{uid}"),
                )
            before = (output / "validator/report.json").read_bytes()
            again = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual((output / "validator/report.json").read_bytes(), before)
            time.sleep(2)
            self.assertFalse(
                (environment / "orphan-marker").exists(),
                "a worker descendant survived launcher cleanup",
            )


if __name__ == "__main__":
    unittest.main()
