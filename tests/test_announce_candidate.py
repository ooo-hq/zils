"""`zils.miner announce-candidate` against a local validator with the base-tier intake."""

import contextlib
import importlib.util
import io
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import zils
from tests.test_base_round import make_cases, sealed
from tests.test_base_tier_intake import RUNNER, candidate
from zils import base_round, miner, models, protocol as wire, testnet, validator
from zils.runtime import signed, signing_key

VALIDATOR_SEED = "0x" + "aa" * 32


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def fake_chain(validator_hotkey, miner_hotkey):
    state = {
        "NetworksAdded": True,
        "SubnetworkN": 2,
        "Uids": {validator_hotkey: 0, miner_hotkey: 1},
        "Keys": {0: validator_hotkey, 1: miner_hotkey},
        "ValidatorPermit": [True, False],
        "SubnetOwnerHotkey": validator_hotkey,
        "CommitRevealWeightsEnabled": True,
        "MinAllowedWeights": 1,
        "MaxWeightsLimit": 65535,
        "WeightsSetRateLimit": 100,
        "WeightsVersionKey": 0,
        "LastUpdate": [100, 0],
    }

    class Chain:
        network, endpoint, block = "test", testnet.ENDPOINT, 250

        def query(self, item, params, *, block=None):
            value = state[item[1]]
            return value.get(params[1]) if isinstance(value, dict) else value

    @contextlib.contextmanager
    def connect(config):
        testnet.check_config(config)
        yield Chain()

    return connect


def local_validator(config, vdir, members, current, registries, condition):
    """The validator's public API over the real `accept`, driven round by round from the test."""
    key = signing_key(config)
    rounds = vdir / "state/rounds"

    class API(wire.Handler):
        def send(self, code, payload):
            self.reply(code, signed(payload, key))

        def do_GET(self):
            with condition:
                if self.path == "/round":
                    self.send(200, current or {"kind": "idle"})
                    return
                match = re.fullmatch(r"/results/([a-f0-9]{32})", self.path)
                path = rounds / match[1] / "result.json" if match else None
                if path is not None and path.exists():
                    self.reply(200, json.loads(path.read_text()))
                    return
                self.send(404, {"error": "result not ready"})

        def do_POST(self):
            try:
                message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with condition:
                    code, payload = validator.accept(
                        config,
                        current,
                        members,
                        registries,
                        rounds,
                        message,
                        self.client_address[0],
                        False,
                    )
                    self.send(code, payload)
                    condition.notify_all()
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.send(400, {"error": str(error)[:200]})

    return API


@unittest.skipUnless(
    importlib.util.find_spec("bittensor_wallet") and importlib.util.find_spec("kev"),
    "use .venv-kev",
)
class AnnounceCandidateTest(unittest.TestCase):
    def setUp(self):
        from bittensor.wallet import Wallet
        from bittensor_wallet import Keypair

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.root = root
        self.vdir = root / "validator"
        (self.vdir / "draws").mkdir(parents=True)
        calibration = make_cases("cal", 60)
        sealed(self.vdir / "draws/first", calibration, make_cases("first", 60))
        sealed(self.vdir / "draws/second", calibration, make_cases("second", 60))
        self.draws = [
            base_round.draw(self.vdir / "draws/first"),
            base_round.draw(self.vdir / "draws/second"),
        ]
        (self.vdir / "reference").mkdir()
        models.write_metadata(self.vdir / "reference", kind="base", model=models.D3)
        self.incumbent = zils.checkpoint_hash(self.vdir / "reference")
        self.candidate = candidate(root / "strong", 0.85)
        self.wallet = {"path": str(root / "wallets"), "name": "fixture", "hotkey": "miner"}
        self.miner_key = (
            Wallet(**self.wallet).regenerate_hotkey(seed=bytes([7]) * 32, suppress=True).hotkey
        )
        self.validator_hotkey = Keypair.create_from_seed(VALIDATOR_SEED).ss58_address
        self.port = free_port()
        self.members = {1: self.miner_key.ss58_address}
        self.config = {
            "validator": "http://127.0.0.1:1",
            "seed": VALIDATOR_SEED,
            "model": models.spec(models.D3),
            "base_revision": models.spec(models.D3)["base_revision"],
            "members": {"1": {"hotkey": self.miner_key.ss58_address, "port": self.port}},
            "base_tier": {
                "share": 0.7,
                "incumbent_sha256": self.incumbent,
                "draws": self.draws,
                "policy": {"bootstrap_resamples": 200},
            },
            "base_round_dir": "draws",
            "base_reference": "reference",
        }
        runner = root / "runner-python"
        runner.write_text(f"#!{sys.executable}\n{RUNNER}")
        runner.chmod(0o700)
        self.log = root / "runner.log"
        self.log.touch()
        self.condition = threading.Condition()
        self.current, self.registries = {}, {"adapter": {}, "candidate": {}}
        self.api = local_validator(
            self.config, self.vdir, self.members, self.current, self.registries, self.condition
        )
        self.enterContext(
            patch.dict(
                os.environ,
                {"ZILS_D3_RUNTIME_PYTHON": str(runner), "ZILS_TEST_RUNNER_LOG": str(self.log)},
            )
        )
        self.enterContext(patch("zils.d3.validate_candidate", Mock()))
        self.enterContext(
            patch(
                "zils.testnet.connect",
                fake_chain(self.validator_hotkey, self.miner_key.ss58_address),
            )
        )

    def args(self, url, **changes):
        return SimpleNamespace(
            validator=url,
            validator_hotkey=self.validator_hotkey,
            wallet_name=self.wallet["name"],
            wallet_hotkey=self.wallet["hotkey"],
            wallet_path=self.wallet["path"],
            candidate=str(self.candidate),
            port=self.port,
            advertise_host=None,
            netuid=579,
            rounds=2,
            poll=0.05,
            **changes,
        )

    def argv(self, url, *extra):
        return [
            "announce-candidate",
            "--validator",
            url,
            "--validator-hotkey",
            self.validator_hotkey,
            "--wallet-name",
            self.wallet["name"],
            "--wallet-hotkey",
            self.wallet["hotkey"],
            "--wallet-path",
            self.wallet["path"],
            "--candidate",
            str(self.candidate),
            "--port",
            str(self.port),
            "--poll",
            "0.05",
            "--rounds",
            "1",
            *extra,
        ]

    def open_round(self, rid, *, intake=True):
        work = self.vdir / "state/rounds" / rid
        (work / "base-submissions").mkdir(parents=True)
        with self.condition:
            for registry in self.registries.values():
                registry.clear()
            self.current.clear()
            self.current.update(kind="round", round_id=rid, status="collecting")
            if intake:
                self.current["base_tier"] = {
                    "incumbent_sha256": self.incumbent,
                    "draws": self.draws,
                }
        return work

    def wait_for_announcement(self):
        with self.condition:
            self.assertTrue(
                self.condition.wait_for(lambda: 1 in self.registries["candidate"], timeout=20)
            )
            return dict(self.registries["candidate"])

    def close_round(self, work, audited):
        args = SimpleNamespace(device="cpu", runtime_python=sys.executable, poll=0.1)
        with self.condition:
            self.current["status"] = "evaluating"
            registry = dict(self.registries["candidate"])
        validator.base_tier_epoch(self.config, self.vdir, work, registry, args, audited)
        report = validator.attach_base_tier(
            self.config, self.vdir, work, {"miners": [], "weights": {}}
        )
        public = {
            "kind": "result",
            "round_id": work.name,
            "weights": {"1": 1.0},
            "miners": [],
            "base_tier": report["base_tier"],
        }
        wire.write_json(work / "result.json", signed(public, signing_key(self.config)))
        with self.condition:
            self.current["status"] = "complete"
        return report["base_tier"]

    def test_announce_fetch_score_and_reannounce_next_round(self):
        audited = validator.base_draws(self.config, self.vdir)
        fetch = Mock(wraps=wire.fetch_checkpoint)
        output = io.StringIO()
        with (
            wire.local_server(self.api) as url,
            patch("zils.validator.wire.fetch_checkpoint", fetch),
            contextlib.redirect_stdout(output),
        ):
            self.config["validator"] = url
            work = self.open_round("a" * 32)
            outcome = {}
            worker = threading.Thread(
                target=lambda: outcome.update(miner.announce_candidate(self.args(url))),
                daemon=True,
            )
            worker.start()
            registry = self.wait_for_announcement()
            claim = registry[1]["claim"]
            self.assertEqual(claim["kind"], "candidate")
            self.assertEqual(claim["sha256"], zils.checkpoint_hash(self.candidate))
            self.assertEqual(claim["endpoint"], f"http://127.0.0.1:{self.port}")
            self.assertEqual(claim["hotkey"], self.miner_key.ss58_address)
            self.assertTrue((work / "base-submissions/1.json").exists())
            decision = self.close_round(work, audited)
            self.assertEqual(decision["pending_winner"]["uid"], 1)
            self.assertEqual(decision["rejected"], {})
            self.assertEqual(fetch.call_count, 1)
            store = self.vdir / "base-tier/store" / claim["sha256"] / "checkpoint"
            self.assertTrue((store / "model.json").exists())
            runs = len(self.log.read_text().splitlines())
            self.assertEqual(runs, 8)  # incumbent and candidate, two draws, two splits
            # The miner reads its result, then announces the same hash in the next round.
            deadline = time.monotonic() + 20
            while "a" * 32 not in output.getvalue() or "complete" not in output.getvalue():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.05)
            work = self.open_round("b" * 32)
            self.assertEqual(self.wait_for_announcement()[1]["claim"]["sha256"], claim["sha256"])
            decision = self.close_round(work, audited)
            worker.join(timeout=20)
            self.assertFalse(worker.is_alive())
        self.assertEqual(fetch.call_count, 1)  # a repeated hash is neither fetched...
        self.assertEqual(len(self.log.read_text().splitlines()), 8)  # ...nor rescored
        self.assertEqual(decision["round_id"], "b" * 32)
        self.assertEqual(outcome["rounds"], ["a" * 32, "b" * 32])
        self.assertEqual(outcome["uid"], 1)
        printed = output.getvalue()
        self.assertIn("announced", printed)
        self.assertIn("pending winner uid 1", printed)
        self.assertIn("eligible True", printed)
        self.assertNotIn("predictions", printed)

    def refused(self, *argv):
        errors = io.StringIO()
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(errors),
            self.assertRaises(SystemExit) as stopped,
        ):
            miner.main(self.argv(*argv))
        self.assertEqual(stopped.exception.code, 1)
        return errors.getvalue()

    def test_refuses_without_intake_bad_candidate_or_private_endpoint(self):
        from bittensor.wallet import Wallet

        with wire.local_server(self.api) as url:
            self.config["validator"] = url
            self.open_round("c" * 32, intake=False)
            self.assertIn("no base-tier intake", self.refused(url))
            self.assertEqual(self.registries["candidate"], {})
            with patch("zils.d3.validate_candidate", side_effect=ValueError("shard differs")):
                self.assertIn("shard differs", self.refused(url))
            Wallet(**{**self.wallet, "hotkey": "stranger"}).regenerate_hotkey(
                seed=bytes([9]) * 32, suppress=True
            )
            self.assertIn(
                "not registered on testnet subnet 579",
                self.refused(url, "--wallet-hotkey", "stranger"),
            )
        # A public validator needs a globally routable artifact endpoint; the check runs offline.
        with self.assertRaisesRegex(ValueError, "globally routable"):
            miner.own_endpoint("http://8.8.8.8:8900", 8901, "192.168.1.20")
        with self.assertRaisesRegex(ValueError, "globally routable"):
            miner.own_endpoint("http://8.8.8.8:8900", 8901, "127.0.0.1")
        self.assertEqual(
            miner.own_endpoint("http://8.8.8.8:8900", 8901, "8.8.4.4"), "http://8.8.4.4:8901"
        )
        self.assertEqual(
            miner.own_endpoint("http://127.0.0.1:8900", 8901, "127.0.0.1"), "http://127.0.0.1:8901"
        )
        with self.assertRaisesRegex(ValueError, "private"):
            miner.own_endpoint("http://127.0.0.1:8900", 8901, "8.8.4.4")
        self.assertIn(
            "globally routable",
            self.refused("http://8.8.8.8:8900", "--advertise-host", "10.0.0.5"),
        )

    def doctor(self, url, *extra):
        argv = [
            "doctor",
            "--validator",
            url,
            "--validator-hotkey",
            self.validator_hotkey,
            "--wallet-name",
            self.wallet["name"],
            "--wallet-hotkey",
            self.wallet["hotkey"],
            "--wallet-path",
            self.wallet["path"],
            "--port",
            str(self.port),
            *extra,
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            try:
                miner.main(argv)
                code = 0
            except SystemExit as stopped:
                code = stopped.code
        return code, output.getvalue()

    def test_doctor_stops_at_the_first_failing_step_with_its_fix(self):
        from bittensor.wallet import Wallet

        with wire.local_server(self.api) as url:
            self.config["validator"] = url
            self.open_round("d" * 32)
            code, out = self.doctor(url, "--candidate", str(self.candidate))
            self.assertEqual(code, 0, out)
            self.assertEqual(out.count("ok  "), 5)
            self.assertIn("ready: uid 1", out)
            self.assertIn(f"artifacts at http://127.0.0.1:{self.port}", out)
            self.assertIn(
                f"announce-candidate \\\n  --validator {url} --validator-hotkey "
                f"{self.validator_hotkey}",
                out,
            )
            self.assertIn(f"--candidate {self.candidate} --port {self.port}", out)

            code, out = self.doctor(url)
            self.assertEqual(code, 1)
            self.assertEqual(out.count("ok  "), 3)
            self.assertIn("fix candidate: no --candidate", out)
            self.assertIn("python -m zils.d3 candidate --adapter", out)
            self.assertNotIn("ready", out)

            with patch("zils.d3.validate_candidate", side_effect=ValueError("shard differs")):
                code, out = self.doctor(url, "--candidate", str(self.candidate))
            self.assertIn("fix candidate: shard differs; merge again", out)

            code, out = self.doctor(url, "--candidate", str(self.vdir / "reference"))
            self.assertIn("fix validator: the candidate is the stock base", out)

            Wallet(**{**self.wallet, "hotkey": "stranger"}).regenerate_hotkey(
                seed=bytes([9]) * 32, suppress=True
            )
            code, out = self.doctor(url, "--wallet-hotkey", "stranger")
            self.assertEqual(out.count("ok  "), 2)
            self.assertIn("fix registration: hotkey", out)
            self.assertIn(
                f"btcli --network test --wallet {self.wallet['name']} --wallet-hotkey stranger"
                f" --wallet-path {self.wallet['path']} tx burned-register --netuid 579",
                out,
            )

            code, out = self.doctor(url, "--wallet-hotkey", "absent")
            self.assertEqual(out.count("ok  "), 1)
            self.assertIn("fix wallet hotkey: no hotkey file", out)
            self.assertIn("wallet create", out)

            code, out = self.doctor(url, "--validator-hotkey", self.miner_key.ss58_address)
            self.assertIn("fix registration: validator has no permit", out)
            self.assertIn("check --validator-hotkey", out)

            real = miner.importlib.import_module

            def without_kev(name, *rest):
                if name == "kev":
                    raise ImportError("No module named kev")
                return real(name, *rest)

            with patch("zils.miner.importlib.import_module", without_kev):
                code, out = self.doctor(url, "--candidate", str(self.candidate))
            self.assertEqual(out.count("ok  "), 0)
            self.assertIn("fix requirements: missing kev", out)
            self.assertIn("requirements/testnet.txt", out)

            self.open_round("e" * 32, intake=False)
            code, out = self.doctor(url, "--candidate", str(self.candidate))
            self.assertIn("fix validator: this validator has no base-tier intake", out)

        code, out = self.doctor("http://127.0.0.1:1", "--candidate", str(self.candidate))
        self.assertEqual(out.count("ok  "), 4)
        self.assertIn("fix validator: cannot reach http://127.0.0.1:1", out)
        code, out = self.doctor(
            "http://8.8.8.8:8900",
            "--candidate",
            str(self.candidate),
            "--advertise-host",
            "10.0.0.5",
        )
        self.assertIn("fix validator: endpoint must match the pinned public IPv4", out)
        self.assertIn("--advertise-host", out)


if __name__ == "__main__":
    unittest.main()
