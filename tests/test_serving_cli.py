"""`zils announce-serving` and `zils serve` against a fake chain; no GPU, no network."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from zils import miner, testnet

ENDPOINT = "http://8.8.8.8:8091"


class Chain:
    """Uids, Keys, Axons and ValidatorPermit, with serve_axon applied at the next block."""

    network, endpoint = "test", testnet.ENDPOINT

    def __init__(self, uids, permits=()):
        self.block, self.uids, self.axons, self.calls = 300, uids, {}, []
        self.permits = [hotkey in permits for hotkey in uids]

    @property
    def keys(self):
        return {uid: hotkey for hotkey, uid in self.uids.items()}

    def query(self, item, params, *, block=None):
        assert params[0] == 579 and block in (None, self.block)
        name = item[1]
        if name == "Uids":
            return self.uids.get(params[1])
        if name == "Keys":
            return self.keys.get(params[1])
        if name == "Axons":
            return self.axons.get(params[1])
        if name == "ValidatorPermit":
            return self.permits
        return None

    def submit_call(self, call, wallet, *, signer, **kwargs):
        from bittensor.result import ExtrinsicResult

        assert (call.module, call.function, signer) == ("SubtensorModule", "serve_axon", "hotkey")
        self.calls.append(call)
        self.block += 1
        self.axons[wallet.hotkey.ss58_address] = {
            "ip": call.params["ip"],
            "port": call.params["port"],
            "ip_type": 4,
        }
        return ExtrinsicResult(True)


@unittest.skipUnless(
    importlib.util.find_spec("bittensor") and importlib.util.find_spec("kev"), "use .venv-kev"
)
class ServingCliTest(unittest.TestCase):
    def setUp(self):
        from bittensor.wallet import Wallet

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.wallet = {"path": str(Path(tmp.name) / "wallets"), "name": "fixture", "hotkey": "miner"}

        def hotkey(name, seed):
            return Wallet(path=self.wallet["path"], name="fixture", hotkey=name).regenerate_hotkey(
                seed="0x" + (bytes([seed]) * 32).hex(),
                suppress=True,
            ).hotkey.ss58_address

        self.hotkey = hotkey("miner", 7)
        self.validator, self.stranger = hotkey("validator", 8), hotkey("stranger", 9)
        self.chain = Chain({self.hotkey: 0, self.validator: 1, self.stranger: 2}, {self.validator})

        @contextlib.contextmanager
        def connect(config, *, raw_calls=False):
            testnet.check_config(config)
            self.raw_calls.append(raw_calls)
            yield self.chain

        self.raw_calls = []
        self.enterContext(patch("zils.testnet.connect", connect))
        self.enterContext(patch.dict("os.environ", {"ZILS_MINER_CONFIG": str(Path(tmp.name) / "m.json")}))

    def run_miner(self, *argv, public_ip="8.8.8.8"):
        flags = [
            "--wallet-name", self.wallet["name"], "--wallet-hotkey", self.wallet["hotkey"],
            "--wallet-path", self.wallet["path"], "--public-ip", public_ip, "--port", "8091",
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                miner.main([argv[0], *flags, *argv[1:]])
                code = 0
            except SystemExit as stopped:
                code = stopped.code
        return code, output.getvalue()

    def test_announce_dry_run_writes_nothing(self):
        code, out = self.run_miner("announce-serving")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.chain.calls, [])
        self.assertIn("dry run: nothing written; run again with --yes", out)

    def test_announce_yes_writes_and_reads_back(self):
        code, out = self.run_miner("announce-serving", "--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.chain.calls), 1)
        self.assertEqual(self.raw_calls, [True])  # the Policy refuses submit_call otherwise
        self.assertIn(ENDPOINT, out)

    def test_announce_same_endpoint_is_a_no_op(self):
        self.run_miner("announce-serving", "--yes")
        code, out = self.run_miner("announce-serving", "--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.chain.calls), 1)
        self.assertIn("already announced", out)

    def test_announce_unregistered_hotkey_fails(self):
        del self.chain.uids[self.hotkey]
        code, out = self.run_miner("announce-serving", "--yes")
        self.assertEqual(code, 1)
        self.assertIn("not registered", out)
        self.assertEqual(self.chain.calls, [])

    def test_non_public_ip_fails_before_any_chain_call(self):
        for command in (["announce-serving", "--yes"], ["serve", "--checkpoint", "ckpt"]):
            code, out = self.run_miner(*command, public_ip="10.0.0.5")
            self.assertEqual(code, 1)
            self.assertIn("public IPv4", out)
        self.assertEqual(self.chain.calls, [])

    def test_serve_refuses_when_axon_does_not_match(self):
        with patch("zils.serving_server.load_served", Mock(return_value=Mock(model="z3-mini"))), patch(
            "zils.serving_server.make_server"
        ) as make:
            code, out = self.run_miner("serve", "--checkpoint", "ckpt")
        self.assertEqual(code, 1)
        self.assertIn("announce-serving", out)
        make.assert_not_called()

    def test_serve_allows_permit_holders_and_gateway(self):
        self.run_miner("announce-serving", "--yes")
        with patch("zils.serving_server.load_served", Mock(return_value=Mock(model="z3-mini"))), patch(
            "zils.serving_server.make_server"
        ) as make:
            code, out = self.run_miner(
                "serve", "--checkpoint", "ckpt", "--gateway-hotkey", "gateway"
            )
        self.assertEqual(code, 0, out)
        self.assertEqual(self.raw_calls[-2:], [False, False])  # serve only reads
        allowed = make.call_args.args[3]
        self.assertEqual(make.call_args.args[4], ("0.0.0.0", 8091))
        self.assertEqual(
            [allowed(self.validator), allowed("gateway"), allowed(self.stranger)],
            [True, True, False],
        )

    def test_permit_set_survives_a_failed_refresh(self):
        reads = [{"a"}, RuntimeError("chain down"), {"b"}]

        def read():
            value = reads.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        allowed = miner.permit_checker(read, ("gateway",))
        self.assertTrue(allowed("a") and allowed("gateway"))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            allowed.refresh()  # fails: keeps {"a"}
        self.assertIn("chain down", out.getvalue())
        self.assertTrue(allowed("a"))
        allowed.refresh()
        self.assertTrue(allowed("b") and not allowed("a"))


if __name__ == "__main__":
    unittest.main()
