"""Open admission: chain-derived members, public endpoints pinned to the caller, submit throttling settings."""

import unittest

from zils import protocol as r, testnet, validator


class PublicAdmissionTest(unittest.TestCase):
    def test_public_endpoints_are_pinned_and_globally_routable(self):
        public = "http://8.8.8.8:8901"
        r.endpoint_ok(public, allowed=[public], public=True)
        self.assertEqual(r.endpoint_port(public), (r.ipaddress.IPv4Address("8.8.8.8"), 8901))
        with self.assertRaises(ValueError):
            r.endpoint_ok(public, allowed=["http://8.8.4.4:8901"], public=True)
        for url in (
            "http://192.168.1.20:8901",
            "http://100.108.80.3:8900",
            "http://127.0.0.1:8901",
            "http://169.254.169.254:80",
            "http://224.0.0.1:80",
        ):
            with self.assertRaises(ValueError):
                r.endpoint_ok(url, allowed=[url], public=True)
        # Private mode is unchanged: a public address is still refused there.
        with self.assertRaises(ValueError):
            r.endpoint_ok(public, allowed=[public])
        with self.assertRaises(ValueError):
            r.endpoint_port("http://example.com:80")

    def test_chain_admission_lists_every_registered_hotkey_except_the_validator(self):
        state = {
            "NetworksAdded": True,
            "SubnetworkN": 4,
            "Uids": {"validator": 0, "miner-a": 1, "miner-b": 2, "miner-c": 3},
            "Keys": {0: "validator", 1: "miner-a", 2: "miner-b", 3: "miner-c"},
            "ValidatorPermit": [True, False, False, False],
            "SubnetOwnerHotkey": "validator",
            "CommitRevealWeightsEnabled": True,
            "MinAllowedWeights": 1,
            "MaxWeightsLimit": 65535,
            "WeightsSetRateLimit": 100,
            "WeightsVersionKey": 0,
            "LastUpdate": [100, 0, 0, 0],
        }

        class Chain:
            network, endpoint, block = "test", testnet.ENDPOINT, 250

            def query(self, item, params, *, block=None):
                value = state[item[1]]
                return value.get(params[1]) if isinstance(value, dict) else value

        config = {
            "chain": {"network": "test", "netuid": 553},
            "validator_hotkey": "validator",
            "admission": "chain",
            "public_endpoints": True,
        }
        result = testnet.preflight(config, Chain())
        self.assertEqual(result["members"], {"1": "miner-a", "2": "miner-b", "3": "miner-c"})
        state["Keys"][2] = ""  # a deregistered slot is simply absent
        self.assertEqual(set(testnet.preflight(config, Chain())["members"]), {"1", "3"})

    def test_round_config_uses_the_chain_snapshot_roster(self):
        config = {
            "chain": {"network": "test", "netuid": 579},
            "admission": "chain",
            "public_endpoints": True,
        }
        members = {1: "miner-a", 3: "miner-c"}
        derived = validator.round_config(config, "chain", members)
        self.assertEqual(
            derived["members"], {"1": {"hotkey": "miner-a"}, "3": {"hotkey": "miner-c"}}
        )
        self.assertNotIn("members", config)
        static = {"members": {"1": {"hotkey": "a", "port": 8901}}}
        self.assertIs(validator.round_config(static, "members", {1: "a"}), static)

    def test_admission_settings_fail_closed(self):
        self.assertEqual(
            validator.admission_settings({"members": {"1": {"hotkey": "a", "port": 8901}}}),
            ("members", False),
        )
        self.assertEqual(
            validator.admission_settings(
                {"chain": {}, "admission": "chain", "public_endpoints": True}
            ),
            ("chain", True),
        )
        with self.assertRaisesRegex(ValueError, "public_endpoints"):
            validator.admission_settings({"chain": {}, "admission": "chain"})
        with self.assertRaisesRegex(ValueError, "chain configuration"):
            validator.admission_settings({"admission": "chain", "public_endpoints": True})
        with self.assertRaisesRegex(ValueError, "admission"):
            validator.admission_settings({"admission": "open"})


if __name__ == "__main__":
    unittest.main()
