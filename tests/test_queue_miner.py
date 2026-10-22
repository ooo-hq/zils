"""The queue miner's chain-admission call: terms version and qualification report, once."""

import json
import tempfile
import unittest
from pathlib import Path

from miner.queue import admit
from zils.cloud import APIError


class FakeClient:
    def __init__(self, error=None):
        self.calls, self.error = [], error

    def call(self, action, body=None):
        self.calls.append((action, body))
        if self.error:
            raise APIError(self.error, "rejected")
        return {"admitted": True, "uid": 103}


class AdmitTest(unittest.TestCase):
    def test_sends_terms_and_report_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "qualification.json"
            report.write_text(json.dumps({"model": "d3-mini-4.5b-v1", "terms_sha256": "f" * 64}))
            client = FakeClient()
            result = admit(
                client, {"coordinator": "x", "terms_sha256": "f" * 64, "qualification": str(report)}
            )
        self.assertEqual(result["uid"], 103)
        self.assertEqual(
            client.calls,
            [
                (
                    "admit",
                    {
                        "terms_sha256": "f" * 64,
                        "qualification": {"model": "d3-mini-4.5b-v1", "terms_sha256": "f" * 64},
                    },
                )
            ],
        )

    def test_operator_registered_nodes_skip_it(self):
        client = FakeClient()
        self.assertIsNone(admit(client, {"coordinator": "x"}))
        self.assertEqual(client.calls, [])

    def test_bad_terms_hash_stops_before_any_call(self):
        client = FakeClient()
        with self.assertRaises(ValueError):
            admit(client, {"terms_sha256": "not-a-digest"})
        self.assertEqual(client.calls, [])

    def test_old_or_closed_coordinator_is_tolerated_other_refusals_are_not(self):
        for status in (404, 409):
            self.assertIsNone(admit(FakeClient(status), {"terms_sha256": "f" * 64}))
        with self.assertRaises(APIError):
            admit(FakeClient(403), {"terms_sha256": "f" * 64})


if __name__ == "__main__":
    unittest.main()


class FitCheckTest(unittest.TestCase):
    def test_refuses_only_what_the_whole_budget_cannot_hold(self):
        from unittest.mock import patch

        from miner.queue import fits
        from zils import models

        long, short = {"max_tokens": 8192}, {"max_tokens": 1000}
        self.assertFalse(fits(models.D3, long, "cuda", budget_mib=12000))
        self.assertTrue(fits(models.D3, short, "cuda", budget_mib=12000))
        with patch("miner.queue.gpu_total_mib", return_value=23028):
            self.assertTrue(fits(models.D3, long, "cuda"))
        with patch("miner.queue.gpu_total_mib", return_value=None):
            self.assertTrue(
                fits(models.D3, long, "cuda")
            )  # unreadable: training's own gate decides
        # No curve, no workload or no GPU: nothing to check here.
        self.assertTrue(fits(models.KEV, long, "cuda", budget_mib=1))
        self.assertTrue(fits(models.D3, None, "cuda", budget_mib=1))
        self.assertTrue(fits(models.D3, long, "cpu", budget_mib=1))
