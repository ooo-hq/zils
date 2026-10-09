"""Real-provider gates must not accept storage or evaluation false positives."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_spaces import check, passed
from scripts.rehearse_image_vertical import require_evaluated
from tests import test_spaces as fixture


class SpacesPreflightTest(unittest.TestCase):
    def test_each_gate_is_required(self):
        report = dict(
            private_read=True,
            multipart_immutable=True,
            roundtrip_bytes=True,
            roundtrip_sha256=True,
            catalog_recovery=True,
            cors=True,
        )
        self.assertTrue(passed(report))
        for key in report:
            self.assertFalse(passed(report | {key: False}), key)
            self.assertFalse(passed({k: v for k, v in report.items() if k != key}), key)
        self.assertFalse(passed(report | {"roundtrip_bytes": 1}))

    def test_missing_candidate_cannot_count_as_quality_evaluation(self):
        for status in ("missing", "invalid", "failed"):
            with self.assertRaises(ValueError):
                require_evaluated(
                    {
                        "miners": [{"uid": 1, "status": status}],
                        "delivery": {"status": "no_qualifying_model"},
                    }
                )
        with self.assertRaises(ValueError):
            require_evaluated({"miners": []})
        require_evaluated(
            {
                "miners": [{"uid": 1, "status": "evaluated"}],
                "delivery": {"status": "no_qualifying_model"},
            }
        )


class PreflightProtocolTest(unittest.TestCase):
    setUpClass = classmethod(fixture.SpacesTest.setUpClass.__func__)
    tearDownClass = classmethod(fixture.SpacesTest.tearDownClass.__func__)
    setUp = fixture.SpacesTest.setUp

    def run_check(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "checkpoint"
            source.write_bytes(b"actual provider protocol bytes")
            return check(source, self.store, "https://frontend.example")

    def test_private_storage_protocol_checks_pass(self):
        self.s3.private, self.s3.cors_origin = True, "https://frontend.example"
        result = self.run_check()
        self.assertTrue(result["passed"], result)

    def test_cors_can_return_only_the_requested_method(self):
        self.s3.private, self.s3.cors_origin = True, "https://frontend.example"
        self.s3.cors_echo_method = True
        result = self.run_check()
        self.assertTrue(result["cors"], result)
        self.s3.cors_denied_methods.add("HEAD")
        self.assertFalse(self.run_check()["cors"])

    def test_idempotent_completion_preserves_the_same_object(self):
        self.s3.private, self.s3.cors_origin = True, "https://frontend.example"
        self.s3.idempotent_completion = True
        self.s3.completion_quotes = False
        result = self.run_check()
        self.assertTrue(result["multipart_immutable"], result)

    def test_public_read_and_bad_cors_fail(self):
        self.s3.private, self.s3.cors_origin = False, "*"
        result = self.run_check()
        self.assertFalse(result["passed"])
        self.assertFalse(result["private_read"])
        self.assertFalse(result["cors"])

    def test_corrupted_roundtrip_fails(self):
        self.s3.private, self.s3.cors_origin = True, "https://frontend.example"
        original = self.store.download

        def corrupt(bucket, path, destination, limit):
            original(bucket, path, destination, limit)
            destination.write_bytes(b"changed")

        with patch.object(self.store, "download", corrupt):
            result = self.run_check()
        self.assertFalse(result["passed"])
        self.assertFalse(result["roundtrip_sha256"])
        self.assertFalse(result["multipart_immutable"])
