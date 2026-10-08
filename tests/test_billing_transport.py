"""Expose actionable credit errors while keeping provider details private."""

import unittest
from unittest.mock import Mock, patch

from zils.cloud import APIError, Supabase


class BillingTransportTest(unittest.TestCase):
    def test_insufficient_funds_is_payment_required_without_database_details(self):
        response = Mock(status_code=400, content=b"{}")
        response.json.return_value = {"code": "P0402", "message": "secret database details"}
        db = Supabase("https://project.supabase.co", "fixture")
        with patch("zils.cloud.requests.request", return_value=response):
            with self.assertRaises(APIError) as failure:
                db.patch("fez_training_jobs", "id=eq.fixture", {"status": "validating"})
        self.assertEqual(failure.exception.status, 402)
        self.assertIn("credit", str(failure.exception))
        self.assertNotIn("secret", str(failure.exception))
        response.close.assert_called_once()

    def test_unrecognized_provider_error_remains_private(self):
        response = Mock(status_code=400, content=b"{}")
        response.json.return_value = {"code": "anything", "message": "secret"}
        with patch("zils.cloud.requests.request", return_value=response):
            with self.assertRaises(APIError) as failure:
                Supabase("https://project.supabase.co", "fixture").rpc("test", {})
        self.assertEqual(failure.exception.status, 409)
        self.assertNotIn("secret", str(failure.exception))
