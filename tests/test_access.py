"""Hosted services reject signed-in customers until their account is approved."""

import unittest

from tests.test_queue import Store
from zils.api_store import Store as KeyStore
from zils.cloud import APIError
from zils.coordinator import Service


class UnapprovedStore(Store):
    def rpc(self, name, values):
        if name == "zils_access_allowed":
            return False
        return super().rpc(name, values)


class AccessTest(unittest.TestCase):
    def test_signed_in_unapproved_customer_cannot_list_or_create_jobs(self):
        service = Service(UnapprovedStore(), "https://training.example.com")
        for method in ("GET", "POST"):
            with self.subTest(method=method):
                with self.assertRaises(APIError) as error:
                    service.customer(method, "/v1/jobs", "owner-token", {})
                self.assertEqual(error.exception.status, 403)

    def test_signed_in_unapproved_customer_cannot_manage_keys(self):
        with self.assertRaises(APIError) as error:
            KeyStore(UnapprovedStore()).session_owner("owner-token")
        self.assertEqual(error.exception.status, 403)

    def test_access_service_failure_never_grants_access(self):
        class UnavailableStore(Store):
            def rpc(self, name, values):
                raise APIError(503, "Unavailable")

        with self.assertRaises(APIError) as error:
            KeyStore(UnavailableStore()).session_owner("owner-token")
        self.assertEqual(error.exception.status, 503)
