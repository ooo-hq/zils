"""API credentials are random, tenant scoped, revocable, and never listed as secrets."""

import hashlib
import importlib
import importlib.util
import unittest
import uuid

from fez.decisions import DecisionError


class MemoryDB:
    def __init__(self):
        self.keys = []
        self.enabled = True
        self.calls = []

    def rpc(self, name, values):
        self.calls.append((name, values))
        if name == "zils_api_create_key":
            self.keys.append({**values["p_key"], "created_at": "now", "revoked_at": None})
            return self.keys[-1]
        if name == "zils_api_auth":
            return [
                k
                for k in self.keys
                if k["id"] == values["p_id"] and not k["revoked_at"] and self.enabled
            ]
        if name == "zils_api_admit":
            return "allowed"
        raise AssertionError(name)

    def rows(self, table, query):
        from urllib.parse import parse_qs

        params = parse_qs(query)
        cursor = params.get("id", ["gt."])[0][3:]
        return sorted((row for row in self.keys if row["id"] > cursor), key=lambda row: row["id"])[
            :100
        ]

    def patch(self, table, query, values):
        key = self.keys[0]
        if f"owner_id=eq.{key['owner_id']}" not in query:
            return []
        key.update(values)
        return [key]


class KeysTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("fez.api_store"), "key store missing")
        return importlib.import_module("fez.api_store")

    def test_keys_are_unrecoverable_unique_and_revocable_without_count_cap(self):
        m = self.module()
        db = MemoryDB()
        store = m.Store(db)
        owner = str(uuid.uuid4())
        issued = [store.create_key(owner, "test") for _ in range(12)]
        self.assertEqual(len({k["key"] for k in issued}), 12)
        for public, stored in zip(issued, db.keys, strict=True):
            self.assertEqual(stored["digest"], hashlib.sha256(public["key"].encode()).hexdigest())
            self.assertNotIn(public["key"], str(stored))
            self.assertEqual(store.authenticate(public["key"])["owner_id"], owner)
        self.assertNotIn("digest", str(store.list_keys(owner)))
        self.assertNotIn(issued[0]["key"], str(store.list_keys(owner)))
        store.revoke_key(owner, issued[0]["id"])
        with self.assertRaises(DecisionError) as cm:
            store.authenticate(issued[0]["key"])
        self.assertEqual(cm.exception.status, 401)
        self.assertEqual(store.authenticate(issued[1]["key"])["owner_id"], owner)
        db.enabled = False
        with self.assertRaises(DecisionError):
            store.authenticate(issued[1]["key"])

    def test_guessing_owner_or_malformed_id_cannot_revoke(self):
        m = self.module()
        store = m.Store(MemoryDB())
        owner = str(uuid.uuid4())
        key = store.create_key(owner, "demo")
        with self.assertRaises(DecisionError) as cm:
            store.revoke_key(str(uuid.uuid4()), key["id"])
        self.assertEqual(cm.exception.status, 404)
        for token in ("", key["key"] + "x", "zils_sk_x_x", "foo"):
            with self.assertRaises(DecisionError):
                store.authenticate(token)
        with self.assertRaises(DecisionError):
            store.revoke_key(owner, "eq.foo&owner_id=not.is.null")

    def test_reservations_use_account_not_key_budget(self):
        m = self.module()
        db = MemoryDB()
        store = m.Store(db)
        owner, key = str(uuid.uuid4()), str(uuid.uuid4())
        store.admit(owner, key, str(uuid.uuid4()), 123)
        name, values = db.calls[-1]
        self.assertEqual(name, "zils_api_admit")
        self.assertEqual(values["p_owner"], owner)
        self.assertEqual(values["p_tokens"], 123)

    def test_key_listing_walks_provider_pages_without_losing_older_keys(self):
        m = self.module()
        from urllib.parse import parse_qs

        owner = str(uuid.uuid4())

        class PaginatedDB:
            calls = 0

            def rows(self, table, query):
                self.calls += 1
                params = parse_qs(query)
                assert params["owner_id"] == ["eq." + owner]
                # A deliberately smaller provider page than the requested limit.
                after = int(uuid.UUID(params["id"][0][3:])) if "id" in params else 0
                return [
                    {
                        "id": str(uuid.UUID(int=i)),
                        "name": "historical-key",
                        "prefix": "zils_sk_",
                        "created_at": "now",
                        "revoked_at": None,
                    }
                    for i in range(after + 1, min(after + 37, 1050) + 1)
                ]

        db = PaginatedDB()
        rows = m.Store(db).list_keys(owner)
        self.assertEqual(len(rows), 1050)
        self.assertEqual(len({row["id"] for row in rows}), 1050)
        self.assertGreater(db.calls, 1)
