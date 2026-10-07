"""Readable names route to one accepted release without moving earlier names."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_adapter_releases import OTHER, OWNER, Source
from tests.test_version_selection import versioned_fixture
from zils.adapter_releases import publish, register, registry_entry
from zils.api import Registry
from zils.decisions import DecisionError
from zils.model_names import is_model_name, model_name, valid_customer_aliases


class ModelNamesTest(unittest.TestCase):
    def test_model_inventory_walks_all_pages_with_account_and_ready_filters(self):
        import uuid

        from zils.coordinator import available_models

        rows = [
            {
                "id": str(uuid.UUID(int=index)),
                "name": f"task-{index}",
                "status": "completed",
                "result": {
                    "delivery": {"status": "accepted"},
                    "workflow": {"state": "ready", "model_id": f"model-{index}"},
                },
            }
            for index in range(1, 102)
        ]
        queries = []

        class Store:
            def rows(self, table, query):
                queries.append(query)
                return rows[100:] if "&id=gt." in query else rows[:100]

        models = available_models(Store(), OWNER)
        self.assertEqual(len(models), 101)
        self.assertEqual(models[-1]["workflow"]["model_id"], "model-101")
        for query in queries:
            self.assertIn(f"owner_id=eq.{OWNER}", query)
            self.assertIn("result->workflow->>state=eq.ready", query)
            self.assertIn("result->delivery->>status=eq.accepted", query)
        self.assertIn("&id=gt." + rows[99]["id"], queries[1])

    def test_names_are_short_readable_and_distinct_for_runs_of_the_same_task(self):
        first = "9f670236-4286-49c8-8fa1-3551316c93c3"
        second = "a82c140b-4286-49c8-8fa1-3551316c93c3"
        self.assertEqual(model_name("Support actions", first), "support-actions-9f670236")
        self.assertNotEqual(
            model_name("Support actions", first), model_name("Support actions", second)
        )
        for text in (
            "x" * 64,
            "which-support-action-should-be-taken-next",
            "---",
            "zils-task-demo",
            "Résumé routing",
        ):
            name = model_name(text, first)
            self.assertLessEqual(len(name), 33)
            self.assertTrue(is_model_name(name, f"zils-adapter-{first}-" + "a" * 64))

    def test_names_and_legacy_ids_keep_their_release_after_an_upgrade(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = versioned_fixture(root / "first")
            second = versioned_fixture(root / "second", first)
            path = root / "registry.json"
            entries = []
            for job, folder in ((first, "first"), (second, "second")):
                release = publish(Source(job, root / folder), job["id"], root / "releases")
                entry = registry_entry(release, "http://127.0.0.1:8921", "TOKEN", name=job["name"])
                self.assertTrue(valid_customer_aliases(entry, release["selection"]))
                register(path, entry, selection=release["selection"])
                entries.append(entry)
            registry = Registry(json.loads(path.read_text())["models"])
            for job, entry in zip((first, second), entries, strict=True):
                name = model_name(job["name"], job["id"])
                self.assertEqual(registry.resolve(name, OWNER)["id"], entry["id"])
                self.assertEqual(registry.resolve(entry["id"], OWNER)["id"], entry["id"])
                with self.assertRaises(DecisionError):
                    registry.resolve(name, OTHER)
            self.assertEqual(
                registry.resolve("zils-task-" + first["id"], OWNER)["id"], entries[1]["id"]
            )

    def test_restricted_aliases_cannot_claim_shared_or_other_run_names(self):
        model = "zils-adapter-9f670236-4286-49c8-8fa1-3551316c93c3-" + "a" * 64
        for aliases in (
            ["zils-shared"],
            ["support-a82c140b"],
            ["support-9f670236", "extra-9f670236"],
        ):
            self.assertFalse(valid_customer_aliases({"id": model, "aliases": aliases}, None))

    def test_short_suffix_collision_cannot_retarget_an_existing_name(self):
        from zils.adapter_releases import merge_registry

        first = {
            "id": "zils-adapter-9f670236-4286-49c8-8fa1-3551316c93c3-" + "a" * 64,
            "aliases": ["support-9f670236"],
            "owners": [OWNER],
            "fingerprint": "b" * 64,
            "url": "http://127.0.0.1:8921",
            "token_env": "TOKEN",
            "description": "fixture",
            "release_date": "2026-10-07",
        }
        second = {**first, "id": first["id"].replace("4286", "1234")}
        original = {"models": [first]}
        before = copy.deepcopy(original)
        with self.assertRaisesRegex(ValueError, "cannot move"):
            merge_registry(original, second)
        self.assertEqual(original, before)
