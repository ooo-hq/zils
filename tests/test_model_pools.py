"""Pool isolation and model-specific checkpoint contracts, with tiny local tensors."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

import zils
from zils import d3, models


class PoolProfilesTest(unittest.TestCase):
    def test_promoted_mini_keeps_its_pool_and_legacy_identity(self):
        self.assertEqual(models.pool_id(models.D3_ZILS_R5), "d3-mini")
        self.assertEqual(
            models.spec(models.D3)["base_revision"], "b54402a963373cc44713f18c9596c580e31cc68b"
        )
        self.assertEqual(models.candidate_files(models.D3, "candidate"), models.D3_CANDIDATE_FILES)
        self.assertEqual(models.artifact_budget("candidate", models.D3), 10 * 1024**3)
        with self.assertRaises(ValueError):
            models.pool_id(models.H2O)

    def test_pool_catalog_does_not_claim_new_hardware_qualification(self):
        catalog = {p["id"]: p for p in models.pool_catalog()}
        self.assertEqual(set(catalog), {"d3-lite", "d3-nano", "d3-mini", "d3-flash", "d3"})
        lite = catalog["d3-lite"]["model"]
        self.assertEqual(models.pool_id(lite), "d3-lite")
        self.assertEqual(models.spec(lite)["max_input_tokens"], 4096)
        self.assertFalse(catalog["d3-lite"]["qualified"])
        # A caller cannot mutate subsequent discovery or a pinned profile.
        catalog["d3-lite"]["qualified"] = True
        self.assertFalse(
            next(p for p in models.pool_catalog() if p["id"] == "d3-lite")["qualified"]
        )

    def test_candidate_layout_and_budget_follow_the_pinned_model(self):
        catalog = {p["id"]: p["model"] for p in models.pool_catalog()}
        self.assertIn("model.safetensors", models.candidate_files(catalog["d3-nano"], "candidate"))
        self.assertNotIn(
            "model.safetensors.index.json", models.candidate_files(catalog["d3-nano"], "candidate")
        )
        full = models.candidate_files(catalog["d3"], "candidate")
        self.assertEqual(len([n for n in full if n.startswith("model-")]), 11)
        self.assertGreater(models.artifact_budget("candidate", catalog["d3"]), 48 * 1024**3)
        self.assertLess(models.artifact_budget("candidate", catalog["d3-lite"]), 3 * 1024**3)

    def test_unsharded_candidate_validates_tensors_and_cannot_cross_pools(self):
        model = "d3-nano-2.2b-v1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, candidate = root / "parent", root / "candidate"
            parent.mkdir()
            candidate.mkdir()
            save_file(
                {"a.weight": torch.zeros(2, 3, dtype=torch.bfloat16)},
                str(parent / "model.safetensors"),
            )
            save_file(
                {"a.weight": torch.ones(2, 3, dtype=torch.bfloat16)},
                str(candidate / "model.safetensors"),
            )
            hashes = {}
            for name in models.D3_CONTRACT_FILES:
                content = b"pinned " + name.encode()
                (parent / name).write_bytes(content)
                (candidate / name).write_bytes(content)
                hashes[name] = hashlib.sha256(content).hexdigest()
            hashes["model.safetensors"] = hashlib.sha256(
                (parent / "model.safetensors").read_bytes()
            ).hexdigest()
            pins = {**models.pins(model), "files": hashes}
            models.write_metadata(candidate, model=model, kind="candidate")
            with (
                patch("zils.models.pins", return_value=pins),
                patch("zils.d3.pinned_base_path", return_value=parent),
            ):
                d3.validate_candidate(candidate, model)
                self.assertEqual(len(zils.checkpoint_hash(candidate)), 64)
                with self.assertRaisesRegex(ValueError, "metadata"):
                    d3.validate_candidate(candidate, models.D3)
                save_file(
                    {"a.weight": torch.ones(3, 2, dtype=torch.bfloat16)},
                    str(candidate / "model.safetensors"),
                )
                with self.assertRaisesRegex(ValueError, "shape|schema"):
                    d3.validate_candidate(candidate, model)
                save_file(
                    {"extra.weight": torch.ones(2, 3, dtype=torch.bfloat16)},
                    str(candidate / "model.safetensors"),
                )
                with self.assertRaisesRegex(ValueError, "keys|tensor"):
                    d3.validate_candidate(candidate, model)


if __name__ == "__main__":
    unittest.main()
