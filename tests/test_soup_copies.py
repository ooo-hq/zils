"""Soup copy checks: streamed weight sketches, copy / re-upload rejection and behavioural twins."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tests.test_bodies import NAMES, SHAPES, checkpoint, weights
from zils import soup_copies

SEED = b"round-seed"
LAYERS = [n for n in NAMES if n.startswith("language_model.layers.")]


def moved(base, seed, scale=0.05):
    """`base` plus a seeded random change on every tensor (bf16)."""
    generator = torch.Generator().manual_seed(seed)
    return {
        n: (t.float() + scale * torch.randn(t.shape, generator=generator)).to(torch.bfloat16)
        for n, t in base.items()
    }


class SketchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base_tensors = weights(1)
        self.base = checkpoint(self.root / "qwen", self.base_tensors, prefix="model.", shards=2)
        self.body_tensors = moved(self.base_tensors, 7)

    def tearDown(self):
        self.tmp.cleanup()

    def test_deterministic_and_layout_independent(self):
        folders = [
            checkpoint(self.root / "a", self.body_tensors, extras=False),
            checkpoint(self.root / "b", self.body_tensors, prefix="model.", shards=3),
        ]
        sketches = [soup_copies.Sketch(SEED, dims=64).sketch(f, self.base, NAMES) for f in folders]
        again = soup_copies.Sketch(SEED, dims=64).sketch(folders[0], self.base, NAMES)
        self.assertEqual(sketches[0].dtype, np.float64)
        self.assertEqual(sketches[0].shape, (64,))
        np.testing.assert_array_equal(sketches[0], sketches[1])
        np.testing.assert_array_equal(sketches[0], again)
        self.assertGreater(np.linalg.norm(sketches[0]), 0)
        other = soup_copies.Sketch(b"other-seed", dims=64).sketch(folders[0], self.base, NAMES)
        self.assertFalse(np.array_equal(sketches[0], other))
        module = soup_copies.sketch(folders[0], self.base, NAMES, seed=SEED, dims=64)
        np.testing.assert_array_equal(module, sketches[0])

    def test_block_size_does_not_change_the_sketch(self):
        folder = checkpoint(self.root / "a", self.body_tensors)
        whole = soup_copies.Sketch(SEED, dims=32).sketch(folder, self.base, NAMES)
        with patch.object(soup_copies, "READ_ELEMENTS", 3):
            pieces = soup_copies.Sketch(SEED, dims=32).sketch(folder, self.base, NAMES)
        np.testing.assert_array_equal(whole, pieces)

    def test_only_language_layers_count_and_sketch_is_linear(self):
        changed = dict(self.base_tensors)
        changed["visual.patch.weight"] = torch.zeros(SHAPES["visual.patch.weight"]).bfloat16()
        changed["language_model.embed_tokens.weight"] = torch.ones(8, 4).bfloat16()
        folder = checkpoint(self.root / "outside", changed)
        sketcher = soup_copies.Sketch(SEED, dims=32)
        np.testing.assert_array_equal(sketcher.sketch(folder, self.base, NAMES), np.zeros(32))
        # Doubling the layer delta doubles the sketch (the projection is linear in the delta).
        delta = {n: torch.zeros(SHAPES[n]) for n in NAMES}
        delta[LAYERS[0]] = torch.full(SHAPES[LAYERS[0]], 0.25)
        once = {n: (t.float() + delta[n]).bfloat16() for n, t in self.base_tensors.items()}
        twice = {n: (t.float() + 2 * delta[n]).bfloat16() for n, t in self.base_tensors.items()}
        one = sketcher.sketch(checkpoint(self.root / "once", once), self.base, NAMES)
        two = sketcher.sketch(checkpoint(self.root / "twice", twice), self.base, NAMES)
        np.testing.assert_allclose(two, 2 * one, atol=0.02 * np.abs(two).max())

    def test_rejects_shape_mismatch(self):
        wrong = {
            n: torch.zeros(5, 4).bfloat16() if n == LAYERS[0] else t
            for n, t in self.body_tensors.items()
        }
        shapes = dict(SHAPES)
        folder = checkpoint(self.root / "wrong", wrong)
        with self.assertRaisesRegex(ValueError, "shape"):
            soup_copies.Sketch(SEED, dims=16).sketch(folder, self.base, shapes)


class GeometryTest(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(soup_copies.cosine([1, 0], [2, 0]), 1.0)
        self.assertAlmostEqual(soup_copies.cosine([1, 0], [0, 3]), 0.0)
        self.assertAlmostEqual(soup_copies.cosine([1, 0], [-1, 0]), -1.0)
        self.assertEqual(soup_copies.cosine([0, 0], [1, 0]), 0.0)

    def test_explained_share(self):
        rng = np.random.default_rng(0)
        a, b, c = rng.standard_normal((3, 256))
        self.assertAlmostEqual(soup_copies.explained_share((a + b) / 2, [a, b, c]), 1.0, places=6)
        self.assertAlmostEqual(soup_copies.explained_share(a, [-a]), 0.0, places=6)
        self.assertLess(soup_copies.explained_share(a, [b, c]), 0.1)
        self.assertEqual(soup_copies.explained_share(a, []), 0.0)
        self.assertEqual(soup_copies.explained_share(np.zeros(4), [np.ones(4)]), 0.0)

    def test_fallback_nnls_matches_scipy(self):
        from scipy.optimize import nnls

        rng = np.random.default_rng(1)
        matrix = rng.standard_normal((64, 5))
        target = matrix @ np.array([0.5, 0.0, 1.5, 0.2, 0.0]) - 0.7 * matrix[:, 1]
        expected, _ = nnls(matrix, target)
        got = soup_copies._nnls_fallback(matrix, target)
        np.testing.assert_allclose(got, expected, atol=1e-6)
        with patch.object(soup_copies, "_scipy_nnls", None):
            share = soup_copies.explained_share(target, list(matrix.T))
        self.assertAlmostEqual(share, soup_copies.explained_share(target, list(matrix.T)), places=6)


class IsCopyTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.earlier = list(rng.standard_normal((3, 4096)))
        self.fresh = rng.standard_normal(4096)
        self.noise = 0.01 * rng.standard_normal(4096)

    def test_exact_copy(self):
        self.assertEqual(soup_copies.is_copy(self.earlier[1].copy(), self.earlier), "cosine")
        self.assertEqual(soup_copies.is_copy(3 * self.earlier[1], self.earlier), "cosine")

    def test_uniform_average_explained_by_earlier(self):
        average = sum(self.earlier) / 3
        self.assertGreater(soup_copies.explained_share(average, self.earlier), 0.999)
        self.assertEqual(
            soup_copies.is_copy(average + self.noise, self.earlier), "explained by earlier"
        )

    def test_soup_and_release_reuploads(self):
        soup = sum(self.earlier) / 3
        self.assertEqual(soup_copies.is_copy(soup + self.noise, [], soup=soup), "soup re-upload")
        self.assertEqual(
            soup_copies.is_copy(soup + self.noise, [], soup=self.fresh, release=soup),
            "release re-upload",
        )

    def test_house_bodies_reject_only_near_exact_re_uploads(self):
        house = self.earlier[2]
        self.assertEqual(
            soup_copies.is_copy(house + self.noise, [], house=[house]), "house re-upload"
        )
        # A real fine-tune of a house model moves away from it and is allowed.
        tuned = house + 0.3 * self.fresh
        self.assertGreater(soup_copies.explained_share(tuned, [house]), 0.8)
        self.assertIsNone(soup_copies.is_copy(tuned, [], house=[house]))

    def test_a_fine_tune_of_the_soup_or_release_is_allowed(self):
        soup = sum(self.earlier) / 3
        tuned = soup + 0.2 * self.fresh
        self.assertGreater(soup_copies.explained_share(tuned, [soup]), 0.8)
        self.assertIsNone(soup_copies.is_copy(tuned, [], soup=soup, release=soup))

    def test_genuinely_different_body_passes(self):
        self.assertIsNone(
            soup_copies.is_copy(
                self.fresh, self.earlier, soup=sum(self.earlier) / 3, release=self.earlier[0]
            )
        )
        # Mostly fresh with a little of an earlier body: its share is under the 0.3 bar.
        self.assertIsNone(soup_copies.is_copy(self.fresh + 0.3 * self.earlier[0], self.earlier))

    def test_two_trainers_fine_tuning_the_same_house_body_are_not_copies(self):
        # Both deltas are mostly the shared house delta, plus a small part of their own (dry run:
        # trainer A explains 0.90 of trainer B alone, but adds only 0.007 beyond the house).
        rng = np.random.default_rng(7)
        house = 10 * rng.standard_normal(4096)
        first = house + 3 * rng.standard_normal(4096)
        second = house + 3 * rng.standard_normal(4096)
        self.assertGreater(soup_copies.explained_share(second, [first]), 0.8)
        self.assertIsNone(soup_copies.is_copy(second, [first], house=[house]))

    def test_a_merge_with_a_house_body_is_judged_on_the_other_hotkeys_part(self):
        other, house = self.earlier[0], self.earlier[2]
        merge = 0.5 * other + 0.5 * house
        # Earlier bodies alone explain only half of it, under the 0.8 bar of the old check.
        self.assertLess(soup_copies.explained_share(merge, [other]), 0.8)
        self.assertEqual(soup_copies.is_copy(merge, [other], house=[house]), "explained by earlier")
        # A light fine-tune of the house body owes nothing to the other hotkey's body.
        tuned = house + 0.3 * self.fresh
        self.assertIsNone(soup_copies.is_copy(tuned, [other], house=[house]))
        # Half fresh, half another hotkey's body: that body's share is about 0.5.
        self.assertEqual(
            soup_copies.is_copy(self.fresh + other, [other], house=[house]), "explained by earlier"
        )

    # Allowed bases: house bodies, the current soup and the release. Once miners hold seats the
    # soup and release are mostly other hotkeys' bodies; a fine-tune of them is the miner's own
    # work (owner decision 10), while copying or re-weighting the seated bodies is not.

    def test_a_fine_tune_of_a_soup_of_other_hotkeys_bodies_is_allowed(self):
        soup = sum(self.earlier) / 3
        tuned = soup + 0.3 * self.fresh
        self.assertGreater(soup_copies.explained_share(tuned, self.earlier), 0.5)
        self.assertIsNone(soup_copies.is_copy(tuned, self.earlier, soup=soup))

    def test_a_fine_tune_of_a_release_of_other_hotkeys_bodies_is_allowed(self):
        release = (self.earlier[0] + self.earlier[1]) / 2
        tuned = release + 0.3 * self.fresh
        self.assertIsNone(soup_copies.is_copy(tuned, self.earlier, release=release))

    def test_a_near_copy_of_a_seated_body_is_still_a_copy(self):
        soup = sum(self.earlier) / 3
        near = self.earlier[0] + 0.15 * self.fresh
        self.assertLess(soup_copies.cosine(near, self.earlier[0]), 0.999)
        self.assertEqual(soup_copies.is_copy(near, self.earlier, soup=soup), "explained by earlier")

    def test_a_re_weighted_mix_of_seated_bodies_is_a_copy(self):
        soup = sum(self.earlier) / 3
        mixes = [
            0.6 * self.earlier[0] + 0.3 * self.earlier[1] + 0.1 * self.earlier[2],
            soup + 0.2 * self.earlier[0],  # the soup with one member's weight raised
        ]
        for mix in mixes:
            with self.subTest():
                self.assertLess(soup_copies.cosine(mix, soup), 0.999)
                self.assertEqual(
                    soup_copies.is_copy(mix + self.noise, self.earlier, soup=soup),
                    "explained by earlier",
                )

    def test_a_near_copy_of_another_hotkeys_house_fine_tune_is_a_copy(self):
        # Real fine-tunes share about 93% of their change with the house body they start from
        # (dry run), so the copied part is a small share of the whole body but all of the part
        # beyond the house.
        rng = np.random.default_rng(11)
        house = 10 * rng.standard_normal(4096)
        other = house + 3 * rng.standard_normal(4096)
        near = other + rng.standard_normal(4096)
        self.assertLess(soup_copies.cosine(near, other), 0.999)
        self.assertEqual(
            soup_copies.is_copy(near, [other], house=[house], soup=(other + house) / 2),
            "explained by earlier",
        )

    def test_two_house_fine_tunes_stay_allowed_with_one_seated_in_the_soup(self):
        rng = np.random.default_rng(7)
        house, pplx = 10 * rng.standard_normal(4096), 5 * rng.standard_normal(4096)
        first = house + 3 * rng.standard_normal(4096)
        second = house + 3 * rng.standard_normal(4096)
        soup = (first + pplx) / 2
        self.assertIsNone(soup_copies.is_copy(second, [first], house=[house, pplx], soup=soup))

    def test_near_exact_re_uploads_of_house_soup_or_release_stay_banned(self):
        soup, release, house = sum(self.earlier) / 3, self.earlier[0], self.earlier[2]
        for target, want in (
            (house + self.noise, "house re-upload"),
            (soup + self.noise, "soup re-upload"),
            (release * 1.01, "release re-upload"),
        ):
            with self.subTest(want=want):
                got = soup_copies.is_copy(target, [], house=[house], soup=soup, release=release)
                self.assertEqual(got, want)

    def test_base_residual_is_the_share_beyond_house_soup_and_release(self):
        soup = sum(self.earlier) / 3
        self.assertLess(soup_copies.base_residual(soup * 2, house=[], soup=soup), 1e-9)
        tuned = soup + 0.3 * self.fresh
        expected = 1 - soup_copies.explained_share(tuned, [self.earlier[2], soup])
        got = soup_copies.base_residual(tuned, house=[self.earlier[2]], soup=soup)
        self.assertAlmostEqual(got, expected)
        self.assertAlmostEqual(soup_copies.base_residual(self.fresh, house=[]), 1.0)

    def test_zero_change_is_rejected(self):
        self.assertEqual(
            soup_copies.is_copy(np.zeros(4096), self.earlier), "no change from the base"
        )

    def test_end_to_end_on_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_tensors = weights(1)
            base = checkpoint(root / "qwen", base_tensors)
            sketcher = soup_copies.Sketch(SEED, dims=64)
            members = [moved(base_tensors, s) for s in (11, 12, 13)]
            sketches = [
                sketcher.sketch(checkpoint(root / f"m{i}", t), base, NAMES)
                for i, t in enumerate(members)
            ]
            soup = {n: (sum(m[n].float() for m in members) / 3).bfloat16() for n in NAMES}
            soup_sketch = sketcher.sketch(checkpoint(root / "soup", soup, shards=2), base, NAMES)
            self.assertEqual(soup_copies.is_copy(soup_sketch, sketches), "explained by earlier")
            self.assertEqual(
                soup_copies.is_copy(soup_sketch, [], soup=soup_sketch), "soup re-upload"
            )
            fresh = sketcher.sketch(
                checkpoint(root / "fresh", moved(base_tensors, 99)), base, NAMES
            )
            self.assertIsNone(soup_copies.is_copy(fresh, sketches, soup=soup_sketch))


def rows(probabilities):
    return [
        {"id": f"item-{i}", "probabilities": {f"o{j}": float(p) for j, p in enumerate(row)}}
        for i, row in enumerate(probabilities)
    ]


def tempered(probabilities, temperature):
    logits = np.log(probabilities) / temperature
    out = np.exp(logits - logits.max(axis=1, keepdims=True))
    return out / out.sum(axis=1, keepdims=True)


class IsTwinTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.a = self.rng.dirichlet(np.ones(4), size=80)

    def test_identical_and_temperature_copies_are_twins(self):
        self.assertTrue(soup_copies.is_twin(rows(self.a), rows(self.a)))
        for temperature in (0.6, 1.6, 1.9):
            with self.subTest(temperature=temperature):
                self.assertTrue(
                    soup_copies.is_twin(rows(self.a), rows(tempered(self.a, temperature)))
                )

    def test_noise_padded_copy_is_twin(self):
        noisy = self.a * np.exp(0.02 * self.rng.standard_normal(self.a.shape))
        noisy /= noisy.sum(axis=1, keepdims=True)
        self.assertTrue(soup_copies.is_twin(rows(self.a), rows(noisy)))

    def test_different_predictions_are_not_twins(self):
        other = self.rng.dirichlet(np.ones(4), size=80)
        self.assertFalse(soup_copies.is_twin(rows(self.a), rows(other)))

    def test_too_few_shared_items_not_judged(self):
        self.assertFalse(soup_copies.is_twin(rows(self.a[:49]), rows(self.a[:49])))
        b = rows(self.a)
        for row in b[:40]:
            row["id"] = "other-" + row["id"]
        self.assertFalse(soup_copies.is_twin(rows(self.a), b))
        self.assertTrue(soup_copies.is_twin(rows(self.a), b, min_items=40))

    def test_jsd_threshold(self):
        noisy = self.a * np.exp(0.3 * self.rng.standard_normal(self.a.shape))
        noisy /= noisy.sum(axis=1, keepdims=True)
        self.assertFalse(soup_copies.is_twin(rows(self.a), rows(noisy)))
        self.assertTrue(soup_copies.is_twin(rows(self.a), rows(noisy), jsd_max=math.inf))


if __name__ == "__main__":
    unittest.main()
