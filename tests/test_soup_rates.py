"""Soup seat rates: house ingredients, scouts (re-uploads of known public models) and trainers."""

import json
import re
import unittest

import numpy as np

from zils import soup_copies, soup_rates

HOUSE_HASH = "1" * 64
KNOWN_HASH = "2" * 64
BODY_HASH = "3" * 64
HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
OTHER = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


class ConstantsTest(unittest.TestCase):
    def test_rates(self):
        self.assertEqual(
            (soup_rates.HOUSE_RATE, soup_rates.SCOUT_RATE, soup_rates.TRAINER_RATE),
            (0.0, 0.25, 1.0),
        )


class HouseRegistryTest(unittest.TestCase):
    def test_registry_file(self):
        house = soup_rates.load_house()
        self.assertEqual(
            [entry["repo"] for entry in house],
            [
                "vllm-sr/d3-mini",
                "perplexity-ai/pplx-decider-flash-v1.2",
                "torchcast-ai/torchcast-decision-4b",
                "Maincode/Matilda-Decision-v1-Mini",
                "PelaAI/KnowLine-4B-Gen5",
            ],
        )
        for entry in house:
            self.assertEqual(set(entry), {"name", "repo", "revision", "body_sha256"})
            self.assertRegex(entry["revision"], "^[a-f0-9]{40}$")
            sha = entry["body_sha256"]
            self.assertTrue(sha is None or re.fullmatch("[a-f0-9]{64}", sha))
        self.assertEqual(len({entry["name"] for entry in house}), len(house))

    def test_load_rejects_malformed(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "house.json"
            for value in (
                {"repo": "a/b"},
                [{"repo": "a/b"}],
                [{"name": "x", "repo": "a/b", "revision": "main", "body_sha256": None}],
            ):
                with self.subTest(value=value):
                    path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):
                        soup_rates.load_house(path)


class RateTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.known_sketch, self.fresh = rng.standard_normal((2, 512))
        self.house = [
            {"name": "h", "repo": "house/h", "revision": "a" * 40, "body_sha256": HOUSE_HASH}
        ]
        self.known = [
            {
                "body_sha256": KNOWN_HASH,
                "sketch": self.known_sketch,
                "hotkey": None,
                "repo": "public/model",
            }
        ]

    def rate(self, sha, sketch, **kwargs):
        options = {"house": self.house, "known": self.known, "hotkey": HOTKEY, **kwargs}
        return soup_rates.rate(sha, sketch, **options)

    def test_house_body_earns_nothing(self):
        self.assertEqual(self.rate(HOUSE_HASH, self.fresh), 0.0)
        self.assertEqual(self.rate(HOUSE_HASH, self.fresh, house=[HOUSE_HASH]), 0.0)

    def test_near_known_model_is_a_scout(self):
        near = self.known_sketch + 0.1 * self.fresh
        self.assertEqual(self.rate(BODY_HASH, near), 0.25)
        self.assertEqual(self.rate(KNOWN_HASH, self.known_sketch), 0.25)

    def test_new_body_is_a_trainer(self):
        self.assertEqual(self.rate(BODY_HASH, self.fresh), 1.0)
        self.assertEqual(self.rate(BODY_HASH, self.known_sketch, known=[]), 1.0)

    def test_only_the_submitters_own_earlier_bodies_are_not_known(self):
        own_hotkey = [{**self.known[0], "hotkey": HOTKEY}]
        self.assertEqual(self.rate(BODY_HASH, self.known_sketch, known=own_hotkey), 1.0)
        other_hotkey = [{**self.known[0], "hotkey": OTHER}]
        self.assertEqual(self.rate(BODY_HASH, self.known_sketch, known=other_hotkey), 0.25)

    def test_a_house_re_upload_is_a_scout_whatever_its_repo(self):
        # The house model's own repo (or anyone able to push to it) earns no trainer rate.
        with self.assertRaises(TypeError):
            self.rate(BODY_HASH, self.known_sketch, repo="public/model")
        self.assertEqual(self.rate(BODY_HASH, self.known_sketch), 0.25)

    def test_threshold_boundary(self):
        near = self.known_sketch + 0.5 * self.fresh
        cosine = soup_copies.cosine(near, self.known_sketch)
        self.assertEqual(self.rate(BODY_HASH, near, cosine_scout=cosine), 0.25)
        self.assertEqual(
            self.rate(BODY_HASH, near, cosine_scout=float(np.nextafter(cosine, 2))), 1.0
        )

    def test_house_entries_without_hash_are_ignored(self):
        house = [{"name": "h", "repo": "house/h", "revision": "a" * 40, "body_sha256": None}]
        self.assertEqual(self.rate(BODY_HASH, self.fresh, house=house), 1.0)


if __name__ == "__main__":
    unittest.main()
