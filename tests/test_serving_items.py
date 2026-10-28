import unittest

from zils import benchmark
from zils.decisions import validate_request
from zils.serving_items import items, round_seed


class ServingItemsTest(unittest.TestCase):
    def test_deterministic_and_count(self):
        self.assertEqual(items(7, 12), items(7, 12))
        self.assertEqual(len(items(7, 12)), 12)
        self.assertNotEqual(items(7, 12), items(8, 12))

    def test_different_miners_get_different_items(self):
        a, b = (round_seed("v", "r1", m, "z3-mini") for m in ("m1", "m2"))
        self.assertNotEqual(a, b)
        self.assertNotEqual(items(a, 8), items(b, 8))

    def test_round_seed_formula(self):
        import hashlib

        want = int(hashlib.sha256(b"v|r|m|x").hexdigest()[:16], 16)
        self.assertEqual(round_seed("v", "r", "m", "x"), want)

    def test_valid_and_clean(self):
        for item in items(1, 200):
            self.assertEqual(item["state"]["note"], "")
            validate_request({**item, "model": "z3-mini"})

    def test_all_families_appear(self):
        seen = set()
        for item in items(3, 200):
            (q,) = item["questions"].values()
            seen.add(q["type"] + ("-" + q["instructions"][:6]))
        types = {s.split("-")[0] for s in seen}
        self.assertEqual(types, {"noul", "choice"})  # score questions came only from severity
        text = " ".join(next(iter(i["questions"].values()))["instructions"] for i in items(3, 200))
        for marker in ("approve exactly", "Select the team", "supports or contradicts"):
            self.assertIn(marker, text)

    def test_severity_family_is_never_served(self):
        # 28 Oct 2026 calibration: every model size answers severity items >= 0.99 sure, so they carry no signal.
        for item in items(3, 400):
            (q,) = item["questions"].values()
            self.assertNotIn("level using these exact rules", q["instructions"])
        self.assertEqual(len(benchmark.ORACLES), 4)

    def test_golden_digest(self):
        # Auditors regenerate items from the seed; zils/benchmark.py is part of that contract.
        import hashlib
        import json

        got = hashlib.sha256(json.dumps(items(0, 20), sort_keys=True).encode()).hexdigest()
        self.assertEqual(got, "1d14f1e94185feb57492a9404581a5bce1b6fe623ac3054e69c24157f1a54d8e")


if __name__ == "__main__":
    unittest.main()
