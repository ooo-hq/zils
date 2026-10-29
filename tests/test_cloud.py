import unittest

from zils.cloud import MAX_DATA_BYTES


class DataLimitTest(unittest.TestCase):
    def test_a_training_split_may_reach_384_mib(self):
        # Claims' full eligibility train split is 281 MB (24,366 cases that each repeat their paper's abstract).
        self.assertEqual(MAX_DATA_BYTES, 384 * 1024**2)
        self.assertGreater(MAX_DATA_BYTES, 280_725_482)


if __name__ == "__main__":
    unittest.main()
