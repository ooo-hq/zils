import unittest

from zils import d3


class MPSTrainingTest(unittest.TestCase):
    def test_only_lite_short_text_is_accepted(self):
        d3.check_mps_training("d3-lite-0.8b-v1", list(range(512)))
        for model, inputs in [
            ("d3-mini-4.5b-v1", [1]),
            ("d3-lite-0.8b-v1", list(range(513))),
            ("d3-lite-0.8b-v1", {"input_ids": [1]}),
        ]:
            with self.subTest(model=model), self.assertRaises(ValueError):
                d3.check_mps_training(model, inputs)


if __name__ == "__main__":
    unittest.main()
