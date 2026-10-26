import argparse
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from zils import teacher

CHOICE = {"type": "choice", "instructions": "Pick.", "criteria": {"A": "one", "B": "two"}}
NOUL = {"type": "noul", "instructions": "True?"}


class FakeModel:
    device = "cpu"

    def __init__(self, logits):
        self.values = torch.tensor(logits)

    def encode(self, state, question):
        return [1, 2, 3], teacher.options(question)

    def logits(self, ids, count):
        return self.values[:count]


class TeacherTest(unittest.TestCase):
    def test_pool_rows_drop_labels_and_accept_both_shapes(self):
        state, question = teacher.pool_row(
            {"state": {"text": "x"}, "question": {**CHOICE, "label": "A"}}
        )
        self.assertEqual((state, question), ({"text": "x"}, CHOICE))
        _, nested = teacher.pool_row(
            {"state": "y", "questions": {"decision": {**NOUL, "label": "true"}}}
        )
        self.assertEqual(nested, NOUL)
        with self.assertRaises(ValueError):
            teacher.pool_row({"question": CHOICE})

    def test_teacher_probabilities_cover_declared_options(self):
        self.assertEqual(
            teacher.teacher_probabilities({"type": "noul", "noul": 0.25}, NOUL),
            {"false": 0.75, "true": 0.25},
        )
        self.assertEqual(
            teacher.teacher_probabilities({"probabilities": {"A": 0.6, "B": 0.4}}, CHOICE),
            {"A": 0.6, "B": 0.4},
        )
        for bad in ({"A": 1.0}, {"A": 0.6, "B": 0.6}, {"A": 0.5, "C": 0.5}):
            with self.assertRaises(ValueError):
                teacher.teacher_probabilities({"probabilities": bad}, CHOICE)

    def test_kl_is_zero_when_the_student_matches_and_positive_otherwise(self):
        target = {"A": 0.7310585786, "B": 0.2689414214}  # softmax of logits [1, 0]
        self.assertAlmostEqual(
            teacher.kl(FakeModel([1.0, 0.0]), {}, CHOICE, target).item(), 0, places=6
        )
        self.assertGreater(teacher.kl(FakeModel([0.0, 3.0]), {}, CHOICE, target).item(), 0.5)

    def test_soup_weights_must_be_a_distribution(self):
        self.assertEqual(
            teacher.parse_parts(["stock:0.5", "/a/b:0.5"]), [("stock", 0.5), ("/a/b", 0.5)]
        )
        for bad in (["stock:0.6", "x:0.6"], ["stock:-0.5", "x:1.5"], []):
            with self.assertRaises(ValueError):
                teacher.parse_parts(bad)

    def test_soup_averages_every_tensor_against_the_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in (("parent", 0.0), ("student", 4.0)):
                (root / name).mkdir()
                save_file(
                    {"w": torch.full((2, 2), value, dtype=torch.bfloat16)},
                    str(root / name / "model.safetensors"),
                )
            args = argparse.Namespace(
                out=str(root / "soup"), parts=["stock:0.75", f"{root / 'student'}:0.25"]
            )
            with (
                patch.object(teacher.d3, "pinned_base_path", return_value=root / "parent"),
                patch.object(teacher.d3, "weight_map", return_value={"w": "model.safetensors"}),
                patch.object(teacher, "copy_contract") as contract,
            ):
                teacher.soup(args)
            with safe_open(str(root / "soup" / "model.safetensors"), framework="pt") as stream:
                value = stream.get_tensor("w")
            self.assertEqual(value.dtype, torch.bfloat16)
            self.assertTrue(math.isclose(float(value[0, 0]), 1.0))
            contract.assert_called_once_with(root / "parent", root / "soup")


if __name__ == "__main__":
    unittest.main()
