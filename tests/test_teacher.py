import argparse
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from zils import models, teacher

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

    def test_start_loads_a_validated_candidate_into_the_backbone(self):
        with tempfile.TemporaryDirectory() as tmp:
            king = Path(tmp)
            save_file(
                {"weight": torch.ones(2, 2), "bias": torch.zeros(2)},
                str(king / "model.safetensors"),
            )
            model = argparse.Namespace(base=torch.nn.Linear(2, 2))
            with (
                patch.object(teacher.d3, "validate_candidate") as validate,
                patch.object(
                    teacher.d3, "weight_map", return_value={"weight": "model.safetensors"}
                ),
            ):
                teacher.load_candidate(model, king)
            validate.assert_called_once()
            self.assertTrue(torch.equal(model.base.weight, torch.ones(2, 2)))

    def test_soup_weights_must_be_a_distribution(self):
        self.assertEqual(
            teacher.parse_parts(["stock:0.5", "/a/b:0.5"]), [("stock", 0.5), ("/a/b", 0.5)]
        )
        for bad in (["stock:0.6", "x:0.6"], ["stock:-0.5", "x:1.5"], []):
            with self.assertRaises(ValueError):
                teacher.parse_parts(bad)

    def soup_parts(self, root, student_readout=0.0):
        for name, value, readout in (("parent", 0.0, 0.0), ("student", 4.0, student_readout)):
            (root / name).mkdir()
            save_file(
                {"w": torch.full((2, 2), value, dtype=torch.bfloat16)},
                str(root / name / "model.safetensors"),
            )
            save_file({"r": torch.full((3, 2), readout)}, str(root / name / "readout.safetensors"))

    def test_soup_averages_trained_answer_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.soup_parts(root, student_readout=2.0)
            args = argparse.Namespace(
                out=str(root / "soup"),
                parts=["stock:0.5", f"{root / 'student'}:0.5"],
                model="d3-mini",
            )
            with (
                patch.object(teacher.d3, "pinned_base_path", return_value=root / "parent"),
                patch.object(teacher.d3, "weight_map", return_value={"w": "model.safetensors"}),
                patch.object(teacher, "copy_contract") as contract,
            ):
                teacher.soup(args)
            readout = contract.call_args.args[2]
            self.assertTrue(torch.equal(readout, torch.full((3, 2), 1.0)))

    def test_soup_averages_every_tensor_against_the_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.soup_parts(root)
            args = argparse.Namespace(
                out=str(root / "soup"),
                parts=["stock:0.75", f"{root / 'student'}:0.25"],
                model="d3-mini",
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
            contract.assert_called_once_with(root / "parent", root / "soup", profile=models.D3)

    def test_soup_resolves_a_pool_name_to_its_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.soup_parts(root)
            args = argparse.Namespace(
                out=str(root / "soup"),
                parts=["stock:0.5", f"{root / 'student'}:0.5"],
                model="d3-lite",
            )
            with (
                patch.object(teacher.d3, "pinned_base_path", return_value=root / "parent") as base,
                patch.object(teacher.d3, "weight_map", return_value={"w": "model.safetensors"}),
                patch.object(teacher, "copy_contract") as contract,
            ):
                teacher.soup(args)
            base.assert_called_once_with(model="d3-lite-0.8b-v1")
            self.assertEqual(contract.call_args.kwargs["profile"], "d3-lite-0.8b-v1")

    def test_trained_answer_layer_is_written_then_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, out = Path(tmp) / "parent", Path(tmp) / "out"
            parent.mkdir(), out.mkdir()
            for name in models.D3_CONTRACT_FILES:
                (parent / name).write_bytes(b"x")
            save_file({"readout": torch.zeros(3, 2)}, str(parent / "readout.safetensors"))
            with patch.object(teacher.d3, "validate_candidate") as validate:
                teacher.copy_contract(parent, out, torch.ones(3, 2, dtype=torch.float64))
            with safe_open(str(out / "readout.safetensors"), framework="pt") as stream:
                value = stream.get_tensor("readout")
            self.assertEqual((value.dtype, float(value.sum())), (torch.float32, 6.0))
            validate.assert_called_once_with(out, models.D3)


if __name__ == "__main__":
    unittest.main()
