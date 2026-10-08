"""Image continuation must use the published checkpoint and correct gradients."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from zils import imajev, models


class TrainingTest(unittest.TestCase):
    def test_partial_accumulation_is_the_mean_of_the_actual_two_examples(self):
        groups = imajev.accumulation_batches(list(range(6)), 4)
        self.assertEqual([len(x) for x in groups], [4, 2])
        parameter = torch.tensor(1.5, requires_grad=True)
        imajev.accumulate_gradients(groups[-1], lambda x: (parameter * x - 2).square())
        # Independent derivative of ((4*w-2)^2 + (5*w-2)^2)/2 at w=1.5.
        self.assertAlmostEqual(parameter.grad.item(), 4 * (4 * 1.5 - 2) + 5 * (5 * 1.5 - 2))
        with self.assertRaises(ValueError):
            imajev.accumulation_batches([1], 0)

    def test_only_float32_language_lora_and_head_are_trainable(self):
        model = torch.nn.Module()
        model.language_model = torch.nn.Module()
        model.language_model.lora_A = torch.nn.Linear(2, 1, bias=False)
        model.visual = torch.nn.Linear(2, 2, bias=False)
        model.visual.requires_grad_(False)
        head = torch.nn.Linear(2, 1, bias=False)
        engine = SimpleNamespace(model=model, readout=head)
        self.assertEqual(len(imajev.trainable_parameters(engine, expected_count=4)), 2)
        model.visual.requires_grad_(True)
        with self.assertRaises(ValueError):
            imajev.trainable_parameters(engine, expected_count=8)
        model.visual.requires_grad_(False)
        head.double()
        with self.assertRaises(ValueError):
            imajev.trainable_parameters(engine, expected_count=4)

    def test_nonfinite_or_wrong_shape_checkpoint_tensors_fail_closed(self):
        expected = {"language_model.lora_A.weight": torch.zeros(2, 3)}
        imajev.validate_tensors(expected, expected)
        for actual in (
            {},
            {**expected, "extra": torch.zeros(1)},
            {"language_model.lora_A.weight": torch.zeros(3, 2)},
            {"language_model.lora_A.weight": torch.full((2, 3), float("nan"))},
            {"language_model.lora_A.weight": torch.zeros(2, 3).half()},
        ):
            with self.subTest(actual=list(actual)), self.assertRaises(ValueError):
                imajev.validate_tensors(actual, expected)

    def test_image_reference_cannot_be_an_empty_base_or_experimental_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaises(ValueError):
                models.write_metadata(path, kind="base", model=models.IMAJEV)
            models.write_metadata(path, model=models.IMAJEV)
            with self.assertRaises(ValueError):
                imajev.verify_starting_checkpoint(path)
