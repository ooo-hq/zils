"""Device-independent readout and opt-in real Apple GPU training compatibility."""

import gc
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from zils import models
from zils.jevk5 import DecisionModel, train


class ReadoutDeviceTest(unittest.TestCase):
    def test_readout_places_tokens_and_last_position_on_the_model_device(self):
        import torch

        # A CPU readout isolates device routing without loading the 4B checkpoint.
        # Public model loading continues to require a supported GPU.
        embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )

        def slot_logits(tokens, last):
            return embedding(tokens)[torch.arange(tokens.shape[0]), last]

        model = DecisionModel.__new__(DecisionModel)
        model.device = "cpu"
        model.runtime = SimpleNamespace(_slot_logits=slot_logits)
        self.assertEqual(model.logits([0, 1], 2).tolist(), [4.0, 5.0])


@unittest.skipUnless(
    os.environ.get("ZILS_TEST_JEVK5_MPS_REFERENCE"),
    "set ZILS_TEST_JEVK5_MPS_REFERENCE for real Mac training verification",
)
class MacTrainingTest(unittest.TestCase):
    def test_training_exports_finite_bf16_adapter_and_reloads_on_mps(self):
        import torch
        from safetensors.torch import load_file

        self.assertTrue(torch.backends.mps.is_available())
        reference = Path(os.environ["ZILS_TEST_JEVK5_MPS_REFERENCE"])
        question = {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {"billing": "Payments and refunds", "access": "Login problems"},
        }
        samples = [
            ("Please refund the duplicate charge.", "billing"),
            ("My password no longer works.", "access"),
            ("I need a copy of my invoice.", "billing"),
            ("My account is locked and I cannot sign in.", "access"),
        ]
        with tempfile.TemporaryDirectory(prefix="zils-mps-training-") as tmp:
            root = Path(tmp)
            data, output = root / "train.jsonl", root / "adapter"
            data.write_text(
                "".join(
                    json.dumps(
                        {
                            "state": text,
                            "questions": {"decision": {**question, "label": label}},
                        }
                    )
                    + "\n"
                    for text, label in samples
                )
            )
            train(
                SimpleNamespace(
                    data=str(data),
                    reference=str(reference),
                    device="mps",
                    seed=553,
                    out=str(output),
                )
            )
            weights = load_file(str(output / "adapter_model.safetensors"))
            self.assertTrue(weights)
            self.assertTrue(all(t.dtype == torch.bfloat16 for t in weights.values()))
            self.assertTrue(all(torch.isfinite(t).all() for t in weights.values()))
            self.assertTrue(any(t.count_nonzero() for k, t in weights.items() if "lora_B" in k))
            self.assertEqual(models.metadata(output)["kind"], "adapter")
            metrics = json.loads((output / "training_metrics.json").read_text())
            self.assertEqual(metrics["optimizer_steps"], 1)
            self.assertEqual(metrics["examples"], 4)
            gc.collect()
            torch.mps.empty_cache()
            model = DecisionModel(output, "mps")
            probabilities = model.predict(samples[0][0], question)
            self.assertEqual(set(probabilities), {"billing", "access"})
            self.assertAlmostEqual(sum(probabilities.values()), 1.0)
            self.assertTrue(all(0 <= p <= 1 for p in probabilities.values()))
            del model
            gc.collect()
            torch.mps.empty_cache()
