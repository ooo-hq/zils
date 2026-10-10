"""H2O profile isolation, native calibration, and mixed-precision training readout."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from zils import models
from zils.h2o import DecisionModel, validate_adapter_config
from zils.miner_grading import trainer_identity, workload_profile


class H2OTest(unittest.TestCase):
    def test_saved_peft_recipe_retains_exact_language_scope(self):
        import torch
        from transformers import PretrainedConfig

        from zils.jevk5 import save_adapter_weights

        base = torch.nn.Module()
        base.config = PretrainedConfig()
        base.model = torch.nn.Module()
        base.model.language_model = torch.nn.Module()
        base.model.language_model.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        name: torch.nn.Linear(2, 2)
                        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                    }
                )
                for _ in range(38)
            ]
        )
        base.model.visual = torch.nn.Linear(2, 2)
        base.lm_head = torch.nn.Linear(2, 255, bias=False)
        targets = {name for name, _ in base.named_modules() if name.endswith("_proj")}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=True),
            patch("zils.h2o.verify_runtime"),
            patch("zils.h2o.base_path", return_value=Path(tmp)),
            patch(
                "zils.h2o.Encoder",
                return_value=SimpleNamespace(label_ids=list(range(255)), tokenizer=None),
            ),
            patch(
                "transformers.Qwen3_5ForConditionalGeneration.from_pretrained",
                return_value=(base, {}),
            ),
        ):
            models.write_metadata(tmp, model=models.H2O, kind="base")
            model = DecisionModel(tmp, "cuda", train=True)
            output = Path(tmp) / "adapter"
            save_adapter_weights(model.peft, output)
            saved = json.loads((output / "adapter_config.json").read_text())
            self.assertEqual(set(saved["target_modules"]), targets)
            self.assertEqual(len(targets), 152)
            self.assertFalse(any(p.requires_grad for p in base.model.visual.parameters()))

    def test_h2o_capacity_floor_and_cuda_requirement(self):
        from zils.runtime import gpu_ready

        self.assertFalse(gpu_ready("mps", model=models.H2O, minimum_mib=1))
        self.assertFalse(gpu_ready("cpu", model=models.H2O))
        with patch("zils.runtime.subprocess.run", return_value=SimpleNamespace(stdout="12288")):
            self.assertTrue(gpu_ready("cuda", model=models.H2O))
            self.assertFalse(gpu_ready("cuda", model=models.H2O, minimum_mib=14000))
        with patch("zils.runtime.subprocess.run", return_value=SimpleNamespace(stdout="12287")):
            self.assertFalse(gpu_ready("cuda", model=models.H2O))

    def test_pin_and_checkpoint_cannot_be_relabelled(self):
        self.assertEqual(models.DEFAULT_TRAINING_MODEL, models.H2O)
        self.assertEqual(
            models.spec(models.H2O)["base_revision"], "acaf0d4ea251e54de928c75ef4352670d33192d3"
        )
        self.assertNotEqual(
            models.profile_identity(models.H2O), models.profile_identity(models.JEVK5)
        )
        self.assertNotEqual(trainer_identity(models.H2O), trainer_identity(models.JEVK5))
        self.assertEqual(models.job_model({}), models.KEV)
        self.assertEqual(workload_profile([2048], model=models.H2O)["model"], models.H2O)
        with tempfile.TemporaryDirectory() as tmp:
            models.write_metadata(tmp, model=models.H2O, kind="base")
            self.assertEqual(models.artifact_files(tmp), ("model.json",))
            models.set_temperature(tmp, 0.5)
            self.assertEqual(models.metadata(tmp)["profile"], models.spec(models.H2O))
            path = Path(tmp) / "model.json"
            path.chmod(0o600)
            value = json.loads(path.read_text())
            value["profile"]["base_revision"] = models.spec(models.JEVK5)["base_revision"]
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "pinned"):
                models.metadata(tmp)

    def test_adapter_recipe_rejects_wrong_base_and_extra_targets(self):
        profile = models.spec(models.H2O)
        valid = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "peft_type": "LORA",
            "base_model_name_or_path": profile["base"],
            "revision": profile["base_revision"],
            "target_modules": ["language.q_proj"],
            "modules_to_save": None,
            "use_dora": False,
            "use_rslora": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter_config.json"
            path.write_text(json.dumps(valid))
            validate_adapter_config(tmp, valid["target_modules"])
            for changes in (
                {"revision": "0" * 40},
                {"base_model_name_or_path": "alibiserikbay/JevK5"},
                {"target_modules": ["language.q_proj", "visual.q_proj"]},
                {"modules_to_save": ["lm_head"]},
                {"use_dora": True},
            ):
                path.write_text(json.dumps({**valid, **changes}))
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "recipe"):
                    validate_adapter_config(tmp, valid["target_modules"])

    def test_readout_has_fp32_gradients(self):
        import torch

        hidden = torch.tensor([[[0.25, -0.5]]], dtype=torch.bfloat16, requires_grad=True)
        model = DecisionModel.__new__(DecisionModel)
        model.device = "cpu"
        model.base = SimpleNamespace(
            model=lambda **kwargs: SimpleNamespace(last_hidden_state=hidden)
        )
        model.readout = torch.tensor([[0.2, -0.7], [0.4, 0.1]], dtype=torch.float32)
        logits = model.logits([1], 2)
        self.assertEqual(logits.dtype, torch.float32)
        torch.nn.functional.cross_entropy(logits[None], torch.tensor([1])).backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(hidden.grad.abs().max().item(), 0)

    def test_native_noul_order_and_scalar_calibration(self):
        import torch

        native = SimpleNamespace(
            probabilities=lambda logits, t: torch.softmax(
                torch.tensor(logits, dtype=torch.float64) / t, 0
            ).tolist(),
            commit_noul=lambda yes, floor: max(yes, floor) if yes >= 0.5 else min(yes, 1 - floor),
        )
        model = DecisionModel.__new__(DecisionModel)
        model.encoder = SimpleNamespace(
            native=native,
            contract=SimpleNamespace(
                temperature_by_type={"choice": 0.75, "noul": 0.8, "score": 0.65}, noul_floor=0.801
            ),
        )
        model.logits = lambda ids, count: torch.tensor([0.1, 0.0])
        with patch("torch.cuda.synchronize"):
            raw = model.probabilities([1], ["true", "false"], {"type": "noul"}, 1.0)
            calibrated = model.probabilities([1], ["true", "false"], {"type": "noul"}, 2.0)
            self.assertAlmostEqual(raw["true"], 0.801)
            self.assertAlmostEqual(calibrated["true"], 0.801**0.5 / (0.801**0.5 + 0.199**0.5))
            choice = model.probabilities([1], ["a", "b"], {"type": "choice"}, 1.0)
            self.assertLess(choice["a"], 0.801)
