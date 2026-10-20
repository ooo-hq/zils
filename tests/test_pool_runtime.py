"""CPU regressions for heterogeneous pinned runtimes; no model weights or GPU qualification."""

import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch
from transformers import PretrainedConfig

from zils import d3, models


class PoolRuntimeTest(unittest.TestCase):
    def test_every_pool_initializes_its_own_lora_dimensions(self):
        for width, model in enumerate(models.D3_POOL_MODELS.values(), start=4):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                required = d3.pins(model)
                count = required.get("target_modules", d3.TARGET_MODULES)
                base = torch.nn.Module()
                base.language_model = torch.nn.Module()
                base.language_model.layers = torch.nn.ModuleList(
                    [
                        torch.nn.ModuleDict(
                            {"q_proj": torch.nn.Linear(width, width + 1, bias=False)}
                        )
                        for _ in range(count)
                    ]
                )
                base.to(torch.bfloat16)
                base.config = PretrainedConfig()
                base.config.text_config = PretrainedConfig()
                runtime = SimpleNamespace(
                    fast=None,
                    attention_mode=required["attention"],
                    temperature=1.0,
                    readout_dtype="float32",
                    prompt="d3",
                    backbone_type="qwen3_5",
                    max_length=4096,
                    backbone=base,
                    readout=torch.zeros(255, width),
                )
                native = SimpleNamespace(
                    D3=SimpleNamespace(from_pretrained=lambda *a, **k: runtime)
                )
                models.write_metadata(tmp, kind="base", model=model)
                with (
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.is_bf16_supported", return_value=True),
                    patch.object(
                        torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
                    ),
                    patch("zils.d3.verify_runtime"),
                    patch("zils.d3.base_path", return_value=Path(tmp)),
                    patch(
                        "zils.d3.Encoder",
                        return_value=SimpleNamespace(native=native, tokenizer=None),
                    ),
                ):
                    loaded = d3.DecisionModel(tmp, "cuda", train=True)
                parameters = [p for p in loaded.peft.parameters() if p.requires_grad]
                self.assertEqual(len(parameters), 2 * count)
                self.assertEqual(sum(p.numel() for p in parameters), count * 16 * (2 * width + 1))

    def test_new_encoder_handles_text_and_images_under_vendor_sdpa(self):
        model = d3.DecisionModel.__new__(d3.DecisionModel)
        model.device = "cpu"
        model.readout = torch.eye(3)
        embeds = torch.zeros(1, 2, 3, dtype=torch.bfloat16)
        output = embeds.clone()
        output[0, -1, 1] = 2
        model.base = Mock(side_effect=AssertionError("new runtime must use its encoder"))
        model.base.get_input_embeddings.return_value = lambda ids: embeds
        entered = []

        @contextlib.contextmanager
        def sdpa(device):
            entered.append(device)
            yield

        encoder = Mock(return_value=output)
        model.runtime = SimpleNamespace(
            encoder=encoder, image_embeddings=Mock(return_value=(embeds, "mrope"))
        )
        model.encoder = SimpleNamespace(native=SimpleNamespace(sdpa_backends=sdpa))
        self.assertEqual(model.logits([1, 2], 3).tolist(), [0, 2, 0])
        self.assertEqual(len(encoder.call_args.args), 2)
        batch = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2),
            "pixel_values": embeds,
        }
        self.assertEqual(model.logits(batch, 3).tolist(), [0, 2, 0])
        self.assertEqual(encoder.call_args.args[2], "mrope")
        self.assertEqual(entered, ["cpu", "cpu"])
        model.base.get_video_features.return_value = SimpleNamespace(pooler_output=[embeds[0] + 1])
        model.base.get_placeholder_mask.return_value = (
            None,
            torch.ones_like(embeds, dtype=torch.bool),
        )
        model.base.compute_3d_position_ids.return_value = "video-mrope"
        video = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values_videos": embeds,
            "video_grid_thw": torch.ones(1, 3),
            "mm_token_type_ids": torch.ones(1, 2),
        }
        self.assertEqual(model.logits(video, 3).tolist(), [0, 2, 0])
        self.assertEqual(encoder.call_args.args[2], "video-mrope")
        self.assertTrue(torch.equal(encoder.call_args.args[0], embeds + 1))

    def test_vendor_packages_and_late_siblings_are_isolated_mini_lite_mini(self):
        before = set(sys.modules)
        self.addCleanup(d3.vendor.cache_clear)
        self.addCleanup(
            lambda: [
                sys.modules.pop(k, None)
                for k in set(sys.modules) - before
                if k.startswith("_zils_d3_")
                or k in ("d3_runtime", "d3_format", "d3_fast", "d3_kernels")
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            roots = {}
            for name, model in (("mini", models.D3), ("lite", "d3-lite-0.8b-v1")):
                root = roots[model] = Path(tmp) / name
                root.mkdir()
                (root / "d3_format.py").write_text(f"NAME = {name!r}\n")
                (root / "d3_kernels.py").write_text(f"NAME = {name!r}\n")
                (root / "d3_fast.py").write_text(
                    "def late():\n    from . import d3_kernels\n    return d3_kernels.NAME\n"
                )
                (root / "d3_runtime.py").write_text(
                    "try:\n    from .d3_format import NAME\n    from . import d3_fast\nexcept ImportError:\n    from d3_format import NAME\n    import d3_fast\n"
                )
            d3.vendor.cache_clear()
            with patch("zils.d3.pinned_base_path", side_effect=lambda model: roots[model]):
                mini = d3.vendor(models.D3)
                lite = d3.vendor("d3-lite-0.8b-v1")
                self.assertIs(d3.vendor(models.D3), mini)
                self.assertEqual((mini.NAME, lite.NAME), ("mini", "lite"))
                self.assertEqual((mini.d3_fast.late(), lite.d3_fast.late()), ("mini", "lite"))
                self.assertNotIn(str(roots[models.D3]), sys.path)
