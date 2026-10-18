"""d3-mini runtime: pins, weight-free encoder contract, FP32 readout and parity."""

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from zils import d3, models


class FakeBackbone(torch.nn.Module):
    def forward(self, input_ids, attention_mask, use_cache, **media):
        self.media = media
        hidden = torch.zeros(1, input_ids.shape[1], 255, dtype=torch.bfloat16)
        hidden[0, -1, 0] = 1
        return SimpleNamespace(last_hidden_state=hidden)


IMAGE, VIDEO = "<|image_pad|>", "<|video_pad|>"


def png(color=(0, 0, 0)):
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


class FakeProcessor:
    """Renders placeholders like the chat template and returns a processor batch of `length` ids."""

    def __init__(self, length=20):
        self.length, self.calls = length, []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return messages

    def __call__(self, text, return_tensors, **media):
        self.calls.append((text, media))
        ids = torch.ones(1, self.length, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": ids, "pixel_values": torch.zeros(2, 3)}


def media_encoder(processor):
    """An Encoder over a fake vendor: image_messages/video_messages render placeholders + state."""
    encoder = d3.Encoder.__new__(d3.Encoder)
    encoder.codes, encoder.root, encoder.processor = ["A", "B"], Path("/pinned"), processor
    encoder.native = SimpleNamespace(
        IMAGE_TOKEN=IMAGE,
        VIDEO_TOKEN=VIDEO,
        _json_value=json.dumps,
        normalize_question=lambda q: SimpleNamespace(keys=list(q["criteria"]), rendered=q),
        image_messages=lambda prompt, state, q, codes, n: IMAGE * n + json.dumps(state),
        video_messages=lambda prompt, state, q, codes, n, v: IMAGE * n + VIDEO * v + str(state),
        load_video=lambda path: SimpleNamespace(
            frames="frames:" + path, metadata=lambda: {"fps": 2}
        ),
        load_processor=Mock(side_effect=AssertionError("text prompts never load the processor")),
        render=lambda tokenizer, prompt, state, q, codes: "text",
    )
    encoder.tokenizer = lambda text, add_special_tokens: {"input_ids": [1, 2, 3]}
    return encoder


QUESTION = {"type": "choice", "criteria": {"a": "a", "b": "b"}}


SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
INDEX = "model.safetensors.index.json"


def weights(root, weight_map, tensors=None):
    """Tiny shards and an index; tensors[shard] overrides the keys a shard actually holds."""
    from safetensors.torch import save_file

    root.mkdir(exist_ok=True)
    (root / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    for shard in SHARDS:
        keys = (tensors or {}).get(shard, [k for k, v in weight_map.items() if v == shard])
        save_file({key: torch.zeros(1) for key in keys}, str(root / shard))


class D3ModuleTest(unittest.TestCase):
    def test_pins_and_constants(self):
        self.assertEqual(d3.MAX_TOKENS, 4096)
        self.assertEqual(d3.PINS["revision"], models.spec(models.D3)["base_revision"])
        self.assertEqual(len(d3.TARGETS), 9)

    def test_pinned_base_path_rejects_hash_mismatch(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"ZILS_D3_BASE_DIR": tmp}),
        ):
            for name in d3.PINS["files"]:
                (Path(tmp) / name).parent.mkdir(parents=True, exist_ok=True)
                (Path(tmp) / name).write_bytes(b"x")
            with self.assertRaisesRegex(OSError, "d3 base file .* differs from its pin"):
                d3.pinned_base_path()

    def test_verify_runtime_rejects_wrong_torchvision(self):
        with (
            patch(
                "zils.d3.importlib.metadata.version",
                side_effect=lambda p: "0.23.0" if p == "torchvision" else d3.PINS["packages"][p],
            ),
            patch("zils.d3.sys.version_info", (3, 12, 12)),
        ):
            with self.assertRaisesRegex(OSError, "torchvision"):
                d3.verify_runtime()

    def test_validate_adapter_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            # jevk5.train records the profile's base (the zils-ai mirror), not the vendor repo.
            good = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "modules_to_save": None,
                "target_modules": list(d3.TARGETS),
                "base_model_name_or_path": models.spec(models.D3)["base"],
                "revision": models.spec(models.D3)["base_revision"],
            }
            (Path(tmp) / "adapter_config.json").write_text(json.dumps(good))
            d3.validate_adapter_config(tmp, d3.TARGETS)
            (Path(tmp) / "adapter_config.json").write_text(
                json.dumps({**good, "revision": "0" * 40})
            )
            with self.assertRaisesRegex(ValueError, "revision"):
                d3.validate_adapter_config(tmp, d3.TARGETS)

    def test_logits_and_probabilities_use_fp32_readout(self):
        model = d3.DecisionModel.__new__(d3.DecisionModel)
        model.device = "cpu"
        model.readout = torch.eye(255, dtype=torch.float32)
        model.base = FakeBackbone().eval()
        self.assertEqual(model.logits([1, 2, 3], 3).dtype, torch.float32)
        probabilities = model.probabilities([1, 2, 3], ["a", "b", "c"], {"type": "choice"}, 1.0)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)
        self.assertEqual(max(probabilities, key=probabilities.get), "a")
        # A model left in training mode is a node fault (exit 78), not the miner's.
        model.base.train()
        with self.assertRaisesRegex(OSError, "eval mode"):
            model.probabilities([1, 2, 3], ["a", "b", "c"], {"type": "choice"}, 1.0)

    def test_setup_invariants_raise_oserror(self):
        runtime = SimpleNamespace(
            fast=None,
            attention_mode=d3.PINS["attention"],
            temperature=d3.PINS["temperature"],
            readout_dtype=d3.PINS["readout_dtype"],
            prompt="d3",
            backbone_type="qwen3_5",
            max_length=2048,
        )
        native = SimpleNamespace(D3=SimpleNamespace(from_pretrained=lambda *a, **k: runtime))
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=True),
            patch("zils.d3.verify_runtime"),
            patch("zils.d3.base_path", return_value=Path(tmp)),
            patch("zils.d3.Encoder", return_value=SimpleNamespace(native=native, tokenizer=None)),
        ):
            models.write_metadata(tmp, kind="base", model=models.D3)
            with self.assertRaisesRegex(OSError, "pinned configuration"):
                d3.DecisionModel(tmp, "cuda")

    def test_vendor_code_comes_from_the_pinned_base_only(self):
        with tempfile.TemporaryDirectory() as pinned, tempfile.TemporaryDirectory() as candidate:
            (Path(pinned) / "d3_runtime.py").write_text(
                "def answer_codes(tokenizer):\n    return ['A'], [1]\n"
            )
            (Path(candidate) / "d3_runtime.py").write_text("raise ImportError('candidate code')\n")
            (Path(candidate) / "decision_config.json").write_text(
                json.dumps({"prompt": "d3", "codes": ["A"]})
            )
            d3.vendor.cache_clear()
            self.addCleanup(d3.vendor.cache_clear)
            self.addCleanup(sys.modules.pop, "d3_runtime", None)
            self.addCleanup(lambda: sys.path.remove(pinned) if pinned in sys.path else None)
            with (
                patch("zils.d3.pinned_base_path", return_value=Path(pinned)),
                patch("transformers.AutoTokenizer.from_pretrained"),
            ):
                encoder = d3.Encoder(Path(candidate))
            self.assertEqual(Path(encoder.native.__file__).parent, Path(pinned))
            self.assertNotIn(candidate, sys.path)

    def test_encode_limits_and_canonical_png(self):
        encoder = d3.Encoder.__new__(d3.Encoder)  # refusals happen before any vendor or processor
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.mp4"
            clip.write_bytes(b"mp4")
            for message, images, videos in (
                ("at most 4 images", [png()] * 5, ()),
                ("at most 1 video", (), [str(clip)] * 2),
                ("canonical PNG", [b"GIF89a" + bytes(32)], ()),
                ("canonical PNG", [png()[:8] + b"not a png"], ()),
                ("canonical PNG", ["not bytes"], ()),
                ("regular MP4 file", (), [tmp]),
            ):
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    encoder.encode({}, QUESTION, images=images, videos=videos)

    def test_text_prompts_stay_processor_free(self):
        encoder = media_encoder(None)
        del encoder.processor
        self.assertEqual(encoder.encode({}, QUESTION), ([1, 2, 3], ["a", "b"]))
        batch, names = encoder.inputs({}, QUESTION)
        self.assertEqual((batch, names), ([1, 2, 3], ["a", "b"]))

    def test_image_and_video_prompts_go_through_the_processor(self):
        processor = FakeProcessor()
        encoder = media_encoder(processor)
        ids, names = encoder.encode({"n": 1}, QUESTION, images=[png(), png((9, 9, 9))])
        self.assertEqual((ids, names), ([1] * 20, ["a", "b"]))
        text, media = processor.calls[-1]
        self.assertEqual(text, [IMAGE * 2 + '{"n": 1}'])
        self.assertEqual([image.mode for image in media["images"]], ["RGB", "RGB"])
        self.assertEqual(media["images"][1].getpixel((0, 0)), (9, 9, 9))
        self.assertNotIn("videos", media)
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.mp4"
            clip.write_bytes(b"mp4")
            batch, _ = encoder.inputs("s", QUESTION, images=[png()], videos=[str(clip)])
            self.assertIn("pixel_values", batch)
            text, media = processor.calls[-1]
            self.assertEqual(text, [IMAGE + VIDEO + "s"])
            self.assertEqual(media["videos"], ["frames:" + str(clip)])
            self.assertEqual(media["video_metadata"], [{"fps": 2}])
        with self.assertRaisesRegex(ValueError, "placeholder"):
            encoder.encode(IMAGE, QUESTION, images=[png()])
        processor.length = 4097  # expanded image tokens count toward the cap; nothing is truncated
        with self.assertRaisesRegex(ValueError, "4096-token"):
            encoder.encode({}, QUESTION, images=[png()])

    def test_predict_runs_a_processor_batch_through_the_fp32_readout(self):
        model = d3.DecisionModel.__new__(d3.DecisionModel)
        model.device = "cpu"
        model.readout = torch.eye(255, dtype=torch.float32)
        model.base = FakeBackbone().eval()
        model.meta = {"temperature": 1.0}
        model.encoder = media_encoder(FakeProcessor())
        probabilities = model.predict({}, QUESTION, images=[png()])
        self.assertEqual(max(probabilities, key=probabilities.get), "a")
        self.assertEqual(model.base.media["pixel_values"].shape, (2, 3))
        self.assertEqual(model.predict({}, QUESTION).keys(), {"a", "b"})
        self.assertEqual(model.base.media, {})

    def test_cli_help_lists_commands(self):
        out = subprocess.run(
            [sys.executable, "-m", "zils.d3", "--help"], capture_output=True, text=True, check=True
        ).stdout
        for word in ("reference", "train", "parity"):
            self.assertIn(word, out)

    def candidate(self, root):
        """A pinned-base fixture and a candidate whose contract files equal the patched pins."""
        parent = {"a.weight": SHARDS[0], "b.weight": SHARDS[1]}
        weights(root / "base", parent)
        candidate = root / "candidate"
        candidate.mkdir()
        models.write_metadata(candidate, kind="candidate", model=models.D3)
        for name in models.D3_CONTRACT_FILES:
            (candidate / name).write_bytes(name.encode())
        weights(candidate, parent)
        pins = {name: hashlib.sha256(name.encode()).hexdigest() for name in d3.PINS["files"]}
        self.enterContext(patch.dict(d3.PINS["files"], pins))
        self.enterContext(patch("zils.d3.pinned_base_path", return_value=root / "base"))
        return candidate, parent

    def test_candidate_hashes_only_contract_files(self):
        with tempfile.TemporaryDirectory() as root:
            tmp, _ = self.candidate(Path(root))
            d3.validate_candidate(tmp)  # MODEL_MANIFEST.json, requirements.txt, d3_server.py absent
            for extra, make in (
                ("extra.py", lambda path: path.write_text("")),
                ("__pycache__", lambda path: path.symlink_to(Path(tmp) / "config.json")),
            ):
                make(Path(tmp) / extra)
                with self.assertRaisesRegex(ValueError, f"unexpected entry: {extra}"):
                    d3.validate_candidate(tmp)
                (Path(tmp) / extra).unlink()
            (Path(tmp) / "d3_runtime.py").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "d3_runtime.py differs"):
                d3.validate_candidate(tmp)

    def test_candidate_weights_must_load_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            candidate, parent = self.candidate(Path(root))
            d3.validate_candidate(candidate)
            for message, weight_map, tensors in (
                ("index keys differ.*c.weight", {**parent, "c.weight": SHARDS[0]}, None),
                ("outside the pinned shards", {**parent, "b.weight": "x/" + SHARDS[1]}, None),
                ("differ from its index.*c.weight", parent, {SHARDS[0]: ["a.weight", "c.weight"]}),
            ):
                with self.subTest(message=message):
                    weights(candidate, weight_map, tensors)
                    with self.assertRaisesRegex(ValueError, message):
                        d3.validate_candidate(candidate)

    def test_write_candidate_merges_the_adapter_into_the_parent_shards(self):
        from safetensors.torch import load_file, save_file

        module = "language_model.layers.0.self_attn.q_proj"
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            base, adapter = root / "base", root / "adapter"
            base.mkdir(), adapter.mkdir()
            torch.manual_seed(0)
            weight, other = torch.randn(4, 3).bfloat16(), torch.randn(2, 2).bfloat16()
            parent = {module + ".weight": SHARDS[0], "visual.other.weight": SHARDS[1]}
            (base / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": parent}))
            save_file({module + ".weight": weight}, str(base / SHARDS[0]))
            save_file({"visual.other.weight": other}, str(base / SHARDS[1]))
            for name in models.D3_CONTRACT_FILES:
                (base / name).write_bytes(name.encode())
            pins = {name: hashlib.sha256(name.encode()).hexdigest() for name in d3.PINS["files"]}
            self.enterContext(patch.dict(d3.PINS["files"], pins))
            self.enterContext(patch("zils.d3.pinned_base_path", return_value=base))
            self.enterContext(patch("zils.d3.TARGET_MODULES", 1))
            a, b = torch.randn(2, 3).bfloat16(), torch.randn(4, 2).bfloat16()
            prefix = "base_model.model." + module
            save_file(
                {prefix + ".lora_A.weight": a, prefix + ".lora_B.weight": b},
                str(adapter / "adapter_model.safetensors"),
            )
            spec = models.spec(models.D3)
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "r": 16,
                        "lora_alpha": 32,
                        "lora_dropout": 0.05,
                        "bias": "none",
                        "base_model_name_or_path": spec["base"],
                        "revision": spec["base_revision"],
                        "modules_to_save": None,
                        "target_modules": [module],
                    }
                )
            )
            models.write_metadata(adapter, model=models.D3)
            out = root / "candidate"
            provenance = d3.write_candidate(adapter, out)
            merged = load_file(str(out / SHARDS[0]))[module + ".weight"]
            expected = (weight.float() + 2 * (b.float() @ a.float())).bfloat16()
            self.assertTrue(torch.equal(merged, expected))
            self.assertTrue(
                torch.equal(load_file(str(out / SHARDS[1]))["visual.other.weight"], other)
            )
            self.assertEqual(
                sorted(p.name for p in out.iterdir()), sorted(models.D3_CANDIDATE_FILES)
            )
            self.assertEqual((out / "d3_runtime.py").read_bytes(), b"d3_runtime.py")
            self.assertEqual(models.metadata(out)["kind"], "candidate")
            self.assertEqual(provenance["merged_tensors"], 1)
            self.assertEqual(provenance["recipe"]["scaling"], 2.0)
            d3.validate_candidate(out)
            save_file(
                {
                    prefix + ".lora_A.weight": a,
                    prefix + ".lora_B.weight": b,
                    "visual.x.lora_A.weight": a.clone(),
                },
                str(adapter / "adapter_model.safetensors"),
            )
            with self.assertRaisesRegex(ValueError, "outside the pinned recipe"):
                d3.write_candidate(adapter, root / "candidate-2")

    def test_promoted_profiles_join_the_d3_family(self):
        """A Zils base promoted from d3 is its own profile: same runtime, own repo, pins and candidates."""
        from zils import image_contract

        r1 = "d3-mini-4.5b-zils-r1"
        spec = {
            **models.spec(models.D3),
            "id": r1,
            "base": "zils-ai/" + r1,
            "base_revision": "0" * 40,
            "parent": {"id": models.D3, "base_revision": models.spec(models.D3)["base_revision"]},
        }
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            self.enterContext(patch.dict(models.TEXT_RUNTIME_MODULES, {r1: ".d3"}))
            self.enterContext(patch.dict(models.SPECS, {r1: spec}))
            self.enterContext(patch.dict(models.PINS_FILES, {r1: "d3-r1-pins.json"}))
            self.enterContext(
                patch.dict(models.CANDIDATE_BASE_ENV, {r1: "ZILS_D3_R1_CANDIDATE_BASE"})
            )
            self.enterContext(
                patch.object(models, "TEXT_ADAPTER_MODELS", (*models.TEXT_ADAPTER_MODELS, r1))
            )
            parent = {"a.weight": SHARDS[0], "b.weight": SHARDS[1]}
            weights(root / "r1-base", parent)
            pins = {
                **d3.PINS,
                "files": {
                    n: hashlib.sha256(b"r1 " + n.encode()).hexdigest() for n in d3.PINS["files"]
                },
            }
            self.enterContext(patch.object(models, "pins", lambda model: pins))
            self.enterContext(
                patch(
                    "zils.d3.pinned_base_path",
                    lambda download=False, model=models.D3: root / "r1-base",
                )
            )
            self.assertEqual(models.d3_family(), (models.D3, r1))
            self.assertIn(r1, models.base_candidate_models())
            self.assertEqual(models.candidate_files(r1, "candidate"), models.D3_CANDIDATE_FILES)
            self.assertEqual(image_contract.capabilities(r1)["max_images"], 4)
            candidate = root / "candidate"
            candidate.mkdir()
            models.write_metadata(candidate, kind="candidate", model=r1)
            for name in models.D3_CONTRACT_FILES:
                (candidate / name).write_bytes(b"r1 " + name.encode())
            weights(candidate, parent)
            d3.validate_candidate(candidate, r1)
            with self.assertRaisesRegex(ValueError, "candidate metadata"):
                d3.validate_candidate(candidate, models.D3)
            with self.assertRaisesRegex(ValueError, "not a d3 profile"):
                d3.profile(models.H2O)

    def test_parity_reports_and_rejects_drift(self):
        reference = [0.7, 0.3]
        model = SimpleNamespace(
            encode=lambda state, question: ([1, 2], ["a", "b"]),
            probabilities=lambda ids, keys, question, t: dict(zip(keys, [0.7 + 4e-7, 0.3 - 4e-7])),
            runtime=SimpleNamespace(probabilities=lambda sequences, counts: [reference]),
        )
        case = {"id": "c1", "state": {}, "question": {"type": "choice"}}
        result = d3.parity(model, [case, case])
        self.assertEqual(result["cases"], 2)
        self.assertAlmostEqual(result["max_difference"], 4e-7)
        reference[:] = [0.3, 0.7]
        with self.assertRaisesRegex(RuntimeError, "c1"):
            d3.parity(model, [case])

    def test_capacity_floor_and_cuda_requirement(self):
        from zils.runtime import gpu_ready

        self.assertFalse(gpu_ready("mps", model=models.D3, minimum_mib=1))
        self.assertFalse(gpu_ready("cpu", model=models.D3))
        with patch("zils.runtime.subprocess.run", return_value=SimpleNamespace(stdout="12288")):
            self.assertTrue(gpu_ready("cuda", model=models.D3))
        with patch("zils.runtime.subprocess.run", return_value=SimpleNamespace(stdout="12287")):
            self.assertFalse(gpu_ready("cuda", model=models.D3))
        with (
            patch.dict(os.environ, {"ZILS_D3_MIN_FREE_MIB": "14000"}),
            patch("zils.runtime.subprocess.run", return_value=SimpleNamespace(stdout="13000")),
        ):
            self.assertFalse(gpu_ready("cuda", model=models.D3))

    def test_validate_inputs_reads_image_rows_under_the_dataset_root(self):
        encoder = SimpleNamespace(calls=[])
        encoder.encode = lambda state, q, images=(): encoder.calls.append(images) or ([1] * 5, [])
        rows = {"train": [{"state": {}, "question": QUESTION, "images": ["photos/a.png"]}]}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("zils.d3.base_path"),
            patch("zils.d3.Encoder", return_value=encoder),
            patch(
                "zils.miner_grading.workload_profile", side_effect=lambda lengths, model: lengths
            ),
        ):
            with self.assertRaisesRegex(ValueError, "image rows need a dataset root"):
                d3.validate_inputs(rows)
            (Path(tmp) / "photos").mkdir()
            (Path(tmp) / "photos/a.png").write_bytes(png())
            self.assertEqual(d3.validate_inputs(rows, root=tmp), [5])
            self.assertEqual(encoder.calls, [[png()]])
            rows["train"][0]["images"] = ["../a.png"]
            with self.assertRaisesRegex(ValueError, "inside"):
                d3.validate_inputs(rows, root=tmp)

    def test_runner_reads_image_bytes_and_passes_media_only_when_present(self):
        from zils import jevk5_runner

        calls = []

        class Model:
            meta = {"temperature": 1.0}

            def __init__(self, checkpoint, device):
                pass

            def predict(self, state, question, **media):
                calls.append(media)
                return {"a": 0.5, "b": 0.5}

        spec = models.spec(models.D3)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.png").write_bytes(png())
            rows = [
                {"id": "text", "state": {}, "question": QUESTION},
                {"id": "media", "state": {}, "question": QUESTION, "images": [f"{tmp}/a.png"]},
                {"id": "clip", "state": {}, "question": QUESTION, "videos": [f"{tmp}/c.mp4"]},
            ]
            argv = ["runner", "--checkpoint", tmp, "--device", "cuda"]
            argv += ["--base", spec["base"], "--base-revision", spec["base_revision"]]
            for requests in (rows, [{**rows[0], "label": "a"}]):
                with (
                    patch.object(sys, "argv", argv),
                    patch("sys.stdin", io.StringIO(json.dumps(requests))),
                    patch("zils.models.checkpoint_model", return_value=models.D3),
                    patch(
                        "zils.models.text_runtime",
                        return_value=SimpleNamespace(DecisionModel=Model),
                    ),
                    contextlib.redirect_stdout(io.StringIO()) as out,
                ):
                    if requests is rows:
                        jevk5_runner.main()
                    else:
                        with self.assertRaisesRegex(ValueError, "only id, state, question"):
                            jevk5_runner.main()
                if requests is rows:
                    predictions = json.loads(out.getvalue())["predictions"]
                    self.assertEqual([p["id"] for p in predictions], ["text", "media", "clip"])
        self.assertEqual(calls, [{}, {"images": [png()]}, {"videos": [f"{tmp}/c.mp4"]}])


class FakeAdapter(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.register_parameter(name, torch.nn.Parameter(torch.zeros(1)))


class Stop(Exception):
    pass


class TrainingRowsTest(unittest.TestCase):
    def write(self, root, images):
        row = {"state": {}, "questions": {"decision": {**QUESTION, "label": "a"}}}
        (root / "train.jsonl").write_text(json.dumps({**row, "images": images}) + "\n")
        return SimpleNamespace(
            data=str(root / "train.jsonl"),
            reference="ref",
            device="cpu",
            seed=1,
            out=str(root / "o"),
        )

    def test_text_profiles_refuse_image_rows_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp, patch("zils.h2o.DecisionModel") as loaded:
            from zils.jevk5 import train

            args = self.write(Path(tmp), ["a.png"])
            with self.assertRaisesRegex(
                ValueError, "h2o-lightning-4b-v1.2.3 does not accept images"
            ):
                train(args, model_id=models.H2O)
            loaded.assert_not_called()

    def test_image_paths_must_stay_inside_the_training_directory(self):
        from zils.jevk5 import train

        with tempfile.TemporaryDirectory() as tmp, patch("zils.d3.DecisionModel") as loaded:
            for path in ("../a.png", "/etc/hosts", "missing.png", 3):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "inside"):
                    train(self.write(Path(tmp), [path]), model_id=models.D3)
            loaded.assert_not_called()

    def test_media_limits_are_refused_before_loading(self):
        from zils.jevk5 import train

        with tempfile.TemporaryDirectory() as tmp, patch("zils.d3.DecisionModel") as loaded:
            (Path(tmp) / "a.png").write_bytes(png())
            (Path(tmp) / "a.gif").write_bytes(b"GIF89a" + bytes(32))
            for message, images in (
                ("at most 4 images", ["a.png"] * 5),
                ("canonical PNG", ["a.gif"]),
            ):
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    train(self.write(Path(tmp), images), model_id=models.D3)
            loaded.assert_not_called()

    def model(self, parameter):
        batch = {"input_ids": torch.ones(1, 3, dtype=torch.long)}
        model = SimpleNamespace(
            peft=FakeAdapter(parameter),
            base=SimpleNamespace(gradient_checkpointing_enable=lambda **kwargs: None),
            seen=[],
        )
        model.encoder = SimpleNamespace(
            inputs=lambda state, q, images: model.seen.append(images) or (batch, ["a", "b"])
        )

        def logits(inputs, count):
            model.seen.append((inputs, count))
            raise Stop

        model.logits = logits
        return model, batch

    def test_image_rows_reach_logits_as_a_processor_batch(self):
        from zils.jevk5 import train

        model, batch = self.model("lora_A")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("zils.d3.DecisionModel", return_value=model),
        ):
            (Path(tmp) / "photos").mkdir()
            (Path(tmp) / "photos/a.png").write_bytes(png())
            with self.assertRaises(Stop):
                train(self.write(Path(tmp), ["photos/a.png"]), model_id=models.D3)
        self.assertEqual(model.seen, [[png()], (batch, 2)])

    def test_frozen_export_references_resolve_to_the_cached_training_images(self):
        import hashlib

        from zils.jevk5 import train

        model, batch = self.model("lora_A")
        sha = hashlib.sha256(png()).hexdigest()
        reference = {"asset_id": "10000000-0000-4000-8000-000000000001", "sha256": sha}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("zils.d3.DecisionModel", return_value=model),
        ):
            (Path(tmp) / "images").mkdir()
            (Path(tmp) / "images" / (sha + ".png")).write_bytes(png())
            with self.assertRaises(Stop):
                train(self.write(Path(tmp), [reference]), model_id=models.D3)
            with self.assertRaisesRegex(ValueError, "frozen training image reference"):
                train(self.write(Path(tmp), [{"sha256": sha}]), model_id=models.D3)
        self.assertEqual(model.seen, [[png()], (batch, 2)])

    def test_training_refuses_a_trainable_vision_tower(self):
        from zils.jevk5 import train

        model, _ = self.model("visual_lora_A")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("zils.d3.DecisionModel", return_value=model),
        ):
            (Path(tmp) / "a.png").write_bytes(png())
            with self.assertRaisesRegex(RuntimeError, "vision tower"):
                train(self.write(Path(tmp), ["a.png"]), model_id=models.D3)


if __name__ == "__main__":
    unittest.main()
