"""Option 5 base candidates: full H2O weights with the parent's contract files, at the artifact interface."""

import contextlib
import hashlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import zils
from zils import models, protocol
from zils.h2o import PINS, DecisionModel, validate_candidate

CONTRACT = {name: b"contract " + name.encode() for name in models.H2O_CONTRACT_FILES}
PARENT_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "transformers_version": "4.57.0.dev0",
    "text_config": {"hidden_size": 2560, "num_hidden_layers": 32, "transformers_version": "5.15.0"},
    "vision_config": {"model_type": "qwen3_5"},
}


def candidate(root, config=PARENT_CONFIG, weights=b"weights"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, data in CONTRACT.items():
        (root / name).write_bytes(data)
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.safetensors").write_bytes(weights)
    models.write_metadata(root, kind="candidate", model=models.H2O)
    return root


def pins_for_fixture():
    files = dict(PINS["files"])
    files.update({name: hashlib.sha256(data).hexdigest() for name, data in CONTRACT.items()})
    return {**PINS, "files": files}


class BaseCandidateTest(unittest.TestCase):
    def test_metadata_files_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = candidate(Path(tmp) / "candidate")
            self.assertEqual(models.metadata(root)["kind"], "candidate")
            self.assertEqual(models.artifact_files(root), models.CANDIDATE_FILES)
            self.assertEqual(
                models.candidate_files(models.H2O, "candidate"), models.CANDIDATE_FILES
            )
            self.assertEqual(models.artifact_budget("candidate"), 10 * 1024**3)
            self.assertEqual(models.artifact_budget("adapter"), zils.MAX_ARTIFACT_BYTES)
            self.assertEqual(models.artifact_kind(root), "candidate")
            digest = zils.checkpoint_hash(root)
            (root / "model.safetensors").write_bytes(b"other weights")
            self.assertNotEqual(zils.checkpoint_hash(root), digest)
            (root / "adapter_model.safetensors").write_bytes(b"lora")
            with self.assertRaisesRegex(ValueError, "mixes"):
                models.artifact_files(root)
            with self.assertRaisesRegex(ValueError, "H2O and d3 profiles only"):
                models.candidate_files(models.IMAJEV, "candidate")
            (Path(tmp) / "jev").mkdir()
            with self.assertRaisesRegex(ValueError, "H2O and d3 profiles only"):
                models.write_metadata(Path(tmp) / "jev", kind="candidate", model=models.IMAJEV)

    def test_download_transfers_every_candidate_file_under_its_own_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = candidate(root / "source")
            requested = []

            class Files(protocol.Handler):
                def do_GET(self):
                    name = self.path.rsplit("/", 1)[-1]
                    requested.append(name)
                    data = (source / name).read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

            with protocol.local_server(Files) as endpoint:
                claim = {"endpoint": endpoint, "sha256": zils.checkpoint_hash(source)}
                protocol.fetch_checkpoint(
                    claim, root / "download", model=models.H2O, kind="candidate"
                )
                self.assertEqual(set(requested), set(models.CANDIDATE_FILES))
                self.assertEqual(zils.checkpoint_hash(root / "download"), claim["sha256"])
                # The adapter contract does not accept a candidate announcement.
                with self.assertRaises((ValueError, OSError)):
                    protocol.fetch_checkpoint(claim, root / "as-adapter", model=models.H2O)

    def test_validation_pins_contract_files_and_text_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent"
            parent.mkdir()
            (parent / "config.json").write_text(json.dumps(PARENT_CONFIG))
            saved = {
                **PARENT_CONFIG,
                "dtype": "bfloat16",
                "use_cache": False,
                "transformers_version": "5.17.0",
                "text_config": {"hidden_size": 2560, "num_hidden_layers": 32},
                "vision_config": {"model_type": "qwen3_5_vision", "rope_parameters": {}},
            }
            good = candidate(root / "good", saved)
            with (
                patch("zils.h2o.PINS", pins_for_fixture()),
                patch("zils.h2o.pinned_base_path", return_value=parent),
            ):
                validate_candidate(good)
                narrower = candidate(
                    root / "narrower",
                    {**saved, "text_config": {**saved["text_config"], "hidden_size": 2048}},
                )
                with self.assertRaisesRegex(ValueError, "architecture"):
                    validate_candidate(narrower)
                other = candidate(
                    root / "other", {**saved, "architectures": ["Qwen3_5ForCausalLM"]}
                )
                with self.assertRaisesRegex(ValueError, "architecture"):
                    validate_candidate(other)
                edited = candidate(root / "edited", saved)
                (edited / "serve_config.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "pinned parent"):
                    validate_candidate(edited)
                with self.assertRaisesRegex(ValueError, "pinned base"):
                    DecisionModel(good, "cuda", train=True)

    def test_promote_base_targets_the_chosen_profile(self):
        for model, prefix, validator in (
            (models.H2O, "h2o-lightning-4b", "zils.h2o.validate_candidate"),
            (models.D3, "d3-mini-4.5b", "zils.d3.validate_candidate"),
            ("d3-lite-0.8b-v1", "d3-lite-0.8b", "zils.d3.validate_candidate"),
        ):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                weights = root / "candidate"
                weights.mkdir()
                for name in models.candidate_files(model, "candidate"):
                    (weights / name).write_text(name)
                models.write_metadata(weights, kind="candidate", model=model)
                digest = zils.checkpoint_hash(weights)
                decision = root / "decision.json"
                winner = {"uid": 7, "sha256": digest}
                first = {
                    "pending_winner": winner,
                    "dataset_sha256": "1" * 64,
                    "evaluated_at": 1000,
                    "model": models.spec(model),
                    "incumbent": {"sha256": "b" * 64},
                }
                (root / "first.json").write_text(json.dumps(first))
                decision.write_text(
                    json.dumps(
                        {
                            **first,
                            "confirmed_winner": winner,
                            "dataset_sha256": "2" * 64,
                            "evaluated_at": 44200,
                            "candidates": [{**winner, "eligible": True}],
                        }
                    )
                )
                (root / "regression.json").write_text(
                    json.dumps(
                        {
                            "incumbent_model": model,
                            "candidate_model": model,
                            "incumbent_sha256": "b" * 64,
                            "candidate_sha256": digest,
                            "jobs": {
                                "task": {"incumbent": {"brier": 0.3}, "candidate": {"brier": 0.2}}
                            },
                        }
                    )
                )
                argv = [
                    "promote_base.py",
                    "--model",
                    model,
                    "--candidate",
                    str(weights),
                    "--decision",
                    str(decision),
                    "--round",
                    "3",
                    "--staging",
                    str(root / "staging"),
                    "--king-hotkey",
                    "hk-new",
                    "--previous-kings",
                    "hk-old,hk-older",
                ]
                argv += [
                    "--first-decision",
                    str(root / "first.json"),
                    "--regression-report",
                    str(root / "regression.json"),
                ]
                argv += [
                    "--config",
                    str(root / "first.json"),
                    "--pool",
                    "d3-mini",
                    "--window-report",
                    str(decision),
                ]
                verified = {
                    "status": "qualified",
                    "winner": {**winner, "hotkey": "hk-new"},
                    "closing": {"decision": json.loads(decision.read_text())},
                }
                out = io.StringIO()
                with (
                    patch(
                        "zils.base_release.active_context",
                        return_value={"model": models.spec(model), "kings": ["hk-old", "hk-older"]},
                    ),
                    patch("zils.base_release.verify_window", return_value=verified),
                    patch.object(sys, "argv", argv),
                    patch(validator) as validate,
                    contextlib.redirect_stdout(out),
                ):
                    runpy.run_path(
                        str(Path(__file__).parents[1] / "scripts/promote_base.py"),
                        run_name="__main__",
                    )
                validate.assert_called_once_with(
                    weights, *([model] if model in models.d3_family() else [])
                )
                result = json.loads(out.getvalue())
                entry = result["proposed_spec_entry"]
                self.assertEqual(entry["id"], f"{prefix}-zils-r3")
                self.assertEqual(result["repo"], f"zils-ai/{prefix}-zils-r3")
                self.assertEqual(entry["parent"]["id"], model)
                self.assertEqual(entry["base"], result["repo"])
                self.assertEqual(entry["max_input_tokens"], models.spec(model)["max_input_tokens"])
                # The same promotion PR crowns the winner: new incumbent, its hotkey prepended.
                self.assertEqual(
                    result["proposed_base_tier"],
                    {"incumbent_sha256": digest, "kings": ["hk-new", "hk-old", "hk-older"]},
                )
                self.assertEqual(result["crowned_uid"], 7)
                registration = result["proposed_registration"]
                self.assertEqual(
                    registration["TEXT_RUNTIME_MODULES"], models.TEXT_RUNTIME_MODULES[model]
                )
                self.assertEqual(
                    registration["RUNTIME_PYTHON_ENV"], models.RUNTIME_PYTHON_ENV[model]
                )
                self.assertIn("_ZILS_R3_CANDIDATE_BASE", registration["CANDIDATE_BASE_ENV"])
                pins_file = json.loads(Path(result["proposed_pins_file"]).read_text())
                self.assertEqual(
                    registration["PINS_FILES"], Path(result["proposed_pins_file"]).name
                )
                self.assertNotIn("README.md", pins_file["files"])
                self.assertEqual(
                    pins_file["files"]["NOTICE"], result["proposed_pin_files"]["NOTICE"]
                )
                self.assertEqual(
                    {k: v for k, v in pins_file.items() if k != "files"},
                    {k: v for k, v in models.pins(model).items() if k != "files"},
                )

    def test_promote_base_refuses_a_candidate_of_another_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "candidate"
            weights.mkdir()
            models.write_metadata(weights, kind="candidate", model=models.H2O)
            decision = root / "decision.json"
            decision.write_text(json.dumps({"confirmed_winner": {"sha256": "x"}}))
            argv = ["promote_base.py", "--model", models.D3, "--candidate", str(weights)]
            argv += ["--decision", str(decision), "--round", "1", "--staging", str(root / "s")]
            argv += [
                "--king-hotkey",
                "hk",
                "--first-decision",
                str(decision),
                "--regression-report",
                str(decision),
            ]
            argv += [
                "--config",
                str(decision),
                "--pool",
                "d3-mini",
                "--window-report",
                str(decision),
            ]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(SystemExit, "not a d3"):
                runpy.run_path(
                    str(Path(__file__).parents[1] / "scripts/promote_base.py"), run_name="__main__"
                )


if __name__ == "__main__":
    unittest.main()
