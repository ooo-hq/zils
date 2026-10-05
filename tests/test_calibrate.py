"""Exercise real temperature fitting; fixture probabilities are not model results."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import zils
from zils import benchmark


@unittest.skipUnless(importlib.util.find_spec("kev"), "use .venv-kev for calibration checks")
class CalibrationTest(unittest.TestCase):
    def test_calibration_uses_only_matching_raw_development_predictions(self):
        self.assertIsNotNone(
            importlib.util.find_spec("zils.calibrate"), "calibration adapter is missing"
        )
        import torch
        from kev.checkpoint import Meta, read_meta, write_meta

        from zils import calibrate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            benchmark.build(data, seed=553)
            cases = benchmark.audit(data)["calibration"]
            source = root / "raw"
            source.mkdir()
            (source / "adapter_config.json").write_text("{}")
            (source / "adapter_model.safetensors").write_bytes(b"test artifact only")
            write_meta(
                source, Meta(base=zils.BASE, head={"fixture": torch.zeros(1)}, temperature=1.0)
            )
            original = zils.checkpoint_hash(source)
            predictions = []
            for case in cases:
                keys = zils.options(case["question"])
                wrong = next(k for k in keys if k != case["label"])
                predictions.append(
                    {
                        "id": case["id"],
                        "elapsed_ms": 1,
                        "probabilities": {
                            k: 0.97 if k == wrong else 0.03 / (len(keys) - 1) for k in keys
                        },
                    }
                )
            report = {
                "dataset_sha256": hashlib.sha256(
                    json.dumps(cases, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
                "miners": [
                    {
                        "uid": 1,
                        "sha256": original,
                        "status": "evaluated",
                        "runtime": {"temperature": 1.0},
                        "predictions": predictions,
                    }
                ],
            }
            result = calibrate.fit(data, report, 1, source, root / "fitted")
            # Confidently wrong observations require softening; positive T cannot change the chosen answer.
            self.assertAlmostEqual(result["temperature"], 4.0)
            self.assertEqual(result["fit_cases"], 112)
            self.assertEqual(result["before"]["accuracy"], result["after"]["accuracy"])
            self.assertLess(result["after"]["brier"], result["before"]["brier"])
            self.assertEqual(zils.checkpoint_hash(source), original)
            fitted = read_meta(root / "fitted")
            self.assertAlmostEqual(fitted.temperature, 4.0)
            self.assertTrue(torch.equal(fitted.head["fixture"], read_meta(source).head["fixture"]))
            self.assertEqual(
                (source / "adapter_model.safetensors").read_bytes(),
                (root / "fitted/adapter_model.safetensors").read_bytes(),
            )
            with self.assertRaises(FileExistsError):
                calibrate.fit(data, report, 1, source, root / "fitted")
            for changed, pattern in (
                ("dataset", "dataset"),
                ("temperature", "temperature"),
                ("hash", "checkpoint"),
            ):
                invalid = copy.deepcopy(report)
                if changed == "dataset":
                    invalid["dataset_sha256"] = "0" * 64
                elif changed == "temperature":
                    invalid["miners"][0]["runtime"]["temperature"] = 2.0
                else:
                    invalid["miners"][0]["sha256"] = "0" * 64
                destination = root / changed
                with self.assertRaisesRegex(ValueError, pattern):
                    calibrate.fit(data, invalid, 1, source, destination)
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
