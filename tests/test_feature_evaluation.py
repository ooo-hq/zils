"""Private box labels stay in the controller; native inference gets verified pixels."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.box_dataset import generate
from zils import core, models


class FeatureEvaluationTest(unittest.TestCase):
    def test_box_inference_receives_only_context_questions_and_verified_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate(root / "draw", 4, 42)
            checkpoint = root / "base"
            checkpoint.mkdir()
            models.write_metadata(checkpoint, model=models.D3_ZILS_R5, kind="base")
            submissions = root / "submissions.json"
            submissions.write_text(json.dumps([core.submission(checkpoint, 1)]))
            args = SimpleNamespace(
                cases=root / "draw/expanded.jsonl",
                images=root / "draw/images",
                submissions=submissions,
                base_revision=models.spec(models.D3_ZILS_R5)["base_revision"],
                timeout=10,
                report=root / "report.json",
                device="cpu",
                runner_python="python",
            )
            seen = []

            def run(command, **kwargs):
                requests = json.loads(kwargs["input"])
                seen.extend(requests)
                self.assertTrue(
                    all(set(r) == {"id", "state", "question", "images"} for r in requests)
                )
                self.assertTrue(
                    all(Path(r["images"][0]).parent == args.images.resolve() for r in requests)
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "runtime": {},
                            "predictions": [
                                {
                                    "id": r["id"],
                                    "elapsed_ms": 1,
                                    "probabilities": {
                                        k: 1 / len(r["question"]["criteria"])
                                        for k in r["question"]["criteria"]
                                    },
                                }
                                for r in requests
                            ],
                        }
                    ),
                    stderr="",
                )

            with (
                patch.dict(os.environ, {"ZILS_D3_RUNTIME_PYTHON": "python"}),
                patch("zils.core.subprocess.run", side_effect=run),
                patch("zils.core.sandbox.wrap", side_effect=lambda c: c),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                core.evaluate(args)
            self.assertEqual(len(seen), 20)
            self.assertEqual(
                json.loads(args.report.read_text())["miners"][0]["status"], "evaluated"
            )
            args.report.unlink()
            asset = next(args.images.glob("*.png"))
            asset.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                core.evaluate(args)


if __name__ == "__main__":
    unittest.main()
