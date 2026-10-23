import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from zils import validator


class EvaluationTimeoutTests(unittest.TestCase):
    def test_scales_with_cases_and_keeps_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            small, large, images = Path(tmp, "s.jsonl"), Path(tmp, "l.jsonl"), Path(tmp, "i.json")
            small.write_text('{"id": 1}\n' * 10)
            large.write_text('{"id": 1}\n\n' * 784)
            images.write_text(json.dumps([{}] * 300))
            self.assertEqual(validator.evaluation_timeout(small), 600)
            self.assertEqual(validator.evaluation_timeout(large), 3 * 784)
            self.assertEqual(validator.evaluation_timeout(images), 900)

    def test_run_evaluate_passes_inner_and_outer_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp, "test.jsonl")
            cases.write_text('{"id": 1}\n' * 784)
            calls = []

            def fake(*a, **k):
                calls.append((a, k))
                Path(tmp, "test.json").write_text("{}")

            args = SimpleNamespace(runtime_python="python", device="cuda")
            with patch("zils.validator.run_child", side_effect=fake):
                validator.run_evaluate(
                    args,
                    Path(tmp),
                    "test",
                    [{"uid": 0}, {"uid": 1}],
                    cases,
                    base_revision="r",
                )
            command, kwargs = calls[0][0][0], calls[0][1]
            self.assertEqual(command[command.index("--timeout") + 1], "2352")
            self.assertEqual(kwargs["timeout"], 120 + (2352 + 120) * 2)


if __name__ == "__main__":
    unittest.main()
