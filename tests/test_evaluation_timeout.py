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


class CapacityWaitTests(unittest.TestCase):
    """ZIL-162: the validator's weight pass must not wait for a soup round to free the GPU."""

    def run_busy(self, wait_for_capacity, clock):
        from zils.runtime import CapacityUnavailable

        with tempfile.TemporaryDirectory() as tmp:
            cases = Path(tmp, "test.jsonl")
            cases.write_text('{"id": 1}\n')
            args = SimpleNamespace(runtime_python="python", device="cuda", poll=5)
            with (
                patch("zils.validator.run_child", side_effect=CapacityUnavailable()),
                patch(
                    "zils.validator.time.sleep", side_effect=lambda s: clock.append(clock[-1] + s)
                ),
                patch("zils.validator.time.monotonic", side_effect=lambda: clock[-1]),
            ):
                return validator.run_evaluate(
                    args, Path(tmp), "test", [{"uid": 0}], cases, base_revision="r",
                    wait_for_capacity=wait_for_capacity,
                )  # fmt: skip

    def test_a_bounded_wait_gives_up_after_its_seconds(self):
        from zils.runtime import CapacityUnavailable

        clock = [0.0]
        with self.assertRaises(CapacityUnavailable):
            self.run_busy(600, clock)
        self.assertGreaterEqual(clock[-1], 600)
        self.assertLess(clock[-1], 600 + 10)

    def test_the_weight_pass_bound_is_ten_minutes(self):
        self.assertEqual(validator.EVALUATION_CAPACITY_WAIT, 600)

    def test_a_busy_gpu_still_yields_a_report_for_the_weights(self):
        from zils.runtime import CapacityUnavailable

        calls = []

        def fake(config, directory, work, registry, args, **kwargs):
            calls.append((registry, kwargs))
            if len(calls) == 1:
                raise CapacityUnavailable()
            return {"miners": [], "weights": {}}

        with patch("zils.validator.evaluate_round", side_effect=fake):
            report = validator.round_report({}, Path("d"), Path("w"), {"1": "claim"}, None)
        self.assertEqual(report, {"miners": [], "weights": {}})
        first, second = calls
        self.assertEqual(first[0], {"1": "claim"})
        self.assertEqual(first[1]["wait_for_capacity"], validator.EVALUATION_CAPACITY_WAIT)
        self.assertFalse(first[1]["baseline_without_submissions"])
        self.assertEqual(second[0], {})  # recorded unscored, without the GPU
        self.assertFalse(second[1]["baseline_without_submissions"])
        self.assertNotIn("wait_for_capacity", second[1])

    def test_no_wait_raises_at_once(self):
        from zils.runtime import CapacityUnavailable

        clock = [0.0]
        with self.assertRaises(CapacityUnavailable):
            self.run_busy(False, clock)
        self.assertEqual(clock, [0.0])


if __name__ == "__main__":
    unittest.main()
