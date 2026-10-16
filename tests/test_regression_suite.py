"""Adapter regression verdicts and the validator-only candidate-base override."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_base_candidate import PARENT_CONFIG, candidate, pins_for_fixture
from zils import h2o, regression_suite


class RegressionSuiteTest(unittest.TestCase):
    def test_verdict_rules(self):
        ok = regression_suite.verdict(
            {
                "a": {"incumbent": 0.30, "candidate": 0.28},
                "b": {"incumbent": 0.20, "candidate": 0.203},
            }
        )
        self.assertTrue(ok["passed"])
        self.assertEqual(set(ok["better"]), {"a"})
        worse = regression_suite.verdict(
            {
                "a": {"incumbent": 0.30, "candidate": 0.25},
                "b": {"incumbent": 0.20, "candidate": 0.21},
            }
        )
        self.assertFalse(worse["passed"])
        self.assertEqual(set(worse["worse"]), {"b"})
        flat = regression_suite.verdict({"a": {"incumbent": 0.30, "candidate": 0.30}})
        self.assertFalse(flat["passed"], "no improvement anywhere is not a pass")

    def test_candidate_base_override_validates_and_does_not_recurse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent"
            parent.mkdir()
            (parent / "config.json").write_text(json.dumps(PARENT_CONFIG))
            good = candidate(root / "good")
            with (
                patch("zils.h2o.PINS", pins_for_fixture()),
                patch("zils.h2o.pinned_base_path", return_value=parent),
                patch(
                    "zils.h2o.settings.get",
                    side_effect=lambda name, default=None: (
                        str(good) if name == "ZILS_H2O_CANDIDATE_BASE" else default
                    ),
                ),
            ):
                self.assertEqual(h2o.base_path(), good.resolve())
                (good / "serve_config.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "pinned parent"):
                    h2o.base_path()


if __name__ == "__main__":
    unittest.main()
