"""Adapter regression verdicts and the validator-only candidate-base override."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_base_candidate import PARENT_CONFIG, candidate, pins_for_fixture
from zils import h2o, models, regression_suite

H2O_PYTHON, D3_PYTHON = "/fixture/h2o-python", "/fixture/d3-python"
BRIER = {("incumbent", "a"): 0.30, ("candidate", "a"): 0.28}


def run_suite(flags, **environ):
    """Run main() with fake training; return (calls, written result)."""
    calls = []

    def fake(job, base_dir, label, out, runtime_python, env, device, module, base_revision):
        calls.append(
            {
                "job": job.name,
                "label": label,
                "reference": base_dir,
                "reference_meta": models.metadata(base_dir),
                "python": runtime_python,
                "module": module,
                "revision": base_revision,
                "env": {k: v for k, v in env.items() if k.startswith("ZILS_")},
            }
        )
        return {"brier": BRIER[(label, job.name)]}

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        argv = ["--suite", "unused", "--out", str(out), "--device", "cpu", *flags]
        scrubbed = {k: v for k, v in os.environ.items() if not k.startswith("ZILS_")}
        with (
            patch("sys.argv", ["regression_suite", *argv]),
            patch.dict(os.environ, scrubbed | environ, clear=True),
            patch("zils.regression_suite.jobs", return_value=[Path("a")]),
            patch("zils.regression_suite.train_and_score", fake),
            patch("builtins.print"),
        ):
            regression_suite.main()
        return calls, json.loads((out / "suite-result.json").read_text())


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

    def test_same_profile_mode_selects_the_models_trainer_and_candidate_env(self):
        for model, module, runtime, candidate_env in (
            (models.H2O, "zils.h2o", "ZILS_H2O_RUNTIME_PYTHON", "ZILS_H2O_CANDIDATE_BASE"),
            (models.D3, "zils.d3", "ZILS_D3_RUNTIME_PYTHON", "ZILS_D3_CANDIDATE_BASE"),
        ):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                weights = Path(tmp) / "candidate"
                weights.mkdir()
                models.write_metadata(weights, kind="candidate", model=model)
                reference = Path(tmp) / "reference"
                flags = [
                    "--model",
                    model,
                    "--incumbent",
                    str(reference),
                    "--candidate",
                    str(weights),
                ]
                calls, result = run_suite(flags, **{runtime: "/fixture/python"})
                incumbent, candidate_run = calls
                for call in calls:
                    self.assertEqual(call["module"], module)
                    self.assertEqual(call["revision"], models.spec(model)["base_revision"])
                    self.assertEqual(call["python"], "/fixture/python")
                    self.assertEqual(call["env"].get(runtime), "/fixture/python")
                self.assertEqual(incumbent["reference"], reference)
                self.assertNotIn(candidate_env, incumbent["env"])
                self.assertEqual(candidate_run["env"][candidate_env], str(weights.resolve()))
                other = ({models.H2O, models.D3} - {model}).pop()
                self.assertNotIn(models.candidate_base_env(other), candidate_run["env"])
                self.assertEqual(
                    {k: candidate_run["reference_meta"][k] for k in ("kind", "model")},
                    {"kind": "base", "model": model},
                )
                self.assertEqual(result["incumbent_model"], model)
                self.assertEqual(result["candidate_model"], model)

    def test_same_profile_mode_rejects_a_candidate_of_another_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "candidate"
            weights.mkdir()
            models.write_metadata(weights, kind="candidate", model=models.H2O)
            flags = ["--incumbent", tmp, "--candidate", str(weights)]
            with self.assertRaisesRegex(SystemExit, "d3-mini-4.5b-v1 base candidate"):
                run_suite(flags)

    def test_cross_profile_mode_trains_each_side_on_its_own_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            h2o_reference, d3_reference = Path(tmp) / "h2o", Path(tmp) / "d3"
            for path, model in ((h2o_reference, models.H2O), (d3_reference, models.D3)):
                path.mkdir()
                models.write_metadata(path, kind="base", model=model)
            flags = [
                "--incumbent-model",
                models.H2O,
                "--incumbent",
                str(h2o_reference),
                "--candidate-model",
                models.D3,
                "--candidate",
                str(d3_reference),
            ]
            environ = {
                "ZILS_H2O_RUNTIME_PYTHON": H2O_PYTHON,
                "ZILS_D3_RUNTIME_PYTHON": D3_PYTHON,
                "ZILS_D3_CANDIDATE_BASE": "/must/be/removed",
            }
            calls, result = run_suite(flags, **environ)
        incumbent, candidate_run = calls
        self.assertEqual((incumbent["job"], candidate_run["job"]), ("a", "a"))
        self.assertEqual(
            (incumbent["module"], incumbent["python"], incumbent["reference"]),
            ("zils.h2o", H2O_PYTHON, h2o_reference),
        )
        self.assertEqual(
            (candidate_run["module"], candidate_run["python"], candidate_run["reference"]),
            ("zils.d3", D3_PYTHON, d3_reference),
        )
        self.assertEqual(incumbent["revision"], models.spec(models.H2O)["base_revision"])
        self.assertEqual(candidate_run["revision"], models.spec(models.D3)["base_revision"])
        self.assertEqual(incumbent["env"]["ZILS_H2O_RUNTIME_PYTHON"], H2O_PYTHON)
        self.assertEqual(candidate_run["env"]["ZILS_D3_RUNTIME_PYTHON"], D3_PYTHON)
        for call in calls:
            self.assertNotIn("ZILS_D3_CANDIDATE_BASE", call["env"])
            self.assertNotIn("ZILS_H2O_CANDIDATE_BASE", call["env"])
        self.assertEqual(result["incumbent_model"], models.H2O)
        self.assertEqual(result["candidate_model"], models.D3)
        self.assertAlmostEqual(result["verdict"]["deltas"]["a"], 0.02)
        self.assertEqual(result["verdict"]["tolerance"], 0.005)
        self.assertTrue(result["verdict"]["passed"])

    def test_cross_profile_mode_needs_both_models_and_no_model(self):
        for flags in (
            ["--incumbent-model", models.H2O],
            ["--incumbent-model", models.H2O, "--candidate-model", models.D3, "--model", models.D3],
        ):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                with patch("sys.stderr"):
                    run_suite([*flags, "--incumbent", "x", "--candidate", "y"])

    def test_train_and_score_runs_the_given_trainer_module(self):
        class Stop(Exception):
            pass

        recorded = []

        def stop(command, env, cwd):
            recorded.append(command)
            raise Stop

        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job"
            job.mkdir()
            (job / "train.jsonl").write_text("")
            with (
                patch("zils.regression_suite._run", stop),
                patch("zils.benchmark.training_rows", return_value=[]),
                self.assertRaises(Stop),
            ):
                regression_suite.train_and_score(
                    job, Path("ref"), "candidate", tmp, "py", {}, "cpu", "zils.d3", "rev"
                )
        self.assertEqual(recorded[0][:4], ["py", "-m", "zils.d3", "train"])

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
