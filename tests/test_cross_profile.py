"""A job may move to a new base: the frozen previous release is scored on its own base."""

import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import zils
from tests.test_fleet_profiles import checkpoint
from tests.test_jobs import POLICY, examples
from zils import benchmark, jobs, models, validator
from zils.version_selection import VERSION, check_promotion, validate

RUNTIME = """import json, sys
from pathlib import Path
from zils import models
with open(RECORD, "a") as record:
    record.write(json.dumps(sys.argv) + "\\n")
path = Path(sys.argv[sys.argv.index("--checkpoint") + 1])
trained = any("trained" in f.read_text() for f in path.iterdir() if f.is_file())
p = .9 if trained else .5
temp = models.temperature(path)
a, b = p ** (1 / temp), (1 - p) ** (1 / temp)
rows = json.load(sys.stdin)
print(json.dumps({"runtime": {"temperature": temp}, "predictions": [
    {"id": r["id"], "probabilities": {"true": a / (a + b), "false": b / (a + b)},
     "elapsed_ms": 1} for r in rows]}))
"""


def revision(model):
    return models.spec(model)["base_revision"]


class CrossProfileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        runtime = self.root / "runtime"
        self.record = self.root / "argv.jsonl"
        runtime.write_text(f"#!{sys.executable}\nRECORD={str(self.record)!r}\n" + RUNTIME)
        runtime.chmod(0o700)
        self.env = {
            **os.environ,
            "ZILS_H2O_RUNTIME_PYTHON": str(runtime),
            "ZILS_D3_RUNTIME_PYTHON": str(runtime),
        }

    def round(self, previous_model, job_model=models.D3, env=None, base_change=None):
        root = self.root
        previous = checkpoint(root / "previous", previous_model)
        sha = zils.checkpoint_hash(previous)
        previous_job = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        selection = {
            "version": VERSION,
            "root_job_id": previous_job,
            "previous": {
                "job_id": previous_job,
                "model_id": f"zils-adapter-{previous_job}-{sha}",
                "sha256": sha,
            },
        }
        if base_change is not None:
            selection["base_change"] = base_change
        directory = root / "validator"
        data = directory / "benchmark"
        manifest = jobs.build(
            data,
            job_id,
            examples(),
            {**POLICY, "previous_job_id": previous_job},
            allow_training_data_export=True,
            model=job_model,
            selection=selection,
        )
        zils.stage(zils.submission(previous, 0), directory / "comparison")
        config = {
            "base_revision": revision(job_model),
            "benchmark_sha256": benchmark.file_hash(data / "manifest.json"),
            "job_id": job_id,
            "job_sha256": benchmark.file_hash(data / "manifest.json"),
            "initial_sha256": sha,
            "members": {"1": {}},
        }
        candidate = checkpoint(root / "candidate", job_model)
        next(f for f in candidate.iterdir() if f.suffix == ".safetensors").write_text("trained")
        registry = {1: {"claim": {"uid": 1, "sha256": "a" * 64, "endpoint": "fixture"}}}

        def fetch(claim, destination, **kwargs):
            zils.stage(zils.submission(candidate, 1), destination)

        work = root / "work"
        work.mkdir()
        args = SimpleNamespace(device="cpu", runtime_python=sys.executable, poll=0.05)
        with patch.dict(os.environ, env or self.env):
            return manifest, validator.evaluate_round(
                config, directory, work, registry, args, fetch_checkpoint=fetch
            )

    def revisions(self):
        calls = [json.loads(line) for line in self.record.read_text().splitlines()]
        return [c[c.index("--base-revision") + 1] for c in calls]

    def test_previous_release_on_another_base_is_scored_on_its_own_base(self):
        manifest, report = self.round(models.H2O, base_change=True)
        self.assertEqual(report["delivery"]["status"], "accepted", report)
        self.assertEqual(report["baseline_model"], models.H2O)
        self.assertEqual(report["model"]["id"], models.D3)
        revisions = self.revisions()
        self.assertEqual(revisions[0], revision(models.H2O))
        self.assertEqual(set(revisions[1:]), {revision(models.D3)})

    def test_missing_previous_interpreter_accepts_nothing_and_names_the_variable(self):
        env = {k: v for k, v in self.env.items() if k not in ("ZILS_H2O_RUNTIME_PYTHON",)}
        env.pop("FEZ_H2O_RUNTIME_PYTHON", None)
        with self.assertRaisesRegex(RuntimeError, "ZILS_H2O_RUNTIME_PYTHON"):
            self.round(models.H2O, env=env, base_change=True)
        self.assertFalse(self.record.exists())

    def test_cross_profile_upgrade_without_base_change_is_refused_before_evaluation(self):
        with self.assertRaisesRegex(ValueError, "requires base_change"):
            self.round(models.H2O)
        self.assertFalse(self.record.exists())

    def test_same_profile_upgrade_records_the_job_model(self):
        _, report = self.round(models.D3)
        self.assertEqual(report["baseline_model"], models.D3)
        self.assertEqual(set(self.revisions()), {revision(models.D3)})

    def test_promotion_crosses_profiles_only_when_the_selection_says_so(self):
        previous_job, job_id, sha = str(uuid.uuid4()), str(uuid.uuid4()), "b" * 64
        previous_id = f"zils-adapter-{previous_job}-{sha}"
        entry_id = f"zils-adapter-{job_id}-{'c' * 64}"
        alias = "zils-task-" + previous_job
        selection = {
            "version": VERSION,
            "root_job_id": previous_job,
            "previous": {"job_id": previous_job, "model_id": previous_id, "sha256": sha},
        }
        incumbent = {
            "id": previous_id,
            "owners": ["owner"],
            "aliases": [alias],
            "profile": models.spec(models.H2O),
        }
        entry = {
            "id": entry_id,
            "owners": ["owner"],
            "aliases": [alias],
            "profile": models.spec(models.D3),
        }
        current = {"models": [incumbent]}
        with self.assertRaisesRegex(ValueError, "cannot move between model profiles"):
            check_promotion(current, entry, selection)
        check_promotion(current, entry, {**selection, "base_change": True})
        legacy = {key: value for key, value in incumbent.items() if key != "profile"}
        check_promotion({"models": [legacy]}, entry, selection)
        image = {**entry, "profile": models.spec(models.IMAJEV)}
        with self.assertRaisesRegex(ValueError, "cannot move between model profiles"):
            check_promotion({"models": [legacy]}, image, selection)

    def test_selection_base_change_must_be_true(self):
        selection = {"version": VERSION, "root_job_id": str(uuid.uuid4()), "previous": None}
        validate({**selection, "base_change": True})
        for value in (False, "true"):
            with self.assertRaisesRegex(ValueError, "Invalid frozen version selection"):
                validate({**selection, "base_change": value})


if __name__ == "__main__":
    unittest.main()
