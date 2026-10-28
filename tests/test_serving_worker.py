"""The d3 inference worker and its JSON-lines client, with a fake model in place of the GPU."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zils import ROOT, checkpoint_hash, models, serving_server, serving_worker

MODEL = models.D3
QUESTION = {"type": "noul", "instructions": None, "criteria": None}
# The fake encodes one token per state character and fails over 10 the way d3.Encoder.inputs does.
FAKE = '''
class Encoder:
    def inputs(self, state, question):
        if state == "hang":
            import time
            time.sleep(30)
        if len(state) > 10:
            raise ValueError("Decision exceeds the 4096-token d3 input limit; shorten the example")
        if state == "bad":
            raise ValueError("SECRET-STATE cannot be encoded")
        return list(range(len(state))), ["true", "false"]


class FakeModel:
    meta = {"temperature": 1.0}
    encoder = Encoder()

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def probabilities(self, ids, keys, question, temperature):
        if ids == [0, 1, 2, 3]:
            raise OSError("CUDA lost SECRET")
        return {"true": 0.25, "false": 0.75}
'''
scope = {}
exec(FAKE, scope)
FakeModel = scope["FakeModel"]
SCRIPT = FAKE + "\nimport sys\nfrom zils import serving_worker\nserving_worker.main(sys.argv[1:], factory=FakeModel)\n"


def checkpoint(root, name, kind="base", model=MODEL):
    path = Path(root) / name
    path.mkdir()
    if kind:
        models.write_metadata(path, kind=kind, model=model)
    return path


class WorkerTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.enterContext(patch.dict(os.environ, {models.candidate_base_env(MODEL): ""}))
        self.loaded = serving_worker.load([checkpoint(self.root, "stock")], FakeModel)[0]

    def ask(self, state, **extra):
        line = json.dumps({"id": 7, "model": MODEL, "questions": {"q1": QUESTION}, "state": state, **extra})
        return serving_worker.answer(self.loaded, line)

    def test_load_describes_a_base_by_its_checkpoint_hash(self):
        path = checkpoint(self.root, "base")
        loaded, served = serving_worker.load([path], FakeModel)
        self.assertEqual(
            served, [{"model": MODEL, "weights_sha256": checkpoint_hash(path), "max_input_tokens": 8192}]
        )
        self.assertEqual(loaded[MODEL].checkpoint, path)

    def test_load_refuses_adapters_and_duplicate_models_before_loading(self):
        made = []
        for paths in (
            [checkpoint(self.root, "adapter", kind=None)],
            [checkpoint(self.root, "a"), checkpoint(self.root, "b")],
        ):
            with self.assertRaises(ValueError):
                serving_worker.load(paths, made.append)
        self.assertEqual(made, [])

    def test_load_refuses_a_candidate_base_override(self):
        path = checkpoint(self.root, "base")
        with patch.dict(os.environ, {models.candidate_base_env(MODEL): "/elsewhere"}):
            with self.assertRaisesRegex(ValueError, "ZILS_D3_CANDIDATE_BASE"):
                serving_worker.load([path], FakeModel)

    def test_answer_predicts_every_question(self):
        self.assertEqual(
            self.ask("abc"),
            {"id": 7, "predictions": {"q1": {"probabilities": {"true": 0.25, "false": 0.75}, "input_tokens": 3}}},
        )

    def test_encoder_length_error_is_too_long(self):
        self.assertEqual(self.ask("x" * 11)["error"]["kind"], "too_long")

    def test_other_errors_are_fixed_messages(self):
        bad = self.ask("bad")
        self.assertEqual(bad["error"], {"kind": "invalid", "message": serving_worker.MESSAGES["invalid"]})
        self.assertEqual(self.ask("abc", model="other")["error"]["kind"], "invalid")
        self.assertEqual(serving_worker.answer(self.loaded, "not json")["error"]["kind"], "invalid")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            runtime = self.ask("abcd")
        self.assertEqual(runtime["error"]["kind"], "runtime")
        self.assertEqual(err.getvalue(), "serving worker: inference failed (OSError)\n")
        self.assertNotIn("SECRET", json.dumps([bad, runtime]))

    def test_run_speaks_json_lines(self):
        stdin = io.StringIO(json.dumps({"id": 1, "model": MODEL, "questions": {"q": QUESTION}, "state": "ab"}) + "\n")
        stdout = io.StringIO()
        serving_worker.run([checkpoint(self.root, "base")], FakeModel, stdin, stdout)
        ready, reply = map(json.loads, stdout.getvalue().splitlines())
        self.assertEqual((ready["ready"], ready["served"][0]["model"]), (True, MODEL))
        self.assertEqual(reply["predictions"]["q"]["input_tokens"], 2)


class WorkerProcessTest(unittest.TestCase):
    """The real worker module in a child process (this interpreter, fake model) behind its client."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = checkpoint(tmp.name, "base")
        self.enterContext(patch.dict(os.environ, {models.candidate_base_env(MODEL): ""}))

    def start(self, timeout=10.0):
        served = serving_server.start_worker([self.path], sys.executable, args=("-c", SCRIPT), timeout=timeout)
        self.addCleanup(served[0].worker.close)
        return served[0]

    def test_protocol_end_to_end(self):
        served = self.start()
        self.assertEqual(
            (served.model, served.weights_sha256, served.max_input_tokens), (MODEL, checkpoint_hash(self.path), 8192)
        )
        predictions = served.predict("abc", {"q1": QUESTION, "q2": QUESTION})
        self.assertEqual(predictions["q2"], {"probabilities": {"true": 0.25, "false": 0.75}, "input_tokens": 3})
        with self.assertRaises(serving_server.WorkerError) as refused:
            served.predict("x" * 11, {"q1": QUESTION})
        self.assertEqual(refused.exception.kind, "too_long")
        self.assertEqual(served.predict("ab", {"q": QUESTION})["q"]["input_tokens"], 2)  # still in step

    def test_dead_worker_raises_oserror_and_signals_exit(self):
        served = self.start()
        served.worker.process.kill()
        self.assertTrue(served.worker.exited.wait(5))
        for _ in range(2):
            with self.assertRaises(OSError):
                served.predict("abc", {"q1": QUESTION})

    def test_timeout_kills_the_worker(self):
        served = self.start(timeout=0.3)
        with self.assertRaisesRegex(OSError, "timed out"):
            served.predict("hang", {"q1": QUESTION})
        self.assertTrue(served.worker.exited.wait(5))

    def test_worker_that_exits_before_ready_fails_to_start(self):
        with self.assertRaisesRegex(ValueError, r"failed to start \(exit 3\)"):
            serving_server.start_worker([self.path], sys.executable, args=("-c", "raise SystemExit(3)"))

    def test_worker_imports_no_chain_or_signing_code(self):
        import subprocess

        probe = (
            "import sys, zils.serving_worker\n"
            "print(sorted(m for m in ('bittensor', 'bittensor_wallet', 'zils.serving') if m in sys.modules))"
        )
        done = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(done.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
