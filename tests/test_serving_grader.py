import ast
import fcntl
import functools
import json
import math
import os
import random
import socket
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from bittensor_wallet import Keypair

from zils import decisions, serving, serving_items, serving_server

MODEL = "d3-mini-4.5b-v1"
SHA = "ab" * 32
REQUEST_KEY = Keypair.create_from_seed("0x" + "44" * 32)
VALIDATOR = Keypair.create_from_seed("0x" + "66" * 32).ss58_address
MINERS = [Keypair.create_from_seed("0x" + f"{70 + i:02x}" * 32) for i in range(5)]
HOUR = 3600


def uniform(question):
    keys = list(decisions.option_descriptions(question))
    return {k: 1 / len(keys) for k in keys}


def reference(questions):
    return {qid: {"probabilities": uniform(q), "input_tokens": 3} for qid, q in questions.items()}


def tilt(probs, d):
    """Shift the first option by `d` in log space: centred-log gap d*(1-1/n)."""
    first = next(iter(probs))
    raw = {k: p * (math.exp(d) if k == first else 1) for k, p in probs.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def answer(questions, behaviour):
    preds = reference(questions)
    for row in preds.values():
        if behaviour == "marginal":
            row["probabilities"] = tilt(row["probabilities"], 0.24)  # inside (mean_pass, mean_fail)
        elif behaviour == "fail":
            row["probabilities"] = tilt(row["probabilities"], 5.0)
        elif behaviour == "mismatch":
            row["probabilities"] = {"bogus": 1.0}
    return {"payload": {"predictions": preds}, "signature": "00" * 64}


class FakeWorker:
    def __init__(self, on_close=None):
        self.closed, self.on_close = False, on_close

    def close(self):
        if self.on_close:
            self.on_close()
        self.closed = True


class FakeServed:
    def __init__(self, worker, model=MODEL, sha=SHA, on_predict=None):
        self.worker, self.model, self.weights_sha256, self.max_input_tokens = (
            worker,
            model,
            sha,
            8192,
        )
        self.calls, self.on_predict = [], on_predict

    def predict(self, state, questions):
        if self.on_predict:
            self.on_predict()
        self.calls.append((state, questions))
        return reference(questions)


def lock_is_free(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


class GraderTestBase(unittest.TestCase):
    def setUp(self):
        from zils import serving_grader

        self.sg = serving_grader
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        (self.tmp / "state").mkdir()
        self.lock = str(self.tmp / "compute.lock")
        env = patch.dict(os.environ, {"ZILS_COMPUTE_LOCK": self.lock})
        env.start()
        self.addCleanup(env.stop)
        self.behaviour = {}  # hotkey -> "pass" | "marginal" | "fail" | "mismatch" | MinerError
        self.catalog = {}  # uid -> (hotkey, endpoint, manifest models or {"error"})
        self.asked, self.events, self.starts = [], [], 0
        self.on_predict, self.on_ask, self.start_error = None, None, None
        self.served_sha = SHA
        quiet = patch.object(serving_grader, "log", lambda text: None)
        quiet.start()
        self.addCleanup(quiet.stop)
        discover = patch.object(serving, "discover", self.discover)
        discover.start()
        self.addCleanup(discover.stop)

    def config(self, **extra):
        return {
            "chain": {"network": "test", "netuid": 579},
            "validator_hotkey": VALIDATOR,
            "request_key_file": str(self.tmp / "key.json"),
            "models": {MODEL: {"reference": str(self.tmp / "ckpt"), "weights_sha256": SHA}},
            "state": str(self.tmp / "state"),
            "questions": 8,
            "verified": 3,
            "lock_wait_seconds": 1,
            **extra,
        }

    def add_miner(self, i, behaviour="pass", served=None):
        hotkey = MINERS[i].ss58_address
        models = (
            served
            if served is not None
            else [{"model": MODEL, "weights_sha256": SHA, "max_input_tokens": 8192}]
        )
        self.catalog[i] = (hotkey, f"http://1.2.3.{i + 4}:8091", models)
        self.behaviour[hotkey] = behaviour
        return hotkey

    def catalog_read(self):
        return {uid: (hk, ep) for uid, (hk, ep, _) in self.catalog.items()}

    def discover(self, axons, clock=time.time, **_):
        out = {}
        for uid, (hk, ep) in axons.items():
            models = self.catalog[uid][2]
            out[uid] = (
                dict(models)
                if isinstance(models, dict)
                else {"hotkey": hk, "endpoint": ep, "models": models}
            )
        return out

    def worker(self):
        self.starts += 1
        self.events.append("start" if lock_is_free(self.lock) else "start-locked")
        if self.start_error:
            raise self.start_error
        self.fake_worker = FakeWorker(
            lambda: self.events.append("close" if lock_is_free(self.lock) else "close-locked")
        )
        self.served = FakeServed(
            self.fake_worker, sha=self.served_sha, on_predict=self.predict_hook
        )
        return [self.served]

    def predict_hook(self):
        self.events.append("predict")
        if self.on_predict:
            self.on_predict()

    def ask(self, endpoint, miner_hotkey, key, model, weights_sha256, body, timeout):
        self.assertIs(key, REQUEST_KEY)
        self.assertEqual((model, weights_sha256, timeout), (MODEL, SHA, self.sg.ASK_TIMEOUT))
        self.asked.append((miner_hotkey, body))
        self.events.append("ask")
        if self.on_ask:
            self.on_ask(miner_hotkey)
        behaviour = self.behaviour[miner_hotkey]
        if isinstance(behaviour, set):  # wrong only on these item indexes
            index = sum(hk == miner_hotkey for hk, _ in self.asked) - 1
            behaviour = "fail" if index in behaviour else "pass"
        if isinstance(behaviour, serving.MinerError):
            raise behaviour
        return answer(body["questions"], behaviour)

    def grader(self, **extra):
        g = self.sg.Grader(
            self.config(**extra), self.catalog_read, self.worker, REQUEST_KEY, ask=self.ask
        )
        return g

    def evidence(self):
        path = self.tmp / "state" / "serving-evidence.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def scores_file(self):
        return json.loads((self.tmp / "state" / "serving-scores.json").read_text())


class OutcomeTests(GraderTestBase):
    def test_pass_marginal_fail_mismatch_are_scored(self):
        keys = {
            b: self.add_miner(i, b) for i, b in enumerate(["pass", "marginal", "fail", "mismatch"])
        }
        before = time.time()
        self.grader().run_round("r1")
        rows = {r["hotkey"]: r for r in self.evidence()}
        for b, hk in keys.items():
            self.assertEqual(rows[hk]["outcome"], b)
        scores = self.scores_file()
        self.assertEqual(scores[keys["pass"]][MODEL]["score"], 1.0)
        self.assertEqual(scores[keys["pass"]][MODEL]["struck_until"], 0)
        self.assertEqual(scores[keys["marginal"]][MODEL]["score"], 0)
        self.assertEqual(scores[keys["marginal"]][MODEL]["struck_until"], 0)
        for b in ("fail", "mismatch"):
            summary = scores[keys[b]][MODEL]
            self.assertEqual(summary["score"], 0)
            self.assertGreaterEqual(summary["struck_until"], before + 24 * HOUR)

    def test_evidence_row_shape_and_non_finite_as_null(self):
        hk = self.add_miner(0, "mismatch")
        self.grader().run_round("r1")
        path = self.tmp / "state" / "serving-evidence.jsonl"
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        [row] = self.evidence()
        self.assertEqual(
            set(row),
            {
                "v",
                "kind",
                "at",
                "round_id",
                "seed",
                "hotkey",
                "model",
                "outcome",
                "mean",
                "worst",
                "latency_s",
                "verified",
                "messages",
            },
        )
        self.assertEqual((row["v"], row["kind"], row["round_id"]), (1, "validator", "r1"))
        self.assertEqual(row["seed"], serving_items.round_seed(VALIDATOR, "r1", hk, MODEL))
        self.assertIsNone(row["mean"])
        self.assertIsNone(row["worst"])
        self.assertEqual(len(row["messages"]), 8)

    def test_miner_error_counts_against_availability_not_a_strike(self):
        hk = self.add_miner(0, serving.MinerError("busy"))
        self.grader().run_round("r1")
        [row] = self.evidence()
        self.assertEqual(row["outcome"], "busy")
        self.assertIsNone(row["verified"])
        summary = self.scores_file()[hk][MODEL]
        self.assertEqual((summary["rounds"], summary["availability"]), (1, 0))
        self.assertEqual((summary["struck_until"], summary["score"]), (0, 0))
        self.assertEqual(self.starts, 0)  # nobody answered: the model is never loaded

    def test_first_miner_error_ends_that_miners_round(self):
        self.add_miner(0, "pass")
        calls = []

        def flaky(miner):
            calls.append(miner)
            if len(calls) == 3:
                raise serving.MinerError("timeout")

        self.on_ask = flaky
        self.grader().run_round("r1")
        self.assertEqual(len(self.asked), 3)
        self.assertEqual(self.evidence()[0]["outcome"], "timeout")

    def test_only_configured_model_and_weights_are_graded(self):
        good = self.add_miner(0)
        self.add_miner(
            1, served=[{"model": MODEL, "weights_sha256": "cd" * 32, "max_input_tokens": 8192}]
        )
        self.add_miner(
            2,
            served=[{"model": "d3-lite-0.8b-v1", "weights_sha256": SHA, "max_input_tokens": 4096}],
        )
        self.add_miner(3, served={"error": "manifest fetch failed"})
        self.grader().run_round("r1")
        self.assertEqual({hk for hk, _ in self.asked}, {good})
        self.assertEqual([r["hotkey"] for r in self.evidence()], [good])


class SecretSubsetTests(GraderTestBase):
    def test_subset_chosen_after_all_answers_and_only_it_is_computed(self):
        hk = self.add_miner(0)
        self.grader().run_round("r1")
        self.assertEqual(
            self.events, ["ask"] * 8 + ["start-locked"] + ["predict"] * 3 + ["close-locked"]
        )
        # the seed is secret because production round_ids are fresh uuid4s
        seed = serving_items.round_seed(VALIDATOR, "r1", hk, MODEL)
        bodies = serving_items.items(seed, 8)
        self.assertEqual([b for _, b in self.asked], bodies)
        verified = sorted(random.Random(seed ^ 1).sample(range(8), 3))
        self.assertEqual(self.evidence()[0]["verified"], verified)
        self.assertEqual(
            self.served.calls, [(bodies[i]["state"], bodies[i]["questions"]) for i in verified]
        )

    def test_compare_sees_only_verified_items(self):
        self.add_miner(0)
        seen = []
        real = self.sg.answer_check.compare

        def spy(ours, theirs):
            seen.append((set(ours), set(theirs)))
            return real(ours, theirs)

        with patch.object(self.sg.answer_check, "compare", spy):
            self.grader().run_round("r1")
        [(ours, theirs)] = seen
        verified = self.evidence()[0]["verified"]
        self.assertEqual(ours, theirs)
        self.assertEqual({int(k.split(":")[0]) for k in ours}, set(verified))

    def test_same_seed_regenerates_same_questions(self):
        hk = self.add_miner(0)
        self.grader().run_round("r1")
        first = [b for _, b in self.asked]
        self.asked.clear()
        self.grader().run_round("r1")
        self.assertEqual([b for _, b in self.asked], first)
        self.asked.clear()
        self.grader().run_round("r2")
        self.assertNotEqual([b for _, b in self.asked], first)
        seed = serving_items.round_seed(VALIDATOR, "r1", hk, MODEL)
        self.assertEqual(first, serving_items.items(seed, 8))


class LockAndReferenceTests(GraderTestBase):
    def hold_lock(self):
        fd = os.open(self.lock, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        self.addCleanup(os.close, fd)
        return fd

    def test_model_loads_and_unloads_inside_the_lock_every_round(self):
        self.add_miner(0)
        self.add_miner(1)
        self.on_ask = lambda _: self.assertTrue(lock_is_free(self.lock))
        self.on_predict = lambda: self.assertFalse(lock_is_free(self.lock))
        g = self.grader()
        for r in ("r1", "r2"):
            self.events.clear()
            g.run_round(r)
            self.assertEqual(
                self.events,
                ["ask"] * 16 + ["start-locked"] + ["predict"] * 6 + ["close-locked"],
            )
            self.assertTrue(self.fake_worker.closed)
        self.assertEqual(self.starts, 2)
        self.assertEqual([r["outcome"] for r in self.evidence()], ["pass"] * 4)
        self.assertTrue(lock_is_free(self.lock))

    def test_busy_lock_skips_the_round_without_asking(self):
        hk = self.add_miner(0)
        g = self.grader()
        g.run_round("r1")
        before = self.evidence()
        fd = self.hold_lock()
        self.asked.clear()
        self.assertEqual(g.run_round("r2"), {"skipped": 1})
        fcntl.flock(fd, fcntl.LOCK_UN)
        self.assertEqual(self.asked, [])
        self.assertEqual(self.evidence(), before)
        self.assertEqual(self.starts, 1)
        self.assertEqual(self.scores_file()[hk][MODEL]["rounds"], 1)  # still rewritten

    def test_unopenable_lock_skips_the_round_without_asking(self):
        self.add_miner(0)
        with patch.dict(os.environ, {"ZILS_COMPUTE_LOCK": str(self.tmp / "missing" / "x.lock")}):
            self.assertEqual(self.grader().run_round("r1"), {"skipped": 1})
        self.assertEqual((self.asked, self.evidence(), self.scores_file()), ([], [], {}))

    def test_lock_timeout_after_asks_never_touches_scores(self):
        a, b = self.add_miner(0), self.add_miner(1)
        held = []
        self.on_ask = lambda _: len(self.asked) == 16 and held.append(self.hold_lock())
        self.grader(lock_wait_seconds=0).run_round("r1")
        self.assertEqual(
            {r["hotkey"]: r["outcome"] for r in self.evidence()},
            {a: "lock_timeout", b: "lock_timeout"},
        )
        self.assertEqual(self.scores_file(), {})
        self.assertEqual(self.starts, 0)

    def test_lock_open_error_after_asks_is_lock_timeout(self):
        a, b = self.add_miner(0), self.add_miner(1)
        missing = str(self.tmp / "missing" / "x.lock")
        self.on_ask = lambda _: (
            len(self.asked) == 16 and os.environ.update(ZILS_COMPUTE_LOCK=missing)
        )
        self.grader().run_round("r1")
        self.assertEqual(
            {r["hotkey"]: r["outcome"] for r in self.evidence()},
            {a: "lock_timeout", b: "lock_timeout"},
        )
        self.assertEqual(self.scores_file(), {})

    def test_worker_crash_keeps_earlier_verdicts_and_never_scores_the_rest(self):
        self.add_miner(0)
        self.add_miner(1)
        self.add_miner(2)
        calls = []

        def crash_on_second_miner():
            calls.append(1)
            if len(calls) == 4:
                raise OSError("serving worker exited")

        self.on_predict = crash_on_second_miner
        g = self.grader()
        g.run_round("r1")
        outcomes = [r["outcome"] for r in self.evidence()]
        self.assertEqual(outcomes, ["pass", "reference_unavailable", "reference_unavailable"])
        self.assertTrue(self.fake_worker.closed)
        self.assertEqual(len(self.scores_file()), 1)
        self.on_predict = None
        g.run_round("r2")
        self.assertEqual(self.starts, 2)
        self.assertEqual([r["outcome"] for r in self.evidence()][3:], ["pass"] * 3)

    def test_worker_refusal_is_reference_unavailable(self):
        def refuse():
            raise serving_server.WorkerError("runtime")

        a, b = self.add_miner(0), self.add_miner(1)
        first = []
        self.on_ask = lambda miner: first or first.append(miner)
        calls = []
        self.on_predict = lambda: (calls.append(1), len(calls) == 1 and refuse())
        self.grader().run_round("r1")
        outcomes = {r["hotkey"]: r["outcome"] for r in self.evidence()}
        other = ({a, b} - set(first)).pop()
        self.assertEqual(outcomes, {first[0]: "reference_unavailable", other: "pass"})
        self.assertEqual(self.starts, 1)
        self.assertEqual(set(self.scores_file()), {other})

    def test_unexpected_reference_error_unloads_and_records_evidence(self):
        def boom():
            raise RuntimeError("bug on our side")

        self.add_miner(0)
        self.on_predict = boom
        self.grader().run_round("r1")
        self.assertEqual(self.evidence()[0]["outcome"], "reference_unavailable")
        self.assertEqual(self.scores_file(), {})
        self.assertTrue(self.fake_worker.closed)
        self.assertTrue(lock_is_free(self.lock))

    def test_start_failure_is_reference_unavailable_and_retried(self):
        self.add_miner(0)
        self.start_error = ValueError("serving worker failed to start")
        g = self.grader()
        g.run_round("r1")
        self.assertEqual(self.evidence()[0]["outcome"], "reference_unavailable")
        self.assertEqual(self.scores_file(), {})
        self.assertTrue(lock_is_free(self.lock))
        self.start_error = None
        g.run_round("r2")
        self.assertEqual(self.starts, 2)
        self.assertEqual(self.evidence()[1]["outcome"], "pass")

    def test_refuses_worker_with_other_weights_or_missing_model(self):
        self.add_miner(0)
        self.served_sha = "cd" * 32
        with self.assertRaises(self.sg.ReferenceMismatch):
            self.grader().run_round("r1")
        self.assertTrue(self.fake_worker.closed)
        self.assertTrue(lock_is_free(self.lock))
        self.served_sha = SHA
        extra = {
            MODEL: {"reference": "x", "weights_sha256": SHA},
            "d3-lite-0.8b-v1": {"reference": "y", "weights_sha256": SHA},
        }
        with self.assertRaises(self.sg.ReferenceMismatch):
            self.grader(models=extra).run_round("r1")
        self.assertTrue(self.fake_worker.closed)

    def test_worker_serving_nothing_is_refused(self):
        self.add_miner(0)
        g = self.grader()
        g.worker = lambda: []
        with self.assertRaises(self.sg.ReferenceMismatch):
            g.run_round("r1")


class LockPathTests(unittest.TestCase):
    def test_lock_path_uses_settings(self):
        from zils import serving_grader

        with patch.object(serving_grader.settings, "get", return_value="/x/lock") as get:
            self.assertEqual(serving_grader.lock_path(), "/x/lock")
        self.assertEqual(get.call_args.args[0], "ZILS_COMPUTE_LOCK")
        self.assertTrue(get.call_args.args[1].endswith(f"fez-compute-{os.getuid()}-cuda.lock"))

    def test_lock_path_honours_legacy_name(self):
        from zils import serving_grader

        env = {k: v for k, v in os.environ.items() if k != "ZILS_COMPUTE_LOCK"}
        with patch.dict(os.environ, {**env, "FEZ_COMPUTE_LOCK": "/legacy/lock"}, clear=True):
            self.assertEqual(serving_grader.lock_path(), "/legacy/lock")


class RoundTests(GraderTestBase):
    def test_round_budget_skips_later_miners(self):
        self.add_miner(0)
        self.add_miner(1)
        self.on_ask = lambda miner: time.sleep(0.15)
        self.grader(round_seconds=1).run_round("r1")
        [asked] = {hk for hk, _ in self.asked}
        self.assertEqual([r["hotkey"] for r in self.evidence()], [asked])

    def test_miner_order_is_shuffled_per_round(self):
        for i in range(5):
            self.add_miner(i)
        g = self.grader()
        firsts = set()
        for r in range(10):
            self.asked.clear()
            g.run_round(f"r{r}")
            firsts.add(self.asked[0][0])
        self.assertGreater(len(firsts), 1)

    def test_partial_cheater_caught_only_on_verified_items(self):
        hk = MINERS[0].ss58_address
        seed = serving_items.round_seed(VALIDATOR, "r1", hk, MODEL)
        verified = sorted(random.Random(seed ^ 1).sample(range(8), 3))
        self.add_miner(0, set(range(8)) - set(verified))
        self.grader().run_round("r1")
        self.assertEqual(self.evidence()[-1]["outcome"], "pass")
        self.asked.clear()
        self.add_miner(0, {verified[1]})
        self.grader().run_round("r1")
        self.assertEqual(self.evidence()[-1]["outcome"], "fail")
        self.assertGreater(self.scores_file()[hk][MODEL]["struck_until"], 0)

    def test_evidence_written_before_scores(self):
        hk = self.add_miner(0)
        real = self.sg.Scores.record

        def record(scores, hotkey, model, result):
            self.assertIn(hotkey, [r["hotkey"] for r in self.evidence()])
            return real(scores, hotkey, model, result)

        with patch.object(self.sg.Scores, "record", record):
            self.grader().run_round("r1")
        self.assertIn(hk, self.scores_file())

    def test_score_file_write_is_atomic(self):
        hk = self.add_miner(0)
        self.grader().run_round("r1")
        before = (self.tmp / "state" / "serving-scores.json").read_text()
        with patch.object(self.sg.os, "replace", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                self.grader().run_round("r2")
        self.assertEqual((self.tmp / "state" / "serving-scores.json").read_text(), before)
        self.assertIn(hk, json.loads(before))
        leftovers = [p.name for p in (self.tmp / "state").iterdir()]
        self.assertEqual(
            sorted(n for n in leftovers if not n.startswith("serving-")), [], leftovers
        )


class ScoresTests(unittest.TestCase):
    def setUp(self):
        from zils import serving_grader

        self.sg = serving_grader
        self.path = Path(tempfile.mkdtemp()) / "rounds.json"

    def result(self, at, outcome, latency=0.5):
        return {"at": at, "outcome": outcome, "mean": 0.0, "worst": 0.0, "latency_s": latency}

    def test_strike_zeroes_score_for_window_then_recovers(self):
        s = self.sg.Scores(self.path, window_hours=24, floor=0.9)
        t0 = 1_000_000
        s.record("hk", MODEL, self.result(t0, "fail"))
        for h in range(1, 24):
            s.record("hk", MODEL, self.result(t0 + h * HOUR, "pass"))
        early = s.summary("hk", MODEL, now=t0 + 23 * HOUR)
        self.assertEqual(early["struck_until"], t0 + 24 * HOUR)
        self.assertEqual(early["score"], 0)
        s.record("hk", MODEL, self.result(t0 + 25 * HOUR, "pass"))
        later = s.summary("hk", MODEL, now=t0 + 25 * HOUR)
        self.assertEqual(later["pass_rate"], 1.0)
        self.assertEqual(later["score"], 1.0)

    def test_availability_counts_busy_timeout_unavailable(self):
        s = self.sg.Scores(self.path, window_hours=24, floor=0.9)
        for i, outcome in enumerate(["pass", "busy", "timeout", "unavailable", "http"]):
            s.record("hk", MODEL, self.result(100 + i, outcome, latency=i))
        summary = s.summary("hk", MODEL, now=200)
        self.assertEqual(summary["rounds"], 5)
        self.assertAlmostEqual(summary["availability"], 0.2)
        self.assertEqual(summary["pass_rate"], 1.0)
        self.assertAlmostEqual(summary["score"], 0.2)
        self.assertEqual(summary["median_latency_s"], 0)

    def test_marginal_lowers_pass_rate_below_floor_without_strike(self):
        s = self.sg.Scores(self.path, window_hours=24, floor=0.9)
        for i in range(9):
            s.record("hk", MODEL, self.result(100 + i, "pass"))
        s.record("hk", MODEL, self.result(110, "marginal"))
        summary = s.summary("hk", MODEL, now=200)
        self.assertEqual((summary["pass_rate"], summary["struck_until"]), (0.9, 0))
        self.assertAlmostEqual(summary["score"], 0.9)
        s.record("hk", MODEL, self.result(111, "marginal"))
        self.assertEqual(s.summary("hk", MODEL, now=200)["score"], 0)

    def test_persists_and_prunes(self):
        s = self.sg.Scores(self.path, window_hours=1, floor=0.9)
        s.record("hk", MODEL, self.result(0, "pass"))
        s.record("hk", MODEL, self.result(2 * HOUR, "pass"))
        again = self.sg.Scores(self.path, window_hours=1, floor=0.9)
        self.assertEqual(again.summary("hk", MODEL, now=2 * HOUR)["rounds"], 1)

    def test_refuses_our_own_failures(self):
        s = self.sg.Scores(self.path, window_hours=24, floor=0.9)
        for outcome in ("reference_unavailable", "lock_timeout"):
            with self.assertRaises(ValueError):
                s.record("hk", MODEL, self.result(1, outcome))


class StubServed:
    model, weights_sha256, max_input_tokens = MODEL, SHA, 8192

    def __init__(self):
        self.worker = FakeWorker()

    def predict(self, state, questions):
        return reference(questions)


class RealServerTests(unittest.TestCase):
    def test_round_against_the_pinned_miner_server(self):
        from zils import serving_grader

        tmp = Path(tempfile.mkdtemp())
        (tmp / "state").mkdir()
        miner = MINERS[0]
        with socket.socket() as probe:  # the manifest signs its endpoint, so pick the port first
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        endpoint = "http://127.0.0.1:%d" % port
        server = serving_server.make_server(
            miner,
            endpoint,
            [StubServed()],
            allowed=lambda caller: caller == REQUEST_KEY.ss58_address,
            bind=("127.0.0.1", port),
        )
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        config = {
            "chain": {"network": "test", "netuid": 579},
            "validator_hotkey": VALIDATOR,
            "request_key_file": str(tmp / "key.json"),
            "models": {MODEL: {"reference": str(tmp), "weights_sha256": SHA}},
            "state": str(tmp / "state"),
        }
        with (
            patch.dict(os.environ, {"ZILS_COMPUTE_LOCK": str(tmp / "lock")}),
            patch.object(serving, "public_endpoint", lambda e: e),
        ):
            g = serving_grader.Grader(
                config,
                lambda: {0: (miner.ss58_address, endpoint)},
                lambda: [StubServed()],
                REQUEST_KEY,
                ask=functools.partial(serving.ask, check=lambda e: e),
            )
            g.run_round("real")
        [row] = [
            json.loads(line)
            for line in (tmp / "state" / "serving-evidence.jsonl").read_text().splitlines()
        ]
        self.assertEqual(row["outcome"], "pass")
        for message in row["messages"]:
            serving.verify(message, "answer", miner.ss58_address, int(time.time()))
        scores = json.loads((tmp / "state" / "serving-scores.json").read_text())
        self.assertEqual(scores[miner.ss58_address][MODEL]["score"], 1.0)


def serving_grader_mismatch():
    from zils import serving_grader

    return serving_grader.ReferenceMismatch()


class Stop(BaseException):
    """Ends main's loop in a test without looking like any error main handles."""


class MainTests(unittest.TestCase):
    def test_imports_no_chain_writes(self):
        from zils import serving_grader

        tree = ast.parse(Path(serving_grader.__file__).read_text())
        used = {
            n.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "testnet"
        }
        self.assertLessEqual(used, {"connect", "serving_axons", "check_config"})
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        self.assertFalse(names & {"write_axon", "write_commitment", "publish_round", "set_weights"})
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
            self.assertNotIn("raw_calls", [k.arg for k in call.keywords])

    def setUp(self):
        self.tmp = tmp = Path(tempfile.mkdtemp())
        (tmp / "state").mkdir()
        (tmp / "ckpt").mkdir()
        (tmp / "key.json").write_text(json.dumps({"secretSeed": "0x" + "44" * 32}))
        self.base = {
            "chain": {"network": "test", "netuid": 579},
            "validator_hotkey": VALIDATOR,
            "request_key_file": str(tmp / "key.json"),
            "models": {MODEL: {"reference": str(tmp / "ckpt"), "weights_sha256": SHA}},
            "state": str(tmp / "state"),
        }
        (tmp / "config.json").write_text(json.dumps(self.base))
        self.lock = str(tmp / "compute.lock")
        env = patch.dict(os.environ, {"ZILS_COMPUTE_LOCK": self.lock})
        env.start()
        self.addCleanup(env.stop)

    def run_main(self, served_sha=SHA, rounds=(), expect=None):
        from zils import serving_grader

        closes, starts = [], []
        worker = FakeWorker(lambda: closes.append(lock_is_free(self.lock)))

        def start_worker(*a, **k):
            starts.append(lock_is_free(self.lock))
            return [FakeServed(worker, sha=served_sha)]

        with (
            patch.object(serving_server, "start_worker", start_worker),
            patch.object(serving_grader.Grader, "run_round", side_effect=[*rounds, Stop()]) as run,
            patch.object(serving_grader.time, "sleep", lambda s: None),
            patch.object(serving_grader, "log", lambda text: None),
            self.assertRaises(expect or serving_grader.ReferenceMismatch),
        ):
            serving_grader.main(["--config", str(self.tmp / "config.json")])
        self.starts, self.closes = starts, closes
        return run

    def test_main_startup_check_loads_and_unloads_under_the_lock(self):
        run = self.run_main(rounds=[serving_grader_mismatch()])
        self.assertEqual((self.starts, self.closes), ([False], [False]))
        self.assertEqual(run.call_count, 1)

    def test_main_runs_anyway_when_lock_busy_at_startup(self):
        fd = os.open(self.lock, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        self.addCleanup(os.close, fd)
        (self.tmp / "config.json").write_text(json.dumps({**self.base, "lock_wait_seconds": 0}))
        run = self.run_main(served_sha="cd" * 32, expect=Stop)
        self.assertEqual(self.starts, [])
        self.assertEqual(run.call_count, 1)

    def test_main_stops_on_reference_mismatch_at_startup(self):
        run = self.run_main(served_sha="cd" * 32)
        self.assertEqual(run.call_count, 0)
        self.assertEqual(self.closes, [False])

    def test_main_logs_other_errors_and_continues(self):
        from zils import serving_grader

        rounds = [ValueError("chain"), RuntimeError("disk"), {}, serving_grader.ReferenceMismatch()]
        self.assertEqual(self.run_main(rounds=rounds).call_count, 4)

    def test_main_refuses_state_that_is_not_a_directory(self):
        from zils import serving_grader

        (self.tmp / "file").write_text("")
        (self.tmp / "config.json").write_text(
            json.dumps({**self.base, "state": str(self.tmp / "file")})
        )
        with (
            patch.object(serving_grader.time, "sleep", side_effect=Stop()),
            self.assertRaises(SystemExit),
        ):
            serving_grader.main(["--config", str(self.tmp / "config.json")])

    def test_main_refuses_missing_paths(self):
        from zils import serving_grader

        tmp = self.tmp
        base = {
            "chain": {"network": "test", "netuid": 579},
            "validator_hotkey": VALIDATOR,
            "request_key_file": str(tmp / "key.json"),
            "models": {MODEL: {"reference": str(tmp / "ckpt"), "weights_sha256": SHA}},
            "state": str(tmp / "state"),
        }
        for broken in (
            {"request_key_file": str(tmp / "nokey.json")},
            {"models": {MODEL: {"reference": str(tmp / "nockpt"), "weights_sha256": SHA}}},
            {"state": str(tmp / "nostate")},
        ):
            path = tmp / "config.json"
            path.write_text(json.dumps({**base, **broken}))
            with self.assertRaises(SystemExit):
                serving_grader.main(["--config", str(path)])

    def test_load_key_matches_platform_format(self):
        from zils import serving_grader

        path = Path(tempfile.mkdtemp()) / "key.json"
        path.write_text(json.dumps({"secretSeed": "0x" + "44" * 32}))
        self.assertEqual(serving_grader.load_key(path).ss58_address, REQUEST_KEY.ss58_address)


class StartWorkerTests(unittest.TestCase):
    def test_worker_serving_nothing_is_closed_and_refused(self):
        import sys

        script = 'import json,sys; print(json.dumps({"ready": True, "served": []}), flush=True); sys.stdin.read()'
        closed = []
        real = serving_server.Worker.close

        def spy(worker):
            closed.append(worker)
            return real(worker)

        with patch.object(serving_server.Worker, "close", spy), self.assertRaises(ValueError):
            serving_server.start_worker([], sys.executable, args=("-c", script))
        self.assertEqual(len(closed), 1)
        self.assertIsNotNone(closed[0].process.poll())


if __name__ == "__main__":
    unittest.main()
