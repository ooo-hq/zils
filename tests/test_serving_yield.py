import fcntl
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from zils import models, serving_yield
from zils.runtime import CapacityUnavailable


class ServingYieldTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        env = patch.dict(
            os.environ, {"ZILS_SERVING_DIRECTORY": str(self.root), "ZILS_SERVING_YIELD_MIB": "9216"}
        )
        env.start()
        self.addCleanup(env.stop)

    def state(self, **values):
        (self.root / "state.json").write_text(json.dumps({"checked_at": time.time(), **values}))

    def test_advertises_reclaimable_capacity_only_from_a_live_serving_controller(self):
        with (
            patch("zils.runtime.gpu_ready", return_value=False),
            patch(
                "zils.serving_yield.subprocess.run",
                return_value=SimpleNamespace(stdout="11019, 20475"),
            ),
        ):
            self.assertFalse(serving_yield.ready("cuda", model=models.D3, minimum_mib=12288))
            self.state(serving=True)
            self.assertTrue(serving_yield.ready("cuda", model=models.D3, minimum_mib=12288))
            self.assertFalse(serving_yield.ready("cuda", model=models.D3, minimum_mib=21000))
            with patch.dict(os.environ, {"ZILS_D3_MIN_FREE_MIB": "22000"}):
                self.assertFalse(serving_yield.ready("cuda", model=models.D3, minimum_mib=12288))
            self.state(serving=False)
            self.assertFalse(serving_yield.ready("cuda", model=models.D3, minimum_mib=12288))
            self.state(serving=True, checked_at=time.time() - 60)
            self.assertFalse(serving_yield.ready("cuda", model=models.D3, minimum_mib=12288))

    def test_free_memory_does_not_bypass_the_stop_acknowledgement(self):
        self.state(serving=False, yielded=["stale.request"])
        with patch("zils.runtime.gpu_ready", return_value=True):
            with self.assertRaises(CapacityUnavailable):
                with serving_yield.yield_gpu("cuda", model=models.D3, timeout=0):
                    self.fail("Unacknowledged work must not start")
        self.assertEqual(list(self.root.glob("*.request")), [])

    def test_invalid_acknowledgement_cannot_start_work(self):
        with (
            patch("zils.serving_yield.state", return_value={"serving": False, "yielded": None}),
            patch("zils.runtime.gpu_ready", return_value=True),
        ):
            with self.assertRaises(CapacityUnavailable):
                with serving_yield.yield_gpu("cuda", timeout=0):
                    self.fail("Malformed acknowledgement must not allow work")

    def test_yield_stays_locked_through_work_and_cleans_up_on_error(self):
        stop = threading.Event()

        def controller():
            while not stop.wait(0.005):
                requests = serving_yield.active_requests(self.root)
                if requests:
                    self.state(serving=False, yielded=requests)
                    return

        thread = threading.Thread(target=controller)
        thread.start()
        try:
            with patch("zils.runtime.gpu_ready", return_value=True):
                with self.assertRaisesRegex(ValueError, "training failed"):
                    with serving_yield.yield_gpu("cuda", model=models.D3, timeout=2, poll=0.01):
                        self.assertEqual(len(serving_yield.active_requests(self.root)), 1)
                        raise ValueError("training failed")
        finally:
            stop.set()
            thread.join()
        self.assertEqual(serving_yield.active_requests(self.root), [])
        self.assertEqual(list(self.root.glob("*.request")), [])

    def test_crashed_request_is_ignored_without_deleting_another_workers_request(self):
        stale = self.root / "stale.request"
        stale.touch()
        active = self.root / "active.request"
        with active.open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(serving_yield.active_requests(self.root), ["active.request"])
            self.assertTrue(stale.exists())
        self.assertEqual(serving_yield.active_requests(self.root), [])

    def test_lost_lease_during_handoff_removes_request(self):
        def expired():
            raise RuntimeError("lease expired")

        with self.assertRaisesRegex(RuntimeError, "lease expired"):
            with serving_yield.yield_gpu("cuda", model=models.D3, check_lease=expired):
                self.fail("Expired work must not run")
        self.assertEqual(list(self.root.glob("*.request")), [])

    def test_dead_parent_keeps_serving_paused_until_its_supervised_children_exit(self):
        request = self.root / "crashed.request"
        request.write_text("old-invocation")
        self.assertEqual(
            serving_yield.active_requests(self.root, {"old-invocation"}), [request.name]
        )
        self.assertEqual(serving_yield.active_requests(self.root, {"new-invocation"}), [])

    def test_unconfigured_workers_keep_existing_capacity_behavior(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("zils.runtime.gpu_ready", return_value=True),
        ):
            self.assertTrue(serving_yield.ready("cpu"))
            with serving_yield.yield_gpu("cpu"):
                self.assertEqual(list(self.root.glob("*.request")), [])

    def test_empty_queue_does_not_request_a_serving_pause(self):
        from miner.queue import run_once

        class Client:
            def call(self, action, body):
                assert action == "claim"
                return {"assignment": None}

        with patch("miner.queue.gpu_ready", return_value=True):
            self.assertFalse(run_once(Client(), self.root, None, "python", "cuda"))
        self.assertEqual(list(self.root.glob("*.request")), [])
