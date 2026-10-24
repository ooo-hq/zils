"""A queue attempt keeps the shared compute lock across evaluation steps; waiting is bounded.

runtime.py is pinned by exam qualification, so the lease lives in base_queue and run_child is
pointed at a private lock while it is held."""

import fcntl
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zils import runtime
from zils.base_queue import ComputeLease
from zils.runtime import CapacityUnavailable, run_child


def held_elsewhere(lock):
    with open(lock, "w") as other:
        try:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


class ComputeLeaseTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.lock = self.root / "compute.lock"
        self.enterContext(patch.dict(os.environ, {"ZILS_COMPUTE_LOCK": str(self.lock)}))
        self.enterContext(patch("zils.runtime.gpu_ready", return_value=True))
        self.enterContext(patch("zils.base_queue.LEASE_POLL", 0.01))

    def lease(self, wait=60):
        return ComputeLease("cuda", wait, self.root / "private.lock")

    def step(self):
        run_child([sys.executable, "-c", "pass"], self.root / "child.log", "cuda", minimum_mib=1)

    def test_lock_is_kept_between_steps_and_released_after(self):
        seen = []
        real = runtime._run_child

        def child(*args, **kwargs):
            seen.append(held_elsewhere(self.lock))
            return real(*args, **kwargs)

        lease = self.lease()
        with patch("zils.runtime._run_child", side_effect=child):
            lease.acquire()
            self.step()
            self.assertTrue(held_elsewhere(self.lock))  # still held between steps
            lease.acquire()  # idempotent
            self.step()
        lease.release()
        self.assertEqual(seen, [True, True])
        self.assertFalse(held_elsewhere(self.lock))
        self.assertEqual(os.environ["ZILS_COMPUTE_LOCK"], str(self.lock))
        self.assertLess(lease.waited, 1)

    def test_busy_lock_waits_a_bounded_time_and_reports_it(self):
        lease = self.lease(0.1)
        with open(self.lock, "w") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            with self.assertRaises(CapacityUnavailable):
                lease.acquire()
        lease.release()
        self.assertGreaterEqual(lease.waited, 0.1)
        self.assertFalse(held_elsewhere(self.lock))
        self.assertEqual(os.environ["ZILS_COMPUTE_LOCK"], str(self.lock))

    def test_without_a_lease_contention_still_defers_immediately(self):
        with open(self.lock, "w") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            with self.assertRaises(CapacityUnavailable):
                self.step()


if __name__ == "__main__":
    unittest.main()
