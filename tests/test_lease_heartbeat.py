"""A lease heartbeat survives a few renewals the coordinator was too slow or too broken to answer, and stops
at once when the coordinator refuses the lease."""

import threading
import unittest

from zils.cloud import APIError
from zils.queue_protocol import RENEW_ATTEMPTS, lease_heartbeat


class Renewals:
    """The opening renewal succeeds; later ones follow the script (an error to raise, or None), then succeed."""

    def __init__(self, *script):
        self.script, self.calls, self.done = list(script), 0, threading.Event()

    def __call__(self):
        self.calls += 1
        if self.calls == 1 or not self.script:
            return
        outcome = self.script.pop(0)
        if not self.script:
            self.done.set()
        if outcome is not None:
            raise outcome


def misses(count):
    return [APIError(503, "Coordinator unavailable.") for _ in range(count)]


class LeaseHeartbeatTest(unittest.TestCase):
    def run_heartbeat(self, renew):
        with lease_heartbeat(renew, interval=0.01) as check_lease:
            self.assertTrue(renew.done.wait(5))
            threading.Event().wait(0.05)  # let the heartbeat act on the last scripted renewal
            check_lease()

    def test_missed_renewals_are_retried_and_a_success_resets_the_count(self):
        few = RENEW_ATTEMPTS - 1
        renew = Renewals(*misses(few), None, *misses(few), None)
        self.run_heartbeat(renew)
        self.assertGreater(renew.calls, 1 + 2 * few + 2)  # still renewing after the script

    def test_repeated_misses_end_the_attempt(self):
        renew = Renewals(*misses(RENEW_ATTEMPTS))
        with self.assertRaisesRegex(APIError, "Lease renewal failed"):
            self.run_heartbeat(renew)
        self.assertEqual(renew.calls, 1 + RENEW_ATTEMPTS)

    def test_a_refusal_ends_the_attempt_at_once(self):
        for error in (
            APIError(409, "Lease expired."),
            APIError(403, "Not admitted."),
            ValueError(),
        ):
            renew = Renewals(error)
            with self.assertRaisesRegex(APIError, "Lease renewal failed"):
                self.run_heartbeat(renew)
            self.assertEqual(renew.calls, 2)


if __name__ == "__main__":
    unittest.main()
