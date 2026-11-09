"""A queue miner names its installed release in every claim, and shows the coordinator's refusal when that release
is too old for customer jobs."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from miner import queue
from tests.test_fleet_profiles import checkpoint
from zils import models
from zils.cloud import APIError

UPDATE = (
    "This miner is at bbbbbbbbbbbb; customer jobs need a current release. Run zils-miner update."
)


class Client:
    def __init__(self):
        self.claims = []

    def call(self, action, body=None):
        if action == "profiles":
            identity = models.profile_identity(models.D3)
            return {"profiles": [{"profile_id": models.D3, "min_free_mib": 0, **identity}]}
        self.claims.append(body)
        return {"assignment": None}


class ReleaseClaimTest(unittest.TestCase):
    def claim(self, release):
        client = Client()
        with tempfile.TemporaryDirectory() as root:
            reference = checkpoint(Path(root) / "reference", models.D3)
            with (
                patch("miner.queue.gpu_ready", return_value=True),
                patch("miner.queue.installed_release", return_value=release),
            ):
                queue.run_once(
                    client,
                    Path(root) / "state",
                    None,
                    "python",
                    "cuda",
                    references={models.D3: reference},
                )
        return client.claims[0]

    def test_the_claim_names_the_installed_release(self):
        self.assertEqual(self.claim("a" * 40)["release"], "a" * 40)

    def test_an_install_without_git_sends_no_release(self):
        self.assertNotIn("release", self.claim(None))


class RefusalTest(unittest.TestCase):
    def response(self, status, error):
        response = Mock(status_code=status, content=b"{}")
        response.json.return_value = {"error": error}
        return response

    def test_an_outdated_release_shows_the_coordinators_message(self):
        client = queue.Client("https://training.example.com", Mock(ss58_address="hotkey"))
        with (
            patch("miner.queue.queue_protocol.sign", return_value={}),
            patch("miner.queue.requests.post", return_value=self.response(426, UPDATE)),
            self.assertRaises(APIError) as caught,
        ):
            client.call("claim", {})
        self.assertEqual(caught.exception.status, 426)
        self.assertEqual(str(caught.exception), UPDATE)
        self.assertIn("Run zils-miner update", queue.attempt_failed(caught.exception))

    def test_other_refusals_stay_generic(self):
        client = queue.Client("https://training.example.com", Mock(ss58_address="hotkey"))
        with (
            patch("miner.queue.queue_protocol.sign", return_value={}),
            patch("miner.queue.requests.post", return_value=self.response(409, "internal detail")),
            self.assertRaises(APIError) as caught,
        ):
            client.call("claim", {})
        self.assertNotIn("internal detail", str(caught.exception))
        self.assertEqual(
            queue.attempt_failed(caught.exception), "Queued worker attempt failed (APIError)."
        )


if __name__ == "__main__":
    unittest.main()
