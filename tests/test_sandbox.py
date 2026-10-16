"""Submission isolation: mode detection, command wrapping and report labelling."""

import unittest
from unittest.mock import patch

from zils import sandbox


class SandboxTest(unittest.TestCase):
    def setUp(self):
        sandbox.mode.cache_clear()

    def tearDown(self):
        sandbox.mode.cache_clear()

    def test_forced_mode_and_rejection(self):
        with patch("zils.sandbox.settings.get", return_value="none"):
            self.assertEqual(sandbox.mode(), "none")
            self.assertEqual(sandbox.wrap(["python", "-m", "x"]), ["python", "-m", "x"])
        sandbox.mode.cache_clear()
        with (
            patch("zils.sandbox.settings.get", return_value="chroot"),
            self.assertRaisesRegex(ValueError, "ZILS_SANDBOX"),
        ):
            sandbox.mode()

    def test_unshare_used_only_when_it_works(self):
        with (
            patch("zils.sandbox.settings.get", return_value=None),
            patch("zils.sandbox.shutil.which", return_value="/usr/bin/unshare"),
            patch("zils.sandbox._works", return_value=True),
        ):
            self.assertEqual(sandbox.mode(), "unshare")
            wrapped = sandbox.wrap(["runner", "--device", "cuda"])
            self.assertEqual(
                wrapped[:5], ["/usr/bin/unshare", "--net", "--map-root-user", "--kill-child", "--"]
            )
            self.assertEqual(wrapped[5:], ["runner", "--device", "cuda"])
        sandbox.mode.cache_clear()
        with (
            patch("zils.sandbox.settings.get", return_value=None),
            patch("zils.sandbox.shutil.which", return_value="/usr/bin/unshare"),
            patch("zils.sandbox._works", return_value=False),
        ):
            self.assertEqual(sandbox.mode(), "none")


if __name__ == "__main__":
    unittest.main()
