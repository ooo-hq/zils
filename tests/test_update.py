"""`zils update` and the doctor's stable-release warning, without GitHub or the installer."""

import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from zils import miner

STABLE, OLD = "a" * 40, "b" * 40


def response(text):
    return patch("zils.miner.urllib.request.urlopen", return_value=io.BytesIO(text.encode()))


class UpdateTest(unittest.TestCase):
    def test_stable_pointer_is_a_full_sha(self):
        sha = (Path(miner.zils.ROOT) / "release/stable").read_text().strip()
        self.assertEqual(miner.commitments.check_revision(sha), sha)
        with response("main\n"), self.assertRaises(ValueError):
            miner.stable_release()

    def test_behind_release_warns_only_when_behind(self):
        with response(STABLE + "\n"), patch("zils.miner.installed_release", return_value=OLD):
            self.assertIn(
                "bbbbbbbbbbbb, the stable release is aaaaaaaaaaaa", miner.behind_release()
            )
        with response(STABLE), patch("zils.miner.installed_release", return_value=STABLE):
            self.assertIsNone(miner.behind_release())
        with patch("zils.miner.urllib.request.urlopen", side_effect=OSError("offline")):
            self.assertIsNone(miner.behind_release())

    def test_update_runs_the_installer_at_the_stable_sha(self):
        runs = []

        def fake_run(cmd, env):
            runs.append(env)
            return SimpleNamespace(returncode=0)

        with (
            response(STABLE),
            patch("zils.miner.installed_release", return_value=OLD),
            patch("zils.miner.zils.ROOT", Path("/home/m/.zils/zils")),
            patch("zils.miner.subprocess.run", fake_run),
        ):
            miner.update()
        self.assertEqual(
            {k: runs[0][k] for k in ("ZILS_HOME", "ZILS_REF", "ZILS_SKIP_SETUP")},
            {"ZILS_HOME": "/home/m/.zils", "ZILS_REF": STABLE, "ZILS_SKIP_SETUP": "1"},
        )
        with (
            response(STABLE),
            patch("zils.miner.installed_release", return_value=STABLE),
            patch("zils.miner.subprocess.run", fake_run),
        ):
            miner.update()
        self.assertEqual(len(runs), 1)
        with (
            response(STABLE),
            patch("zils.miner.installed_release", return_value=OLD),
            patch("zils.miner.zils.ROOT", Path("/src/zils-update")),
            self.assertRaises(ValueError),
        ):
            miner.update()


if __name__ == "__main__":
    unittest.main()
