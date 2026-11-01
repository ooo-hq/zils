"""The installer links `zils-miner` and retires the old `zils` link, with git and uv stubbed out."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/install.sh"

# git "clones" an empty checkout and copies this repo's pyproject.toml in on checkout; uv
# "installs" a console script for every name under [project.scripts], plus btcli.
STUBS = {
    "git": f"""#!/bin/sh
case "$1" in
  clone) for last; do :; done; mkdir -p "$last/.git" ;;
  -C) [ "$3" = checkout ] && cp '{ROOT}/pyproject.toml' "$2/" ;;
esac
exit 0
""",
    "uv": """#!/bin/sh
case "$1" in
  --version) echo "uv 0.0.0" ;;
  venv) for last; do :; done; mkdir -p "$last/bin" ;;
  pip)
    for name in $(sed -n '/^\\[project.scripts\\]/,/^\\[/p' pyproject.toml | grep ' = ' | cut -d' ' -f1) btcli; do
      printf '#!/bin/sh\\n' >".venv-kev/bin/$name"; chmod +x ".venv-kev/bin/$name"
    done ;;
esac
exit 0
""",
    "curl": "#!/bin/sh\nexit 1\n",
    "uname": "#!/bin/sh\necho Darwin\n",
}


class InstallScriptTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name)
        self.stubs = self.home / "stubs"
        self.stubs.mkdir()
        for name, body in STUBS.items():
            (self.stubs / name).write_text(body)
            (self.stubs / name).chmod(0o755)
        self.bin = self.home / ".local/bin"
        self.venv_bin = self.home / ".zils/zils/.venv-kev/bin"

    def install(self):
        env = {
            "HOME": str(self.home),
            "PATH": f"{self.stubs}:/usr/bin:/bin",
            "SHELL": "/bin/sh",
            "ZILS_REF": "a" * 40,
            "ZILS_SKIP_SETUP": "1",
            "NO_COLOR": "1",
        }
        done = subprocess.run(
            ["sh", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout

    def test_fresh_install_links_zils_miner_not_zils(self):
        out = self.install()
        self.assertEqual(os.readlink(self.bin / "zils-miner"), str(self.venv_bin / "zils-miner"))
        self.assertEqual(os.readlink(self.bin / "btcli"), str(self.venv_bin / "btcli"))
        self.assertFalse((self.bin / "zils").exists() or (self.bin / "zils").is_symlink())
        self.assertIn("Next: zils-miner doctor", out)

    def test_upgrade_removes_the_old_miner_zils_link(self):
        self.bin.mkdir(parents=True)
        (self.bin / "zils").symlink_to(self.venv_bin / "zils")
        out = self.install()
        self.assertFalse((self.bin / "zils").is_symlink())
        self.assertIn("zils-miner", out)

    def test_another_zils_command_is_left_alone(self):
        self.bin.mkdir(parents=True)
        other = self.home / "npm/bin/zils"
        other.parent.mkdir(parents=True)
        other.write_text("#!/bin/sh\n")
        (self.bin / "zils").symlink_to(other)
        self.install()
        self.assertEqual(os.readlink(self.bin / "zils"), str(other))


if __name__ == "__main__":
    unittest.main()
