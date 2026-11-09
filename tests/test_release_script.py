"""scripts/release.py: the checks before a stable release and the notes it writes, on a throwaway git repo."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import release


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.email", "test@example.com")
        git(self.repo, "config", "user.name", "test")
        self.base = self.commit({"zils/d3.py": "pinned\n", "miner/queue.py": "a\n"}, "Base")

    def commit(self, files, message):
        for name, text in files.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", message)
        return git(self.repo, "rev-parse", "HEAD")

    def test_a_miner_only_release_says_soup_miners_need_nothing(self):
        target = self.commit({"miner/queue.py": "b\n"}, "Miners delete customer data")
        text = release.notes(self.base, target, required=True, repo=self.repo)
        self.assertIn(f"from {self.base[:7]} to **{target[:7]}**", text)
        self.assertIn("**Required for miners taking customer jobs.**", text)
        self.assertIn("24 hours", text)
        self.assertIn("Soup miners: nothing changes", text)
        self.assertIn(f"- {target[:7]} Miners delete customer data", text)
        self.assertNotIn("pinned runtime files change", text)

    def test_pinned_files_and_terms_are_called_out(self):
        target = self.commit(
            {"zils/d3.py": "changed\n", "docs/mining-terms.md": "v2\n", "zils/soup.py": "x\n"},
            "Change the chassis",
        )
        text = release.notes(self.base, target, required=False, repo=self.repo)
        self.assertIn("**Optional.**", text)
        self.assertIn("pinned runtime files change: zils/d3.py", text)
        self.assertIn("terms_sha256", text)
        self.assertIn("Validators and soup miners run changed code: zils/soup.py", text)
        self.assertNotIn("Soup miners: nothing changes", text)

    def test_a_release_must_contain_the_current_one(self):
        git(self.repo, "checkout", "-q", "--orphan", "elsewhere")
        other = self.commit({"miner/queue.py": "c\n"}, "Unrelated")
        with self.assertRaisesRegex(ValueError, "does not contain the current stable"):
            release.check(self.base, other, repo=self.repo)
        with self.assertRaisesRegex(ValueError, "full 40-character"):
            release.check(self.base, other[:7], repo=self.repo)
        with self.assertRaisesRegex(ValueError, "already the stable"):
            release.check(self.base, self.base, repo=self.repo)


if __name__ == "__main__":
    unittest.main()
