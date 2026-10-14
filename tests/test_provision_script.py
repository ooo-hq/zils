"""The provisioner is idempotent, checks the driver first and never touches secrets."""

import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/provision_node.sh"
PYTHON = "/Users/ken/Projects/OOO/Zils/zils/.venv-kev/bin/python"
PHASES = [
    "driver",
    "accounts",
    "source",
    "environments",
    "references",
    "units",
    "qualify",
    "image",
]


class ProvisionScriptTest(unittest.TestCase):
    def dry_run(self):
        return subprocess.run(
            ["bash", str(SCRIPT)],
            env={**os.environ, "ZILS_NODE_DRY_RUN": "1"},
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(SCRIPT)]).returncode, 0)

    def test_dry_run_lists_phases_in_order_and_is_stable(self):
        first, second = self.dry_run(), self.dry_run()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, second.stdout)
        phases = [
            line.split(":")[1].strip()
            for line in first.stdout.splitlines()
            if line.startswith("phase:")
        ]
        self.assertEqual(phases, PHASES)

    def test_script_never_reads_secret_locations(self):
        text = SCRIPT.read_text()
        forbidden = (
            "coldkey",
            "SUPABASE_SERVICE_ROLE_KEY",
            "ZILS_RUNTIME_TOKEN",
            "/etc/zils-node/hotkey",
            "/etc/zils-node/wallets",
        )
        for item in forbidden:
            self.assertNotIn(item, text, item)

    def test_dry_run_accepts_truthy_words(self):
        env = {**os.environ, "ZILS_NODE_DRY_RUN": "yes"}
        result = subprocess.run(
            ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        phases = [
            line.split(":")[1].strip()
            for line in result.stdout.splitlines()
            if line.startswith("phase:")
        ]
        self.assertEqual(phases, PHASES)

    def test_dry_run_is_normalised_and_every_mutation_is_guarded(self):
        text = SCRIPT.read_text()
        self.assertIn('case "$DRY" in 0|"") DRY=0;; *) DRY=1;; esac', text)
        # Each phase body (and the final image step) only mutates behind a "$DRY" = 0 guard.
        self.assertEqual(text.count('if [ "$DRY" = 0 ]; then'), 8)
        self.assertNotIn("DRY:-1", text)

    def test_qualify_clears_stale_adapter_before_running(self):
        text = SCRIPT.read_text()
        qualify = text[text.index("phase qualify") : text.index("phase image")]
        clear = qualify.index('rm -rf "$STATE/qualification/adapter"')
        self.assertIn('"$STATE/qualification/train.jsonl"', qualify)
        self.assertLess(clear, qualify.index("-m scripts.qualify_node"))

    def test_service_user_can_read_node_config_and_owns_caches(self):
        text = SCRIPT.read_text()
        self.assertIn("-o root -g zils-compute /etc/zils-node\n", text)
        self.assertIn("-m 0770 -o root -g zils-compute /etc/zils-node/fleet", text)
        for cache in ("triton", "inductor", "hf"):
            self.assertIn(f'"$STATE/cache/{cache}"', text)
        self.assertIn('chown -R zilsminer:zilsminer "$STATE/cache"', text)

    def test_qualifier_runs_as_a_module_that_can_import_zils(self):
        text = SCRIPT.read_text()
        self.assertIn("-m scripts.qualify_node", text)
        self.assertNotIn("scripts/qualify_node.py", text)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        result = subprocess.run(
            [PYTHON, "-m", "scripts.qualify_node", "--help"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_qualify_runs_in_the_unit_sandbox_as_the_service_user(self):
        text = SCRIPT.read_text()
        qualify = text[text.index("phase qualify") : text.index("phase image")]
        for item in (
            "systemd-run",
            "User=zilsminer",
            "ProtectSystem=strict",
            "EnvironmentFile=/etc/zils-node/node.env",
        ):
            self.assertIn(item, qualify)
        self.assertLess(qualify.index("chown -R zilsminer:zilsminer"), qualify.index("systemd-run"))

    def test_apt_does_not_request_unavailable_interpreters(self):
        text = SCRIPT.read_text()
        self.assertNotIn("python3.13", text.split("apt-get install", 1)[1].splitlines()[0])
        self.assertNotIn("$HOME/", text)

    def test_managed_interpreters_live_where_the_service_user_can_exec_them(self):
        text = SCRIPT.read_text()
        self.assertIn("UV_PYTHON_INSTALL_DIR=/opt/uv-python", text)
        self.assertIn("chmod -R a+rX /opt/uv-python", text)
        env = text[text.index("phase environments") : text.index("phase references")]
        self.assertLess(
            env.index("UV_PYTHON_INSTALL_DIR"), env.index("uv venv"), "export before any uv call"
        )
        self.assertLess(env.index("requirements/h2o.txt"), env.index("chmod -R a+rX"))
