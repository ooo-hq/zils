"""The provisioner is idempotent, checks the driver first and never touches secrets."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/provision_node.sh"
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
            [sys.executable, "-m", "scripts.qualify_node", "--help"],
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

    def test_service_user_can_read_the_environments_whatever_the_inherited_umask(self):
        text = SCRIPT.read_text()
        self.assertLess(text.index("umask 022"), text.index("phase driver"))
        env = text[text.index("phase environments") : text.index("phase references")]
        self.assertIn("chmod -R a+rX .venv-kev .venv-h2o .venv-d3", env)
        self.assertLess(env.index("requirements/d3.txt"), env.index("a+rX .venv-kev"))

    def test_d3_runtime_follows_h2o_in_environments_references_qualify_and_image(self):
        text = SCRIPT.read_text()
        env = text[text.index("phase environments") : text.index("phase references")]
        self.assertLess(env.index(".venv-h2o"), env.index(".venv-d3"))
        self.assertIn("uv venv --python 3.12 .venv-d3", env)
        self.assertIn(
            "--python .venv-d3/bin/python --torch-backend=cu130 -r requirements/d3.txt", env
        )
        refs = text[text.index("phase references") : text.index("phase units")]
        self.assertLess(refs.index("zils.h2o reference"), refs.index("zils.d3 reference"))
        self.assertIn("-m zils.d3 reference --out models/d3-reference", refs)
        qualify = text[text.index("phase qualify") : text.index("phase image")]
        self.assertEqual(qualify.count("\n      qualify_runtime "), 2)
        self.assertLess(qualify.index(".venv-h2o"), qualify.index("--model d3-mini-4.5b-v1"))
        self.assertIn("--runtime-python $ROOT/.venv-d3/bin/python", qualify)
        self.assertIn('rm -rf "$STATE/qualification-d3/adapter"', qualify)
        self.assertIn("--out $STATE/qualification-d3", qualify)
        image = text[text.index("phase image") :]
        self.assertIn(
            "PINS=$(sha256sum zils/h2o-pins.json requirements/h2o.txt | sha256sum | cut -c1-16)",
            image,
        )
        self.assertIn(
            "D3_PINS=$(sha256sum zils/d3-pins.json requirements/d3.txt | sha256sum | cut -c1-16)",
            image,
        )
        self.assertIn("d3_pins_sha256_16", image)

    def test_source_checks_out_the_fetched_ref_not_a_stale_local_branch(self):
        """An old snapshot's local `main` must not win over the fetched origin/main (zils-node-2, 19 Oct)."""
        import tempfile

        def git(*args, cwd):
            return subprocess.run(
                ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
            ).stdout.strip()

        with tempfile.TemporaryDirectory() as tmp:
            origin, node = Path(tmp) / "origin", Path(tmp) / "node"
            origin.mkdir()
            git("init", "-q", "-b", "main", cwd=origin)
            ident = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
            git(*ident, "commit", "-q", "--allow-empty", "-m", "old", cwd=origin)
            old = git("rev-parse", "HEAD", cwd=origin)
            git("clone", "-q", str(origin), str(node), cwd=tmp)  # the snapshot: local main at `old`
            git(*ident, "commit", "-q", "--allow-empty", "-m", "new", cwd=origin)
            git("tag", "v1", cwd=origin)
            new = git("rev-parse", "HEAD", cwd=origin)
            git("fetch", "-q", "--tags", "origin", cwd=node)
            function = subprocess.run(
                ["sed", "-n", "/^checkout_ref() {/,/^}/p", str(SCRIPT)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            self.assertIn("checkout_ref", function)
            for ref, expected in (("main", new), (old, old), ("v1", new)):
                with self.subTest(ref=ref):
                    subprocess.run(
                        ["bash", "-c", function + '\ncheckout_ref "$1" "$2"', "_", str(node), ref],
                        check=True,
                    )
                    self.assertEqual(git("rev-parse", "HEAD", cwd=node), expected)
            self.assertEqual(git("rev-parse", "main", cwd=node), old)  # stale branch left alone

    def test_later_phases_run_from_the_checked_out_script(self):
        text = SCRIPT.read_text()
        source = text[text.index("phase source") : text.index("phase environments")]
        self.assertIn("exec bash $ROOT/scripts/provision_node.sh", source)
        self.assertIn('[ -z "${ZILS_NODE_REEXEC:-}" ]', source)
