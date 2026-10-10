"""Miners and validators remain usable without the hosted platform installed."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path


class RepositoryBoundaryTest(unittest.TestCase):
    def test_core_does_not_ship_or_import_hosted_services(self):
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "zils/api.py",
            "zils/coordinator.py",
            "zils/billing.py",
            "supabase",
            "website",
            "packages",
        ):
            self.assertFalse((root / relative).exists(), relative)
        for package in ("zils", "miner"):
            for source in (root / package).glob("*.py"):
                for node in ast.walk(ast.parse(source.read_text())):
                    names = (
                        [a.name for a in node.names]
                        if isinstance(node, ast.Import)
                        else [node.module or ""]
                        if isinstance(node, ast.ImportFrom)
                        else []
                    )
                    self.assertFalse(
                        any(
                            n.split(".")[0] in {"zils_platform", "zils_sdk", "stripe", "boto3"}
                            for n in names
                        ),
                        source,
                    )

    def test_worker_and_validator_imports_without_product_dependencies(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules.update({name: None for name in ('zils_platform', 'zils_sdk', 'stripe', 'boto3')}); import miner.queue, zils.validator, zils.fleet, zils.releases, zils.jev_manifest",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
