"""Wrap the per-submission inference runner in the strongest isolation this host offers.

Submissions are untrusted weights. The runner already receives no labels and runs offline; this layer removes
network access entirely and ties the runner's lifetime to its parent. The mode actually used is recorded in
every evaluation report so an operator can see when a host falls back to `none`.
"""

import functools
import shutil
import subprocess

from . import settings

MODES = ("unshare", "none")


def _works(command):
    try:
        return subprocess.run(command, capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@functools.cache
def mode():
    """`ZILS_SANDBOX` forces a mode; otherwise use unprivileged network namespaces when the host allows them."""
    requested = settings.get("ZILS_SANDBOX")
    if requested is not None:
        if requested not in MODES:
            raise ValueError(f"ZILS_SANDBOX must be one of {MODES}")
        return requested
    unshare = shutil.which("unshare")
    if unshare and _works([unshare, "--net", "--map-root-user", "--kill-child", "--", "true"]):
        return "unshare"
    return "none"


def wrap(command):
    """Return `command` prefixed with the isolation launcher for the active mode."""
    if mode() == "unshare":
        # New network namespace with only loopback; the child dies with the launcher on timeout.
        return [shutil.which("unshare"), "--net", "--map-root-user", "--kill-child", "--", *command]
    return list(command)
