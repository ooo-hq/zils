"""Stable releases: what miners get from `zils-miner update` (docs/releases.md).

  scripts/release.sh notes <sha> --required|--optional   print the release notes against the current stable
  scripts/release.sh open <sha> --required|--optional    check <sha>, open the PR that moves release/stable to it
  scripts/release.sh publish                             after that PR merges: the GitHub release for the stable sha

<sha> is a full commit already pushed to ooo-hq/zils (usually a release-<base>-<name> branch built on the current
stable). Our own fleet's part of a release (our workers, the coordinator's accepted releases) is not in this repo.
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "ooo-hq/zils"
PINNED = ("zils/d3.py", "zils/core.py", "zils/runtime.py", "requirements/d3.txt")
TERMS = "docs/mining-terms.md"


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def check(base, target, *, repo=ROOT):
    """A release names a full commit that contains the current stable one, so nothing it shipped is dropped."""
    if not re.fullmatch("[0-9a-f]{40}", target):
        raise ValueError("Give the release as a full 40-character commit sha")
    if target == base:
        raise ValueError(f"{target[:7]} is already the stable release")
    ancestor = ["git", "-C", str(repo), "merge-base", "--is-ancestor", base, target]
    if subprocess.run(ancestor, capture_output=True).returncode:
        raise ValueError(
            f"{target[:7]} does not contain the current stable release {base[:7]}; build it on top of {base[:7]}"
        )


def notes(base, target, *, required, repo=ROOT):
    changed = git(repo, "diff", "--name-only", base, target).split("\n")
    code = [f for f in changed if f.startswith(("zils/", "requirements/"))]
    miner = [f for f in changed if f.startswith("miner/")]
    pinned = [f for f in changed if f in PINNED]
    lines = [
        f"Moves `release/stable` from {base[:7]} to **{target[:7]}**.",
        "",
        "## Release notes",
        "",
    ]
    if required:
        lines.append(
            "**Required for miners taking customer jobs.** Run `zils-miner update`, then restart the miner. The"
            " previous release keeps customer jobs for 24 hours after this release goes out; after that its claims"
            " are refused."
        )
    else:
        lines.append(
            "**Optional.** `zils-miner doctor` warns until you update; nothing is refused."
        )
    lines.append("")
    if miner:
        lines.append("- Miner code changes: " + ", ".join(miner))
    if pinned:
        lines.append(
            f"- **The pinned runtime files change: {', '.join(pinned)}.** Exams and the soup chassis change with"
            " them; soup miners should update before the next round."
        )
    if TERMS in changed:
        lines.append(
            "- **The miner terms change.** Miners accept them by setting the new `terms_sha256` in the miner"
            " config; the coordinator's terms hash must change with this release."
        )
    if rest := [f for f in code if f not in pinned]:
        lines.append("- Validators and soup miners run changed code: " + ", ".join(rest))
    if not code:
        lines.append(
            f"- Soup miners: nothing changes. `zils/` and the pinned runtime files are the same as {base[:7]}."
        )
    if "scripts/install.sh" in changed:
        lines.append("- The installer changes (`scripts/install.sh`).")
    log = git(repo, "log", "--no-merges", "--format=- %h %s", "--abbrev=7", f"{base}..{target}")
    lines += [
        "",
        "## Commits",
        "",
        log,
        "",
        "Say what each change means for miners before merging.",
    ]
    return "\n".join(lines) + "\n"


def stable():
    git(ROOT, "fetch", "-q", "origin", "main")
    return git(ROOT, "show", "origin/main:release/stable")


def checked(target):
    """The current stable sha, once `target` is a pushed full commit that contains it."""
    base = stable()
    if re.fullmatch("[0-9a-f]{40}", target):
        git(ROOT, "fetch", "-q", "origin", target)
    check(base, target)
    return base


def gh(*args):
    return subprocess.run(
        ["gh", *args, "-R", REPO], check=True, capture_output=True, text=True
    ).stdout.strip()


def open_release(target, required):
    base = checked(target)
    text, branch = notes(base, target, required=required), f"release-stable-{target[:7]}"
    with tempfile.TemporaryDirectory() as temp:
        tree = Path(temp) / "tree"
        git(ROOT, "worktree", "add", "-q", "--detach", str(tree), "origin/main")
        try:
            (tree / "release/stable").write_text(target + "\n")
            git(tree, "commit", "-q", "-am", f"Release {target[:7]}")
            git(tree, "push", "-q", "origin", f"HEAD:refs/heads/{branch}")
        finally:
            git(ROOT, "worktree", "remove", "--force", str(tree))
    print(gh("pr", "create", "--head", branch, "--title", f"Release {target[:7]}", "--body", text))


def publish():
    target = stable()
    body = gh(
        "pr",
        "list",
        "--state",
        "merged",
        "--head",
        f"release-stable-{target[:7]}",
        "--json",
        "body",
        "-q",
        ".[0].body",
    )
    if not body:
        raise ValueError(
            f"No merged release-stable-{target[:7]} pull request; open it with `release.sh open`"
        )
    print(
        gh(
            "release",
            "create",
            f"stable-{target[:7]}",
            "--target",
            target,
            "--title",
            f"Release {target[:7]}",
            "--notes",
            body,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("notes", "open", "publish"))
    parser.add_argument("sha", nargs="?")
    kind = parser.add_mutually_exclusive_group()
    kind.add_argument(
        "--required", action="store_true", help="customer-job miners must update (24 h grace)"
    )
    kind.add_argument(
        "--optional", action="store_true", help="the doctor warns; nothing is refused"
    )
    args = parser.parse_args()
    try:
        if args.command == "publish":
            return publish()
        if not args.sha or not (args.required or args.optional):
            parser.error(f"{args.command} needs <sha> and --required or --optional")
        if args.command == "notes":
            base = checked(args.sha)
            print(notes(base, args.sha, required=args.required), end="")
        else:
            open_release(args.sha, args.required)
    except (ValueError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        parser.exit(1, f"release: {detail.strip()}\n")


if __name__ == "__main__":
    main()
