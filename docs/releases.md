# Releases

Miners run the stable release: the commit named in [`release/stable`](../release/stable) on `main`.
`zils-miner update` installs it, and `zils-miner doctor` warns when an install is on another commit.

## Required and optional releases

Every release says which it is, in its notes on the
[releases page](https://github.com/ooo-hq/zils/releases).

- **Required:** miners taking customer jobs must update. A customer-job claim names the miner's installed
  commit, and customer jobs go only to accepted releases. After a required release, the previous release
  keeps customer jobs for 24 hours; then its claims are refused with
  `This miner is at ...; customer jobs need a current release. Run zils-miner update.`
- **Optional:** nothing is refused; the doctor warns until you update.

Soup miners do not claim customer jobs, so the release check does not apply to them. The notes say when a
release changes what soup miners run: the pinned runtime files (`zils/d3.py`, `zils/core.py`,
`zils/runtime.py`, `requirements/d3.txt`), scoring, or the miner terms.

## Making a release (maintainers)

A release is a commit built on top of the current stable one, pushed to a `release-<base>-<name>` branch.

```bash
scripts/release.sh notes <sha> --required   # or --optional: preview the notes
scripts/release.sh open <sha> --required    # opens the PR that moves release/stable
scripts/release.sh publish                  # after it merges: the GitHub release
```

`open` refuses a commit that does not contain the current stable release, so nothing already released is
dropped. Edit the notes in the pull request to say what each change means for miners before merging.
