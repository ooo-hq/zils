# Budgeted base validation

A validator can receive nominations from hundreds of independent miners without
performing every download and evaluation in its weight-publication loop. Enable
an explicit local budget and run the base-evaluation worker separately. This is
opt-in and applies to `base_pools`; the synchronous path remains available when
`base_validation_queue` is absent.

Miners choose their training cadence. The validator queues one nominated immutable
checkpoint per registered hotkey. Unstarted replacements keep their queue position;
once an attempt starts, its checkpoint remains frozen through retry/confirmation.
A newer nomination is remembered for that miner's next turn. An unchanged completed
checkpoint does not consume another turn. The queue preserves first observed
checkpoint ownership across nomination changes and prevents a new UID occupant
from inheriting the previous hotkey's pending work.

## Configure a validator

Start with a configured, qualified pool validator as described in
[base elections](base-election.md) and [validators](validators.md). Keep its exam,
wallet and runtime provisioning unchanged. Add this top-level configuration block:

```json
"base_validation_queue": {
  "window_seconds": 3600,
  "max_candidates": 4,
  "max_seconds": 1800,
  "max_download_bytes": 25769803776,
  "max_cache_bytes": 107374182400
}
```

These are illustrative operator budgets, not a measured capacity recommendation.
All five values are required positive integers. Download and cache budgets must
fit the largest enabled candidate profile. The budgets are shared across all
base pools on this validator; other validators maintain independent budgets.

| Setting | Enforcement |
| --- | --- |
| `window_seconds` | Persistent local accounting window; restarting does not reset it. |
| `max_candidates` | Maximum attempts, including retries, in that window. |
| `max_seconds` | Total wall time reserved for metadata, download, validation and evaluation. Unused time is refunded after an observed completion; a crashed worker keeps its reservation charged. |
| `max_download_bytes` | Reserve the profile's maximum candidate bytes before each uncached download, including failures. Metadata requests retain their existing size/time limits. |
| `max_cache_bytes` | Retained candidate-weight cache limit. Least-recently-used weights may be evicted; private scores, receipts and configured incumbent references are retained. |

The worker interrupts an attempt at its remaining wall-time allowance. Evaluator
subprocess cleanup can add up to 12 seconds. The cache limit excludes reference
models, logs, evidence and temporary staging: the worker also requires free space
for two candidate-sized staging copies plus 2 GiB. Monitor those retained files;
this is not a total-filesystem quota. An evicted checkpoint may need downloading
again for confirmation, while its prior bound evaluation evidence remains cached.

## Run the worker

After deploying this source, restart the validator with the new config at a safe
round boundary, then start the separate worker under the same service identity:

```sh
/path/to/zils/.venv-kev/bin/python -m zils.base_queue \
  --config /srv/validator/config.json \
  --runtime-python /path/to/zils/.venv-d3/bin/python \
  --device cuda
```

Use actual absolute paths from the installed validator. The config directory must
be writable by that service account. The worker needs the validator's existing
read access to private exams/reference files and outbound artifact access. Use
the same `ZILS_COMPUTE_LOCK` and runtime/cache environment as its evaluator. Keep
one worker per validator directory. `--once` executes at most one bounded attempt
and exits; without it, the worker checks every five seconds. A service manager
must terminate the entire worker control group on shutdown so a killed parent
cannot leave an evaluator running. The worker itself never submits chain weights.

The validator records frozen chain nominations with no candidate HTTP or GPU work
in `base_pools_epoch`, then publishes using available results and the unchanged
king history. The worker routes, downloads and evaluates candidates asynchronously.
Existing adapter/feature evaluation and chain I/O are not bypassed or given new
budgets by this feature; shared GPU capacity can still delay those separate tasks.
Do not run a legacy synchronous base evaluator against the same store concurrently.

For rollback, stop the queue worker and its children first. Remove the config block
and restart the validator at a safe round boundary. Retain `base-queue/` and the
pool stores for recovery; disabling budgets does not require deleting evidence.

## Recovery and inspection

State is atomically saved in `base-queue/state.json`, with attempt receipts under
`base-queue/attempts/`. Queue position, frozen nominations, completed results and
reserved resource usage survive restarts. One worker lock prevents overlapping
attempts; short state transactions allow intake during an evaluation.

A lost worker's attempt returns behind existing waiters without refunding its
reservation. Capacity, transport and timeout failures defer to the next budget
window; invalid candidates are recorded and completed. A qualifying first draw
waits until the existing 12-hour confirmation boundary. New miners continue to
receive turns while those candidates wait. Changing pool profiles, draws or path
configuration invalidates queue decisions and pending work without resetting the
resource allowance. Private historical evaluation receipts remain on disk.

Inspect queue status without publishing private exam data:

```sh
python -c 'import json,sys; s=json.load(open(sys.argv[1])); print(json.dumps({"budget":s["budget"], "states":{h:r["status"] for h,r in s["entries"].items()}}, indent=2))' /srv/validator/base-queue/state.json
```

Public round reports continue to contain aggregate decisions, not exam inputs or
per-case predictions. A stopped worker leaves nominations pending; it does not
change existing king rewards or imply that a candidate failed an exam.

## Verification and limits

CPU tests cover 256 distinct queued miners, independent validator state, budgets,
nomination replacement, UID reassignment, restart recovery, cache eviction and
weight calculation without base HTTP/evaluation. A local artifact-server fixture
exercises the actual queue, fetch, evaluator subprocess, scoring and confirmation
path with synthetic weights and a stand-in inference engine. This establishes
software behavior, not real-model throughput or production multi-validator capacity.

Shared finalist announcements, agreement on promotion, and a production
256-miner capacity rehearsal remain separate work. No queue result automatically
crowns a model or bypasses the existing release and regression gates.
