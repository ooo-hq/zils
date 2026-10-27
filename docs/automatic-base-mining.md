# Scheduled base mining

`python -m zils.base_miner --config /absolute/path/miner.json` runs one bounded
public-data training attempt, merges it into a full checkpoint, publishes an
immutable Hugging Face revision and commits it with the miner's own registered
hotkey. It uses the same training and candidate validation as `zils.d3` and
`zils commit-candidate`. It does not elect winners, move channels or migrate
customer adapters. Set `"network": "finney"` and the subnet's `netuid`; mainnet
has no default netuid. Set `interval_seconds` at or above your pool's submission
cooldown ([model pools](model-pools.md)); a distinct candidate committed inside
the cooldown is refused by validators.

This is a bootstrap recipe: one epoch of the existing LoRA trainer, with a new
seed for each attempt. Repeating seeds/data variants is not evidence of model
improvement. Validators independently assess each submitted checkpoint.

Each miner chooses its own training schedule and recipe. The subnet does not
require miners to take turns or wait a day between training attempts. This runner
does not wait for a validator decision before starting its next completed-cycle
attempt; training cadence and election confirmation timing are separate.

## Configuration

Install the regular miner and d3 runtime first (see [mining](mining.md)). Prepare
an operator-owned public training export in the format accepted by `zils.d3 train`
and a reference using `python -m zils.d3 reference --out /srv/miner/reference`.
Never supply customer exports or validator-private exams as base-training data.
Compute the export's SHA-256, then save a JSON configuration:

```json
{
  "network": "finney",
  "netuid": <NETUID>,
  "model": "d3-mini-4.5b-v1",
  "wallet_name": "YOUR_WALLET",
  "wallet_hotkey": "YOUR_HOTKEY",
  "wallet_path": "/srv/miner/wallets",
  "hotkey": "YOUR_REGISTERED_SS58_ADDRESS",
  "repo": "YOUR_HF_ACCOUNT/YOUR_CANDIDATE_REPO",
  "data": "/srv/miner/train.jsonl",
  "data_sha256": "SHA256_OF_YOUR_PUBLIC_TRAINING_EXPORT",
  "reference": "/srv/miner/reference",
  "runtime_python": "/srv/miner/zils/.venv-d3/bin/python",
  "state": "/srv/miner/state",
  "compute_lock": "/var/lock/zils-training/cuda.lock",
  "yield_file": "/run/zils-base-miner/training",
  "interval_seconds": 43200,
  "max_training_seconds": 21600,
  "start_after": 0
}
```

Create the state/yield parent directories and shared lock with permissions for
the miner. Every GPU user on the host must use that same lock. The serving
controller must stop its own runtime while the yield file exists and restore it
when the file disappears. Configure service cleanup to remove this file after
termination, including forced termination. The runner waits up to 180 seconds
for serving to yield, rather than training into insufficient free memory.

Run the preflight under the intended service account:

```sh
python -m zils.base_miner --config /srv/miner/miner.json --check
```

Provide publishing authentication through a protected service environment or
Hugging Face token store. Check the token's expiry and arrange renewal before
it expires. Training subprocesses receive offline model-cache settings, not the
publishing token. The host must contain the pinned model cache and have at least
30 GiB free before an attempt begins.

Run the command from a service manager. A timer can check every minute: the
runner starts another attempt when the preceding attempt has completed and the
local compute lock is available. `interval_seconds` defaults to zero; a positive
value is an operator-selected pause after a committed attempt finishes. `start_after`
is an optional operator-selected UTC Unix timestamp, defaulting to no delay. Neither
setting coordinates miners or imposes a subnet-wide training schedule.

Validators may enforce a per-pool [submission cooldown](model-pools.md): a
hotkey's next distinct commitment to that pool is refused until the cooldown has
passed since its last accepted one, measured in chain blocks. A refused candidate
still cost its full training and upload. Set `interval_seconds` to at least the
pool's cooldown, plus a small margin because the chain counts blocks rather than
seconds; the next commitment then lands after the cooldown, since training
follows the pause. With the default of zero, a short training run produces
candidates that are refused.
The existing commitment remains until a new immutable candidate is published
and committed. A failed or uncertain publication still requires reconciliation;
this runner does not queue further training behind an unresolved attempt.
Use a service timeout of eight hours and kill the whole service process group
on timeout. Keep existing customer adapter queue workers running.

## Receipts and recovery

`state/attempt.json` records the latest stage; `state/attempts/<id>/` retains
training/merge logs, adapter, checkpoint and receipt. A successful attempt ends
in `committed` with the exact revision, checkpoint hash and chain block.
The upload revision is saved before the chain write. After an ambiguous chain
response the runner reads back once and never blindly writes again. An interrupted
or failed attempt blocks new training until an operator reconciles its artifacts
and publication; a service restart alone cannot discard the uncertainty.

Retain the original receipt when repairing an attempt. For a saved uploaded
revision, use `zils commit-candidate --revision <exact-sha>` with the same
candidate and wallet; it reads the chain first and leaves an existing matching
commitment alone. Only mark the receipt committed after verified readback.
Training failures require an explicit operator decision before archiving the
incomplete latest receipt and allowing a fresh attempt.

Artifacts are retained rather than deleted automatically. Archive completed
attempts before the free-disk guard is reached. A future approved base needs a
new registered profile/reference and explicit runner configuration; this runner
does not infer a new parent from a channel label.
