# Run a customer-job miner

A customer-job miner claims customer training jobs from the hosted queue at
`https://training.zils.ai`, downloads each job's training examples, trains an
adapter on z3-mini (`d3-mini-4.5b-v1`) and submits signed artifacts. Every
candidate is graded on held-out examples the miner never sees. The worker
connects over outbound HTTPS; it needs no inbound port and no database
credentials.

Registering a hotkey on mainnet subnet 127 is the admission
([miner admission](miner-admission.md)): there is no application and no operator
approval. The path, end to end:

1. Register a hotkey on subnet 127.
2. Install the current stable release.
3. Accept the miner terms.
4. Qualify the host.
5. Start the worker (`python -m miner.queue`).
6. Pass the qualification jobs; customer jobs follow.
7. Get paid from the adapter track.

## 1. Register and check the hardware

Register a hotkey on subnet 127 with the [registration guide](bittensor-registration.md).
Keep the coldkey and recovery phrases off the miner host; the worker reads only
the hotkey file.

Use Linux or WSL 2 with Git, a BF16-capable NVIDIA GPU and its CUDA driver. The
worker needs at least 12,288 MiB (12 GB) of free GPU memory; the runtime was
checked on an RTX 4090, where z3-mini training at its 8,192-token input cap peaks
at about 12 GB, so 20 GB cards qualify. Allow at least 25 GB of free disk for
models and dependencies, plus space for job data and candidates. CPU and Apple
MPS training are not supported. Customer jobs run on z3-mini only.

## 2. Install the stable release

Customer jobs go only to miners on an accepted release (see [releases](releases.md)).
The installer checks out the commit named in [`release/stable`](../release/stable)
and builds both runtimes: `.venv-kev` (Python 3.13, the worker) and `.venv-d3`
(Python 3.12, CUDA 13.0, the pinned d3 runtime). `ZILS_SKIP_SETUP=1` skips the
soup-miner setup questions:

```bash
curl -fsSL https://zils.ai/install.sh | ZILS_SKIP_SETUP=1 sh
cd ~/.zils/zils
export ZILS_D3_RUNTIME_PYTHON="$PWD/.venv-d3/bin/python"
.venv-d3/bin/python -m zils.d3 reference --out models/d3-reference
```

The reference command downloads and verifies the pinned base weights (9.1 GB)
and writes `models/d3-reference/model.json`. The pins are `zils/d3-pins.json`
(vendor release `zils-ai/d3-mini-4.5b-v1`, 8,192-token inputs, up to four images
or one video per decision).

Every claim names the commit the worker is installed at. A checkout without git
history sends no commit and gets no customer jobs. A manual `git clone` works
only when it is checked out at the commit in `release/stable`; update it with git.

## 3. Accept the miner terms

Read the [miner terms](mining-terms.md). You accept them by sending their
SHA-256 with your worker's claims. Take the hash from the file in your install,
and check it matches the one the coordinator publishes (`admission.terms_sha256`
in `GET https://training.zils.ai/v1/config`):

```bash
sha256sum docs/mining-terms.md
curl -s https://training.zils.ai/v1/config
```

When the terms change, the coordinator refuses the old hash. Update to the new
release, read the new terms, and restart with the new hash in both the
qualification report and the worker config.

## 4. Qualify the host

The qualification run checks the pinned runtime, trains one step at the
8,192-token cap, reloads the adapter and compares its readout with the vendor's.
It writes `qualification.json`, which the worker sends to the coordinator:

```bash
.venv-kev/bin/python -m scripts.qualify_node --model d3-mini-4.5b-v1 \
  --runtime-python .venv-d3/bin/python --out .private/qualification-d3 \
  --terms-sha256 <hash from step 3>
```

Use a fresh `--out` folder for each run. A report that failed, is for another
profile, or carries another terms hash is refused at admission.

## 5. Configure and start the worker

Create a private state directory:

```bash
mkdir -p .private
chmod 700 .private
```

Save this as `.private/queue-miner.json`, replacing every placeholder with your
own wallet details:

```json
{
  "coordinator": "https://training.zils.ai",
  "hotkey": "YOUR_PUBLIC_HOTKEY_SS58_ADDRESS",
  "wallet": {
    "name": "YOUR_WALLET_NAME",
    "hotkey": "YOUR_HOTKEY_NAME",
    "path": "/absolute/path/to/your/wallets"
  },
  "terms_sha256": "HASH_FROM_STEP_3",
  "qualification": "/absolute/path/to/.private/qualification-d3/qualification.json",
  "graded_scheduling": true,
  "graded_min_free_mib": 12288
}
```

```bash
chmod 600 .private/queue-miner.json
```

The wallet directory must contain the named hotkey on this machine; the worker
checks that its public address matches `hotkey` before it starts.
`graded_scheduling` makes the worker send a signed heartbeat every 15 seconds,
which is how the coordinator sees that it is online and has free GPU memory.
When another service shares the card, add `"gpu_budget_mib"`: the worker then
refuses jobs whose longest example would not fit in that budget, and the
coordinator offers them to another miner. If that service also trains or
evaluates on the GPU, set the same `ZILS_COMPUTE_LOCK` path for both.

Start the worker from `~/.zils/zils` (set the same environment variable in a
service supervisor):

```bash
export ZILS_D3_RUNTIME_PYTHON="$PWD/.venv-d3/bin/python"
.venv-kev/bin/python -m miner.queue \
  --config .private/queue-miner.json --state .private/queue-miner \
  --reference models/d3-reference --device cuda
```

At startup the worker admits itself: it sends the terms hash and the
qualification report to the coordinator, signed by the hotkey. The coordinator
checks that the hotkey is registered on subnet 127, the hash is current and the
report passed, then creates your worker record. A hotkey that is deregistered
loses access within five minutes. The worker refuses to start when a
`--reference` directory is missing or belongs to a retired profile, or when
`ZILS_D3_RUNTIME_PYTHON` is unset.

## 6. Qualification jobs, then customer jobs

After admission the coordinator queues a qualification job for your hotkey: a
synthetic job that grades the host's training quality, reliability and speed.
Zils pays for up to three qualification runs per hotkey every 30 days, one at a
time: when one finishes, the next is queued. Customer jobs are offered once three
qualification reports pass in the job's size band
([miner grading](miner-grading.md)). Chain-admitted miners get open-tier
customer jobs only; private-tier data never leaves Zils hardware.

The worker checks for work every ten seconds. It can stay quiet for a long time
when no job is waiting or when it is waiting for GPU memory; an idle worker is
not a failure.

A successful submission prints `Submitted candidate for job ...`. That means the
artifacts reached the coordinator; the validator decides acceptance after
evaluation. Installing the miner does not guarantee work or acceptance.

When a required release ships, the previous release keeps customer jobs for 24
hours. After that the worker logs
`Queued worker attempt failed: This miner is at ...; customer jobs need a current release. Run zils-miner update.`
Run `zils-miner update`, then restart the worker.

## 7. Get paid

Customer jobs are paid from the adapter track, 20% of emissions under the
launch configuration. When an open-tier adapter you trained is delivered,
validators count the customers' paid API spend on it for 30 days, settled over
72-hour windows. Miners as a group earn at most half of that spend; the rest
burns. Only hotkeys registered on the subnet earn. The evidence is signed by the
platform; see [validators §7](validators.md#7-pay-the-adapter-track-from-paid-adapter-use).

## Inspect and restart

Job data, frozen candidates and training logs live under
`.private/queue-miner/<job-id>/`. Keep this directory private and keep it across
restarts. Logs identify the individual training log for each attempt.

The worker deletes a job's training rows (`miner-training.jsonl`), its training
photos (`images/`) and any partial download as soon as it leaves the job: after
it submits, when the lease is lost or the job is cancelled or fails, and when it
hands the job back for lack of GPU memory. Anything a crash left behind is
deleted before the next claim. A retry downloads the rows again under the same
hash check. The trained candidate, the raw training output and the logs stay for
the validator; the [miner terms](mining-terms.md) require deleting those too
within 7 days of the job finishing, and the worker does not do that yet.

Stop a foreground worker with Ctrl+C. Restart with the same configuration and
state path so completed candidates can be reused. Only one worker process can
hold a given state directory. Use a service supervisor for continuous operation.
`--once` performs one claim attempt and exits; it is not a readiness check or a
wait for the next available job.

If a job references a different base model, stop. Do not change model
identifiers or hashes to make an incompatible reference pass.

## Image jobs

Image jobs are z3-mini jobs with a frozen image intake: the same reference,
runtime and qualification cover them, and no separate image runtime is
installed (see the platform's
[image training](https://github.com/ooo-hq/zils-platform/blob/main/docs/image-training.md)
guide for the intake and asset flow). Image workers receive only training
photos; validation photos and labels stay with the processor.

## Fleet nodes

Operators standing up a rented node can run `scripts/provision_node.sh` in place
of steps 2 and 4 (it installs the same runtimes and writes a qualification
report) and configure the worker through `/etc/zils-node/role.json` and
`scripts/apply_role.sh` in place of step 5. See [fleet nodes](fleet-nodes.md)
and the [runbook](fleet-node-runbook.md).

## Soup pools

Customer jobs are one way to earn. The other is the soup pools, which take 80%
of emissions: upload a fine-tune to Hugging Face and commit it on chain with
`zils-miner commit-body`. See [mining](mining.md#soup-pool).
