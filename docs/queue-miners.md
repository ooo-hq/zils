# Run a JevK5 queue miner

A queue miner downloads assigned training examples, trains a JevK5 adapter, and
submits signed artifacts for independent evaluation. It connects to the
coordinator over outbound HTTPS. It needs an approved hotkey and model hardware;
it does not need Supabase credentials or an inbound public server.

This guide is for **customer training with JevK5 4B**. For the **Kev 0.8B
Bittensor testnet fleet**, use [registration](bittensor-registration.md) and
[bundle setup](mining.md). Queue assignments do not publish Bittensor weights
or establish on-chain earnings.

## 1. Confirm access and hardware

Get the following from the operator before downloading models:

- The coordinator's exact HTTPS base URL and confirmation that it uses
  `jevk5-4b-v0.3`.
- Approval for your hotkey's public SS58 address. The operator registers the
  worker and approves assignments through [miner administration](supabase-training.md#approve-miners-and-run-a-queued-miner).
- The wallet name, hotkey name, and local wallet directory for the hotkey you
  control. Provision the hotkey privately on the miner host; keep the coldkey
  and recovery phrases off the host and out of the repository.

Use Linux or WSL 2 with Git, Python 3.13, `uv`, a BF16-capable NVIDIA GPU and its
CUDA driver. Allow at least 25 GB of free disk for models and dependencies,
plus space for assigned data and candidates. The runtime was checked on an
RTX 4090; memory needs depend on input length. Apple silicon miners can instead
follow the [Mac installation and qualification guide](mac-miners.md), including
the measured limitations of a 16 GiB M4. CPU training is unsupported.

The operator assigns a local queue UID. On-chain registration is not required
for this queue, and a Bittensor UID alone does not authorize job access.

## 2. Install and verify the reference

From a fresh clone:

```bash
git clone https://github.com/ooo-hq/zils.git
cd zils
uv venv --python 3.13 .venv-kev
uv pip install --python .venv-kev/bin/python --torch-backend=cu128 \
  -r requirements/model.txt -r requirements/rehearsal.txt \
  -r requirements/testnet.txt
.venv-kev/bin/python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()'
.venv-kev/bin/python -m zils.jevk5 reference --out models/jevk5-reference
```

The Bittensor SDK is installed here to read the wallet hotkey; these commands
send no chain transactions. The reference command downloads and verifies the
pinned base weights and writes `models/jevk5-reference/model.json`.
See [model identity and cache options](jevk5-queue.md#install-and-create-the-reference)
when reusing an existing download.

## 3. Configure your worker

From the repository root, create a private state directory:

```bash
mkdir -p .private
chmod 700 .private
```

Save this as `.private/queue-miner.json`, replacing every placeholder with the
operator-confirmed URL and your own wallet details:

```json
{
  "coordinator": "https://YOUR_COORDINATOR_HOST",
  "hotkey": "YOUR_PUBLIC_HOTKEY_SS58_ADDRESS",
  "wallet": {
    "name": "YOUR_WALLET_NAME",
    "hotkey": "YOUR_HOTKEY_NAME",
    "path": "/absolute/path/to/your/wallets"
  }
}
```

The wallet directory must contain the named hotkey on this machine. The worker
checks that its public address matches `hotkey` before it starts.

```bash
chmod 600 .private/queue-miner.json
```

Use a dedicated miner operating-system account. Never load the validator's
`.private/training.env` or Supabase service-role key into it. If training and
evaluation share a GPU, have the operator configure the same
[compute lock and capacity settings](jevk5-queue.md#configure-the-services)
for both services before starting.

## 4. Start the miner

Run from the repository root:

```bash
.venv-kev/bin/python -m miner.queue \
  --config .private/queue-miner.json --state .private/queue-miner \
  --reference models/jevk5-reference --device cuda
```

The worker checks for assigned work every ten seconds. It can remain quiet when
no assignment is available or a configured capacity gate is waiting for GPU
memory. Ask the operator to check worker approval, job assignment, and capacity
before treating an idle worker as a failure.

A successful submission prints `Submitted candidate for job ...`. That means
the artifacts reached the coordinator; acceptance is determined by the validator
after evaluation. Installing the miner does not guarantee work or acceptance.

## 5. Inspect and restart

Job data, frozen candidates, and training logs live under
`.private/queue-miner/<job-id>/`. Keep this directory private and retain it across
restarts. Logs identify the individual training log for each attempt.

Stop a foreground worker with Ctrl+C. Restart with the same configuration and
state path so completed candidates can be reused. Only one worker process can
hold a given state directory. Use a service supervisor for continuous operation.
`--once` performs one claim attempt and exits; it is not a readiness check or a
wait for the next available job.

If a job references a different base model, stop and resolve the operator's
configuration using the [model transition guide](jevk5-queue.md#existing-jobs-and-rollback).
Do not change model identifiers or hashes to make an incompatible reference pass.
