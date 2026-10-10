# Run a text or image queue miner

A queue miner downloads assigned training examples, trains the pinned adapter, and
submits signed artifacts for independent evaluation. It connects to the
coordinator over outbound HTTPS. It needs an approved hotkey and model hardware;
it does not need Supabase credentials or an inbound public server.

The initial setup below is for **H2O Lightning 4B text training**; add **ImaJev 4B image
training** with the steps below. For a **Bittensor testnet fleet**, use
[registration](bittensor-registration.md) and
[bundle setup](mining.md). Queue assignments do not publish Bittensor weights
or establish on-chain earnings.

Operators can enable [graded job assignment](miner-grading.md) after qualification.
Miners opt into signed availability only after the coordinator upgrade; existing
configurations retain their current request flow.

## 1. Confirm access and hardware

Get the following from the operator before downloading models:

- The coordinator's exact HTTPS base URL and confirmation that it uses
  `h2o-lightning-4b-v1.2.3`.
- Approval for your hotkey's public SS58 address. The operator registers the
  worker and approves assignments through [miner administration](https://github.com/ooo-hq/zils-platform/blob/main/docs/supabase-training.md#approve-miners-and-run-a-queued-miner).
- The wallet name, hotkey name, and local wallet directory for the hotkey you
  control. Provision the hotkey privately on the miner host; keep the coldkey
  and recovery phrases off the host and out of the repository.

Use Linux or WSL 2 with Git, Python 3.13, `uv`, a BF16-capable NVIDIA GPU and its
CUDA driver. Allow at least 25 GB of free disk for models and dependencies,
plus space for assigned data and candidates. The runtime was checked on an
RTX 4090; memory needs depend on input length. H2O uses the isolated CUDA runtime
below. Apple silicon miners retain the
[JevK5 installation guide](mac-miners.md) for explicitly assigned legacy jobs.
H2O CPU and Apple MPS training are unsupported.

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
uv venv --python 3.12 .venv-h2o
uv pip install --python .venv-h2o/bin/python --torch-backend=cu130 -r requirements/h2o.txt
export ZILS_H2O_RUNTIME_PYTHON="$PWD/.venv-h2o/bin/python"
.venv-h2o/bin/python -m zils.h2o reference --out models/h2o-reference
```

The Bittensor SDK is installed here to read the wallet hotkey; these commands
send no chain transactions. The reference command downloads and verifies the
pinned base weights and writes `models/h2o-reference/model.json`.
See the [H2O model contract](h2o-queue.md) for the immutable revision, runtime,
and cache override. The operator must explicitly qualify this hotkey for H2O;
an existing JevK5 approval does not qualify it for H2O.

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

Run from the repository root (also set this environment variable in a service supervisor):

```bash
export ZILS_H2O_RUNTIME_PYTHON="$PWD/.venv-h2o/bin/python"
.venv-kev/bin/python -m miner.queue \
  --config .private/queue-miner.json --state .private/queue-miner \
  --reference models/h2o-reference --device cuda
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
configuration using the [model transition guide](h2o-queue.md#existing-jobs).
Do not change model identifiers or hashes to make an incompatible reference pass.

## Add the image vertical

Complete [image runtime installation and worker qualification](https://github.com/ooo-hq/zils-platform/blob/main/docs/image-training.md#worker-and-service-setup)
before advertising image support. The operator must approve this hotkey for the
exact `imajev-4b-v1` profile/runtime hashes using maximum-context measurements.
Registering a text worker or downloading image weights does not grant that approval.

Install the isolated image runtime and download its pinned release from the
repository root:

```bash
uv venv --python 3.13 .venv-image
uv pip install --python .venv-image/bin/python --torch-backend=cu128 -r requirements/imajev.txt
.venv-image/bin/python -m scripts.download_imajev --out models/imajev-reference
.venv-image/bin/python -c 'from zils.imajev import create_reference; create_reference("models/imajev-reference", "models/imajev-starting-checkpoint")'
export ZILS_IMAGE_RUNTIME_PYTHON="$PWD/.venv-image/bin/python"
export ZILS_IMAGE_REFERENCE="$PWD/models/imajev-reference"
.venv-kev/bin/python -m miner.queue \
  --config .private/queue-miner.json --state .private/queue-miner \
  --reference models/h2o-reference \
  --reference models/imajev-starting-checkpoint --device cuda
```

Stop the existing worker before restarting it with both references and the same
state directory. One process trains jobs serially, using the matching reference
and separate image runtime. Claims include only installed profiles that have
current operator approval and enough free GPU memory. Image workers receive only
training photos; validation photos and labels remain on the processor.

Keep image admission disabled until the operator has verified training, evaluation,
and activation. A worker with both references installed does not establish launch
readiness. Retain a Kev reference only for explicitly supported legacy assignments.
