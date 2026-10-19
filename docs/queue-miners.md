# Run a text or image queue miner

A queue miner downloads assigned training examples, trains the pinned adapter, and
submits signed artifacts for independent evaluation. It connects to the
coordinator over outbound HTTPS. It needs an approved hotkey and model hardware;
it does not need Supabase credentials or an inbound public server.

The setup below is for **d3-mini 4.5B** (`d3-mini-4.5b-v1`), the base on both
channels since 17 October 2026; it trains text and image jobs. H2O Lightning 4B
is retired (keep its runtime only if the operator still routes legacy H2O jobs to
you) and the ImaJev image profile is retired: image jobs run on d3. For a
**Bittensor testnet fleet**, use
[registration](bittensor-registration.md) and
[bundle setup](mining.md). Queue assignments publish no Bittensor weights by
themselves. A testnet validator can pay delivered queue work from its adapter
share ([validators §7](validators.md#7-pay-the-adapter-track-from-hosted-queue-work),
off by default); only hotkeys registered on that subnet earn.

Operators can enable [graded job assignment](miner-grading.md) after qualification.
Miners opt into signed availability only after the coordinator upgrade; existing
configurations retain their current request flow.

## 1. Confirm access and hardware

Get the following from the operator before downloading models:

- The coordinator's exact HTTPS base URL and confirmation that its channels
  name `d3-mini-4.5b-v1` (`GET /v1/config`, `channels.stable` and `channels.latest`).
- Approval for your hotkey's public SS58 address. The operator registers the
  worker and approves assignments through [miner administration](https://github.com/ooo-hq/zils-platform/blob/main/docs/supabase-training.md#approve-miners-and-run-a-queued-miner).
- The wallet name, hotkey name, and local wallet directory for the hotkey you
  control. Provision the hotkey privately on the miner host; keep the coldkey
  and recovery phrases off the host and out of the repository.

Use Linux or WSL 2 with Git, Python 3.13, `uv`, a BF16-capable NVIDIA GPU and its
CUDA driver. Allow at least 25 GB of free disk for models and dependencies,
plus space for assigned data and candidates. The runtime was checked on an
RTX 4090 (d3 training peaks about 10.5 GB; the worker needs 12,288 MiB free).
d3 and H2O use isolated CUDA runtimes below. CPU and Apple MPS training are
unsupported; the retired JevK5 Mac path no longer exists.

The operator assigns a local queue UID. It is not your subnet UID. On-chain
registration is not required for this queue, and a Bittensor UID alone does not
authorize job access; to earn on chain for queue work, register the same hotkey
on the subnet.

Operators standing up a rented node can run `scripts/provision_node.sh` in
place of step 2 (it performs the same installation and writes a qualification
report) and configure the worker through `/etc/zils-node/role.json` and
`scripts/apply_role.sh` in place of steps 3–4. See
[fleet nodes](fleet-nodes.md) and the [runbook](fleet-node-runbook.md).

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
uv venv --python 3.12 .venv-d3
uv pip install --python .venv-d3/bin/python --torch-backend=cu130 -r requirements/d3.txt
export ZILS_D3_RUNTIME_PYTHON="$PWD/.venv-d3/bin/python"
.venv-d3/bin/python -m zils.d3 reference --out models/d3-reference
# Only if the operator still routes legacy H2O jobs to you:
uv venv --python 3.12 .venv-h2o
uv pip install --python .venv-h2o/bin/python --torch-backend=cu130 -r requirements/h2o.txt
export ZILS_H2O_RUNTIME_PYTHON="$PWD/.venv-h2o/bin/python"
.venv-h2o/bin/python -m zils.h2o reference --out models/h2o-reference
```

The Bittensor SDK is installed here to read the wallet hotkey; these commands
send no chain transactions. The reference command downloads and verifies the
pinned base weights (9.1 GB for d3) and writes `models/d3-reference/model.json`.
The pins are `zils/d3-pins.json` (vendor release `zils-ai/d3-mini-4.5b-v1`,
4,096-token inputs, up to four images or one video per decision). The operator
must explicitly qualify this hotkey for d3 on your actual hardware
(`scripts.qualify_node --model d3-mini-4.5b-v1` writes the report); an approval
for another profile does not qualify it for d3.

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

Two optional keys apply once [chain admission](miner-admission.md) is on:
`"terms_sha256"` (the SHA-256 of [the miner terms](mining-terms.md) you accept)
and `"qualification"` (path to the `qualification.json` written by
`scripts.qualify_node --terms-sha256 <same hash>`). The worker then admits
itself at startup and sends the hash with every claim. Operator-registered
nodes leave both out.

```bash
chmod 600 .private/queue-miner.json
```

Use a dedicated miner operating-system account. Never load the validator's
`.private/training.env` or Supabase service-role key into it. If training and
evaluation share a GPU, have the operator set the same `ZILS_COMPUTE_LOCK` path for
both services before starting: an administrator-owned directory with group
read/write access whose parent miners cannot replace. Without it, locking is per
user and device.

## 4. Start the miner

Run from the repository root (also set this environment variable in a service supervisor):

```bash
export ZILS_D3_RUNTIME_PYTHON="$PWD/.venv-d3/bin/python"
.venv-kev/bin/python -m miner.queue \
  --config .private/queue-miner.json --state .private/queue-miner \
  --reference models/d3-reference --device cuda
```

Add `--reference models/h2o-reference` and export
`ZILS_H2O_RUNTIME_PYTHON="$PWD/.venv-h2o/bin/python"` only if you provisioned the
legacy H2O runtime in step 2. The miner refuses to start when a `--reference`
directory is missing or its runtime interpreter variable is unset.

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

## Image jobs

Since 17 October 2026 image jobs are d3 jobs with a frozen image intake: the same
d3 reference, runtime and qualification cover them, and no separate image runtime
is installed (the ImaJev profile is retired; see the platform's
[image training](https://github.com/ooo-hq/zils-platform/blob/main/docs/image-training.md)
guide for the intake and asset flow). Image workers receive only training photos;
validation photos and labels remain on the processor. Retain a Kev reference only
for explicitly supported legacy assignments.

## Base-tier candidates

Queue work is customer adapters. The network's primary job, improving the pinned
base itself, is the [base tier](base-candidates.md): a full-weight candidate you
train and merge yourself, examined on a sealed holdout by a fleet validator and
paid from the chain's weight vector (king of the hill: only the miners of
promoted bases are paid). It needs a hotkey
registered on testnet subnet 579, a public Hugging Face repo and
`zils.miner commit-candidate`; see
[compete from outside](mining.md#compete-in-the-base-tier-from-outside) and
[open admission](validators.md#open-admission-base-tier).
