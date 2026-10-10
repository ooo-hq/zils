# Run a text or image fleet miner

Local and Bittensor testnet fleets support **JevK5 4B text** and **ImaJev 4B image**
jobs. Each fleet pins one frozen job and its matching model reference. Existing
Kev 0.8B bundles remain supported. For workers that claim different customer jobs,
use the [queue miner guide](https://github.com/ooo-hq/zils/blob/main/docs/queue-miners.md).
Queue and fleet miners use the same trainers but different credentials and startup commands.

After the one-time setup below, run this in your assigned miner folder:

```bash
./start-miner
```

The miner trains one candidate per round, submits its signed checkpoint, receives
the validator's result, and waits for the next round. Stop with Ctrl+C, or use
`./start-miner --rounds 1` for a single result. The first start downloads the
pinned text base model. Image miners require the separately installed, verified
image runtime described below. Later starts reuse the cache.

## Prepare bundles on the validator

For a text fleet, complete the [JevK5 model setup](https://github.com/ooo-hq/zils/blob/main/docs/jevk5-queue.md#install-and-create-the-reference).
Prepare authorized, independent training/calibration/test JSONL files using the
[customer data format](https://github.com/ooo-hq/zils/blob/main/docs/customer-jobs.md#prepare-data).
The following freezes those files and an example acceptance policy; choose your
policy before evaluating candidates:

```bash
.venv-kev/bin/python -m zils.jobs --job-id text-v1 \
  --model jevk5-4b-v0.3 --train .private/train.jsonl \
  --calibration .private/calibration.jsonl --test .private/test.jsonl \
  --min-accuracy 0.80 --min-brier-improvement 0.01 \
  --allow-training-data-export --out .private/jobs/text-v1
.venv-kev/bin/python -m zils.fleet init --out .private/fleet-local \
  --benchmark .private/jobs/text-v1 --checkpoint models/jevk5-reference \
  --host VALIDATOR_PRIVATE_IPV4
```

Replace `VALIDATOR_PRIVATE_IPV4` with the validator's numeric private address.
Use new output paths or reuse an existing audited benchmark. The default ports
are 8900 for the validator and 8901–8903 for three miners. Assign one
`miner-N.tar.gz` to each machine.

Only training data goes into a miner bundle. The validator keeps calibration
cases, test cases, and full reports. Local bundles contain disposable signing
keys and must stay private. [Testnet bundles](testnet.md) contain public wallet
references instead; provision each hotkey separately.

### Image fleets

Complete the pinned runtime and maximum-context hardware qualification in
[image training](https://github.com/ooo-hq/zils-platform/blob/main/docs/image-training.md#worker-and-service-setup)
for every participating miner and the validator. Use a fresh published starting
checkpoint, not an earlier customer adapter. The operator supplies an audited
image-job directory built by `zils.image_jobs.build`, its canonical image cache
(`<sha256>.png`), and measured admission limits. Those are private inputs; fleet
initialization does not download customer photos or resolve hosted predecessor jobs.

```bash
.venv-kev/bin/python -m zils.fleet init --out .private/fleet-image \
  --benchmark .private/jobs/image-v1 --checkpoint models/imajev-starting-checkpoint \
  --images .private/image-cache --min-free-mib QUALIFIED_FREE_MIB \
  --max-seconds QUALIFIED_TRAINING_SECONDS --host VALIDATOR_PRIVATE_IPV4
```

Replace both qualification placeholders with the operator's measurements. Memory
admission must cover peak reserved GPU memory plus 512 MiB and be at least
12,288 MiB; the training deadline must be 1–3,600 seconds. Use limits sufficient
for every host in this fleet. Bundles receive only training photos; calibration
and test photos stay with the validator. Export verifies image hashes and sizes.

Each model uses its own fleet directory and ports. Never combine text and image
scores into one reward vector. Only one validator may publish for a hotkey/subnet;
do not run competing vertical publishers that overwrite each other's weights.

For a legacy Kev rehearsal, keep `zils.benchmark build` and `models/reference`.
Do not relabel a Kev manifest as a different model.

## Set up each machine once

1. Unpack its assigned archive and enter the `miner-N` folder.
2. Install Git and Python 3.13, then install the pinned dependencies:

   ```bash
   python3.13 -m venv .venv-kev
   .venv-kev/bin/python -m pip install -r requirements/model.txt -r requirements/rehearsal.txt
   ```

   With `uv`, use `uv venv --python 3.13 .venv-kev` and
   `uv pip install --python .venv-kev/bin/python -r requirements/model.txt -r requirements/rehearsal.txt`.

3. Allow the assigned miner TCP port from the validator's private address.
   Allow validator TCP 8900 from the miner machines.
4. Run `./start-miner --rounds 1`. It can start before the validator and will retry.

JevK5 needs a qualified CUDA or Apple MPS host. Image training requires qualified
CUDA hardware; CPU fallback applies only to Kev. Use `--device cuda` or `mps`
explicitly. To reuse an environment or cache, set `ZILS_PYTHON`
to its absolute Python path and `HF_HOME` to the cache directory.

For image bundles, also install and verify the isolated runtime on each host:

```bash
uv venv --python 3.13 .venv-image
uv pip install --python .venv-image/bin/python --torch-backend=cu128 -r requirements/imajev.txt
.venv-image/bin/python -m scripts.download_imajev --out models/imajev-reference
export ZILS_IMAGE_RUNTIME_PYTHON="$PWD/.venv-image/bin/python"
export ZILS_IMAGE_REFERENCE="$PWD/models/imajev-reference"
./start-miner --device cuda --rounds 1
```

The validator needs these same image runtime variables. Set the shared
`ZILS_COMPUTE_LOCK` and the qualified `ZILS_IMAGE_GPU_MEMORY_FRACTION` when sharing
a GPU, as described in the image training guide. Low capacity makes a miner wait;
it never permits a smaller unverified memory allowance.

### Windows / NVIDIA GPU

Run the miner inside WSL 2, using the supported NVIDIA driver installed on
Windows. Do not install a Linux NVIDIA display driver inside WSL. Check CUDA
from the miner environment:

```bash
.venv-kev/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA is not ready"; print(torch.cuda.get_device_name(0))'
```

The validator must reach the miner directly at its private IPv4 address.
Default WSL NAT does not support this callback layout without additional
network setup. Use [WSL mirrored networking](https://learn.microsoft.com/en-us/windows/wsl/networking#mirrored-mode-networking)
and limit the Windows/Hyper-V firewall rule to the assigned port and validator.
See [NVIDIA's WSL setup](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
Native Windows service execution and NAT traversal are not implemented.

## Start the validator

From the repository root:

```bash
.venv-kev/bin/python -m zils.fleet validator \
  --config .private/fleet-local/validator/config.json --rounds 1
```

Remove `--rounds 1` for continuous rounds. The validator evaluates after every
miner submits or the collection deadline expires. It verifies signatures and
hashes, calibrates each candidate, and scores the separate evaluation cases.
Local mode never writes weights to Bittensor. Testnet mode needs registered
wallets and explicit publication; follow the [testnet guide](testnet.md).

## Results and restarts

Miner logs, frozen candidates, and results are in `state/jobs/`. Full validator
reports are in `validator/state/rounds/`. Restarting a miner reuses completed
candidates; an interrupted training epoch starts again. The validator reloads
accepted submissions after restart. Service locks prevent duplicate launches,
and a per-device compute lock serializes work on shared hardware.

Keep started bundles at their original location: saved checkpoint paths are
absolute. Archive completed rounds between runs if disk space fills; training
and evaluation stop when less than 2 GiB is free. Regenerate bundles to upgrade
code; existing bundles are standalone snapshots.

This is a closed development fleet with a fixed one-epoch recipe. Every round
starts from the same reference with different seeds; winners are not promoted
automatically. The reused synthetic benchmark measures development progress.
Signed HTTP authenticates messages but does not encrypt them; use a trusted
private LAN. Evaluation subprocesses do not isolate hostile checkpoints.
