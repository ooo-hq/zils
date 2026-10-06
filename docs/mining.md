# Run a Kev fleet miner

This guide covers the **Kev 0.8B local and Bittensor testnet fleet**.
For customer training, use the [JevK5 queue miner guide](https://github.com/ooo-hq/zils/blob/main/docs/queue-miners.md).
The two workflows use different models, credentials, and startup commands.

After the one-time setup below, run this in your assigned miner folder:

```bash
./start-miner
```

The miner trains one candidate per round, submits its signed checkpoint, receives
the validator's result, and waits for the next round. Stop with Ctrl+C, or use
`./start-miner --rounds 1` for a single result. The first start downloads the
pinned base model. Later starts reuse the cache.

## Prepare bundles on the validator

From the repository root, after installing dependencies and downloading the
reference as described in the [project README](https://github.com/ooo-hq/zils#setup):

```bash
.venv-kev/bin/python -m zils.benchmark build --out .private/benchmarks/local
.venv-kev/bin/python -m zils.fleet init --out .private/fleet-local \
  --benchmark .private/benchmarks/local --host VALIDATOR_PRIVATE_IPV4
```

Replace `VALIDATOR_PRIVATE_IPV4` with the validator's numeric private address.
Use new output paths or reuse an existing audited benchmark. The default ports
are 8900 for the validator and 8901–8903 for three miners. Assign one
`miner-N.tar.gz` to each machine.

Only training data goes into a miner bundle. The validator keeps calibration
cases, test cases, and full reports. Local bundles contain disposable signing
keys and must stay private. [Testnet bundles](testnet.md) contain public wallet
references instead; provision each hotkey separately.

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

The miner selects CUDA, then Apple MPS, then CPU. Use `--device cuda`, `mps`, or
`cpu` to select explicitly. To reuse an environment or cache, set `ZILS_PYTHON`
to its absolute Python path and `HF_HOME` to the cache directory.

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
