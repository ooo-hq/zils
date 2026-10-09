# Run a JevK5 miner on Apple silicon

A Mac miner trains adapters and submits them to the coordinator over outbound
HTTPS. The validator evaluates them separately, and the serving system publishes
accepted models. Running this worker does not start a prediction server or
establish on-chain earnings.

## Requirements and measured limits

Use Apple silicon, macOS 14 or newer, Python 3.13, Git, and `uv`. Allow at least
25 GB free disk for the pinned model and dependencies, plus training artifacts
and swap. Obtain an approved hotkey and the coordinator URL from the operator.
This guide supports the **JevK5 text profile**, not Imajev image training.

The tested machine was an **M4 Mac mini with 16 GiB unified memory**, running
macOS 26.5.2 and the pinned PyTorch 2.8.0 runtime. Other chips, operating-system
versions, and memory sizes need their own qualification before receiving work.

| Synthetic verification | Result |
| --- | --- |
| Four short routing examples, training and adapter reload | Passed in 37 seconds |
| Four 2,048-token examples, 16 outcomes, training and reload | Passed in 229 seconds; training itself took 197 seconds |
| Sampled maximum Metal driver allocation during the long test | 12.82 GB decimal; sampled every 50 ms, not an exact peak |
| Long-test system memory pressure | Approximately 9.4 GB of swap in use; 16 GiB has limited headroom |
| Mac-produced BF16 adapter loaded by the existing CUDA loader | Passed strict tensor validation and a 16-outcome prediction |

These are compatibility and capacity probes, not throughput guarantees or
evidence of customer-task accuracy. The long test deliberately repeats synthetic
text; larger datasets and other inputs can behave differently. CUDA and MPS
probabilities were not bit-identical: the largest absolute difference on the
single cross-device probe was 0.002524. Independent validator evaluation remains
required. Keep memory-heavy applications closed while mining, and do not disable
PyTorch's MPS memory limit to force an oversized job to run.

## 1. Install the runtime

From a fresh clone:

```bash
git clone https://github.com/ooo-hq/zils.git
cd zils
uv venv --python 3.13 .venv-kev
uv pip install --python .venv-kev/bin/python \
  -r requirements/model.txt -r requirements/rehearsal.txt \
  -r requirements/testnet.txt
.venv-kev/bin/python -c 'import torch; assert torch.backends.mps.is_available(); assert torch.backends.mps.is_macos_or_newer(14, 0)'
.venv-kev/bin/python -m zils.jevk5 reference --out models/jevk5-reference
```

The reference command verifies the same base weights used by CUDA miners and
validators. No model conversion is needed. The optimizer retains FP32 LoRA
parameters and exports the existing approximately 29 MB BF16 adapter format.

## 2. Verify this Mac and configure its identity

Run the real four-example training, export, and reload test:

```bash
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=1 \
  ZILS_TEST_JEVK5_MPS_REFERENCE=models/jevk5-reference \
  .venv-kev/bin/python -m unittest tests.test_jevk5_mps.MacTrainingTest -v
```

This test checks finite saved BF16 tensors, a nonzero LoRA update, and a normalized
prediction after reload. It uses short examples; it does not qualify the full
2,048-token envelope on an untested Mac. An operator should check representative
maximum-size inputs before assigning such jobs.

Follow [Configure your worker](queue-miners.md#3-configure-your-worker) to create
`.private/queue-miner.json` with your coordinator URL and private hotkey wallet.
Keep the wallet on this host; supply only its public address to the operator for
registration. The operator must approve the worker and its assignments. The
miner receives no Supabase service credentials, calibration labels, or test labels.

## 3. Start training work

From the repository root:

```bash
caffeinate -i .venv-kev/bin/python -u -m miner.queue \
  --config .private/queue-miner.json --state .private/queue-miner \
  --reference models/jevk5-reference --device mps --no-download
```

The worker polls every ten seconds and remains quiet when no approved assignment
is available. Training processes use the Apple GPU, with CPU fallback enabled for
unsupported operations. Only one training process per user/device holds the
compute lock. `caffeinate -i` prevents idle system sleep while the worker runs;
explicit sleep and logout still interrupt it. Stop a foreground worker with Ctrl+C.

## 4. Optionally run after login

After the foreground check, stop that worker and create a per-user LaunchAgent
from the repository root. This command embeds absolute paths for this checkout;
it refuses to overwrite an existing agent. Keep the checkout at that location.

```bash
.venv-kev/bin/python - <<'PY'
import os
import plistlib
from pathlib import Path

root = Path.cwd().resolve()
assert (root / '.private/queue-miner.json').is_file()
directory = Path.home() / 'Library/LaunchAgents'
directory.mkdir(exist_ok=True)
path = directory / 'ai.zils.mac-miner.plist'
value = {
    'Label': 'ai.zils.mac-miner',
    'ProgramArguments': [
        '/usr/bin/caffeinate', '-i', str(root / '.venv-kev/bin/python'),
        '-u', '-m', 'miner.queue',
        '--config', str(root / '.private/queue-miner.json'),
        '--state', str(root / '.private/queue-miner'),
        '--reference', str(root / 'models/jevk5-reference'),
        '--device', 'mps', '--no-download',
    ],
    'WorkingDirectory': str(root),
    'RunAtLoad': True,
    'KeepAlive': True,
    'ThrottleInterval': 30,
    'ProcessType': 'Background',
    'StandardOutPath': str(root / '.private/miner.stdout.log'),
    'StandardErrorPath': str(root / '.private/miner.stderr.log'),
}
with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
    plistlib.dump(value, stream)
PY
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/ai.zils.mac-miner.plist"
launchctl print "gui/$(id -u)/ai.zils.mac-miner"
```

This starts at login and restarts after a process failure. It does not run before
login. Inspect `.private/miner.stdout.log`, `.private/miner.stderr.log`, and the
per-job training logs under `.private/queue-miner/`. Restarting with the same
state directory preserves completed candidates for upload retries.

To stop the background worker:

```bash
launchctl bootout "gui/$(id -u)/ai.zils.mac-miner"
```

Run the `bootstrap` command again to start it. Removing the plist after stopping
it prevents future login starts; retain the private wallet and job state.
