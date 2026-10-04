# Local development

Run commands from the repository root. Use Python 3.13 and Make on macOS, Linux,
or WSL 2. Linting and tests need no model weights, GPU, wallet, or chain access.

## Install development dependencies

Create `.venv-kev` if it does not exist, using `uv venv --python 3.13 .venv-kev`.
Install the pinned tools and dependencies, including the optional SDK used by
the testnet tests:

```bash
uv pip install --python .venv-kev/bin/python \
  -r requirements/dev.txt -r requirements/model.txt \
  -r requirements/rehearsal.txt -r requirements/testnet.txt
uv pip install --python .venv-kev/bin/python --no-deps -r requirements/jevk5-source.txt
make check
```

`make check` runs Ruff linting, a formatting check, and the full Python suite.
When `website/package.json` is present, it also runs the website checks; install
Node.js 22+ and npm for that part.
Its dependency preflight fails if an optional runtime is missing, so integration
tests cannot silently skip because the SDK or model library was not installed.
Use `make lint` for the fast checks, `make format` to sort imports and format
Python, and `make test` for the suite. Override `PYTHON` to use another environment,
for example `make check PYTHON=python` in an activated virtual environment.

The process tests run real HTTP, signatures, checkpoint transfer, calibration,
and restart handling. Training and inference use fixture workers; chain RPC is
faked. Tests need no GPU and never submit transactions. Passing them does not
establish model quality or GPU performance.

## Continuous integration

[GitHub Actions](https://github.com/ooo-hq/fez/actions/workflows/checks.yml) runs
lint/format checks and the full Python suite on pull requests and pushes to
`main`. CI installs CPU PyTorch and runs model workers offline. When website
source is present, a separate job checks JavaScript syntax, runs its tests, and
builds the static site. Separate jobs exercise the training/API migrations and
bulk worker in disposable PostgreSQL, and the official Python/JavaScript
TypeSafe clients against the local gateway. See [API verification](decision-api.md#verification-and-measured-scope).
Run the same website checks locally with
`make check-website` using Node.js 22+ and npm.

The workflow uses read-only repository permissions and pinned Action revisions.
Repository branch-protection settings determine whether these checks are required
before merging; the workflow itself does not change those settings.

## Score the reference checkpoint

Complete the [model installation steps](../README.md#setup) to download
`models/reference` and cache the pinned base model before these examples.

```bash
.venv-kev/bin/python -m fez submit \
  --checkpoint models/reference --uid 1 > submissions-reference.json
.venv-kev/bin/python -m fez evaluate \
  --submissions submissions-reference.json --cases examples/cases.jsonl \
  --base-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --runner-python .venv-kev/bin/python --device cpu --report round-local.json
```

Use fresh output paths. For multiple candidates, combine their submission objects
into one JSON list; every UID and checkpoint hash must be distinct. The public
example questions are diagnostics, not an emissions benchmark.

The commands below also default to `cpu`. For GPU execution, replace that device
with `mps` on Apple Silicon or `cuda` on a configured NVIDIA host. See
[miner setup](mining.md) for Windows/WSL requirements.

## Train a smoke-test candidate

This four-example run verifies training and checkpoint handling. It is not a
model-quality experiment. Select an unused output directory:

```bash
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
.venv-kev/bin/python -m kev.train \
  --data examples/train-smoke.jsonl \
  --base Qwen/Qwen3.5-0.8B-Base \
  --base_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --init_from models/reference \
  --epochs 1 --lr 2e-5 --batch 1 --accum 1 --dtype fp32 --device cpu \
  --p_none 0 --p_none_distract 0 --p_distract 0 --seed 42 \
  --out models/fez-probe
```

The resulting checkpoint contains the adapter and decision head. Training resets
its saved temperature to 1.0. For a quality experiment, use separate training,
calibration, and test cases as described in the [benchmark guide](benchmark.md).

## Rehearse two existing checkpoints

After the smoke-test training command succeeds:

```bash
.venv-kev/bin/python -m scripts.rehearsal run \
  --checkpoints models/reference models/fez-probe \
  --device cpu --timeout 1800 --out runs/rehearsal
```

This starts two miners and one validator as separate processes, then stops all
three when evaluation finishes or fails. It generates disposable signing keys
and saves `validator/report.json` beneath the output directory. CPU evaluation
can take several minutes; the command allows up to 30 minutes per checkpoint.

This rehearsal submits existing checkpoints. The [persistent fleet](mining.md)
trains a new candidate each round. Neither local mode writes chain weights.
