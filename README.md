<p align="center">
  <img src="subnet.png" alt="Fez mascot wearing a fez" width="96">
</p>

<h1 align="center">Fez</h1>

<p align="center">
  <a href="https://fez.chat/model">fez.chat/model</a>
</p>

**A small decision model with a Bittensor training competition.**
Fez returns probabilities for yes/no decisions, choices, and scores without
generating text. The current model is an experimental 0.8B candidate.
This repository contains the miner, validator, benchmark, and testnet integration.

Miners fine-tune the model and submit checkpoints. The validator runs those
checkpoints on its own evaluation data, scores their probabilities, and assigns
weights. The resulting adapter and decision head can be loaded with the pinned
base model for applications to use.

**Status:** local training and evaluation work on Apple Silicon and an RTX 4090.
The first closed training-to-chain round completed on **testnet subnet 579**, with
three fresh miner checkpoints and verified revealed weights. See the
[recorded result and limitations](docs/testnet-round-001.md).
There is no published Fez model release or automatic winner promotion yet.

**Zils API:** the decision protocol/subnet is now named Zils; Fez remains the
chat application. The [new decision API](docs/decision-api.md) adds authenticated
TypeSafe-style inference and durable bulk jobs using shared JevK5. Its local
verification and deployment limits are documented separately from this training
pilot; it does not automatically serve miner adapters. Existing package and
training resource names remain compatible.

[Documentation](docs/README.md) covers operation, development, evaluation, and
measured results.

## How it works

1. **Train:** each miner fine-tunes the pinned reference and freezes a candidate.
2. **Submit:** the miner signs its checkpoint hash and serves its three artifact files.
3. **Evaluate:** the validator verifies the bytes, calibrates confidence, and runs its private test cases.
4. **Reward:** probability quality determines weights; testnet publication requires an explicit flag.

The current fleet uses a fixed one-epoch training recipe with different seeds.
It exercises the whole loop; model improvements still require controlled training
experiments. Validators measure model outputs themselves rather than accepting
miner-reported scores. See the [evaluation contract](docs/evaluation.md).

## Setup

### Fine-tune on your own data

Install the [Fez fine-tuning skill](skills/fez-finetune/SKILL.md) into your coding
agent:

```bash
npx skills add ooo-hq/fez@fez-finetune
```

Then ask: “Fine-tune a 4B Fez candidate on my labelled support tickets.”
The skill supports 0.8B, 4B, and 9B starting checkpoints, data preparation,
calibration, baseline comparisons, and optional serving. Cloud training requires
a Modal account and an agreed compute budget.
Installing the skill starts no training and needs no GPU.

Outputs are experimental Fez candidates. Published Fez weights are not yet
available, and the current subnet accepts only its pinned 0.8B architecture;
4B/9B own-data experiments do not change that contract.

### Repository setup

The public dashboard is available at [fez.chat/model](https://fez.chat/model).
For a standalone local preview, see [`website/`](website/README.md). With
Node.js 22+ and Python 3 installed, run `npm --prefix website run preview` and
open <http://127.0.0.1:4173>. No model environment or wallet is needed.

Use macOS, Linux, or WSL 2 with Python 3.13, Git, and `uv` installed.
Run these commands from the repository root:

```bash
git clone https://github.com/ooo-hq/fez.git
cd fez
uv venv --python 3.13 .venv-kev
uv pip install --python .venv-kev/bin/python -r requirements/model.txt -r requirements/rehearsal.txt
.venv-kev/bin/python -m scripts.download_models
```

Dependencies, the model runtime, the public reference checkpoint, and the base
model are pinned. Downloads happen once; model workers run from the local cache.
For Windows GPU and network setup, follow the [miner guide](docs/mining.md).
The Bittensor SDK is optional until [testnet setup](docs/testnet.md).

## Run a local fleet

For authorized business data, the experimental [customer job workflow](docs/customer-jobs.md)
adds a frozen job manifest, calibrated baseline comparison, and a local export
only when a candidate meets acceptance thresholds. Each fleet configuration
currently handles one job. The [Supabase training queue](docs/supabase-training.md)
adds private customer uploads and a shared pool of approved workers. Live upload
and validation checks have passed; real training through the queue remains unverified.

1. Generate a benchmark and three miner bundles. Use new output directories:

   ```bash
   .venv-kev/bin/python -m fez.benchmark build --out .private/benchmarks/local
   .venv-kev/bin/python -m fez.fleet init --out .private/fleet-local \
     --benchmark .private/benchmarks/local --host 127.0.0.1
   ```

2. Start the validator for one round:

   ```bash
   .venv-kev/bin/python -m fez.fleet validator \
     --config .private/fleet-local/validator/config.json --rounds 1
   ```

3. In separate terminals at the repository root, start each miner. Repeat for
   `miner-2` and `miner-3`:

   ```bash
   FEZ_PYTHON="$PWD/.venv-kev/bin/python" HF_HOME="$PWD/.cache/huggingface" \
     .private/fleet-local/miner-1/start-miner --rounds 1
   ```

The device is selected automatically: CUDA, Apple MPS, then CPU. Services sharing
one device run their model work sequentially. Reports appear in
`.private/fleet-local/validator/state/rounds/`. No chain writes occur in local mode.
For separate machines, generate bundles using the validator's private IPv4
address and follow [miner setup](docs/mining.md). After setup, each miner needs
one command: `./start-miner`.

## Repository layout

```text
fez/             Scoring, benchmark, calibration, validator, and testnet code
miner/           Training and signed checkpoint submission
scripts/         Pinned model download and two-miner rehearsal
tests/          Scoring, protocol, process, and chain integration checks
docs/           Setup details, benchmark methodology, and experiment history
skills/         Installable own-data fine-tuning workflow
website/        Static public dashboard and recorded benchmark comparison
examples/       Public diagnostic cases and smoke-training data
requirements/   Pinned model, signing, and optional testnet dependencies
```

## Development

After setup, install the lint tools and SDK used by the testnet tests:

```bash
uv pip install --python .venv-kev/bin/python -r requirements/dev.txt -r requirements/testnet.txt
uv pip install --python .venv-kev/bin/python --no-deps -r requirements/jevk5-source.txt
make check
```

Tests exercise real signatures, HTTP transfers, subprocesses, and restart
recovery, with fixture model workers and fake chain RPC. They do not train a
model or send transactions. `make check` requires all test dependencies, runs
Ruff lint/format checks, and executes the full suite. GitHub Actions runs the same
checks on pull requests and pushes to `main`. Use `make format` to format Python.

See [local development](docs/development.md) for checkpoint scoring and the
shorter two-miner rehearsal. Commands use `python -m ...`; old flat script paths
have been replaced. Existing standalone miner bundles keep their bundled code;
regenerate bundles when upgrading them.

## Model provenance

Fez fine-tunes published [Kev](https://github.com/jaredpalmer/kev) checkpoints
and uses its pinned training and serving tools. The current 0.8B candidate builds
on Qwen3.5-0.8B-Base through Kev. The public JevBench comparison uses the unchanged
published checkpoint as its baseline to measure what Fez's training changed.
Exact source and model revisions are recorded in the
[experiment methodology](docs/experiments.md#runtime-and-reference-models).

## Results and limits

The latest larger-data experiment scored **976/1,120 correct (87.14%)**, versus
958/1,120 (85.54%) for the previous Fez candidate. Equal-source Brier loss fell
9.59%, while high-confidence mistakes increased from 18 to 23. This is an
experimental comparison, not a general claim that Fez beats Kev.
[Experiment history](docs/experiments.md#optimized-4090-training-and-a-larger-dataset)
records the datasets, settings, and tradeoffs.

On the separate [JevBench public comparison](docs/jevbench-public.md), current
Fez and published Kev tied at 147/231 correct (63.64%); Fez's confidence quality
regressed. No official JevBench rank has been measured.

The [synthetic benchmark](docs/benchmark.md) has shared templates and is reused
for development. The current private-LAN services are for operator-controlled
checkpoints; their subprocesses are not an untrusted-model security sandbox.
Public discovery, independent hidden evaluation, and model promotion remain
[future work](docs/roadmap.md).

Model weights, wallets, private datasets, bundles, and raw runs are excluded
from Git. A fresh clone downloads the public reference and generates new
local data; it does not contain the experimental Fez checkpoints.
