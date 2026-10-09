<h1 align="center">
  <a href="https://zils.ai">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/assets/zils-logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="docs/assets/zils-logo-light.svg">
      <img alt="Zils" src="docs/assets/zils-logo-light.svg" width="196">
    </picture>
  </a>
</h1>

<p align="center">
  <a href="https://zils.ai">Website</a> ·
  <a href="https://zils.ai/model">Research</a> ·
  <a href="docs/README.md">Documentation</a>
</p>

**Train decision models on your data. Get probabilities through a simple API.**

Zils returns probabilities for yes/no decisions, choices, and scores. Train a
model on labelled examples, measure it against its starting checkpoint, and
serve accepted models through an authenticated API. This repository contains
the API, training queue, miners, validators, and Bittensor testnet integration.

## Start here

| I want to… | Start with |
| --- | --- |
| Call a model or submit a bulk job | [Decision API](docs/decision-api.md) and [example request](examples/zils-api/request.json) |
| Train on my own data | [Data format](docs/customer-jobs.md#prepare-data) and [JevK5 training workflow](docs/automatic-training.md) |
| Run a miner | [Miner setup](#miner-setup) |
| Run a validator | [Validator setup](#validator-setup) |
| Inspect measured improvements | [Results and limits](#results-and-limits) |

## How training works

1. **Prepare:** separate labelled examples into training, calibration, and test
   sets. Freeze the data and acceptance criteria before training.
2. **Train:** an approved miner receives the training split and produces a
   candidate adapter. Calibration and test examples stay with the evaluator.
3. **Evaluate:** the validator checks the candidate against the starting model
   on held-out examples. A version upgrade compares against the previous
   accepted customer model.
4. **Serve:** when configured, the automatic workflow verifies an accepted
   release and makes it available to its owner through the API.

A run can finish with `no_qualifying_model`. Training does not guarantee an
improvement. See [acceptance criteria](docs/customer-jobs.md#inspect-the-result),
[version selection](docs/version-selection.md), and
[model activation](docs/automatic-training.md). Operators can opt into
[miner grading and job assignment](docs/miner-grading.md) across an approved pool.

## Miner setup

A miner trains candidates and submits signed artifacts. Choose the workflow
before installing dependencies:

| Workflow | Requirements | Setup guide |
| --- | --- | --- |
| **Customer training — JevK5 4B** | Linux or WSL 2, a BF16-capable NVIDIA GPU, an approved hotkey, and the coordinator URL | [Install, configure, and run a queued miner](docs/queue-miners.md) |
| **Customer training on Mac — JevK5 4B** | Apple silicon, macOS 14+, an approved hotkey, and the coordinator URL; tested on M4 with 16 GiB | [Install and qualify an Apple GPU miner](docs/mac-miners.md) |
| **Bittensor testnet — Kev 0.8B** | Operator-confirmed testnet registration, an assigned miner bundle, and private connectivity to the validator | [Register a miner](docs/bittensor-registration.md), then [install its bundle](docs/mining.md#set-up-each-machine-once) |

For customer training, the guide creates `models/jevk5-reference` and
`.private/queue-miner.json`. Start the configured worker from the repository root:

```bash
.venv-kev/bin/python -m miner.queue \
  --config .private/queue-miner.json --state .private/queue-miner \
  --reference models/jevk5-reference --device cuda
```

Use `--device mps` on a qualified Mac. The queue miner uses outbound HTTPS and
needs no Supabase service credential.
For the testnet fleet, run `./start-miner --rounds 1` from the assigned,
configured bundle. Joining either workflow requires operator approval.

## Validator setup

A validator runs its own evaluation of submitted models. Choose the matching
workflow and complete its configuration before starting:

| Workflow | Requirements | Setup guide |
| --- | --- | --- |
| **Customer training — JevK5 4B** | A matching GPU/reference model, protected Supabase operator credentials, and the configured training project | [Queue validator setup](docs/validators.md#jevk5-queue-validator) |
| **Bittensor testnet — Kev 0.8B** | A registered, eligible validator hotkey, an explicit miner roster, and private network access | [Testnet validator setup](docs/validators.md#bittensor-testnet-validator) |

After completing queue validator setup, load its protected environment and
start the evaluator from the repository root:

```bash
set -a
. .private/training.env
set +a
.venv-kev/bin/python -m zils.coordinator process \
  --state .private/queue-processor \
  --reference models/jevk5-reference --device cuda
```

The queue evaluator verifies artifacts, calibrates predictions, and records
acceptance results. Its scores stay in the training service. The separate
Bittensor validator publishes weights only after explicit testnet configuration
and publication. Keep evaluation data and service credentials on the validator.

## Bittensor training competition

The closed testnet fleet trains candidates, scores their probability quality,
and turns those scores into miner weights. The current recipe starts each
candidate from the same Kev 0.8B reference for one epoch, using different seeds.
See the [evaluation contract](docs/evaluation.md) for the reward calculation.

The [first verified round](docs/testnet-round-001.md) used three miners on
**testnet subnet 579** on September 25, 2026. That is a recorded experiment;
confirm the current subnet and admission with the operator before registering.
Mainnet participation, public miner discovery, and automatic promotion of subnet
winners are not implemented. JevK5 customer training is a separate queue and
does not publish Bittensor weights.

## Setup

### Fine-tune on your own data

For the JevK5 customer workflow, follow [queue setup](docs/jevk5-queue.md) and
[automatic training](docs/automatic-training.md). Prepare separate training,
calibration, and test files using the [customer data format](docs/customer-jobs.md#prepare-data).
Approved miners can read the training data assigned to them.

For standalone Kev 0.8B, 4B, or 9B experiments, install the
[Zils fine-tuning skill](skills/zils-finetune/SKILL.md) into your coding agent:

```bash
npx skills add ooo-hq/zils@zils-finetune
```

Then ask: “Fine-tune a 4B Zils candidate on my labelled support tickets.”
The skill covers training, calibration, baseline comparisons, and optional
serving. Installation starts no training. Cloud runs require a Modal account
and an agreed compute budget. These experiments do not change the testnet
fleet's pinned 0.8B model contract.

### Repository setup

This setup prepares the **Kev 0.8B local/testnet fleet**. Use the guides above
for JevK5 miners, validators, or API deployment.

Install Git, Python 3.13, and `uv` on macOS, Linux, or WSL 2. From the directory
where you keep projects:

```bash
git clone https://github.com/ooo-hq/zils.git
cd zils
uv venv --python 3.13 .venv-kev
uv pip install --python .venv-kev/bin/python \
  -r requirements/model.txt -r requirements/rehearsal.txt
.venv-kev/bin/python -m scripts.download_models
```

This downloads the pinned Kev reference and base model. Later model work uses
the local cache. Follow the [miner guide](docs/mining.md) for hardware and
network setup; registered fleets also need the [testnet SDK](docs/testnet.md).

## Run a local fleet

After repository setup, run one validator and three miners on the same machine.
This rehearsal uses synthetic data and sends no chain transactions.

<details>
<summary>Local fleet commands</summary>

1. Generate the benchmark and bundles. Use fresh output directories:

   ```bash
   .venv-kev/bin/python -m zils.benchmark build --out .private/benchmarks/local
   .venv-kev/bin/python -m zils.fleet init --out .private/fleet-local \
     --benchmark .private/benchmarks/local --host 127.0.0.1
   ```

2. Start the validator:

   ```bash
   .venv-kev/bin/python -m zils.fleet validator \
     --config .private/fleet-local/validator/config.json --rounds 1
   ```

3. In separate terminals at the repository root, start each miner. Repeat for
   `miner-2` and `miner-3`:

   ```bash
   ZILS_PYTHON="$PWD/.venv-kev/bin/python" HF_HOME="$PWD/.cache/huggingface" \
     .private/fleet-local/miner-1/start-miner --rounds 1
   ```

Services select CUDA, Apple MPS, then CPU, and serialize model work on a shared
device. Reports appear in `.private/fleet-local/validator/state/rounds/`.
For separate machines, follow [private fleet setup](docs/mining.md#prepare-bundles-on-the-validator).

</details>

## Model provenance

Customer training and the decision API use pinned **JevK5 4B** weights and
runtime revisions. See [model setup](docs/jevk5-queue.md#install-and-create-the-reference)
and [API setup](docs/decision-api.md). The local/testnet fleet uses **Kev 0.8B**,
based on Qwen3.5-0.8B-Base; its revisions are recorded in the
[experiment methodology](docs/experiments.md#runtime-and-reference-models).

A fresh clone downloads upstream reference weights. Experimental adapters,
private datasets, wallets, and raw runs are excluded from Git. There is no
published general-purpose Zils checkpoint to download.

## Results and limits

Results belong to the task, dataset, and model named in each study.

| Study | Measured result | Scope |
| --- | --- | --- |
| [ABCD support decisions](https://zils.ai/model#support-study) | Trained JevK5: **79.2%**, unchanged JevK5: **57.8%** on the same 500 test conversations | One recipe and seed on public role-play conversations; the research adapter is not deployed |
| [Larger-data Kev experiment](docs/experiments.md#optimized-4090-training-and-a-larger-dataset) | **976/1,120 (87.14%)** correct, up from **958/1,120 (85.54%)**; Brier loss fell **9.59%**, while high-confidence mistakes rose from **18 to 23** | Experimental comparison with the previous candidate |
| [Public JevBench comparison](docs/jevbench-public.md) | Trained and unchanged Kev tied at **147/231 (63.64%)**; probability quality regressed | No official JevBench rank measured |

The [synthetic fleet benchmark](docs/benchmark.md) reuses templates for
development. Production capacity, open participation, and isolated evaluation
of hostile checkpoints are not established by these experiments. API and
workflow verification is documented separately in the
[API guide](docs/decision-api.md#verification-and-measured-scope) and
[training guide](docs/automatic-training.md#verification-and-limits).

## Development

Use Python 3.13, `uv`, Make, and Node.js 22+ with npm. From a cloned repository,
create `.venv-kev` if it does not exist, then install the test dependencies:

```bash
uv venv --python 3.13 .venv-kev
uv pip install --python .venv-kev/bin/python \
  -r requirements/dev.txt -r requirements/model.txt \
  -r requirements/rehearsal.txt -r requirements/testnet.txt -r requirements/api.txt
uv pip install --python .venv-kev/bin/python --no-deps -r requirements/jevk5-source.txt
make check
```

Skip the environment-creation command when reusing an existing environment.
Checks cover lint, formatting, Python behavior, and the static website. Model
work uses fixtures and chain RPC is simulated; the default suite needs no GPU,
model downloads, or chain transactions. See [development](docs/development.md)
for database, SDK, and optional GPU checks.

Use the `zils` Python package and `ZILS_*` settings. Regenerate standalone miner
bundles when upgrading their code.

## Repository layout

| Directory | Contents |
| --- | --- |
| `zils/` | Decision API, training coordination, evaluation, and testnet integration |
| `miner/` | Queued workers and fleet miners |
| `docs/` and `examples/` | Setup guides, experiment records, and example requests |
| `scripts/`, `requirements/`, and `tests/` | Model downloads, pinned dependencies, and checks |
| `skills/` and `website/` | Agent fine-tuning workflow and standalone research preview |

The main website is maintained in [ooo-hq/zils-web](https://github.com/ooo-hq/zils-web).
For the bundled research preview, see [website setup](website/README.md).
