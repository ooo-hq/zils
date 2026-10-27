<h1 align="center">
  <a href="https://zils.ai">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/assets/zils-logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="docs/assets/zils-logo-light.svg">
      <img alt="Zils" src="docs/assets/zils-logo-light.svg" width="196">
    </picture>
  </a>
</h1>

**Miner and validator software for Zils decision-model training.**

Miners train candidate models. Validators verify their artifacts, calibrate
probabilities, evaluate held-out examples, and compute miner scores. This
repository contains those workers, their shared model and protocol contracts,
and the guarded Bittensor chain integration.

## Start here

| Role | Guide |
| --- | --- |
| Run a d3 queue miner on NVIDIA hardware | [Queued miners](docs/queue-miners.md) |
| Compete in the base tier on subnet `<NETUID>` | [Registration](docs/bittensor-registration.md), then [base tier from outside](docs/mining.md#compete-in-the-base-tier-from-outside) |
| Run a validator | [Validator setup](docs/validators.md) |
| Develop or evaluate the subnet | [Local development](docs/development.md), [evaluation contract](docs/evaluation.md) |

d3 bases serve text and image jobs; a d3 queue miner downloads a job's frozen
training photos itself. Kev 0.8B jobs remain supported.
Each fleet pins one job and model. Text and image scores are not combined into one
reward vector.

Two tracks pay from one weight vector. In the **base tier**, a miner
registered on Bittensor mainnet subnet `<NETUID>` uploads a full-weight d3 candidate to Hugging Face
and commits it on chain with `zils commit-candidate`; registration is the admission.
Start with `curl -fsSL https://zils.ai/install.sh | sh`, then `zils doctor` names each next step. Only kings are paid: the last five distinct registered miners
whose candidates were promoted split the base share, and an unearned share is
burned ([base candidates](docs/base-candidates.md),
[two-track payment](docs/mining.md#two-tracks-the-base-tier-pays-first)).
Each pool may set a minimum spacing between one hotkey's submissions.
**Adapter rounds** pay the rest. Customer jobs in the hosted queue go to approved
operators.

## Repository scope

| Repository | Owns |
| --- | --- |
| **zils** — this repository | Miners, validators, training, evaluation, model integrity, signed protocols, chain weights |
| [zils-platform](https://github.com/ooo-hq/zils-platform) | Hosted APIs, account/billing logic, storage, database migrations, queue orchestration, serving |
| [zils-sdk](https://github.com/ooo-hq/zils-sdk) | Python/TypeScript clients, CLI, MCP |
| [zils-web](https://github.com/ooo-hq/zils-web) | Website, playground, customer dashboard |

Platform and SDK repositories may require organization access. Miner setup does
not require either repository, a payment SDK, or a Supabase service credential.
The platform uses this repository's evaluation code as a dependency. See
[repository boundaries](docs/repositories.md) for the split and migration details.

## Miner setup

Use [queued miner setup](docs/queue-miners.md) to create a verified d3 reference
and private hotkey configuration. Then, from this repository:

```sh
export ZILS_D3_RUNTIME_PYTHON="$PWD/.venv-d3/bin/python"
.venv-kev/bin/python -m miner.queue \
  --config .private/queue-miner.json --state .private/queue-miner \
  --reference models/d3-reference --device cuda
```

d3 requires CUDA. The miner uses
outbound HTTPS and receives only approved training data and signed upload URLs.
Keep calibration/test data and operator service credentials off miner accounts.
For a closed chain fleet, run `./start-miner --rounds 1` from its assigned
bundle after completing the [fleet guide](docs/mining.md).

## Validator setup

Follow [validator setup](docs/validators.md) to configure the correct workflow.
A configured Bittensor validator evaluates one round with:

```sh
.venv-kev/bin/python -m zils.fleet validator \
  --config .private/chain-fleet/validator/config.json --rounds 1
```

This previews weights without publishing. Chain submission remains a separate,
explicit step. Validators check model hashes, allowed tensors, calibration,
held-out quality, and registered identities. A valid training run can produce
`no_qualifying_model`; miners are not guaranteed acceptance.

The hosted queue processor lives in zils-platform and calls the same evaluation
code. Its deployment and database access are separate from miner installation.

## Setup

### Repository setup

For the Kev local fleet, install Git, Python 3.13, and `uv` on macOS,
Linux, or WSL 2:

```sh
git clone https://github.com/ooo-hq/zils.git
cd zils
uv venv --python 3.13 .venv-kev
uv pip install --python .venv-kev/bin/python \
  -r requirements/model.txt -r requirements/rehearsal.txt
.venv-kev/bin/python -m scripts.download_models
```

For d3 fleets, follow the [model-specific fleet setup](docs/mining.md).
References are downloaded from pinned upstream revisions. Private
checkpoints, wallets, datasets, and generated bundles are excluded from Git.

## Development

Install the [development dependencies](docs/development.md), then run:

```sh
make check
```

Checks exercise workers, signed submissions, evaluation, and chain behavior
with local fixtures. They need no live chain or hosted account. Optional GPU
checks are explicitly gated. Node.js and hosted database services are not
needed for this repository's checks.

## Results and limits

[Experiment reports](docs/experiments.md)
and the [public JevBench comparison](docs/jevbench-public.md) preserve results,
including regressions. Synthetic fixture tests establish behavior, not model
quality, hostile-checkpoint isolation, or production capacity.

See [documentation](docs/README.md), [open milestones](docs/roadmap.md), and the
[public model page](https://zils.ai/model).
