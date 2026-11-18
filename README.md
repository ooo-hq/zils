<p align="center">
  <a href="https://zils.ai">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/assets/zils-wordmark-periwinkle.svg">
      <img alt="Zils" src=".github/assets/zils-wordmark-purple.svg" width="200">
    </picture>
  </a>
</p>

<p align="center">
  <strong>Miner and validator software for Zils decision-model training.</strong><br>
  Miners train candidate models. Validators verify, calibrate and score them<br>
  on Bittensor mainnet subnet 127.
</p>

<p align="center">
  <a href="https://github.com/ooo-hq/zils/actions/workflows/checks.yml"><img alt="Checks" src="https://github.com/ooo-hq/zils/actions/workflows/checks.yml/badge.svg?branch=main"></a>
  <a href="docs/mining.md"><img alt="Bittensor subnet 127" src="https://img.shields.io/badge/Bittensor-subnet%20127-4b44c8"></a>
  <a href="https://huggingface.co/zils-ai"><img alt="Models on Hugging Face: zils-ai" src="https://img.shields.io/badge/Hugging%20Face-zils--ai-8f88ff"></a>
  <a href="docs/development.md"><img alt="Python 3.13" src="https://img.shields.io/badge/python-3.13-ff7a2f"></a>
</p>

<p align="center">
  <a href="https://zils.ai">Website</a> ·
  <a href="https://docs.zils.ai">Docs</a> ·
  <a href="docs/mining.md">Mining guide</a> ·
  <a href="docs/validators.md">Validator guide</a> ·
  <a href="https://zils.ai/model">Models</a> ·
  <a href="https://discord.gg/a9PJ9wchGK">Discord</a> ·
  <a href="https://github.com/ooo-hq/zils/issues">Issues</a>
</p>

This repository holds the miner and validator workers, their shared model and
protocol contracts, and the Bittensor subnet integration.

| Role | Guide |
|---|---|
| Mine on mainnet subnet 127 | [Mining on Zils](docs/mining.md), starting with [registration](docs/bittensor-registration.md) |
| Commit soup bodies | [Soup pool](docs/mining.md#soup-pool) |
| Train customer jobs on NVIDIA hardware | [Customer-job miners](docs/queue-miners.md) |
| Run a validator | [Validator setup](docs/validators.md) |
| Develop or evaluate the subnet | [Local development](docs/development.md), [evaluation contract](docs/evaluation.md) |

**Contents:** [How miners earn](#how-miners-earn) ·
[Quick start](#quick-start) · [Miner setup](#miner-setup) ·
[Validator setup](#validator-setup) ·
[Repository setup](#repository-setup) · [Development](#development) ·
[Results and limits](#results-and-limits) ·
[Repository scope](#repository-scope)

## How miners earn

Mainnet subnet 127 has been live since 3 November 2026. Registering a hotkey on
127 is the admission; there is no application or operator approval. Under the
launch configuration (`zils/data/base-pools-launch.json`) miners earn two ways,
paid from one weight vector:

- **Soup pools (80%).** Four pools, one per z3 size: z3-lite 0.8B and z3-mini
  4.5B are open, z3-flash 8.4B and z3 26B open later and burn their share until
  then. A miner uploads a fine-tune to Hugging Face and commits it on chain with
  `zils-miner commit-body`. Each round, validators average the best bodies into
  a soup and pay its five seats and one crown.
- **Customer jobs (20%, the adapter track).** A registered miner claims customer
  training jobs from the hosted queue with `python -m miner.queue` and is paid
  from paid API use of the open-tier adapters it delivered.

Anything unearned burns to the subnet owner's uid.

## Quick start

```sh
# Install the stable release and the zils-miner command
curl -fsSL https://zils.ai/install.sh | sh

# Save the network once; later chain commands use it
zils-miner setup --network finney --netuid 127
```

`zils-miner` never assumes mainnet, so run `setup` once before any chain
command. Then follow [mining](docs/mining.md#install) for soup bodies or
[customer-job miners](docs/queue-miners.md) for queue work.

## Miner setup

Soup miners follow [mining](docs/mining.md#install). Customer-job miners follow
[customer-job miners](docs/queue-miners.md) to install the stable release, qualify
the host and write a private hotkey configuration. Then, from the install:

```sh
export ZILS_D3_RUNTIME_PYTHON="$PWD/.venv-d3/bin/python"
.venv-kev/bin/python -m miner.queue \
  --config .private/queue-miner.json --state .private/queue-miner \
  --reference models/d3-reference --device cuda
```

z3 training requires CUDA. The worker uses outbound HTTPS and receives only
training data and signed upload URLs. Keep calibration/test data and operator
service credentials off miner accounts. For a closed development fleet, run
`./start-miner --rounds 1` from its assigned bundle after completing the
[fleet guide](docs/mining.md#closed-development-fleet).

## Validator setup

Follow [validator setup](docs/validators.md) to configure the correct workflow.
A configured Bittensor validator evaluates one round with:

```sh
.venv-kev/bin/python -m zils.fleet validator \
  --config .private/testnet-fleet/validator/config.json --rounds 1
```

This previews weights without publishing. Chain submission remains a separate,
explicit step. Validators check model hashes, allowed tensors, calibration,
held-out quality, and registered identities. A valid training run can produce
`no_qualifying_model`; miners are not guaranteed acceptance.

The hosted queue processor lives in zils-platform and calls the same evaluation
code. Its deployment and database access are separate from miner installation.

## Repository setup

Install Git, Python 3.13, and `uv` on macOS, Linux, or WSL 2:

```sh
git clone https://github.com/ooo-hq/zils.git
cd zils
uv venv --python 3.13 .venv-kev
uv pip install --python .venv-kev/bin/python \
  -r requirements/model.txt -r requirements/rehearsal.txt
.venv-kev/bin/python -m scripts.download_models
```

`scripts.download_models` fetches the small Kev 0.8B reference that the CPU
examples in [development](docs/development.md) use. For z3, follow
[customer-job miner setup](docs/queue-miners.md#2-install-the-stable-release).
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
needed for this repository's checks. [CI](.github/workflows/checks.yml) runs
them on every pull request.

## Results and limits

The [first verified testnet round](docs/testnet-round-001.md) (25 September 2026)
is recorded evidence, not a statement of current subnet admission or uptime.
[Experiment reports](docs/experiments.md) and the
[public JevBench comparison](docs/jevbench-public.md) preserve results,
including regressions. Synthetic fixture tests establish behavior, not model
quality, hostile-checkpoint isolation, or production capacity.

See [documentation](docs/README.md), [open milestones](docs/roadmap.md), and the
[public model page](https://zils.ai/model).

## Repository scope

| Repository | Owns |
|---|---|
| **zils** (this repository) | Miners, validators, training, evaluation, model integrity, signed protocols, chain weights |
| [zils-platform](https://github.com/ooo-hq/zils-platform) | Hosted APIs, account/billing logic, storage, database migrations, queue orchestration, serving |
| [zils-sdk](https://github.com/ooo-hq/zils-sdk) | Python/TypeScript clients, CLI, MCP |
| [zils-web](https://github.com/ooo-hq/zils-web) | Website, playground, customer dashboard |
| [zils-docs](https://github.com/ooo-hq/zils-docs) | Public documentation site (`docs.zils.ai`) |

Platform and SDK repositories may require organization access. Miner setup does
not require either repository, a payment SDK, or a Supabase service credential.
The platform uses this repository's evaluation code as a dependency. See
[repository boundaries](docs/repositories.md) for the split and migration details.

## Community and support

- [Discord](https://discord.gg/a9PJ9wchGK): questions, ideas and announcements
- [GitHub issues](https://github.com/ooo-hq/zils/issues): bugs in the miner, validator and evaluation code
- [Contact form](https://zils.ai/contact): accounts, billing and anything private
- [X (@zils_ai)](https://x.com/zils_ai): news
