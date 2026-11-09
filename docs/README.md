# Miner and validator documentation

This repository covers miners, validators, evaluation, signed protocols, and
subnet participation on Bittensor mainnet subnet 127. The models are the z3
family; code and config names keep `d3` (z3-mini is `d3-mini-4.5b-v1`).
Start with the guide matching your role.

| Task | Guide |
| --- | --- |
| Start mining | [Mining on Zils](mining.md), [registration](bittensor-registration.md), [miner terms](mining-terms.md) |
| Commit soup bodies | [Soup pool](mining.md#soup-pool), [model pools](model-pools.md) |
| Train customer jobs | [Customer-job miners](queue-miners.md), [admission](miner-admission.md), [grading](miner-grading.md) |
| Update a miner | [Releases](releases.md) |
| Serve z3 checkpoints as a miner | [Serving miners](serving-miners.md) |
| Operate a validator | [Validator setup](validators.md), [evaluation contract](evaluation.md), [serving validators](serving-validators.md) |
| Run a king pool or submit a base candidate | [King pools](mining.md#king-pools), [base candidates](base-candidates.md), [scheduled base mining](automatic-base-mining.md) |
| Run a closed fleet | [Development fleet](mining.md#closed-development-fleet), [chain fleet](testnet.md), [local customer jobs](customer-jobs.md) |
| Prepare models or training data | [d3 reference](queue-miners.md#2-install-the-stable-release), [customer data contract](customer-jobs.md#prepare-data), [version selection](version-selection.md) |
| Compare tuned open models and miner hardware tradeoffs | [Model research](model-research.md) |
| Develop and inspect evidence | [Development](development.md), [benchmark](benchmark.md), [experiments](experiments.md), [JevBench](jevbench-public.md), [recorded testnet round](testnet-round-001.md) |
| Collect fresh questions labelled by real outcomes | [Zils Outcomes](current.md) |

[Repository boundaries](repositories.md) explains where hosted services, client
SDKs, and the website live. [Roadmap](roadmap.md) lists remaining subnet work.

Paths in operator examples are relative to this repository. Keep private data,
wallets, generated bundles, and checkpoints outside Git. Historical reports
retain the former Fez names and identifiers to preserve their evidence.
