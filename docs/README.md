# Documentation

Zils provides a decision API, customer model training, and a closed Bittensor
testnet training fleet. Start with the [repository README](../README.md) for an
overview, then choose the guide for your role below. Customer training uses
JevK5 4B; the testnet fleet uses Kev 0.8B and requires operator admission.

| Task | Documentation |
| --- | --- |
| Serve typed decisions and bulk jobs | [Zils decision API](decision-api.md) |
| Run a JevK5 miner | [JevK5 miner setup](queue-miners.md) |
| Register on Bittensor testnet | [Miner registration](bittensor-registration.md), [closed fleet setup](testnet.md) |
| Run a validator | [Queue and Bittensor validators](validators.md) |
| Train on authorized business data | [Customer decision jobs](customer-jobs.md) |
| Connect customer uploads to approved miners | [Supabase training queue](supabase-training.md) |
| Automatically train and activate customer adapters | [Automatic training workflow](automatic-training.md) |
| Select a first model or upgrade an active version | [Customer version selection](version-selection.md) |
| Develop and evaluate checkpoints | [Local development](development.md), [evaluation contract](evaluation.md) |
| Understand benchmark methodology | [Synthetic benchmark](benchmark.md) |
| Inspect measured results | [Experiment results](experiments.md), [public JevBench comparison](jevbench-public.md), [aggregate JSON](data/jevbench-public-001.json), [verified testnet round](testnet-round-001.md) |
| Review planned capabilities | [Roadmap](roadmap.md) |

Example paths in setup guides are relative to the repository root. Private
datasets, wallets, generated bundles, checkpoints, and raw experiment records
are excluded from Git. Reports identify where private inputs prevent exact
reproduction from a public checkout; measured hardware is included where it
affects interpretation.

Zils was formerly named Fez. Historical experiment reports and aggregate JSON
retain their original names and identifiers to preserve the evidence record.
