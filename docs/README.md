# Miner and validator documentation

This repository covers workers, evaluation, signed protocols, and Bittensor
participation on subnet `<NETUID>`. One model is pinned per job. Start with the guide matching your role.

| Task | Guide |
| --- | --- |
| Operate a miner | [Queued miners](queue-miners.md), [closed fleet](mining.md) |
| Operate a validator | [Validator setup](validators.md), [evaluation contract](evaluation.md) |
| Join the subnet | [Registration](bittensor-registration.md) |
| Prepare models or training data | [d3 reference](queue-miners.md#2-install-and-verify-the-reference), [customer data contract](customer-jobs.md), [version selection](version-selection.md) |
| Compare tuned open models and miner hardware tradeoffs | [Official Jev comparisons and B300 research](model-research.md) |
| Submit a replacement base | [Base candidates](base-candidates.md), [compete from outside](mining.md#compete-in-the-base-tier-from-outside), [two-track payment](mining.md#two-tracks-the-base-tier-pays-first), [model pools](model-pools.md) |
| Develop and inspect evidence | [Development](development.md), [benchmark](benchmark.md), [experiments](experiments.md), [JevBench](jevbench-public.md) |

[Repository boundaries](repositories.md) explains where hosted services, client
SDKs, and the website live. [Miner grading](miner-grading.md) describes the
quality/assignment policy. [Roadmap](roadmap.md) identifies remaining subnet work.

Paths in operator examples are relative to this repository. Keep private data,
wallets, generated bundles, and checkpoints outside Git. Historical reports
retain the former Fez names and identifiers to preserve their evidence.
