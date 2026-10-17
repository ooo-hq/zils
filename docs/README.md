# Miner and validator documentation

This repository covers workers, evaluation, signed protocols, and testnet
participation. Customer workers and fleets support d3-mini text and image jobs
(the default for new jobs since 17 October 2026), the retiring H2O text profile
(frozen jobs only) and the Imajev image profile, with one model pinned per job. Start with the guide matching your role.

| Task | Guide |
| --- | --- |
| Operate a miner | [Queued miners](queue-miners.md), [closed fleet](mining.md) |
| Operate a validator | [Validator setup](validators.md), [evaluation contract](evaluation.md) |
| Join the testnet | [Registration](bittensor-registration.md), [testnet operation](testnet.md) |
| Prepare models or training data | [H2O reference](queue-miners.md#2-install-and-verify-the-reference), [H2O model contract](h2o-queue.md), [customer data contract](customer-jobs.md), [version selection](version-selection.md) |
| Compare tuned open models and miner hardware tradeoffs | [Official Jev comparisons and B300 research](model-research.md) |
| Submit a replacement base (implemented, no round run) | [Base candidates](base-candidates.md) |
| Develop and inspect evidence | [Development](development.md), [benchmark](benchmark.md), [experiments](experiments.md), [JevBench](jevbench-public.md), [recorded testnet round](testnet-round-001.md) |

[Repository boundaries](repositories.md) explains where hosted services, client
SDKs, and the website live. [Miner grading](miner-grading.md) describes the
quality/assignment policy. [Roadmap](roadmap.md) identifies remaining subnet work.

Paths in operator examples are relative to this repository. Keep private data,
wallets, generated bundles, and checkpoints outside Git. Historical reports
retain the former Fez names and identifiers to preserve their evidence.
