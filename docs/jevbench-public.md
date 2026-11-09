# Fez versus Kev on JevBench's public questions

> **Dated record (24 September 2026).** Kev and Fez are the earlier 0.8B line,
> before z3; neither is served, trained or rewarded by the subnet today. The page
> is kept as history.

Both frozen models ran all **231 public items** on the RTX 4090, using the
same Kev runtime, FP32 precision, and JevBench's native HTTP adapter and scorer.
All responses had valid probability distributions.

| Metric | Published Kev 0.8B | Fez candidate |
| --- | ---: | ---: |
| Correct / 231 | 147 (63.64%) | 147 (63.64%) |
| Easy / 48 | 48 | 48 |
| Original / 72 | 58 | 57 |
| Hard / 111 | 41 | 42 |
| Brier loss, lower is better | 0.451648 | 0.463608 |
| ECE, lower is better | 0.044050 | 0.113615 |
| Wrong answers at ≥90% confidence | 3 | 8 |
| Median / p95 localhost HTTP latency | 52.78 / 209.36 ms | 42.08 / 205.26 ms |

**Accuracy tied, and Fez's confidence quality regressed.** Fez corrected six
Kev mistakes and introduced six. These results do not establish a general
improvement over Kev or a reliable speed improvement from one timing pass.

For context, JevBench's stored outcomes for **Jev 1.13.0** show **200/231
correct (86.58%)** on these same public item IDs. That is a published external
measurement, not a new Jev run or a hardware-matched comparison.
[Upstream public outcomes](https://github.com/fstandhartinger/jevbench/blob/26eb72d4e0e60d8ace0adfc77a384063442561cd/results/v1.2/jevbench-v1.2-per-task.json).

## Scope and reproducibility

The [public aggregate data](data/jevbench-public-001.json) records checkpoint
hashes, temperatures, runtime versions, dataset hash, and upstream revision.
Fez is the calibrated larger-data candidate from the previous 4090 experiment;
Kev is the untouched published 0.8B revision `9a45d25e`. Both retain their
saved temperatures. Neither was trained or calibrated on JevBench.

The [pinned harness](https://github.com/fstandhartinger/jevbench/tree/26eb72d4e0e60d8ace0adfc77a384063442561cd)
provides 48 easy, 72 original, and 111 hard public items. Its withheld and sealed
sets are absent, so there is **no official rank or composite score** here.
The previous Fez result of 87.14% used a different dataset and cannot be compared
directly with this accuracy.

Each model ran once, serially, Kev then Fez, after three synthetic warmup calls.
Prefix caching, CUDA graphs, fused serving kernels, and date augmentation were
disabled. Timing includes localhost HTTP, encoding, inference, and response
serialization; it excludes loading and warmup. No hosted cost was measured.

Every public item fit the 8,192-token row limit without truncation. An exact
normalized-state audit found no matches in the recorded local training sets;
semantic overlap and upstream pretraining contamination remain unknown.

Raw responses, per-item results, frozen inputs, and verification logs are retained
as private experiment records. Only aggregate data is published. The experimental
Fez checkpoint is not distributed, so the full comparison cannot be reproduced
from this checkout alone. Checks on the GPU host and an independent local check
verified coverage, request construction, response hashes, checkpoint integrity,
and recomputed scores. No training or chain write ran during the comparison.
