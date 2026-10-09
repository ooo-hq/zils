# Experiment results

These are development measurements, not release certifications. No experiment
below promoted a Fez release or published chain weights. Different datasets have
different difficulty and aggregation rules; compare models within a result table.
The [flight-delay adapter experiment](flight-delay-001.md) compares an unchanged
JevK5 base, a trained adapter, and historical rates on public BTS records. The
adapter improved over the base but did not meet the requirement to beat both
references. The [public JevBench report](jevbench-public.md) covers a separate
external decision benchmark.

The historical experiments below were recorded on September 24, 2026 (US Eastern). Historical private
corpora, raw reports, and experimental checkpoint files are not distributed in
this repository, so a fresh clone cannot exactly reproduce those runs. Public
examples and the benchmark generator support new experiments. The JevBench report
publishes [aggregate data and provenance](data/jevbench-public-001.json).

## Runtime and reference models

Runs use Kev source `30c619b0527501cfdd448cb6eb9887e2af454603` and the
Qwen3.5-0.8B base revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
The original public Kev checkpoint revision is
`54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8`; later comparisons also use
`9a45d25eb2ab761841196625383fa1dff0e56c1e`. These are different reference releases.

Unless stated otherwise, inference uses Torch FP32. Early GPU evaluations used
an Apple M4 Pro with 24 GiB unified memory; later training and inference also used
an NVIDIA RTX 4090. Hardware is stated per comparison. Subnet evaluator timings
include encoding and synchronized inference but exclude loading and HTTP serving.
JevBench measures localhost HTTP and is reported separately.

## Public diagnostic cases

On 40 handwritten public cases, two copies of the original Kev checkpoint
changed only the saved temperature. The first runs used CPU inference.

| Temperature | Correct / 40 | Family-macro Brier ↓ | Wrong at ≥90% confidence |
| --- | ---: | ---: | ---: |
| Published: 2.4061 | 28 | 0.462489 | 0 |
| 1.0 | 28 | 0.455276 | 3 |

The 32 clean cases favored the published temperature on Brier (0.438190 versus
0.442291); eight additional cases reused facts. Temperature changed probability
scores without learning or correcting answers. One of four option-order pairs
changed a correct answer to a wrong one; four instruction-injection pairs changed
no answers. These small samples do not establish robustness.

A follow-up expanded the options on eight severity cases after errors had been
inspected. Accuracy rose from 2/8 to 4/8; Brier fell from 0.849818 to 0.711699.
There were no ≥90%-confidence errors, but performance still failed the uniform
Brier floor. This exploratory retest does not establish unseen-case improvement.

The cases are [diagnostics.jsonl](../examples/diagnostics.jsonl) and
[severity-explicit.jsonl](../examples/severity-explicit.jsonl). Their SHA-256
hashes were respectively `b0d2ada1c44acd0feadede1f2c7335ab243a013754c0bb54cde245bb6112ebbc`
and `f7b763270d20b7c8133a4bb77b5ce4ed81debeb82f4691b791aa454c39af12a4`.

## Frozen benchmark baseline

The first synthetic bundle contained 224 training, 112 calibration, and 224 test
questions, with 112 test pairs across 16 scenarios. Files were frozen before
baseline inference. A training-export label correction preserved all case files.
The [benchmark guide](benchmark.md) describes the shared templates and pairing.

The original published checkpoint scored 146/224 (65.18%), family-macro Brier
0.458716 versus uniform 0.645833, and zero ≥90%-confidence mistakes. Clean-case
accuracy was 80/112; 58/112 pairs had both answers correct. Family totals were
policy 42/64, routing 44/64, evidence 21/48, and severity 39/48.

Median/p95 inference was 173.0/177.0 ms on M4 Pro; model loading was 1.65 seconds.
These are workload-specific observations, not a Jev comparison or serving SLA.

## First trained candidate: fez-candidate-001

The original checkpoint trained on 224 examples for one epoch and 56 optimizer
steps: learning rate 2e-5, batch 1, accumulation 4, seed 553, MPS FP32. Choice
permutation remained enabled; distractor insertion was disabled. Training took
104.28 seconds including saving and excluding initial loading. No examples were
dropped or truncated. Peak process RSS was 4.87 GiB and sampled device allocation
3.46 GiB; these overlap in unified memory and must not be added.

Both models used the same 112-case calibration procedure, selecting temperatures
2.0705298477 for the reference and 1.0717734625 for the candidate. The candidate's
calibration NLL fit slightly worsened calibration Brier (0.269228 to 0.269503).
The 224-case test was then evaluated without further adjustment.

| Measure | Calibrated reference | Fez candidate 001 |
| --- | ---: | ---: |
| Correct / 224 | 146 (65.18%) | 183 (81.70%) |
| Clean cases correct / 112 | 80 | 91 |
| Family-macro Brier ↓ | 0.460147 | 0.294902 |
| Wrong at ≥90% confidence | 0 | 7 |
| Pair agreement / 112 | 81 | 106 |
| Both answers in a pair correct / 112 | 58 | 89 |
| Median / p95 ms, M4 Pro | 172.01 / 174.24 | 172.05 / 174.38 |

Brier improved 35.91%, but all seven confident errors were in routing. The
candidate's family counts were policy 51/64, routing 52/64, evidence 36/48, and
severity 44/48. This demonstrates improvement on related synthetic tasks, not
broad real-world quality. Checkpoint hash:
`dca402cef751ecafbc74e816b6de044a84eae736f52a334ff5ee2fc946d1e22d`.

## Published Kev 0.8B update comparison

The original and newer published revisions ran on the same 224 development
questions with their published temperatures and M4 Pro FP32 runtime.

| Measure | Original release | Newer release |
| --- | ---: | ---: |
| Correct / 224 | 146 (65.18%) | 154 (68.75%) |
| Family-macro Brier ↓ | 0.458716 | 0.425394 |
| Wrong at ≥90% confidence | 0 | 1 |
| Median / p95 ms | 176.66 / 179.29 | 176.96 / 179.33 |

Evidence and routing improved; severity fell from 39/48 to 33/48 and clean-case
accuracy from 80/112 to 75/112. The new release corrected 26 errors and introduced
18. No training or local calibration was performed for this comparison.

## Matched Fez fine-tuning from both Kev releases

Both starting releases used the candidate-001 recipe and calibration procedure.
Training took 98.74/98.73 seconds on M4 Pro, including saving and excluding loading.
Both processed 30,397 forward tokens without truncation. Calibration selected
1.0717734625 for the old-base candidate and 1.4640856959 for the new-base candidate.

| Measure | Fez from original Kev | Fez from newer Kev |
| --- | ---: | ---: |
| Correct / 224 | 183 (81.70%) | 186 (83.04%) |
| Family-macro Brier ↓ | 0.294902 | 0.282746 |
| Wrong at ≥90% confidence | 7 | 6 |
| Clean cases correct / 112 | 91 | 92 |
| Median / p95 ms | 176.20 / 179.29 | 167.90 / 179.50 |

The newer-base candidate fixed six errors and introduced three. Severity and
policy improved; routing regressed. One seed and reused synthetic cases do not
establish a robust winner or a speedup.

## Fresh authored scenarios for the two trained candidates

Both calibrated candidates ran without further fitting on 64 new synthetic
scenarios, 16 per task family. Cases were checked for exact prompt/state overlap
against 608 earlier cases and frozen before inference. Labels and rationales were
assistant-authored and reviewed by the same author, not independently adjudicated.

| Measure | Fez from original Kev | Fez from newer Kev |
| --- | ---: | ---: |
| Correct / 64 | 56 (87.50%) | 56 (87.50%) |
| Family-macro Brier ↓ | 0.232845 | 0.194603 |
| Wrong at ≥90% confidence | 5 | 2 |
| Median / p95 ms, M4 Pro | 120.01 / 131.75 | 119.66 / 127.60 |

Both missed seven of the same cases; each corrected one other case. The newer-base
candidate had lower Brier in all four families. This is a small, related-task
comparison with one training seed, not a general accuracy or speed claim. These
cases became exposed development data after the run.

## Larger RTX 4090 Fez fine-tune

Three candidates trained from the newer-base Fez checkpoint on 2,878 labeled
records, using BF16 forward passes with FP32 master weights, batch 4 and
accumulation 4. Training-only probes selected the batch size. The runs took
645.34 seconds total; the selected two-epoch recipe peaked at 17.30 GiB.
Selection used macro-source Brier on a separate development partition after
calibration-only temperature fitting, before opening the 1,208-question test.
The selected model scored 1,032/1,208 (85.43%) versus 1,005/1,208 (83.20%) for an
equally recalibrated starting Fez model. Macro-source Brier fell from 0.314314 to
0.292107 (7.07%); ≥90%-confidence mistakes fell from 13 to 10.

Most gains came from generated policy decisions; news and BoolQ each lost one
correct answer. Paired 95% bootstrap intervals excluded zero for aggregate Brier
and accuracy within this corpus and one training seed. Public upstream overlap
and shared policy templates remain limitations.

On the older reused benchmark, accuracy rose from 186/224 to 195/224 and confident
mistakes fell from six to zero. Repeated RTX 4090 inference medians were 42.34 ms
for the candidate and 42.71 ms for the control on 128 training inputs, excluding
loading, encoding, and networking. This supports retained speed, not a speedup.

## Direct published Kev comparison after the 4090 sweep

The published Kev revision `9a45d25e` then ran on the same 1,208-question test:
1,008 correct (83.44%), versus selected Fez's 1,032 (85.43%). The previous 83.20%
control was an older Fez fine-tune, not published Kev. No new training or selection
occurred for this comparison.

Matched calibration produced macro-source Brier 0.312965 for Kev and 0.292107 for
Fez, a 6.66% reduction. Confident mistakes tied at ten. The paired Brier interval
excluded zero; the equal-source accuracy interval included zero. All net additional
correct answers came from generated policy decisions. This is task-specific
improvement, not parity with Kev across its full benchmark suite.

## Optimized 4090 training and a larger dataset

An FLA 0.5.2 training-kernel overlay removed the gated-delta-attention fallback.
Forward and gradient checks agreed within 0.48% relative L2 error on tested shapes.
Matched short-workload throughput improved 35.2% at batch 4 and 69.4% at batch 8;
longer-input measurements favored batch 4. Oversized attempts were stopped, and
an allocator cap limited GPU memory use.

The recipe was fixed before evaluation: one epoch, learning rate 2e-5, Brier
weight 0.25, effective batch 16, seed 553. Sources were BoolQ, AG News, MNLI,
SST5, and generated policy decisions. Calibration used 560 fresh questions;
normalized text/state overlap checks and grouped synthetic siblings separated
the new holdout from earlier local splits.

Training processed 21,694 records (18,816 fresh plus 2,878 replay), or 2.34 million
forward tokens, in 865.10 seconds. Peak allocation was 13.52 GiB; median GPU
utilization was 44%, median power 187.76 W, peak power 256.43 W, and maximum
temperature 59°C. This configuration did not saturate the RTX 4090.

| Fresh 1,120-question test | Previous Fez | Larger-data Fez |
| --- | ---: | ---: |
| Correct | 958 (85.54%) | 976 (87.14%) |
| Equal-source Brier ↓ | 0.270160 | 0.244245 |
| Wrong at ≥90% confidence | 18 | 23 |

Both used the same fresh calibration method. All five sources gained correct
answers; paired grouped 95% intervals excluded zero for macro Brier and accuracy.
Brier improved 9.59%, while high-confidence mistakes increased. Public upstream
contamination and shared templates remain limitations. This was not a new Kev
comparison. The candidate passed a 32-case validator compatibility check and
remains experimental; the test is now exposed.

## External JevBench public comparison

Frozen larger-data Fez and published Kev 0.8B each scored **147/231 (63.64%)** on
JevBench's released public items using the same RTX 4090 and FP32 runtime. Fez
corrected six errors and introduced six. Brier and ECE worsened; confident errors
increased from three to eight. Saved temperatures were retained, with no
JevBench-directed fitting or training.

The [comparison report](jevbench-public.md) provides the full table, timing scope,
public-only limitation, and published aggregate data. There is no official rank.

## Training and network integration checks

These checks establish the local protocol and hardware path, not general model
quality. They used the repeated synthetic development sets.

| Check | Hardware and workload | Observed result |
| --- | --- | --- |
| Training smoke test | M4 Pro, four examples, FP32, batch 1, one epoch | Four steps in 7.84 s excluding load/save; all adapter/head tensors changed and stayed finite; the six-case smoke score tied the original baseline |
| Two-miner submission rehearsal | M4 Pro, 40 diagnostic cases | Reference 28/40, smoke candidate 29/40; Brier 0.462489/0.456368; signed transfer, hash verification, evaluation and proposed weights completed |
| Three-miner training fleet | M4 Pro, sequential device access, 224 test cases | Complete round in 464.76 s; candidate accuracies 186/224, 179/224, 186/224; Brier 0.293305, 0.287240, 0.254880; confident errors 4, 3, 8 |
| Cross-host Apple Silicon miner | M4 with 16 GiB training; M4 Pro validating | Round in 280.13 s; 178/224 correct, Brier 0.321996, four confident errors; validator median/p95 173.85/183.19 ms |
| Windows/WSL NVIDIA miner | RTX 4090 training; M4 Pro validating | 56 steps on 224 examples in 38.75 s; peak GPU allocation 4.18 GiB; 183/224 correct, Brier 0.286728, six confident errors; validator median 167.39 ms |

Timing scope differs between rows. Cross-host inference times belong to the
validator, not the training GPU. Single-participant weight 1.0 is normalization,
not evidence of winning a competition. Signed announcements, transferred hashes,
input integrity, calibration, recomputed scores and returned results were checked.
The three-miner weights were 31.99%, 32.54%, and 35.47%, reflecting probability
quality rather than accuracy alone. No chain publication was exercised.
