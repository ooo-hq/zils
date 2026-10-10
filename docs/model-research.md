# Tuned open models and the B300 study

Task-specific adapters on the open-weight JevK5 4B model outperformed
**TypeSafe's official hosted Jev 1.13.0** on our recorded support-action and
mate-in-one tests. This demonstrates that tuning an open model can make it
competitive on a defined task. It does not establish a general model ranking or
guarantee that another customer dataset will improve.

The October 10, 2026 B300 follow-up explored the practical next question: which
open models provide useful accuracy with lower latency or memory use for miners?
That study compared locally run models; it did not rerun the hosted Jev API test.

## Direct comparisons with official TypeSafe Jev

The tuned models below use separate task-specific adapters on JevK5 4B. They are
different from both the unchanged JevK5 weights and TypeSafe's hosted Jev API.

| Recorded task | Cases | Tuned open model | Official Jev 1.13.0 | Accuracy gain, paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Next support action, ABCD | 500 conversations | 396/500 (79.2%) | 353/500 (70.6%) | +8.6 percentage points [+4.8, +12.2] |
| Mate in one, Lichess puzzles | 512 positions | 260/512 (50.78%) | 223/512 (43.55%) | +7.23 percentage points [+2.15, +12.30] |

The support adapter trained on 1,024 separate conversations and was selected on
development data. Both systems received the same test inputs, instructions, and
choices. The hosted comparison reused verified Zils predictions, sent no test
labels, and made no post-test model or prompt adjustments. The prompt had been
developed for Zils and was not separately optimized for Jev.

The support result is an accuracy gain, not an improvement in every metric.
Official Jev had lower calibration error and higher accuracy among its smaller
set of answers assigned at least 90% probability. Different confidence coverage
means those subsets are not equivalent. Local GPU timing and hosted API latency
are also not directly comparable.

The chess adapter trained on 2,048 positions. Its test concerns immediate
checkmate selection, not full-game strength. Deterministic chess-rules software
scored 100%, above both learned systems. The original reports preserve training
splits, uncertainty, API retries, and model identities:

- [Support comparison and methodology](https://zils.ai/model#jev-comparison)
- [Support aggregate record](https://zils.ai/model/abcd-002-jev.json)
- [Chess comparison and methodology](https://zils.ai/model/chess-study)
- [Chess aggregate record](https://zils.ai/model/chess-001.json)

## B300: models practical for miners

All final ABCD decisions retain the full input context and all 30 actions. Each
cohort contains 500 conversations, with one decision per conversation. The two
cohorts were **retrospective for the H2O and 2B follow-ups**: earlier results had
already been inspected. Checkpoint selection still used development data only.

H2O 4B and JevK5 2B used the same ordered 4,096 examples and native 15/16-option
training views as the JevK5 4B comparator. Each new model had two epoch candidates.
Development accuracy, then lower Brier score, then fewer exposures selected H2O
epoch 1 and JevK5 2B epoch 2. The selected 4B reference used epoch 2.

| Selected adapter | Test accuracy | Historical accuracy | Test Brier ↓ | B300 test median | Test peak CUDA allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2O 4B, 4,096 examples, epoch 1 | 415/500 (83.0%) | 418/500 (83.6%) | 0.2883 | 59.7 ms | 8.82 GiB |
| JevK5 4B, 4,096 examples, epoch 2 | 412/500 (82.4%) | 424/500 (84.8%) | 0.3115 | 178.4 ms | 8.24 GiB |
| JevK5 2B, 4,096 examples, epoch 2 | 404/500 (80.8%) | 422/500 (84.4%) | 0.3370 | 135.9 ms | 3.76 GiB |

H2O had **2.99× lower median latency** than the selected JevK5 4B adapter on the
test cohort. Its accuracy difference was +0.6 points [−1.2, +2.4] on that cohort
and −1.2 [−3.6, +1.4] on the historical cohort. The accuracy evidence is
inconclusive; the speed result makes H2O a useful candidate for further miner
tests.

JevK5 2B used **54% less peak CUDA allocation** than the 4B adapter. Its accuracy
differences were −1.6 points [−4.4, +1.2] on the test cohort and −0.4 [−3.2, +2.2]
on the historical cohort. That supports testing it on smaller GPUs, without
proving equivalent accuracy or guaranteeing that it fits a 4 GB card.

An interval spanning zero demonstrates neither superiority nor equivalence.
Brier is the sum of squared errors across the 30 predicted action probabilities;
lower is better. The downloadable reports also include stock models, the earlier
1,024-example H2O adapter, development scores, and all paired comparisons.

## Larger models and full-weight training

The original test cohort was fresh in identifiable local records for the
initial 4B/9B matrix. It became retrospective when the later H2O/2B trials were
chosen. The historical cohort was already reused.

| Original training comparison | Test accuracy | Historical accuracy |
| --- | ---: | ---: |
| JevK5 4B stock | 52.6% | 57.2% |
| JevK5 4B full weights, 1,024 examples × 1 epoch | 65.8% | 65.4% |
| JevK5 4B adapter, 1,024 examples × 1 epoch | 75.0% | 79.6% |
| Selected JevK5 4B adapter, 4,096 examples × 2 epochs | 82.4% | 84.8% |
| Selected JevK5 9B adapter, 4,096 examples × 1 epoch | 80.8% | 83.6% |

Selected 9B minus selected 4B was −1.6 points [−4.2, +0.8] on the original test.
The fixed full-weight 4B recipe trailed its matched 1,024-example adapter by 9.2
points [−13.6, −5.0]. There was no clear benefit from the selected 9B model, and
the full-weight recipe performed worse than matched LoRA. These are results for
the tested recipes, not proof that larger models or full-weight training cannot
help.

The 27B teacher trial was canceled before any complete development predictions.
Its gate was not evaluated; no 27B accuracy or distillation benefit is claimed.

## Hardware and interpretation limits

1. **B300 was temporary research hardware.** New latency results include prompt
   preparation, tokenization, and synchronized single-case inference; they
   exclude loading, warmup, transport, and queues. These checkpoints have not
   received new RTX 4090, quantization, concurrency, or production validation.
2. **CUDA allocation is not required card capacity.** H2O resets its inference
   peak after loading and warmup; Jev peaks include loading allocations. The
   scopes differ, and allocation excludes some driver and process overhead.
3. **Earlier 4090 evidence remains relevant.** An October 9 ABCD run measured the
   older H2O adapter at 257 ms versus 874 ms for Jev. It used an older runtime and
   H2O temperature 0.8. It supports feasibility and the direction of the speed
   advantage, not exact performance for these new checkpoints. Earlier Bitcast,
   reply-reserve, and chess comparisons were mixed.
4. **The experiments do not isolate parameter count.** There is one training
   seed (553), fixed recipes, and different native prompts, decision protocols,
   calibration, and update counts. JevK5 2B is v0.2; the 4B comparator is v0.3.
   New H2O training uses temperature 0.75; the older H2O adapter used 0.8. Public
   pretraining overlap is unknown. Paired intervals use 2,000 conversation
   resamples, measure cohort rather than training-seed uncertainty, and are not
   adjusted for multiple comparisons.
5. **Numerical and recovery exceptions are preserved.** The 9B/4,096 run exceeded
   its 0.03 probability-error bound on its longest input (fast BF16 vs FP32:
   0.04114; reference BF16 vs FP32: 0.05380). Final actions agreed; an identical
   16-option replay reduced fast-vs-FP32 error to 0.01468. A recorded exception
   permitted the benchmark without claiming FP32 equivalence. H2O saved both
   checkpoints before a CUDA-memory release assertion failed; unchanged weights
   were evaluated in fresh processes after fixing a layer-name validation
   mismatch. H2O training peak memory and uninterrupted runtime are unavailable.

The temporary GPU was deleted after verified retrieval. Cumulative estimated
compute was **$37.95**. This is a runtime estimate, not a provider invoice or a
price for customer training. Research results do not promote an adapter or
replace per-job acceptance tests.

## Reports and reproducibility

- [B300 study](https://zils.ai/model/b300-study)
- [H2O / 2B full report](https://zils.ai/model/b300-2026-10-10/miner-report.md)
  and [aggregate JSON](https://zils.ai/model/b300-2026-10-10/miner-results.json)
- [Original 4B / 9B full report](https://zils.ai/model/b300-2026-10-10/report.md)
  and [aggregate JSON](https://zils.ai/model/b300-2026-10-10/results.json)
- [Publication manifest and hashes](https://zils.ai/model/b300-2026-10-10/manifest.json)

The reports pin model revisions and preserve aggregate metrics, development
selection, and source digests. B300 raw conversations, labels, per-case
predictions, and trained weights are not distributed. The public aggregates do
not reproduce inference from a fresh checkout.
