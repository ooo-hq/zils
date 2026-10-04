# Checkpoint evaluation

## Contract

A local submission contains `uid`, an absolute `checkpoint` directory, and
`sha256`. Only `adapter_config.json`, `adapter_model.safetensors`, and `head.pt`
are copied into the evaluation snapshot. Hashing covers their filenames and
contents. Symlinks are refused; download adapters with `local_dir` to obtain real
files. Duplicate UIDs and identical checkpoint hashes reject the local batch.
This detects exact duplicates only; it is not plagiarism or Sybil resistance.

Each JSONL case has `id`, `family`, `state`, `question`, and `label`. Questions use
Kev's `noul`, `choice`, or `score` format. Labels are strings: `false`/`true`, a
Choice option key, or a Score level index. Each question needs at least two
outcomes. All benchmark metadata and ground-truth labels stay out of the inference
request. Model confidence fields are never treated as measured accuracy.

The scorer calculates multiclass Brier loss for every question, then averages
within each task family and across families equally. Score questions are scored
as categorical distributions in v1. Accuracy, high-confidence mistakes, and
per-question p95 inference latency are diagnostics.

```text
skill  = max(0, 1 - mean_family_brier / mean_family_uniform_brier)
weight = skill / sum(eligible_skills)
```

If no candidate beats the uniform baseline, the proposed weight map is empty.
Missing answers, invalid probabilities, changed artifacts, and failed model runs
are ineligible. Missing runtime packages or unavailable base-model caches abort
the round; they must not manufacture a bad score for a miner.

This is a starting reward rule for local experiments. Brier is a proper scoring
rule, but thresholding and relative payouts do not by themselves establish an
incentive-compatible mechanism. Uniform prediction is a floor, not a sufficient
production baseline. Benchmark refresh, copying resistance, and submission timing
need explicit rules before opening the competition.


## Timing

The runner keeps each model loaded across its evaluation batch and makes typed
decisions without text generation. `model_load_ms` records loading and device
synchronization separately. `median_ms` and `p95_ms` include request encoding,
forward inference, and synchronization; they include the first question and
exclude model loading, checkpoint transfer, and application HTTP overhead.
These timings are diagnostic and do not affect rewards in this rubric.

The measured Mac M4 Pro FP32 path took approximately **122 ms median / 180 ms
p95 per question** on the latest 40-case run. Model loading is measured separately.
This is not a Jev comparison. The competition selects downloadable weights;
an application can keep its chosen Zils checkpoint resident without routing every
decision through the miner/validator network. CUDA kernels, serving precision,
and batching need a separate, quality-checked benchmark on the target hardware.
