# Synthetic benchmark

The benchmark compares decision accuracy, probability quality, and latency using
explicit synthetic rules. It uses the [checkpoint evaluator](evaluation.md) and
the subnet's reward calculation. It is a development benchmark, not an independent
measure of general-purpose capability or resistance to benchmark gaming.

## Build a benchmark

Complete the [installation steps](../README.md#setup), then run from the repository
root. The output directory must not already exist:

```bash
.venv-kev/bin/python -m zils.benchmark build --out .private/benchmarks/local
.venv-kev/bin/python -m zils.benchmark audit --benchmark .private/benchmarks/local
```

Each build chooses a new seed and writes a manifest containing that seed, the
generator hash, file hashes, and counts. Preserve the same generated bundle for
all candidates in a comparison. A new build has the same structure but different
cases; it will not reproduce the historical scores in [experiment results](experiments.md).
If the README's fleet setup already created this directory, audit and reuse it.

## Data and separation

| File | Questions | Paired groups | Rule scenarios | Use |
| --- | ---: | ---: | ---: | --- |
| `train.jsonl` | 224 | 112 | 16 | Training cases and answer rules |
| `miner-training.jsonl` | 224 | 112 | 16 | The same cases in Kev's native training format |
| `calibration.jsonl` | 112 | 56 | 8 | Development and temperature calibration |
| `test.jsonl` | 224 | 112 | 16 | Frozen evaluation |

Related cases and perturbations of one scenario stay in one split. Policy fields,
evidence subjects, severity contexts, and routing domains are separated between
splits. Instruction templates and underlying rule families are shared, so this
measures transfer within those tasks rather than performance on new task families.

Only `miner-training.jsonl` is intended for distribution to miners. Its labels
use native boolean, choice-key, and integer values. Keep the full bundle, seed,
calibration/test cases, and prediction reports with the validator operator. Files
have owner-only permissions and `.private/` is ignored by Git; this does not
isolate processes running under the same OS account.

The public generator allows training on similar cases. A secret seed is not an
anti-gaming mechanism. An open competition needs independently authored private
scenarios and isolated evaluation.

## Answer rules and limits

| Family | Rule | Test cases |
| --- | --- | ---: |
| Policy | Age limit AND (verification OR exception) AND no exclusion; unknown facts cannot establish a condition | 64 |
| Routing | Highest-priority active issue; explicitly denied issues do not count | 64 |
| Evidence | Exact stated attributes, contradiction, missing information and explicit negation | 48 |
| Severity | Two numeric thresholds plus an overriding flag; higher severity takes precedence | 48 |

Deterministic functions generate answers; independent boundary tests check those
functions. Case metadata allows the audit to recompute labels. This is generated
ground truth, not an independent human annotation study.

Each base case has one variant: reordered Choice options or a note attempting to
override the rules. Requested answers in the notes are sampled independently of
truth. The test contains 112 clean cases, 84 note variants, and 28 option-order
variants. Noul and Score outcome order is preserved.

The 224 test questions contain only 112 pairs across 16 scenarios. Do not treat
all questions as independent observations when computing confidence intervals.
Variant slices have different task mixes; use matched pairs to study perturbations.

## Evaluate a checkpoint

Create a manifest for the reference installed during setup, then evaluate it:

```bash
.venv-kev/bin/python -m zils submit \
  --checkpoint models/reference --uid 1 > submissions-reference.json
.venv-kev/bin/python -m zils evaluate \
  --submissions submissions-reference.json \
  --cases .private/benchmarks/local/test.jsonl \
  --base-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --runner-python .venv-kev/bin/python --device cpu \
  --report .private/benchmarks/local/reference-test.json
.venv-kev/bin/python -m zils.benchmark summarize \
  --benchmark .private/benchmarks/local \
  --report .private/benchmarks/local/reference-test.json \
  --out .private/benchmarks/local/reference-summary.json
```

Select `mps` or `cuda` instead of `cpu` for supported GPUs. Use fresh report paths
for subsequent evaluations. The summary rejects reports from a different dataset
and recomputes metrics from saved probabilities, including family/variant slices,
pair agreement, and the fraction of pairs with both answers correct.

The audit checks hashes, labels, exact-prompt overlap, scenario separation,
paired-case integrity, and the native training export. It does not establish
semantic deduplication or absence of upstream training contamination.

## Calibrate a trained candidate

Use training cases to fit weights and calibration cases to choose temperature.
The [development guide](development.md#train-a-smoke-test-candidate) produces
`models/fez-probe`, whose saved temperature is 1.0. To exercise calibration with
that artifact and the generated bundle:

```bash
.venv-kev/bin/python -m zils submit \
  --checkpoint models/fez-probe --uid 2 > submissions-candidate.json
.venv-kev/bin/python -m zils evaluate \
  --submissions submissions-candidate.json \
  --cases .private/benchmarks/local/calibration.jsonl \
  --base-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --runner-python .venv-kev/bin/python --device cpu \
  --report .private/benchmarks/local/candidate-calibration.json
.venv-kev/bin/python -m zils.calibrate \
  --benchmark .private/benchmarks/local \
  --report .private/benchmarks/local/candidate-calibration.json --uid 2 \
  --checkpoint models/fez-probe --out models/fez-probe-calibrated
```

This demonstrates the procedure, not a meaningful training comparison. For a
quality experiment, train on `miner-training.jsonl` first. The fitter requires
predictions from the supplied checkpoint at temperature 1.0, verifies the
calibration split, and refuses an existing output directory. Adapter/head tensors
stay unchanged; the copied checkpoint records its new temperature and provenance.

The objective is equal-family negative log likelihood over 81 log-spaced
temperatures from 0.25 to 4, with a probability floor of 1e-9. All calibration
variants participate. Lower fitted NLL does not guarantee lower test Brier loss.
Positive temperature scaling preserves answer rankings.

Create a new submission for the calibrated checkpoint and repeat the evaluation
sequence on `test.jsonl` using fresh report paths. Compare candidates using the
same data, device, precision, runtime, and declared calibration procedure. Run
inference from saved checkpoints to measure probabilities and latency.

Repeated tuning against exposed test errors turns that test into development
data. Refresh the held-out scenarios before making a new generalization claim.
The [persistent fleet](mining.md) currently reuses its frozen benchmark each round.
