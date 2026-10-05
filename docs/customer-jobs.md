# Customer decision jobs

Zils can run a customer dataset through its existing approved-miner competition:
train candidates, calibrate each candidate and the starting checkpoint, compare
their test results, and export a qualifying candidate. This is an experimental
local workflow, not a hosted customer service or a published model release.

This page describes the local bundle workflow, where one fleet configuration
pins one job. For a shared pool that claims different customer jobs without
regenerating bundles, use the [Supabase training queue](supabase-training.md).
It adds private uploads and approved-miner assignments; cross-job chain rewards
remain unimplemented. The existing synthetic fleet remains supported.

## Prepare data

Complete the Python 3.13 [repository setup](../README.md#repository-setup), including
the model download. Work from the repository root on macOS, Linux, or WSL 2.
Use only data authorized for training and for export to the configured miners.
The export flag records an operator decision; it does not provide encryption,
confidential compute, legal authorization, or isolation from miner operators.

Create three JSONL files under ignored `.private/` storage: `train.jsonl`,
`calibration.jsonl`, and `test.jsonl`. Each line uses the existing decision-case
format plus a required source `group_id`:

```json
{"id":"ticket-001","group_id":"conversation-001","family":"support-routing","state":{"text":"Please send my invoice."},"question":{"type":"choice","criteria":{"billing":"Billing and invoices","technical":"Technical support"}},"label":"billing"}
```

Split related conversations, documents, or source records together before
creating these files. Case IDs must be globally unique; groups and exact prompts
cannot cross splits. Calibration and test must have the same task families,
all present in training. These checks cannot identify semantic duplicates or
incorrectly assigned source groups. The operator remains responsible for labels
and a representative split. The exporter sends only training states, questions,
and labels to miners. Calibration/test cases and acceptance policy stay with
the validator; miners receive the job ID and manifest hash.

For a reproducible plumbing example, generate synthetic inputs first. These are
not customer data and cannot establish customer-task quality:

```bash
.venv-kev/bin/python -m zils.benchmark build --out .private/job-example-input
.venv-kev/bin/python -m zils.jobs \
  --job-id example-decisions-v1 \
  --train .private/job-example-input/train.jsonl \
  --calibration .private/job-example-input/calibration.jsonl \
  --test .private/job-example-input/test.jsonl \
  --min-accuracy 0.80 --min-brier-improvement 0.01 \
  --allow-training-data-export --out .private/jobs/example-decisions-v1
```

For actual work, replace those three input paths with the prepared customer
files and set acceptance thresholds before evaluating candidates. Brier
improvement is an absolute decrease in equal-family Brier loss, not a percentage.
The manifest freezes the data hashes, job ID, and acceptance policy. Use new
output directories and a new job version when changing inputs or thresholds.

## Run the job

```bash
.venv-kev/bin/python -m zils.fleet init \
  --benchmark .private/jobs/example-decisions-v1 \
  --checkpoint models/reference --host 127.0.0.1 \
  --out .private/fleet-example-decisions-v1
.venv-kev/bin/python -m zils.fleet validator \
  --config .private/fleet-example-decisions-v1/validator/config.json --rounds 1
```

In separate terminals, start each miner as in the [fleet setup](../README.md#run-a-local-fleet):

```bash
ZILS_PYTHON="$PWD/.venv-kev/bin/python" HF_HOME="$PWD/.cache/huggingface" \
  .private/fleet-example-decisions-v1/miner-1/start-miner --rounds 1
```

Repeat for `miner-2` and `miner-3`. Each bundle contains the job's training data;
distribute it only to approved operators. The architecture and training recipe
remain pinned to the existing 0.8B model and one-epoch recipe. No automatic model
download, training, or chain transaction is triggered by building the job manifest.
Fleet initialization requires the downloaded reference. Testnet fleets still
use the existing explicit roster and publication flags.

## Inspect the result

Each validator round's `report.json` contains `job_id`, `job_sha256`, a `baseline`
evaluation, and a `delivery` decision. The starting checkpoint's temperature is
reset in a copy and fitted on the job's calibration split; candidates use that
same calibration procedure. Neither fit uses test labels. A failed baseline
aborts evaluation rather than permitting delivery without a valid comparison.

A candidate qualifies only if it beats the uniform floor, meets `min_accuracy`,
strictly improves the calibrated baseline's Brier loss, and meets the configured
minimum Brier improvement. The lowest-Brier qualifying candidate wins; accuracy
then UID break ties. Otherwise `delivery.status` is `no_qualifying_model`.

The hosted queue additionally supports [explicit customer version upgrades](version-selection.md).
Those runs compare with the previous accepted model's exact serving checkpoint,
evaluated on the new test set. The local fleet workflow above continues to use
its configured reference; it cannot resolve hosted predecessor job IDs.

An `accepted` result includes a `checkpoint` path relative to the round directory.
That directory contains the calibrated adapter and head, plus `release.json`
with job/round identity, model hashes, base revision, and acceptance information.
Keep the round report alongside it for full measured results. This is a local
artifact export: it neither publishes weights nor deploys an inference service.

Miner rewards remain the existing normalized skill scores within a round.
Customer acceptance is separate: miners can receive proposed weights even when
no model meets the delivery criteria. Scores from different customer jobs must
not be pooled as if they measured the same task. Repeated rounds reuse test data;
acceptance is a measured threshold on that set, not a guarantee of generalization
or statistical significance. Reserve independent final evaluation data for real
deployment decisions. The separate [Zils decision API](decision-api.md) serves approved shared JevK5
weights in a local pilot. Deployment of these training artifacts, automated release
approval/rollback, billing, and cross-job reward allocation remain future work.
