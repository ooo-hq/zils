# Customer model accuracy comparisons

Completed training results expose `dataset_counts` and the selected candidate’s
metrics. The starting comparator may be the pinned base checkpoint or a previous
accepted customer model. Neither is TypeSafe’s hosted Jev. The selection test
influenced which candidate was accepted, so it is not an independent final test.

## Optional independent Jev test

An authenticated owner can create one immutable comparison for a completed,
accepted JevK5 job:

1. `POST /v1/jobs/{id}/comparison` with `allow_typesafe_export: true`,
   `unseen_examples: true`, and `consent_version: "typesafe-evaluation-v1"`.
2. Upload a new JSONL file to the returned signed storage destination. The file
   contains the same fields as training examples: `id`, `group_id`, `family`,
   `state`, `question`, and `label`. Limits: 500 examples, 5 MiB. The examples must
   represent the intended workload and retain its decision question.
3. `POST /v1/jobs/{id}/comparison/submit`. Poll the authenticated
   `GET /v1/jobs/{id}/comparison` for status and results.

Creation and submission are idempotent. Existing files, consent and results cannot
be replaced through the customer API. Results never change training acceptance,
the serving model, or the selected model alias. An unsuccessful or interrupted
comparison is retained; no winner is declared and no paid requests are retried
automatically. Operators must investigate before any manual intervention.

The processor verifies the accepted checkpoint and its provenance, rejects IDs,
source groups and exact prompts found in this job or its known model ancestry,
and checks input limits before calling TypeSafe. These checks cannot detect all
semantic duplicates or training elsewhere; the owner must confirm the examples
were never used for training, selection or tuning. Duplicate prompts within the
new file are also rejected.

Both models receive the same state and question. Jev requests send neither labels
nor case/group IDs. The provider endpoint is fixed to
`https://api.typesafe.ai/v1/systemone`, and the requested and returned version must
both be `jev-1.13.0`. Provider credentials remain on the processor and are removed
from model subprocess environments. The selected checkpoint is evaluated without
further training or calibration.

## Interpreting the evidence

Accuracy is correct answers divided by all new examples. Choice accuracy uses
Jev’s returned choice. Yes/no and score accuracy use the highest-probability label
(lexicographic tie break); this is classification accuracy, not numeric score error.
Brier is the mean per-example squared probability error. Jev probability totals
within 0.98–1.02 are normalized to accommodate rounding; other invalid outputs fail
the entire comparison.

The accuracy difference includes a paired cluster bootstrap interval (2,000
resamples of source groups, fixed seed, 2.5th/97.5th percentile endpoints). A
directional verdict requires at least 30 source groups and an interval excluding
zero. This is an approximate interval, especially for small or highly uneven
groups; it is not a guarantee of future performance. Small tests and intervals
covering zero say “no clear difference.” Full model identity, file hash, counts,
paired wins/losses and every answer are retained. Unfavorable results remain visible.

## Operator setup

Apply `supabase/migrations/202610080001_training_comparisons.sql` before enabling
this feature. The new table is accessible only through the service role. Customer
reads and writes require the training API’s owner check. The existing private
training-data bucket stores `JOB_ID/comparison/input.jsonl` without overwrite.

Set `ZILS_JEV_COMPARISON_ENABLED=1` on the training API and training processor.
Set `TYPESAFE_API_KEY` only on the processor, using your existing protected service
environment mechanism. Do not enable the API before the configured processor is
running. Jev calls consume the operator’s TypeSafe allowance. This integration is
off by default and adds no runtime dependency.

Existing training work takes priority. Comparison claims are atomic and have a
one-hour expiry; expired attempts fail instead of being replayed. GPU inference
uses the existing shared compute lock. A provider-stage deadline bounds paid work
to 20 minutes. A local private audit directory contains trained and Jev predictions
for completed attempts, while the database retains owner-visible results.

Validate locally with `make check` and `make check-queue-db`. The latter starts a
disposable PostgreSQL cluster, not a production database. No tests require a
TypeSafe key or submit paid requests. Use a public, genuinely unseen fixture for a
separate deployment smoke test after consent.
