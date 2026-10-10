# TypeSafe Jev comparison

The model page can compare an approved text model with pinned TypeSafe
`jev-1.13.0` on the **entire same held-out test split**. The existing starting
model or previous-version evaluation still controls approval. Jev does not
affect acceptance, miner rewards, calibration, or API activation.

Text submissions with `allow_jev_comparison: true` record permission and a pending
comparison. The website's submission permission explains that test inputs and
questions are sent to TypeSafe. Labels, training examples, and calibration data
are never included in these requests. Older jobs without that permission are
not automatically enrolled. An operator may enroll a specific existing job only
with its owner's authorization.

Apply `supabase/migrations/202610090003_jev_comparison.sql` before upgrading the
training API and starting the comparison worker. The nullable column inherits
the job table's owner-only read policy and service-only writes. It is separate
from immutable release provenance; completed-job timestamps and results do not
change. No GPU or additional model download is needed.

Run a separate CPU service with the same storage configuration as the training
API and a server-only `TYPESAFE_API_KEY`:

```bash
mkdir -p .private/jev-comparisons
.venv-kev/bin/python -m zils.jev_comparison --state .private/jev-comparisons
```

`--once` processes at most one eligible job. `--job UUID` processes a particular
already-authorized job and exits. Keep one worker per persistent private state
directory. Atomic database claims prevent overlapping workers from evaluating
the same job. Restarting preserves completed predictions. A missing cache or
an interrupted call with an unknown billing outcome stops that comparison for
operator review instead of silently repeating calls.

The worker verifies the frozen manifest and test-file hashes, uses the original
questions unchanged, validates response model identity and probabilities, and
scores with the same accuracy/tie-breaking and equal-family Brier calculation
as training. Results are published only after every test example completes.
Scores, test count, model version, evaluation date, provenance hashes, and input
token usage are saved; raw responses and labels are not returned by the job API.
Page views only read saved results. Both positive and negative accuracy differences
are displayed.

Initial automatic limits are 2,000 test examples and 16 MiB of combined input and
question JSON. Larger evaluations are marked skipped rather than comparing a
sample against full-test accuracy. Image models are excluded because Jev is
text-only. Rate-limited/overloaded requests receive bounded backoff; other failures
do not retry automatically or block model access. Charges belong to the operator's
TypeSafe account; no customer Zils balance is charged by this worker.

References: [API](https://docs.typesafe.ai/api),
[model version and input support](https://docs.typesafe.ai/models), and
[privacy](https://typesafe.ai/legal/privacy-policy).
