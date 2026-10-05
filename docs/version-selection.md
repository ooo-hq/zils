# Customer model version selection

Customer selection uses the model the new version is intended to replace.
It is separate from miner rewards and from additional baselines in research
experiments.

| Run | Comparison | Required result |
| --- | --- | --- |
| First version of a task | Calibrated pinned base | Positive uniform skill, the customer's minimum accuracy, and a strict Brier improvement meeting the frozen minimum |
| Upgrade of a task | The named accepted, API-ready customer version | The same requirements, measured against that version on the new run's test data |
| No qualifying candidate | No replacement | Existing API models and aliases remain unchanged |

The lowest-Brier qualifying candidate wins; accuracy and then worker UID break
ties. `min_brier_improvement` is an absolute decrease in the queue's equal-family
Brier loss, not a relative percentage. First-version status does not waive a
quality requirement. Research-only checkpoints do not become accepted customer
jobs by changing these rules; their original results remain unchanged.

## Request an upgrade

First versions use the existing `POST /v1/jobs` request. To upgrade a particular
model, include `previous_job_id` in its acceptance policy. Replace the placeholder
with the UUID of an accepted, API-ready job owned by the signed-in customer:

```json
{
  "name": "flight-v2",
  "acceptance": {
    "min_accuracy": 0.80,
    "min_brier_improvement": 0.01,
    "previous_job_id": "REPLACE_WITH_PRIOR_READY_JOB_UUID"
  },
  "allow_training_data_export": true
}
```

Use the same training-service authentication and upload/submit endpoints as an
initial run. No new credentials or database migration are required. The caller
must identify the predecessor explicitly: names, upload order, and the most
recent job in an account do not establish that two jobs concern the same task.
Requests without a predecessor create independent task histories. Existing
dashboard submissions that omit this field retain that behavior.

Validation freezes the predecessor's job ID, model ID, checkpoint hash, and root
job ID in the private manifest. Workers still receive only the training export
and train a fresh adapter against the pinned base; they receive neither the
acceptance criteria nor another model's evaluation data.

During evaluation, the processor verifies and downloads the predecessor's
accepted release. It scores those exact weights and their existing serving
temperature on the new test examples. It does not reuse an older reported score
or refit the deployed version's temperature. Candidates are calibrated on the
new calibration split. Initial runs retain the equally calibrated base comparison.
An unavailable or changed predecessor fails closed; it never silently falls
back to the base.

Use representative test examples that neither the new adapter nor the incumbent
was trained or calibrated on. The existing split audit checks the current job;
operators remain responsible for semantic overlap and repeated use of evaluation
data across jobs. Acceptance measures this dataset, not guaranteed future quality.

## Activation and API names

After runtime verification, new versioned releases receive an immutable model ID
and a stable alias, `zils-task-ROOT_JOB_UUID`. The ready workflow response includes
`model_id` and `model_alias`. Send the alias as `model` in the existing prediction
API to follow accepted upgrades; send the immutable ID to pin one version.

Registration holds the existing registry lock while checking the alias's current
target. The target must still be the frozen predecessor. If another run has
already advanced it, activation enters `needs_review`; it cannot overwrite the
newer version. Start a new run against that active version. Exact retries are
idempotent, and earlier immutable IDs remain callable by their owner, including
for in-flight bulk jobs. Cross-account alias movement is rejected.

Legacy accepted releases remain valid. An explicit upgrade can establish their
first task alias after verifying that predecessor is registered for the owner.
Deploy the updated processor, publisher/workflow, adapter runtime, and registry
registration command together before submitting versioned jobs. Old job
manifests and completed evaluation records are not rewritten.

An operator's deliberate rollback using existing registry administration remains
separate from automated quality selection. Preserve immutable releases and keep
the stable task alias when rolling back.

## Verification scope

`make check` includes deterministic selection tests, exact-incumbent evaluation,
ownership and provenance checks, stale promotion, legacy compatibility, and an
HTTP upload-to-upgrade test with fixture model execution. These checks verify
selection behavior; they do not establish that a real customer adapter improves.
