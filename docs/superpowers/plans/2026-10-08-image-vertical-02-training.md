# Client Image Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a client's reviewed, privately uploaded image examples into a compatible miner job and trustworthy held-out comparison.

**Architecture:** Freeze the image profile and content-addressed manifest before assignment. Export only training images to qualified miners. Evaluate full known-plus-unknown distributions privately and apply the client's frozen quality targets.

**Tech Stack:** Existing Python queue/validator, Supabase/PostgreSQL, pinned Imajev/PyTorch, Next.js image intake and Playwright.

**Spec:** [Approved design](../specs/2026-10-08-image-vertical-design.md).

## Global Constraints

Apply every [master-plan constraint](2026-10-08-image-vertical.md#global-constraints). “Approved miners receive only training rows and short-lived access to the corresponding training image assets.” “For binary classification, extend the frozen policy with a named positive class, minimum positive recall and maximum normal false-positive rate.” Existing acceptance targets remain required.

## Review Focus

- Different image content with identical text must remain valid, while renamed duplicate pixels across splits must fail (task 6).
- A filename containing the answer must never become model evidence automatically (tasks 6–7).
- Group-preserving splits can become impossible despite enough rows; explain the missing independent outcome groups (task 7).
- A miner must not claim an unsupported model and fail after downloading its training data (task 8).
- Equal aggregate accuracy can hide a recall regression or dropped unknown mass; reject candidates that miss frozen targets (task 9).

---

## Task 6: Image-job identity, manifests and private exports

**Files:** Create backend `zils/image_jobs.py`, `tests/test_image_jobs.py` and `supabase/migrations/202610080003_image_jobs.sql`; modify `zils/coordinator.py`, `zils/jobs.py`, `zils/core.py`, `zils/benchmark.py`, `zils/version_selection.py` and `scripts/check_queue_db.py`.

**Interfaces:**
- `image_jobs.case_fingerprint(case: dict, canonical_sha256: str) -> str` hashes state, question and verified canonical content.
- `image_jobs.build(root: Path, job: dict, splits: dict[str,list[dict]], assets: dict[str,dict], policy: dict) -> dict` writes a frozen image manifest, split JSONL and train-only export.
- `image_jobs.audit(root: Path, manifest: dict) -> dict[str,list[dict]]` verifies profile, files, canonical/pixel hashes, groups and counts before use.
- Each dataset row's `image` is `{asset_id}`. Its private frozen asset entry contains canonical SHA256, decoded-pixel SHA256, dimensions and purpose/job binding. The worker export replaces storage internals with an opaque asset ID and canonical hash.
- Add `zils_create_profile_job(p_owner uuid,p_name text,p_acceptance jsonb,p_model jsonb)`. It freezes the registry-validated profile at creation and returns the existing job shape plus model metadata.

- [ ] Write a pure fingerprint regression:

```python
import unittest
from zils.image_jobs import case_fingerprint

class ImageFingerprintTest(unittest.TestCase):
    def test_image_content_is_part_of_identity(self):
        row = {"state": {}, "question": {
            "type": "choice", "criteria": {"normal": None, "damaged": None}
        }}
        self.assertNotEqual(case_fingerprint(row, "a"*64),
                            case_fingerprint(row, "b"*64))
        renamed = {**row, "filename": "damaged.jpg"}
        self.assertEqual(case_fingerprint(row, "a"*64),
                         case_fingerprint(renamed, "a"*64))
```

- [ ] Add manifest tests with two groups: same canonical/pixel hash across splits fails even with different filenames; two distinct images with identical state/question pass. Reject foreign-owner assets, an asset from another job, stale hashes, incomplete/expired images, inconsistent labels and source groups crossing splits.
- [ ] Add an export test that serializes the worker payload and proves calibration/test IDs, labels, signed URLs and source filenames are absent. A signed worker image-download endpoint must reject any requested ID not listed in that lease's frozen train export.
- [ ] Run `.venv-kev/bin/python -m unittest tests.test_image_jobs tests.test_jobs tests.test_version_selection -v` and verify the new behavior fails.
- [ ] Add a nullable profile column to existing jobs; do not rewrite historical manifests. The new coordinator path validates an explicitly allowed model spec and calls the new profile-job RPC. Legacy jobs keep their recorded model semantics. Validation selects the frozen reference by job model, not whichever reference is now configured as default.
- [ ] Implement image manifest build/audit with deterministic canonical JSON. Stage immutable verified bytes or a frozen asset catalog accessible only to the trusted processor; manifest hashes must cover all row-to-content bindings. Keep the text benchmark's exact-prompt rule unchanged and dispatch image cases to their own fingerprint rule.
- [ ] Extend `benchmark.training_rows`/worker export dispatch so image references are preserved; never accidentally drop the image field while keeping a text-only record. Reject manifest/profile changes at training, evaluation and publication boundaries.
- [ ] Extend version selection to reject cross-family predecessors and compare new holdout hashes with retained predecessor training/calibration hashes. Refuse exact overlap when provenance is available; retain the existing semantic-overlap limitation.
- [ ] Run targeted tests and `make check-queue-db`. Commit as `feat: freeze image jobs and isolate training exports`.

**Implementation anchor (fingerprint keeps outcome order but excludes filenames):**

```python
import hashlib
import json

def case_fingerprint(case, canonical_sha256):
    question = case["question"]
    payload = {
        "state": case["state"], "question": question,
        "outcome_order": list(question["criteria"]),
        "image_sha256": canonical_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
```

Pass the store-verified digest, not a digest asserted by the client. Duplicate-pixel and group-overlap checks remain separate from this exact-input fingerprint.


## Task 7: Labelled image intake and group-safe split review

**Files (frontend):** Create `lib/image-training.ts`, `components/image-training-intake.tsx`, `tests/image-training.test.cjs` and `tests/browser/image-training.spec.ts`; modify `components/training-intake.tsx`, `components/training-dashboard.tsx`, `lib/training.ts`, `lib/training-csv.ts` and `package.json`. Backend counterparts remain task 6's coordinator and manifest code.

**Interfaces:**
- `ImageExample = {id:string; filename:string; file:File; label:string; groupId:string; state:Record<string,unknown>}`.
- `prepareImageTraining(examples:ImageExample[], question:object, outcomes:string[], seed:string): PreparedImageTraining` returns `{splits, counts, distribution, seed, proportions}` with per-split example arrays. The final upload creates the job first, obtains owner/job-bound image slots, finalizes assets, and writes JSONL rows referring to returned asset IDs.
- `ImageTrainingIntake({owner, token, trainingApi, imageApi, onSubmitted})` uses the existing dashboard and submission lifecycle. All data-upload actions require the user's reviewed labels and export consent.

- [ ] Write a TypeScript/Node test using 18 examples, three outcomes and six independent groups per outcome. Assert every outcome occurs in each split and every group belongs to one split. Use an impossible fixture with one group carrying all examples of one outcome and assert a useful independence error.
- [ ] Test exact filename mapping: `part.png` appearing in two folders is ambiguous without relative paths; label-folder names may supply labels but must not enter `state`. Excluded/unlabelled files must not upload. Changing labels after a job is frozen must require a new draft/job.
- [ ] Add a browser test that chooses Images, uploads generated photos and a CSV, corrects a missing label, confirms the item grouping, selects a positive class and quality targets, and sends a job. Assert request.model is the image profile, metadata is uploaded before byte slots, and only finalized opaque asset IDs enter dataset JSONL.
- [ ] Run `npm test` and `npm run test:browser -- tests/browser/image-training.spec.ts` to establish the missing flow.
- [ ] Implement exact file/CSV mapping and thumbnail review without adding model/hardware controls. Reuse the existing grouped splitter's 70%/15%/15% targets and outcome-coverage-first behavior; record seed, actual assignments and achieved proportions. Extract only the pure grouping helper if necessary; retain existing text output unchanged.
- [ ] Freeze the same image decision instructions and outcomes across this job. Require JPEG/PNG, group confirmation, 2–16 outcomes and server-advertised limits. Show per-class counts and split counts before any training upload. Require explicit binary positive-class/recall/false-alarm targets, plus current accuracy/Brier targets.
- [ ] Implement bounded upload concurrency of two files, abort on account change, resumable missing-object lookup, and no overwrite of finalized assets. Retry only missing transfers; never resend the whole dataset with mutable object keys. Show expiry/retention dates and a “Start a new run” action for a frozen-data change.
- [ ] Expand the frontend model schema and version-correct artifact list for Imajev. Use server capability metadata to hide unsupported controls, while relying on server checks for enforcement.
- [ ] Run frontend tests, type check and both text/image intake browser tests. Commit as `feat: add reviewed image dataset onboarding`.

**Implementation anchor (row construction after successful asset finalization):**

```typescript
const row = {
  id: example.id,
  group_id: example.groupId,
  family: 'image-classification',
  state: example.state,
  question,
  label: example.label,
  image: { asset_id: asset.id },
};
```

`example` is an ImageExample, `question` is the reviewed choice question, and `asset` is the verified task-2 public asset record. Neither example.filename nor a folder label is inserted into state. The server rebuilds trusted manifest bindings from asset.id.


## Task 8: Compatible worker claims and real image training

**Files:** Modify backend `zils/coordinator.py`, `zils/workflow.py`, `zils/runtime.py`, `miner/queue.py`, `miner/worker.py`, `scripts/check_queue_db.py`; extend `zils/imajev.py`; create `zils/imajev_runner.py`, `supabase/migrations/202610080004_image_worker_profiles.sql`, `tests/test_image_queue.py` and `tests/test_imajev_training.py`.

**Interfaces:**
- `zils_claim_profile_training(p_hotkey text,p_supported_profiles text[])` claims only the intersection of the worker's operator-verified profiles, its advertised installed profiles and the job's frozen model.
- `GET /v1/config` retains its current fields and adds available image capabilities/limits. They indicate configured support, not a promise of a free worker.
- Signed worker image fetch: `POST /v1/workers/image-downloads` with `{job_id,lease_token,asset_ids}`, at most 100 IDs, returns train-only canonical-hash/read-URL pairs. Validate all IDs before issuing any grant.
- `imajev.train(reference: Path, training: Path, images: Path, out: Path, seed: int, device: str) -> dict` returns example/update counts, durations, resource measurements, artifact hashes and reload consistency.
- `python -m zils.imajev_runner --checkpoint PATH --cases PATH --images PATH --device cuda` produces the existing prediction-report envelope with native full distributions and pinned runtime identity.

- [ ] Extend `tests.test_queue.Store` with worker-profile records and test two workers: text-only and image-qualified. After enqueuing one job of each profile, assert neither receives the other's job. Repeat against PostgreSQL with two concurrent claimers and stale leases; update the database harness to run the new migration/checks.
- [ ] Test unsupported-profile rejection before data URL creation, cancellation during URL refresh, guessed holdout IDs, expired lease submission, and capacity deferral without consuming an attempt. Legacy claim behavior must never claim an image job.
- [ ] Add training tests for accumulation groups `[4, 2]` on six examples, trainable-parameter filtering, finite LoRA/head tensors and readout reload. A partial final accumulation uses its actual divisor:

```python
from zils.imajev import accumulation_batches
self.assertEqual([len(x) for x in accumulation_batches(list(range(6)), 4)],
                 [4, 2])
```

Define `accumulation_batches(rows: list, size: int) -> list[list]` in `zils/imajev.py` and use it in the trainer; add a numerical gradient test against an independently computed two-example mean for the partial batch, not just this batching test.

- [ ] Run `.venv-kev/bin/python -m unittest tests.test_image_queue tests.test_imajev_training tests.test_queue tests.test_workflow -v` before implementation.
- [ ] Store operator-verified profile qualification with exact runtime/profile hashes and verification time. A miner can advertise installed profiles only within that allowlist. Select the job inside the same locking/fencing transaction as the lease. Extend the old claim route to exclude image jobs, preserving legacy text callers.
- [ ] Route the coordinator/processor and workflow by frozen profile. Add repeatable installed reference paths to the miner CLI, matching the processor's reference map. Capacity checks happen before claim and again inside the shared compute lock; the per-profile runtime remains offline.
- [ ] Implement the tested fresh-published-checkpoint continuation recipe with actual-size final accumulation and deterministic seed/order. Carry images through native preprocessing; verify the expected trainable names/count, frozen vision tower/base and finite nonzero gradients. Write safetensors plus readout/model metadata, then reload and compare logits before submission.
- [ ] Use a new image child process/environment and measured resource/time bounds; retain parent timeout, log redaction, cancellation and lease-renewal handling. A timeout never becomes a partial accepted candidate. Do not stop any serving process.
- [ ] Run targeted tests and disposable database tests. Run one bounded four-image GPU preflight through the new trainer; discard its updates. Commit as `feat: route qualified miners to pinned image training jobs`.

**Implementation anchor (partial accumulation uses its actual group length):**

```python
def accumulation_batches(rows, size):
    if size < 1:
        raise ValueError("Accumulation size must be positive")
    return [rows[start:start + size] for start in range(0, len(rows), size)]

for batch in accumulation_batches(training_rows, 4):
    optimizer.zero_grad(set_to_none=True)
    for row in batch:
        loss = loss_for_example(row)
        (loss / len(batch)).backward()
    torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
    optimizer.step()
```

Define the local `loss_for_example(row: dict) -> torch.Tensor` inside imajev.train: verify/load the row's frozen image, call the engine's native preparation/candidate logits, and compute cross entropy at the label's index in declared-outcomes-plus-unknown order. `training_rows` is the shuffled frozen export; `parameters` is the verified trainable LoRA/head list.


## Task 9: Unknown-aware evaluation and frozen quality gates

**Files:** Create backend `zils/image_metrics.py` and `tests/test_image_metrics.py`; modify `zils/core.py`, `zils/calibrate.py`, `zils/validator.py`, `zils/jobs.py` and `zils/version_selection.py`; extend `tests/test_image_queue.py` and `tests/test_version_selection.py`.

**Interfaces:**
- `score_rows(labels: list[str], distributions: list[dict[str,float]], families: list[str], outcome_order: list[str]) -> dict` returns existing accuracy/Brier/skill fields plus NLL, native confusion counts, unknown rate and per-class recall with denominators.
- `fit_image_temperature(logits: list[list[float]], target_indices: list[int], families: list[str]) -> float` fits macro-family NLL over 81 log-spaced values in [0.25,4].
- `image_policy_passes(metrics: dict, policy: dict) -> bool` enforces optional image constraints in addition to the existing selection checks. Binary policies require `positive_class`, `min_positive_recall` and `max_false_positive_rate`. Multi-class policies may have `min_class_recall` mapping labels to [0,1] targets.

- [ ] Add the independent scalar oracle:

```python
import unittest
from zils.image_metrics import score_rows

class ImageMetricsTest(unittest.TestCase):
    def test_unknown_is_an_error_and_stays_in_brier(self):
        report = score_rows(
            ["damaged"],
            [{"normal": .2, "damaged": .3, "__unknown__": .5}],
            ["inspection"],
            ["normal", "damaged"],
        )
        self.assertEqual(report["accuracy"], 0)
        self.assertAlmostEqual(report["brier"], .78)
        self.assertEqual(report["unknown_rate"], 1)
```

- [ ] Add equal-accuracy candidate fixtures reflecting the research tradeoff: 97/100 positive recall with 29/100 normal false alarms versus 87/100 recall with 19/100 false alarms. Under minimum recall .95, the second candidate cannot qualify even if its Brier is lower. Add the opposite policy to show the choice follows frozen customer targets, not a hard-coded preference.
- [ ] Test no-positive denominators as invalid for a required positive-class gate, unknown abstentions counted against recall, nonfinite predictions, missing rows, identical test IDs/order for baseline/candidate, and fit restricted to calibration data.
- [ ] Run `.venv-kev/bin/python -m unittest tests.test_image_metrics tests.test_image_queue tests.test_version_selection -v` and establish new failures.
- [ ] Implement full-distribution scores and family aggregation without renormalizing away unknown. The uniform floor assigns 1/N to each known answer and zero to unknown; Brier reference is 1−1/N per case. Preserve the existing public/reward skill contract, max(0, 1 - brier / uniform_brier), while retaining the uncensored Brier and NLL values. A zero-skill candidate cannot qualify. Macro-family aggregation remains consistent with the existing queue.
- [ ] Dispatch image cases through the image metrics/temperature fitter while preserving the legacy text evaluator. Keep native logits/distributions private. Fit positive temperature on calibration only and assert argmax ranking is unchanged except explicitly handled exact numerical ties.
- [ ] Extend policy validation with exact allowed image fields, finite [0,1] bounds, named-class existence and profile checks. Freeze policy in the manifest. Require existing accuracy, positive skill and Brier rules as well as the image gates before winner selection.
- [ ] Score an upgrade's frozen incumbent weights and serving calibration on the new test cases; do not refit the incumbent or silently fall back to stock. Include comparison provenance and image metrics in the immutable result/release evidence.
- [ ] Run targeted tests and `make check`. Complete a fixture HTTP job from upload to no-qualifying result as well as an accepted fixture job; neither fixture proves real accuracy. Commit as `feat: evaluate image adapters with recall and false-alarm gates`.

**Implementation anchor (retain unknown in each case's loss):**

```python
def full_brier(gold, probabilities):
    if gold == "__unknown__":
        raise ValueError("This profile evaluates answerable labelled cases")
    return sum((p - float(key == gold)) ** 2
               for key, p in probabilities.items())
```

Add `full_brier(gold: str, probabilities: dict[str,float]) -> float` in image_metrics.py and call it only after validating the full distribution. Require outcome_order to match exactly the known distribution keys. Compute the chosen label from outcome_order followed by unknown; it is not the argmax of the conditional public response.
