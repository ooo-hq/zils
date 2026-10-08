# Private Image Model Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make qualifying image adapters callable only by their client, expose them in the frontend, and verify the complete image vertical without interrupting text service.

**Architecture:** Extend the existing accepted-release publisher and version-selection workflow by model profile. The separate image runtime verifies immutable artifacts and serially switches the complete adapter/head/calibration state. The shared gateway keeps owner checks, stable aliases and usage accounting.

**Tech Stack:** Existing Python release/registry/workflow services, pinned Imajev engine, Supabase, Next.js training dashboard and Playwright.

**Spec:** [Approved design](../specs/2026-10-08-image-vertical-design.md).

## Global Constraints

Apply every [master-plan constraint](2026-10-08-image-vertical.md#global-constraints). “Only mark a job ready after the image runtime verifies the exact release and the gateway registers it for the correct owner.” “Switching back to stock restores the published Imajev adapter and head; merely disabling LoRA would select the wrong baseline.” Preserve existing aliases and historical results.

## Review Focus

- Mixed JevK5/Imajev artifacts or a changed readout must fail publication before any API registration (task 10).
- Stock, client A and client B predictions must never inherit one another's head, adapter or calibration (task 11).
- A late completed upgrade must not overwrite a newer accepted task version (task 10).
- An accepted artifact may exist while activation fails; the UI must not report it as API-ready (task 12).
- A real GPU trial that fails quality gates remains negative evidence; do not weaken policy or replace the incumbent to make the demo pass (task 13).

---

## Task 10: Profile-aware release publication and atomic activation

**Files:** Modify backend `zils/adapter_releases.py`, `zils/models.py`, `zils/workflow.py`, `zils/version_selection.py`, `zils/api.py`; create `tests/test_image_releases.py` and extend `tests/test_workflow.py`, `tests/test_version_selection.py`.

**Interfaces:**
- Existing `publish(store, job_id, releases)` and `registry_entry(release, url, token_env, alias=None)` gain profile dispatch, preserving existing text outputs.
- An image release binds `model`, base/adapter/runtime/preprocessor pins, artifact hashes, canonical task question, calibration, owner, job/comparison identity and acceptance evidence. Its fingerprint covers all executable/scoring inputs.
- Workflow config adds an image-profile runtime/release-root mapping. Resolve activation destination by the frozen profile, not a filename prefix or client-supplied URL.
- The existing immutable model ID/task-alias mechanism remains the external identity. Reject a predecessor from another model profile or owner.

- [ ] Add publication tests that construct a qualifying image job using a fixture candidate and matching frozen manifest. Change exactly the readout hash, base revision, preprocessing policy, owner, temperature, acceptance policy or outcome names; each mutation must prevent registration.
- [ ] Add a test for a no-qualifying image job proving `publish` returns no release and the existing registry/alias remains byte-for-byte unchanged. Preserve the same test for JevK5.
- [ ] Add concurrent stale-upgrade tests: freeze predecessor A for candidates B and C, activate B, then attempt C. C must enter needs-review without moving the alias or removing immutable A/B access.
- [ ] Run `.venv-kev/bin/python -m unittest tests.test_image_releases tests.test_adapter_releases tests.test_version_selection tests.test_workflow -v` and establish failures for the unsupported image profile.
- [ ] Replace JevK5-only constants in publication with a small explicit profile dispatch. Validate the exact permitted artifact names, regular-file/no-symlink rules, aggregate limits, known tensor names/shapes/dtypes and finite values. Use safetensors for all new image tensors; do not accept pickle or arbitrary model code.
- [ ] Bind the serving task contract to the canonical question used by the image job. The client-specific runtime enforces matching outcome names/instructions; stock prediction supports any valid image question. Expose the owned task question to its frontend panel so the user need not reconstruct it.
- [ ] Reuse the existing acceptance recheck and stale-alias transaction under the registry lock. Image activation verifies runtime release ID/fingerprint before registration and ready status. Failure retains the accepted artifact and reports activation-failed for retry; it never falls back to another runtime/model.
- [ ] Normalize optional registry capabilities consistently on first load and reload. Do not allow capability/profile/owner mutation of an existing immutable ID. Old entries with no capabilities still represent their exact legacy text releases.
- [ ] Run targeted tests and `make check-api`. Commit as `feat: publish and activate verified private image releases`.

**Implementation anchor (model family must follow the frozen job):**

```python
model = models.job_model(job)
if model not in (models.JEVK5, models.IMAJEV):
    raise ValueError("This profile cannot publish a customer API release")
spec = models.spec(model)
if job["result"].get("model") != spec:
    raise ValueError("Evaluated model differs from the frozen job")
```

The existing publication function already has job. Apply these checks before choosing the profile's artifact validator, runtime revision, prompt contract and destination. Recompute release identity from verified files; do not accept a result-supplied fingerprint.


## Task 11: Complete image adapter switching and isolation

**Files:** Extend backend `zils/imajev.py`, `zils/image_server.py` and `tests/test_image_server.py`; create `tests/test_image_adapter_gpu.py`.

**Interfaces:**
- Extend `ImageEngine` with `activate_release(path: Path, release: dict) -> None`. It loads LoRA tensors, decision readout and serving calibration only after release verification.
- `ImageRuntime` discovers verified image releases atomically and queues release activation with each prediction. Stock is an immutable release with its published adapter/head/calibration; no `disable_adapter` shortcut.
- A failed activation invalidates the active slot and returns an error; prediction cannot proceed using previously active weights.

- [ ] Add a recording engine fixture that stores a triple `(adapter_digest, head_digest, temperature)`. Submit A, stock, B and A requests and assert the engine observes the corresponding complete triple on every prediction. Include distinct head hashes even when adapters share the same structure.
- [ ] Add concurrent request tests asserting activation and prediction do not interleave. A corrupted B readout must fail B, and the next A request must reload A successfully rather than reuse uncertain state.
- [ ] Add a fake-engine test for a stock request after A: require stock's published LoRA/head, not a disabled adapter. Keep this as a dedicated regression because the existing text runtime correctly uses a different stock behavior.
- [ ] Run `.venv-kev/bin/python -m unittest tests.test_image_server -v` and verify adapter scenarios fail against stock-only code.
- [ ] Implement bounded serial execution and complete-state activation. Hash-check new release files before adding them to the catalog; reject mixed profiles, changed files or incompatible task contracts. Keep canonical base/vision weights frozen and shared within this image runtime.
- [ ] Avoid an unbounded adapter/tensor cache. Start with one GPU slot and immutable artifacts on disk; measure load-plus-prediction latency separately from warm prediction latency. If retaining a stock CPU snapshot, include its bytes in the startup memory admission and document the bound.
- [ ] Add a real GPU test entry point with explicit arguments: `python -m tests.test_image_adapter_gpu --stock PATH --adapter PATH --images PATH --out PATH`. It records predictions before/after switching, verifies exact release identity and checks stock results return to the baseline within the chosen numeric tolerance. It must refuse an output directory that already exists and never register a model publicly.
- [ ] Run the targeted tests and one bounded authorized GPU switching check. Commit as `feat: isolate stock and client image adapter state`.

**Implementation anchor (inside ImageRuntime's serial execution owner):**

```python
self.active_release = None
self.engine.activate_release(path, release)
self.active_release = release["release_id"]
prepared = self.engine.prepare(image, request["state"], question)
return self.engine.predict(prepared, temperature=release["temperature"])
```

This is the core of new `ImageRuntime._predict_locked(path: Path, release: dict, image: Path, request: dict, question: dict) -> dict`. It executes only within the bounded serial queue. Any activation exception leaves active_release unset; no subsequent prediction may assume the old slot is safe. Both stock and private releases go through this path.


## Task 12: Image results, activation status and client model testing

**Files (frontend):** Modify `lib/training.ts`, `lib/training-status.ts`, `components/training-dashboard.tsx`, `components/training-run-status.tsx`, `components/image-decision-panel.tsx`; create `components/image-training-results.tsx` and `tests/browser/image-models.spec.ts`; extend `tests/training-model.test.cjs`, `tests/training-status.test.cjs`. Backend: extend `zils/coordinator.py:public_job` and owner-authorized model listing without returning private predictions.

**Interfaces:**
- Public job image metrics include `count`, `accuracy`, `brier`, `nll`, `unknown_rate`, per-class denominators/recall and confusion counts for the baseline and evaluated candidates. Raw cases, URLs and predictions remain private.
- `ImageTrainingResults({job})` displays these aggregates and the frozen acceptance thresholds.
- Reuse `ImageDecisionPanel` from slice 1 with the verified `workflow.model_id` and owned task question. Enable it only after `workflow.state === "ready"`.

- [ ] Add parser tests for a completed no-qualifying image job, accepted-but-activating job, activation failure, ready job and stale-upgrade needs-review state. Reject a JevK5 artifact list returned for an image release.
- [ ] Add a Playwright flow showing equal 84% accuracy for two adapters with different recalls/false-alarm rates. Assert both tradeoffs are visible and the no-qualifying result has no “Use this model” or private prediction action.
- [ ] Add a ready-job browser test that uploads a fresh image through the existing panel and sends the private immutable model ID. Switch to another signed-in account and assert model details, thumbnails and pending requests are cleared.
- [ ] Run `npm test` and `npm run test:browser -- tests/browser/image-models.spec.ts` to establish new UI failures.
- [ ] Extend API/frontend schemas with versioned image metrics, frozen image policy and correct artifact names. Keep stock-vs-trained labels clear and show denominators rather than implying a percentage is a guaranteed customer outcome.
- [ ] Implement the results component and exact activation-status messaging. Show the API model ID only after runtime/gateway confirmation, plus a task alias when valid. Never derive an endpoint/model name locally from a job ID.
- [ ] Show retention dates and expired-image errors with a re-upload action. Distinguish finished training from API availability and preserve saved negative results.
- [ ] Run frontend unit tests, type check, lint, build and image/text browser flows. Commit as `feat: expose image training results and private model predictions`.

**Implementation anchor (readiness guard, exported from training-status):**

```typescript
export function isImageReady(job: Job): boolean {
  return job.model?.id === 'imajev-4b-v1'
    && job.status === 'completed'
    && job.result?.delivery.status === 'accepted'
    && job.workflow?.state === 'ready'
    && Boolean(job.workflow.model_id);
}
```

Import Job from lib/training. The dashboard uses this guard before passing workflow.model_id to ImageDecisionPanel. An accepted download alone is not sufficient.


## Task 13: Full rehearsal, release evidence and rollout package

**Files:** Create backend `scripts/rehearse_image_vertical.py`, `tests/test_image_vertical.py`, `docs/image-decisions.md`, `docs/image-training.md` and `examples/image-vertical/` configuration examples; update `docs/development.md`, `docs/README.md` and environment examples. Keep actual credentials, data, host details and rehearsal outputs in ignored private storage.

**Interfaces:**
- `python -m scripts.rehearse_image_vertical --config PATH --dataset PATH --out PATH` runs only against the explicitly configured isolated services. Config contains two already-authorized test owners, API/coordinator endpoints, expected text/image fingerprints and the frozen image policy. Secrets are loaded from protected environment references, not public files.
- Output `report.json` records step status, asset/model ownership checks, immutable IDs/hashes, timing/memory, evaluation outcome, serving result when qualified, and before/after text health. It must preserve failures and refuse to overwrite an existing output directory.

- [ ] Build `tests/test_image_vertical.py` by extending the local HTTP/storage fixture pattern from `tests/test_queue.py`. Exercise upload/finalize, submit, qualified claim, train-only image fetch, candidate upload, calibration/evaluation, accepted publication, runtime verification and owner prediction. Also test the negative candidate and activation-failed paths.
- [ ] Add a full-flow isolation test with a second owner and a text-only worker. Neither can obtain image holdouts or the first owner's private image model. Check no signing secret, session token, URL or source image enters captured service logs.
- [ ] Run `.venv-kev/bin/python -m unittest tests.test_image_vertical -v`, then complete backend checks with `make check` and `make check-queue-db`. Run frontend `npm test`, `npm run types:check`, `npm run lint`, `npm run build` and image/text Playwright flows. Record existing baseline failures separately.
- [ ] Implement the rehearsal driver with explicit phase/result recording, deadline checks, private output permissions and cleanup limited to assets/jobs created by that rehearsal. Freeze and hash the data/policy before model execution. Never tune the test threshold to turn a failed candidate into a pass.
- [ ] Run the actual approved image pipeline on authorized research/client data with one trained candidate and fresh holdouts. Establish acceptance criteria before inference, retain all scores, and measure complete job time separately from training-loop time. Use the real new queue/gateway path, not the previous standalone research script. On the existing 24 GB host, the isolated rehearsal may keep its own new image-serving process unloaded during training/evaluation, then start it for activation/prediction. Touch only processes created by the rehearsal; never stop the existing text service. Record this as sequential resource validation, not evidence that two persistent runtimes and training fit concurrently.
- [ ] If the candidate qualifies, verify image adapter publication, same-account API inference on fresh images, another-account denial, and text-service health/fingerprint. If it does not qualify, preserve the negative result and report that real successful activation remains unverified. Passing fixture activation alone is not evidence of a real trained release.
- [ ] Recheck actual GPU capacity with the intended persistent image/text services. Test queue deferral under insufficient capacity; never interrupt text inference. New paid compute or production migration remains a separate deployment decision with the staged evidence available for review.
- [ ] Document fresh-install pins, image asset expiry, classification limits, worker qualifications, upload/session/API contracts, unknown semantics and expected errors. Provide placeholder-only environment examples with exact keys and comments; no real endpoints, tokens or host paths. Include migration ordering, feature flags default-off, health verification and rollback by disabling new image admissions while preserving old text routing.
- [ ] Prepare a concise deployment checklist linking measured evidence and outstanding limits. Request any needed production/paid-resource authorization only once the proposed changes and rollback are concrete.
- [ ] Commit as `test: verify and document the complete image vertical`. Perform one fresh whole-change review of both repositories using the execution method selected by the user. Fix actionable findings and rerun affected checks. The overall task is complete only when all approved slices work and any remaining operational limitation is explicitly reported.

**Implementation anchor (rehearsal output cannot overwrite earlier evidence):**

```python
def create_run_directory(destination):
    root = Path(destination)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    return root
```

Define `create_run_directory(destination: str | Path) -> Path` in the rehearsal script and call it before any remote mutation. Define `rehearse(config: dict, dataset: Path, out: Path) -> dict` for the recorded sequence above. Persist each completed step immediately and write terminal failure details with redacted messages; do not remove failed evidence during cleanup.


## Coverage map

| Spec requirement | Owning tasks |
|---|---|
| One app/account/key, text compatibility, image wire contract | 1, 4, 5, 12 |
| Pinned base/runtime/processor, immutable per-job identity | 1, 3, 6, 8 |
| Private assets, decoding limits, expiry and deletion | 2, 4, 5, 13 |
| Label review, grouped splits, duplicate and predecessor checks | 6, 7 |
| Training-only worker access and compatible claims | 6, 8 |
| Native training/head artifacts and partial accumulation | 8 |
| Unknown-aware calibration, metrics and customer quality policy | 1, 9, 12 |
| Verified private release, incumbent comparison and alias safety | 9, 10 |
| Separate runtime and complete stock/customer state switching | 3, 11 |
| Frontend private prediction and honest activation state | 5, 12 |
| Real end-to-end evidence, text availability and rollout | 13 |
