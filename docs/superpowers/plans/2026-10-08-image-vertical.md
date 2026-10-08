# Image Vertical Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver image prediction, client image training and private image-model activation beside the existing JevK5 service.

**Architecture:** Preserve the shared account, gateway, queue and release workflow. Add verified private image assets, a pinned Imajev profile and a separate image runtime. Deliver three dependent slices, each independently testable; all three are required to complete the approved scope.

**Tech Stack:** Python 3.13, unittest, PostgreSQL/Supabase Auth and Storage, PyTorch 2.8.0, Transformers 5.17.0, PEFT 0.21.0, safetensors, Pillow 11.3.0, torchvision 0.23.0 with matching CUDA build; Next.js 16.3.0, React 19, TypeScript, Zod and Playwright.

**Spec:** [Approved design](../specs/2026-10-08-image-vertical-design.md).

## Global Constraints

Every slice incorporates these exact requirements from the approved spec:

- “Initial scope is one still image per example, optional accompanying structured information, and one classification question with 2–16 named outcomes.”
- “Start with JPEG and PNG.” “Existing text request and response shapes remain valid and unchanged.”
- “Freeze the selected profile when a job is created, before upload or assignment.”
- “Initial proposed limits: 10 MiB encoded input, 16 million decoded pixels, and 8,192 pixels on either edge; one image per example; at most 1,024 training, 256 calibration and 512 test examples; 1 GiB aggregate canonical image bytes per training job.”
- “Temporary prediction assets expire after 24 hours. Incomplete drafts expire after 24 hours. Training assets and manifests are retained for 30 days after a terminal job state, with the retention date visible to the customer.”
- “Signed read URLs expire within 10 minutes.” “Only verified canonical bytes reach model code.”
- “Keep model preprocessing aligned with the experiment: full image, no task-specific crop, minimum 65,536 and maximum 400,000 processor pixels, and at most 4,096 total model tokens per question.”
- “Initial image recipe: one epoch, batch one, accumulation four, AdamW learning rate 2e-5, zero weight decay, gradient clip one, FP32 LoRA/head parameters and BF16 frozen base.”
- “Never stop the text runtime to make room.” “Image requests support the realtime lane initially.”
- Preserve owner isolation, early-access checks, lease fencing, immutable releases, stale-upgrade rejection and held-out data boundaries. No customer images in public fixtures, logs or public commits.
- Read each repository's AGENTS.md. Before frontend code, read the installed Next.js documentation referenced there. Reuse existing dependencies except the pinned image decoder/runtime additions; do not install image packages into the running text service.

## Review Focus

These additional failure cases are assigned tests in the slice plans:

1. Browser token refresh or switching accounts during an upload: retain the same owner's draft across refresh, cancel and clear private state on owner change. Slice 1 task 5.
2. Upload completion races with cancellation/expiry or crashes after canonical bytes are written: complete idempotently without reviving an expired job or leaking an orphaned object. Slice 1 task 2.
3. Images have identical instructions but different content, or different filenames contain identical pixels: accept the first case and reject cross-split duplicates in the second. Slice 2 task 6.
4. A customer image request is followed by stock prediction or another customer's request: switch LoRA, head and calibration together and restore the published stock adapter, not a bare base. Slice 3 task 11.
5. A prediction URL expires between preparation and execution, or the image runtime has insufficient memory: fail/retry within the request budget, finalize usage correctly, and never return a text-model answer as a fallback. Slice 1 task 4 and slice 3 task 13.

---

## Execution bases and workspaces

Backend paths in these plans are relative to the backend repository. Start from commit `972fe04`, which contains the approved spec and the current backend implementation; the immediately preceding product-code commit is `6602a12`. Frontend paths are relative to the web repository. The inspected frontend source is commit `e8f783b`, with separate uncommitted user edits in home/model/train pages and site navigation.

At execution time, use isolated feature checkouts through the worktree skill and available native tools. The current task directory is not itself a Git repository; select the actual repository explicitly. Do not edit or discard the user's uncommitted frontend changes. Build from the named committed frontend base and reconcile overlapping changes before proposing a merge. Keep implementation work out of the early-access source checkout used to write this plan.

Record baseline checks before changes. Backend uses `make check` and `make check-queue-db` with a configured Python 3.13 environment; frontend uses `npm test`, `npm run types:check`, `npm run lint` and `npm run build` with Node 24. Database checks use a disposable local cluster, never the production database. Record missing dependencies or existing failures separately from regressions.

## Plan sequence

| Slice | Deliverable | Tasks | Dependency |
|---|---|---|---|
| [1: Stock image predictions](2026-10-08-image-vertical-01-prediction.md) | An approved signed-in client uploads a photo and receives a real stock Imajev result through the existing gateway | 1–5 | Existing text/account services |
| [2: Client image training](2026-10-08-image-vertical-02-training.md) | Reviewed image datasets reach a compatible miner, train, and receive held-out results | 6–9 | Slice 1 asset/profile contracts |
| [3: Private image models](2026-10-08-image-vertical-03-activation.md) | Accepted adapters become account-bound API models and work in “Try your model” | 10–13 | Slice 2 evaluation/artifacts |

Implement slices in this order. A locally complete slice does not authorize production deployment or end the overall task. Use one shared interface vocabulary; the exact signatures are defined in the owning task below and carried forward by name.

## File boundaries

| Responsibility | Backend files | Frontend files |
|---|---|---|
| Model identity and image decision contract | `zils/models.py`, new `zils/image_contract.py` | `lib/playground.ts`, new `lib/images.ts` |
| Canonicalization and private asset lifecycle | new `zils/image_assets.py`, `zils/image_store.py`, `zils/image_cleanup.py`; additive SQL migration | new `components/image-decision-panel.tsx` |
| Pinned inference/training engine | new `zils/imajev.py`, `zils/imajev_runner.py`, `zils/image_server.py` | no model execution in browser |
| Dataset manifests and quality evaluation | new `zils/image_jobs.py`, `zils/image_metrics.py`; existing jobs/calibration/validator dispatch | new `lib/image-training.ts`, `components/image-training-intake.tsx` |
| Scheduling, accepted release and activation | existing coordinator/runtime/worker/workflow/release/version-selection modules | existing training dashboard/status/model-download modules |

Do not introduce a general plugin framework or refactor unrelated text paths. New image modules implement only the approved profile.

## Cross-slice interfaces

The wire image request is `{model, state, questions, images:[{asset_id}]}`. It has exactly one choice question with 2–16 outcomes; the inherited text request has no images field. The runtime's trusted envelope adds verified image metadata outside customer-controlled state.

The image response retains the existing typed answer plus `unknown_probability` and `abstained`. Conditional probabilities over known choices follow the pinned upstream conversion; training/evaluation store the full known-plus-unknown distribution. The frontend shows “Needs review” when abstained regardless of the most likely known label.

`GET /v1/image-models` and `POST /v1/image-decisions` are session-authenticated browser conveniences. They resolve the signed-in owner and call the same registry/evaluator with `key_id=None`; existing usage storage supports that value. They never mint or expose a shared API key. Programmatic clients continue using their own key with `GET /v1/models` and `POST /v1/systemone`.

Image asset endpoints accept either a verified session or a Zils API key. A `zils_sk_` token is validated only as an API key, without falling back to session parsing on failure. All other credentials go through Supabase session verification. Require early access and an enabled API account on either path. When a signed-in eligible owner has no API-account row yet, create it transactionally without re-enabling an existing disabled account.

The public frontend capability switch is display only. Backend feature flags `ZILS_IMAGES_ENABLED` and `ZILS_IMAGE_TRAINING_ENABLED` default to false and enforce admission. Private requests are still authorized against owner and model independently of feature flags.

## Completion evidence and review

Each task has a failing behavior test, minimal implementation, passing targeted tests and a focused commit. Broad checks run at slice boundaries, and again after subsequent changes only when warranted. Fixture tests establish contracts; the final real GPU rehearsal establishes actual image execution and training.

Native execution is recommended: the gateway, job manifest and model identity changes depend closely on shared interfaces, and there are two repositories to keep aligned. A fresh whole-change reviewer should examine tenant boundaries, unknown semantics, rollback and text compatibility before deployment is proposed.

After plan approval and execution-method selection, use the corresponding required execution skill. No product code or deployment has been changed while preparing this plan.
