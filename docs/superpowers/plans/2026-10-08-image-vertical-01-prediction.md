# Stock Image Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let approved signed-in clients upload a photo and receive a stock Imajev classification through the existing Zils gateway and frontend.

**Architecture:** Private verified assets feed a separate pinned image runtime. The gateway resolves account ownership, image content and model identity before reserving compute. The browser uses its existing authenticated session.

**Tech Stack:** Python/unittest, Supabase/PostgreSQL, pinned PyTorch/Imajev, Pillow, Next.js/React/TypeScript/Zod/Playwright.

**Spec:** [Approved design](../specs/2026-10-08-image-vertical-design.md).

## Global Constraints

Read and apply the [master plan's Global Constraints](2026-10-08-image-vertical.md#global-constraints), including its exact image limits, pins, expiry and compatibility requirements. “Only verified canonical bytes reach model code.” “Existing text request and response shapes remain valid and unchanged.” “Never stop the text runtime to make room.”

## Review Focus

- Account switch during an upload must abort private work and clear images; ordinary token refresh must not erase the same owner's draft (task 5).
- Completion racing expiry/cancellation must not make the asset ready (task 2).
- Canonical pixels may expand far beyond compressed file size; bound decoding and output separately (task 2).
- Unknown may dominate while a conditional known-label probability looks high; the interface must say Needs review (tasks 1, 5).
- Prepare/predict asset expiry and image-runtime failure must finalize usage correctly, without a text fallback (tasks 3–4).

---

## Task 1: Pinned image profile and wire contract

**Files:** Modify backend `zils/models.py`, `zils/decisions.py`, `zils/api.py`; create `zils/image_contract.py`, `tests/test_image_contract.py`; extend `tests/test_models.py` and `tests/test_decisions.py`.

**Interfaces:**
- Produce `models.IMAJEV = "imajev-4b-v1"` and `models.spec(models.IMAJEV) -> dict` with the approved base/adapter/runtime and processor pins.
- Produce `image_contract.validate_image_request(body: dict) -> dict`; exactly one asset reference and one choice question, 2–16 nonempty distinct outcomes excluding reserved `__unknown__`.
- Produce `image_contract.make_image_response(release_id: str, body: dict, predictions: dict) -> dict` using full known-plus-unknown runtime probabilities and input-token accounting.
- Extend `Registry` with optional immutable capabilities. Missing capabilities normalize to legacy text behavior for comparisons, without rewriting old files or fingerprints.

- [ ] Write a contract test that distinguishes the native and conditional distributions:

```python
import unittest
from zils import image_contract

class ImageContractTest(unittest.TestCase):
    def test_unknown_is_preserved(self):
        body = {"model": "image-release", "state": {}, "images": [
            {"asset_id": "10000000-0000-4000-8000-000000000001"}
        ], "questions": {"inspection": {
            "type": "choice", "criteria": {"normal": None, "damaged": None}
        }}}
        result = image_contract.make_image_response("image-release", body, {
            "inspection": {"probabilities": {
                "normal": .2, "damaged": .3, "__unknown__": .5
            }, "input_tokens": 442}
        })
        answer = result["answers"]["inspection"]
        self.assertAlmostEqual(answer["probabilities"]["damaged"], .6)
        self.assertEqual(answer["unknown_probability"], .5)
        self.assertTrue(answer["abstained"])
        self.assertEqual(result["usage"]["input_tokens"], 442)
```

- [ ] Add rejection cases for zero/two images, two questions, non-choice questions, 17 outcomes, reserved unknown label, invalid IDs, NaN and a probability sum outside tolerance. Add a model-format roundtrip proving legacy reference/candidate hashes and file lists remain unchanged.
- [ ] Run `.venv-kev/bin/python -m unittest tests.test_image_contract tests.test_models tests.test_decisions -v`. Expect the new image behavior to fail before implementation; existing text behavior must pass.
- [ ] Implement separate image validation while leaving the no-images path on the existing validator. Delegate image response conversion to the same formulas used by the pinned upstream `to_response`: conditional known probabilities, unknown-scaled concentration, and abstention from the full distribution. For this pinned decision-head profile, use declared outcome order followed by unknown; equal logits select the first candidate, as in the experiment and upstream result_from_logits with token_ids=None. Test a normal/unknown tie and do not let JSON dictionary sorting change the winner.
- [ ] Implement artifact-format dispatch by explicit model metadata; image candidates require LoRA config/weights, readout config/weights and model/calibration metadata. Reject mixed formats, unknown revisions and missing files. Do not infer every `model.json` checkpoint is JevK5.
- [ ] Run the targeted tests and `make check-api`. Commit only the profile/contract changes with message `feat: define pinned image model and decision contract`.

**Implementation anchor (new profile constants in `zils/image_contract.py`):**

```python
IMAGE_CAPABILITIES = {
    "modalities": ["image", "text"],
    "question_types": ["choice"],
    "max_images": 1,
    "max_questions": 1,
    "max_options": 16,
    "max_input_tokens": 4096,
    "min_pixels": 65536,
    "max_pixels": 400000,
    "unknown_key": "__unknown__",
    "option_order": "declared_then_unknown",
}
```

Use these profile limits after structural validation; never apply them to legacy text requests.


## Task 2: Verified private assets, bounded decoding and expiry

**Files:** Create backend `zils/image_assets.py`, `zils/image_store.py`, `zils/image_cleanup.py`, `requirements/images.txt`, `supabase/migrations/202610080002_image_assets.sql`, `tests/test_image_assets.py`, `tests/image_database.py`; modify `scripts/check_queue_db.py`, `zils/cloud.py` as needed for bounded object removal.

**Interfaces:**
- `CanonicalImage` is a frozen dataclass with `data: bytes` (metadata-free PNG), `source_sha256: str`, `sha256: str`, `pixel_sha256: str`, `width: int` and `height: int`.
- `canonicalize(source: bytes) -> CanonicalImage` performs validated JPEG/PNG decoding, EXIF transpose, RGB conversion and bounded PNG encoding. Reject animated images.
- `ImageStore(db: Supabase)` exposes `create(owner, purpose, job_id, filename, source_bytes, source_sha256) -> dict`, `complete(owner, asset_id) -> dict`, `resolve(owner, asset_id, purpose=None) -> dict`, `delete_unused(owner, asset_id) -> None` and `cleanup(now, limit=100) -> dict`. Owners/IDs are UUID strings; returned public records omit internal object paths and signed read URLs.
- Store state transitions are `uploading -> verifying -> ready`, with terminal `failed`, `expired` or `deleted`. A database lease owns finalization; expiry/cancellation wins over a stale finalizer.

- [ ] Write the canonicalization oracle using pixels rather than encoded-file equality:

```python
import io
import unittest
from PIL import Image
from zils.image_assets import canonicalize

class CanonicalImageTest(unittest.TestCase):
    def test_metadata_is_removed_without_changing_pixels(self):
        source = io.BytesIO()
        Image.new("RGB", (8, 8), (10, 20, 30)).save(source, format="PNG")
        first = canonicalize(source.getvalue())
        second = canonicalize(first.data)
        self.assertEqual(first.pixel_sha256, second.pixel_sha256)
        self.assertEqual((first.width, first.height), (8, 8))
        self.assertEqual(Image.open(io.BytesIO(first.data)).info, {})
```

- [ ] Add image tests for EXIF rotation, truncated JPEG, disguised non-image, animation, over-10-MiB encoded input, over-16-million-pixel header, excessive edge length and excessive canonical output. Verify limits before pixel allocation and within the encoder.
- [ ] Add database checks in `tests/image_database.py:run(psql_command: list[str]) -> None`, called by the disposable cluster harness after the existing migrations. Test service-only mutation, owner reads, broad Storage-policy resistance, disabled-account preservation and two concurrent completions. Use actual PostgreSQL transactions, not only an in-memory double.
- [ ] Implement `requirements/images.txt` with `Pillow==11.3.0` and install it only in the new decoder/test environment. Define all approved limits as constants in `image_assets.py`; verify format/dimensions, use `ImageOps.exif_transpose`, decode in a bounded worker with a timeout, and hash canonical decoded pixels after conversion.
- [ ] Implement additive tables, private bucket policies and RPCs. Reserve per-job canonical-byte/count budget atomically at finalize time; do not trust browser totals. Finalization records source and canonical hashes and publishes ready only after immutable canonical upload succeeds. Track both source and canonical keys before writes so cleanup can recover a crash between object write and DB completion. Record the actual issued upload-grant expiry. On cancellation, block access immediately but retain cleanup metadata until grants have expired, so a late upload cannot create an untracked orphan.
- [ ] Make complete idempotent: an identical ready asset returns its metadata; altered bytes or an expired/cancelled job never replace it. Ready content is immutable. A referenced training asset cannot be deleted as “unused.”
- [ ] Implement `python -m zils.image_cleanup --once` with 24-hour prediction/draft expiry and 30-day terminal-training retention. Claim expired rows with a bounded lease, remove tracked source/canonical objects, then mark expired; retry partial deletion safely. Records remain inaccessible once expired, even if deletion is delayed. Keep live-job assets protected and limit unfinished training uploads by active-job and aggregate reservation rules. For temporary predictions, enforce an initial per-owner limit of 20 active assets, 256 MiB reserved bytes and 100 upload grants per rolling 24 hours; return 429 before issuing another grant. Reserve the maximum allowed output size until finalization replaces it with actual size.
- [ ] Run `.venv-kev/bin/python -m unittest tests.test_image_assets -v` and `make check-queue-db`. Commit as `feat: add verified private image assets and lifecycle`.

**Implementation anchor (finalization must be conditional, not a blind update):**

```sql
update public.zils_image_assets
set state = 'ready', canonical_sha256 = p_sha256,
    pixel_sha256 = p_pixels, canonical_bytes = p_bytes
where id = p_asset and owner_id = p_owner
  and state = 'verifying' and finalize_token = p_token
  and expires_at > clock_timestamp()
returning id;
```

This statement belongs inside the service-only finalization RPC after locking the owner/job budget and checking job cancellation. Zero rows means the lease/expiry check lost; do not return ready. Define matching columns/arguments in the additive migration. Source/canonical object writes precede this transition and remain tracked for cleanup if it loses.


## Task 3: Pinned stock Imajev runtime

**Files:** Create backend `requirements/imajev.txt`, `scripts/download_imajev.py`, `zils/imajev.py`, `zils/image_server.py` and `tests/test_image_server.py`; extend `zils/runtime.py` for explicit image-profile preparation.

**Interfaces:**
- `ImageEngine(reference: Path, device: str)` owns the pinned processor/model and exposes `prepare(image: Path, state: dict | str, question: dict) -> dict` and `predict(prepared: dict, temperature: float = 1.0) -> dict`. Prediction returns full known-plus-unknown probabilities and actual input tokens.
- `ImageRuntime(engine, releases, image_store_origin: str)` exposes `dispatch(method, path, token, body, request_id) -> tuple[int, dict]` for authenticated `/health`, `/prepare` and `/v1/systemone`, following existing runtime envelopes.
- `python -m scripts.download_imajev --out PATH` verifies the approved base, published adapter/head, runtime and processor revisions before writing the stock release manifest. No floating downloads at service startup.

- [ ] Create a recording fake engine in `tests/test_image_server.py` with `prepare` and `predict` implementing the signatures above. Have it return `{"probabilities":{"normal":.8,"damaged":.1,"__unknown__":.1},"input_tokens":442}`. Test that an invalid hash, URL outside the configured store, unauthenticated runtime request, wrong fingerprint or excessive context prevents `predict` from being called.
- [ ] Add a test with a valid image response that asserts returned tokens equal the processor count, not text-character estimation. Add a decode/request timeout case and two concurrent calls to prove one execution owner.
- [ ] Run `.venv-kev/bin/python -m unittest tests.test_image_server -v` and confirm the missing runtime behavior fails.
- [ ] Build the new runtime environment from the tested pins: torch 2.8.0 with matching torchvision 0.23.0/CUDA, transformers 5.17.0, peft 0.21.0, safetensors 0.8.0 and Pillow 11.3.0. Pin the upstream Imajev source by the approved Git commit, and import its processor/prompt/readout code rather than reproducing the architecture.
- [ ] Implement a bounded canonical-image downloader that checks exact origin/path, refuses redirects and arbitrary URLs, verifies the canonical hash and rechecks decoding limits. Keep signed URLs and image bodies out of logs. Preparation and execution bind to the same asset hash and release fingerprint.
- [ ] Use the pinned full-image preprocessing and 4,096-token limit. Serialize any GPU work during preparation and prediction; release prepared tensors on timeout/failure. Use the existing request concurrency/deadline patterns. Load stock Imajev's adapter and decision head; do not serve the bare Qwen checkpoint.
- [ ] Run targeted tests, then a bounded local GPU stock-image smoke test through the new runtime on existing authorized research images. Verify health/fingerprint, finite distributions, actual image tokens and unchanged text-service health; record results privately. Queue/defer if capacity is insufficient.
- [ ] Commit as `feat: serve pinned stock Imajev decisions in an isolated runtime`.

**Implementation anchor (stock initialization within the verified-reference loader):**

```python
engine = TorchDecision(str(base_dir), "cuda",
                       dtype=torch.bfloat16, max_length=4096)
engine.model = PeftModel.from_pretrained(
    engine.model, str(stock_dir), is_trainable=False
)
if not engine.enable_readout(stock_dir, trainable=False, codes=256):
    raise ValueError("The pinned image decision head is missing")
engine.processor._vision().image_processor.size = SizeDict(
    shortest_edge=65536, longest_edge=400000
)
engine.model.eval()
```

Here `base_dir` and `stock_dir` are outputs of `resolve_reference(reference: Path) -> tuple[Path,Path]`, added in `zils/imajev.py`. It resolves only the locally verified base/stock locations from the downloader manifest and validates their frozen hashes. Imports are the same pinned TorchDecision, PEFT and Transformers classes used by the research trial.


## Task 4: Gateway authorization, browser session routes and usage

**Files:** Modify backend `zils/api.py`, `zils/api_store.py`, `zils/decision_http.py`, `zils/batches.py`; create `zils/image_api.py` and `tests/test_image_api.py`; extend `tests/test_api.py`, `tests/test_decision_http.py` and `tests/test_batches.py`.

**Interfaces:**
- `ImageApi(api_store, image_store, registry, evaluate)` implements `dispatch(method, path, bearer, body, request_id) -> tuple[int, dict]` for image asset/session routes.
- `Store.ensure_account(owner: str) -> None` creates a missing account only after early-access authorization, without changing an existing disabled account.
- `Gateway.evaluate` retains its existing parameters and branches on a validated image request. `key_id=None` is allowed only from the verified browser-session route.
- Public routes: `POST /v1/image-assets`, `POST /v1/image-assets/{id}/complete`, `DELETE /v1/image-assets/{id}`, `GET /v1/image-models`, `POST /v1/image-decisions`. Programmatic image inference remains `POST /v1/systemone` with a Zils API key.

- [ ] Extend the existing local HTTP gateway fixture with two owners and two assets. Send an image request using the wrong owner's asset and assert `404` plus zero runtime calls and zero usage admission. Test the same for an inaccessible private model, expired asset and forged body-supplied owner.
- [ ] Test that a browser session can predict without creating any API key; assert the admission key is `None`. Test a disabled account and revoked API key, including that an invalid `zils_sk_` credential is never retried as a session token.
- [ ] Test expiry between prepare and predict: simulate `prepare` returning 442 and prediction returning an expired-object error. Assert a failed usage row, no completed charge and no text-runtime calls. A retry must resolve a fresh internal read URL for the same immutable content, not swap images.
- [ ] Run `.venv-kev/bin/python -m unittest tests.test_image_api tests.test_api tests.test_decision_http tests.test_batches -v` and confirm image-route failures.
- [ ] Implement image-only route dispatch before legacy API-key-only dispatch. Require early access, owner checks, enabled account and feature flags. Keep API-key and browser-session authorization paths explicit. Resolve canonical assets before the model runtime receives a trusted envelope; never forward customer object paths.
- [ ] Extend the HTTP handler's configurable allowed methods to support DELETE for this gateway while retaining the old default for other services. Add DELETE to exact-origin CORS configuration only where enabled. Other endpoints still reject unsupported methods.
- [ ] Preserve `prepare -> admit -> predict -> finish_usage`, with the image processor's actual reservation. Return a controlled error if actual tokens exceed reservation. Reject image bulk input at submission, before a durable job or usage reservation is created.
- [ ] Run targeted tests and `make check-api`. Commit as `feat: authorize and account for image predictions through the gateway`.

**Implementation anchor (inside gateway dispatch for the browser-only route):**

```python
if path == "/v1/image-decisions" and method == "POST":
    owner = self.store.session_owner(bearer)
    self.store.ensure_account(owner)
    image_contract.validate_image_request(body)
    return 200, self.evaluate(owner, None, body, request_id)
```

Run the image feature/access checks before this branch and retain owner/model/asset checks in evaluate itself. There is no API-key creation, fallback shared credential or client-supplied owner in this route.


## Task 5: Signed-in image prediction panel

**Files (frontend):** Create `lib/images.ts`, `components/image-decision-panel.tsx`, `components/image-decision-panel.module.css`, `tests/images.test.cjs`, `tests/browser/image-decisions.spec.ts`; modify `components/training-dashboard.tsx`, `lib/playground.ts` and `package.json`. Read the installed Next.js docs before editing.

**Interfaces:**
- `imageApi(baseUrl: string, storageUrl: string, token: () => Promise<string>, request = fetch)` returns `models(signal?)`, `createAsset(input, signal?)`, `completeAsset(id, signal?)`, `deleteAsset(id, signal?)` and `predict(body, signal?)`, matching task 4 routes.
- `ImageDecisionPanel({owner, token, apiUrl, storageUrl, modelId?, question?})` uses the existing authenticated training dashboard. `modelId` omitted selects the accessible stock image release; an explicit ID selects a private release later. Here owner/apiUrl/storageUrl/modelId are strings, token is () => Promise<string>, and optional question is the frozen image choice question returned by the owned model listing. Define that ImageQuestion type in lib/images.ts and use it consistently in the API client and panel.
- Public asset type: `{id, state, sha256?, width?, height?, expires_at}`. Never expose source/canonical object paths.

- [ ] Add network tests using injected fetch: reject signed URLs outside the configured Storage origin; do not retry immutable uploads with overwrite enabled; do not send images through `/api/playground` or its shared server key.
- [ ] Add a Playwright case intercepting all service traffic (following the training-intake fixture). After signing in, choose Images, upload a generated PNG, enter Normal/Damaged, and return this fixture:

```json
{"model":"image-stock","answers":{"inspection":{"type":"choice","choice":"damaged","probabilities":{"normal":0.4,"damaged":0.6},"confidence":0.02,"unknown_probability":0.5,"abstained":true}},"usage":{"input_tokens":442,"output_tokens":0}}
```

Assert “Needs review” is visible and “Damaged” is not rendered as an automated result. Assert no API-key creation request occurred.

- [ ] Add browser cases for expired session during finalize, owner change during upload, ordinary token refresh, corrupt image, feature unavailable and a narrow mobile viewport. Use AbortController and revoke object URLs when the owner changes or a file is replaced.
- [ ] Run `npm test` and `npm run test:browser -- tests/browser/image-decisions.spec.ts` to establish failures for the absent panel.
- [ ] Implement the image API client using current token retrieval and exact allowed Storage origin; use direct signed PUT for bytes and JSON metadata for the gateway. Parse image responses with explicit unknown/abstained fields while leaving text response validation unchanged.
- [ ] Implement accessible Text/Images controls within the existing signed-in dashboard, thumbnail, filename, question, outcomes and Analyze action. Keep model version/hardware details out of the primary flow. Show clear progress for upload/finalize/prediction and retain a useful retry action.
- [ ] Add the new pure TypeScript module to the existing test compilation command. Run frontend tests, type check, lint, build and the new browser tests; verify the existing training-intake and text-playground browser flows.
- [ ] Commit as `feat: let signed-in clients try image decisions`. Slice 1 ends only when the panel reaches the new real stock runtime in a private smoke test; do not enable production flags.

**Implementation anchor (explicit view-state mapping in `lib/images.ts`):**

```typescript
export function imageAnswerLabel(answer: {
  choice: string; abstained: boolean; unknown_probability: number;
}): string {
  return answer.abstained ? 'Needs review' : answer.choice;
}
```

Use this only after the full response schema validates finite normalized probabilities. The panel must not separately render a confident-looking choice badge when this function returns Needs review.
