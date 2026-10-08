# Image decisions beside JevK5: implementation design

Status: approved by the user on 2026-10-08. Implementation planning follows; product code and deployment have not changed.

## Outcome and scope

Clients use the existing Zils account, dashboard and API key to classify images, train an image model on their own labelled examples, and use an accepted private image model from the frontend or API. Text decisions continue using the existing JevK5 4B releases. Image decisions start from the pinned Imajev 4B checkpoint and use a separate runtime and compatible miner pool.

The user approved this direction and the sequence: first make stock image prediction work through the frontend and existing gateway; then connect image training, evaluation and private adapter activation. The whole vertical is the intended outcome. Shipping the first slice alone does not complete that outcome.

Initial scope is one still image per example, optional accompanying structured information, and one classification question with 2–16 named outcomes. Start with JPEG and PNG. The interface presents capability names, “Text” and “Images”; customers do not need to select model repositories or hardware. Image generation, audio, video, boxes, segmentation, reference-photo comparisons and durable image bulk inference are separate capabilities and are not part of this release.

## Review summary

- One Zils app and API key; Text uses JevK5, Images uses Imajev.
- First release handles one photo and a classification question; clients can try the stock model before training.
- Private image training follows the existing approved-worker, held-out evaluation and private activation flow.
- Existing text service stays available; image work queues when capacity is unavailable.
- Delivery includes all three slices below, with stock prediction first and private trained-model prediction last.

## Existing foundation and required extensions

The backend already provides authenticated jobs, private signed uploads, approved-worker leases, held-out evaluation, immutable model releases, per-account model authorization, release fingerprint verification and activation. Its newer source also supports incumbent-model comparisons and stable task aliases. These mechanisms remain authoritative.

Several implementations still assume JevK5: `zils/models.py`, the active training reference in `zils/coordinator.py`, worker selection in `zils/workflow.py`, artifact publication in `zils/adapter_releases.py`, and adapter switching in `zils/adapter_server.py`. The gateway currently accepts exactly state/model/questions, and validates probabilities over the supplied outcomes. Those assumptions need explicit versioned image support.

The frontend already has a training wizard, reviewed spreadsheet examples, progress, model IDs and API keys. `lib/training.ts` recognizes only Kev and JevK5 artifact formats. `lib/playground.ts` accepts text decision requests. Extend these contracts rather than creating a separate customer account or dashboard.

## Customer experience

1. Choose Text or Images. The Images panel offers “Try an image” immediately for eligible signed-in accounts and “Train on my images” when image training is enabled.
2. Describe the decision and its possible answers. Upload photos and a CSV with filename, answer and optional item/group reference. Folder labels can be imported as a convenience; item groups still require explicit review. Show thumbnails, counts, missing labels and unreadable files before submission.
3. Review the split and data-sharing notice. The service reserves calibration and test groups before training. Related views stay together. Ask the customer whether avoiding missed positives or false alarms matters more, and freeze the resulting acceptance targets before training.
4. Show upload, validation, queue, training, evaluation and activation progress using the existing run dashboard. Insufficient image-worker capacity remains a queued state, not an unexplained training failure.
5. Show baseline versus candidate metrics, plus per-class recall and confusion counts. Only an accepted, runtime-verified release becomes “Ready.” Its page includes “Try your model,” the existing API-key experience, and its private model ID. Uncertain image answers display “Needs review.”

The first interface is available to existing approved accounts. The public text playground is unaffected. Images and labels are not sent through a public shared playground credential.

## Model profiles and immutable identity

Add an image profile, proposed internal ID `imajev-4b-v1`, alongside the existing text profiles. The image profile freezes all of the following:

- Imajev checkpoint `mohit67890/imajev-4b` revision `f8d8234cebc6c99065c07731e59716dc0a6e27ab`, base `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, and upstream runtime revision `ccf586d43d2a580319b6535c893668904d909eb9`.
- Supported request type, image decoder/preprocessing revision, maximum pixels/context, prompt version and option-order policy.
- Starting adapter and decision-head hashes, training recipe version, exact candidate file list and candidate tensor validation rules.
- Calibration and unknown-answer semantics, resource requirements, and compatible runtime identity.

Freeze the selected profile when a job is created, before upload or assignment. Do not let a later deployment-wide default change the model for an in-progress job. Existing jobs without the new field retain their historical identity and interpretation.

A new client image job starts from the published Imajev checkpoint. It does not inherit the Real-IAD experimental adapter or a different client's adapter. The tested recipe continues the published rank-64 LoRA and decision head with the Qwen base and vision tower frozen. It is not an instruction to apply a JevK5 adapter to a vision model or to stack arbitrary adapters.

The image candidate contains adapter configuration, adapter safetensors, decision-readout safetensors and metadata, and versioned model/calibration metadata. The release fingerprint covers these files plus base/runtime/preprocessor identity. Existing Kev and JevK5 formats remain readable with unchanged fingerprints.

## Image assets and private storage

Use an additive Supabase table for asset metadata and a private image bucket. Reuse existing owner authentication and early-access checks. The DigitalOcean gateway/coordinator handles metadata and authorization; image bytes upload directly using immutable signed Storage URLs.

An asset has an opaque ID, owner, purpose (temporary prediction or training job), job ID when applicable, state, storage location, byte count, source SHA256, decoded/canonical-image SHA256, format, dimensions, creation and expiry timestamps. Original filenames are display/mapping metadata, never filesystem paths or trusted URLs.

Proposed API additions are `POST /v1/image-assets` to create bounded upload slots, `POST /v1/image-assets/{id}/complete` to verify a completed upload, and `DELETE /v1/image-assets/{id}` for unused assets. The browser authenticates with its Supabase session; programmatic clients use their existing Zils API key. Both paths resolve the same owner and enforce the same access and quota checks. No browser receives a service-role key.

Finalization verifies actual bytes, file type, decoded dimensions and hashes, normalizes EXIF orientation, converts supported color modes to RGB, and removes metadata from the canonical image. Apply orientation before discarding metadata. Only verified canonical bytes reach model code. Enforce limits before and during decoding, rather than trusting MIME types or extensions.

Initial proposed limits: 10 MiB encoded input, 16 million decoded pixels, and 8,192 pixels on either edge; one image per example; at most 1,024 training, 256 calibration and 512 test examples; 1 GiB aggregate canonical image bytes per training job. Reject oversized canonical output. The existing maximum active-job count continues to apply. These are product limits to validate under load, not measured guarantees about every miner.

Keep model preprocessing aligned with the experiment: full image, no task-specific crop, minimum 65,536 and maximum 400,000 processor pixels, and at most 4,096 total model tokens per question. The runtime performs the pinned processor operation; browser thumbnails are never the training input.

Temporary prediction assets expire after 24 hours. Incomplete drafts expire after 24 hours. Training assets and manifests are retained for 30 days after a terminal job state, with the retention date visible to the customer. Accepted model artifacts remain until explicit model retirement; their manifests preserve provenance hashes, not a hidden promise of retained training images. A cleanup worker deletes expired bytes and marks their records expired, with idempotent retries. Signed read URLs expire within 10 minutes. Reject expired assets even if stale metadata is cached.

New policies must protect image objects despite any existing broad Storage policies. Unauthorized, absent and expired asset references return the same not-found behavior. The runtime accepts only trusted internally constructed references to verified assets, never arbitrary customer-supplied remote URLs.

## Dataset format, splitting and exports

Each image example adds a verified image-asset reference to the existing ID/group/family/state/question/label record. The frozen manifest binds the asset's canonical content hash, pixel dimensions, purpose and preprocessing version. Image identity participates in the example fingerprint: the same question and empty state on two different photos is valid. Identical canonical image bytes or decoded pixels across splits are rejected; hashing only filenames or question text is insufficient.

The customer supplies item/group identity when several photographs may show the same item. Server validation is authoritative even when the browser has already checked the data. Automatic splits reuse the existing training wizard’s split proportions, are stratified where possible and group-preserving, and record the exact proportions, seed and group assignments in the frozen manifest. Refuse a split that cannot represent every outcome in training, calibration and test; explain which class needs more independent examples. Group labels do not prove physical independence, and the interface does not claim otherwise.

For an upgrade, verify the new holdouts do not reuse available image hashes from the predecessor's training/calibration manifests. Preserve the existing requirement to name the intended predecessor explicitly and compare the exact accepted version. Hash checks do not replace customer responsibility for semantic or physical overlap.

Approved miners receive only training rows and short-lived access to the corresponding training image assets. Do not issue a common archive or URL covering calibration/test objects. Validators retain separate credentials and fetch holdout images privately. Reissue download access only while the assignment lease remains valid; cancellation blocks new grants and submissions but cannot recall downloaded bytes.

## Worker scheduling and image training

Add operator-approved model/profile capabilities to worker registration. A worker's self-reported GPU model alone is not proof of compatibility. Probe the pinned runtime, image decoding, candidate save/reload and maximum supported input configuration before admitting it to the image pool.

Selection must match a job's frozen profile before granting its lease or image access. Extend the queue's selection and claiming transaction, not just the frontend filter. Preserve signed requests, lease fencing, attempt bounds and cancellation. The processor/evaluator also resolves references by job profile instead of one global active reference.

Begin with one approved image worker and one validator. A miner may support text, images or both. Different hardware is allowed when it passes the selected profile's measured capacity requirements. A smaller image base would be a separately pinned profile with its own adapters and evaluation, not a silent substitution for Imajev 4B.

Initial image recipe: one epoch, batch one, accumulation four, AdamW learning rate 2e-5, zero weight decay, gradient clip one, FP32 LoRA/head parameters and BF16 frozen base. Use actual accumulation size for a partial final batch. Preserve native unknown in the training loss and evaluator. Verify finite tensors, nonzero gradients, expected trainable parameter names and counts, and candidate reload consistency.

Training/evaluation processes share the existing compute lock and capacity gates. Never stop the text runtime to make room. Validate per-phase budgets with the maximum admitted image/context configuration. The pilot measured 64-image training at about 43 seconds and 256 at about 140 seconds on one RTX4090; these are not customer latency promises or minimum-hardware certification.

## Evaluation and acceptance

Reuse the current base-versus-candidate and incumbent-upgrade rules, with image-aware inputs and unknown-aware scoring. Fit candidate confidence only on calibration data; never choose a threshold from test results. For upgrades, preserve the exact incumbent's serving calibration as in the existing version-selection contract.

Report overall and per-class accuracy, recall, confusion counts, unknown rate, Brier loss and NLL with their denominators. Score Brier/NLL using the full native distribution including unknown, and count abstentions as incorrect for answerable examples. Do not obtain an apparent improvement by dropping unknown mass or hiding review cases.

For binary classification, extend the frozen policy with a named positive class, minimum positive recall and maximum normal false-positive rate. Require these controls in the image-training form instead of silently choosing quality targets on behalf of the client. Existing accuracy and Brier requirements still apply. Multi-class jobs use optional explicit per-class minimum recall. Candidates must pass every requested constraint before ordinary winner selection; no qualifying result preserves the existing active version.

The research result of 84% accuracy is evidence to build the workflow, not a production acceptance target. The 256-image run tied the 64-image run while detecting fewer defects; acceptance tests must include this failure case.

## Prediction API and separate runtime

Extend `/v1/systemone` with an optional `images` list containing exactly one verified asset ID for this release. Existing text request and response shapes remain valid and unchanged. Select by an explicit accessible model ID, never by guessing a model from MIME type. Reject images for text-only profiles and reject a missing image for the image profile.

The gateway checks asset owner/purpose/state and model ownership before preparing inference. Resolve the asset to a canonical hash and a short-lived internal read reference; bind preparation and execution to that same content and immutable model release. Runtime network access is restricted to the configured image store/internal gateway path. Prepare computes the image-plus-text token reservation using the actual pinned processor before GPU execution. Extend usage accounting to include image tokens and preserve existing budget/failed-request semantics.

Serve image releases in a separate Imajev process with its own health and catalog, reachable through the existing authenticated gateway. Each call selects an immutable release. One serialized execution queue owns all changes to the selected image adapter, decision head and calibration. Switching back to stock restores the published Imajev adapter and head; merely disabling LoRA would select the wrong baseline. Checkpoints and tensors are verified before entering the serving catalog.

Keep the upstream distinction between probabilities over allowed answers and `unknown_probability`; add `abstained` to image responses. The gateway must not discard unknown or fail its response validation because of it. Use the upstream pinned conversion semantics and cover them with tests. When abstained, the frontend displays “Needs review” and does not present the most likely allowed label as an automated decision. Confidence is not displayed as a guaranteed correctness rate.

Image requests support the realtime lane initially. Existing text bulk calls remain unchanged. Reject image bulk requests explicitly until asset lifetime, owner checks, frozen content and retry behavior are implemented for durable batches.

Do not assume the current shared GPU can host two persistent bases plus a training process. Admission considers all loaded models and worst-case activations. With insufficient capacity, jobs queue or the image feature remains unavailable. A separate image-serving GPU is the operational option for predictable simultaneous text/image service; provisioning or paid capacity is a deployment decision.

## Activation, compatibility and rollout

Extend publication and runtime verification by model profile. An accepted image adapter receives the existing account-bound immutable model ID and, for versioned tasks, the existing stable task alias. Only mark a job ready after the image runtime verifies the exact release and the gateway registers it for the correct owner. Preserve stale-upgrade checks and prevent cross-model-family predecessor selection.

The registry gains versioned capability metadata without changing the meaning or fingerprints of existing entries. Normalize absent capability fields to the legacy text contract and preserve atomic reload and immutable-release validation. New image features are disabled by default until storage, gateway and runtime checks pass.

Deliver the vertical in three reviewable slices: (1) verified assets, stock image runtime, gateway and signed-in “Try an image”; (2) image-labelled intake, manifests, compatible workers and held-out evaluation; (3) publication, private activation and “Try your model.” Every slice has runnable acceptance evidence; the last slice is required for the requested client-training outcome.

Roll out additive database resources and backend compatibility before enabling UI controls. Existing text model names, saved jobs, keys, artifact downloads and behavior remain supported. Turning off image feature flags stops new image admissions and UI entry points without changing text routing. Existing running jobs finish or follow the normal cancellation path; preserve historical outcomes and artifacts.

## Verification required before calling this complete

- Contract/storage tests: ownership, expired assets, signed uploads, immutable hashes, changed bytes, malformed/oversized images, canonical orientation, and exact/group overlap; no holdout URLs in worker responses.
- Queue/training tests: compatible claims, mismatch refusal, lease expiry/cancellation, capacity deferral, nonzero image-driven updates, candidate artifact validation, save/reload, unknown scoring and refusal of a higher-false-negative candidate when recall fails.
- Gateway/runtime tests: legacy text compatibility, model/asset tenant isolation, real image-token reservation, stock/customer adapter/head switching, unknown response preservation, runtime fingerprint mismatch and explicit image-bulk rejection.
- Browser tests: upload, thumbnail/label review, understandable validation errors, training progress, baseline/candidate metrics, no-qualifying outcome and accepted-model photo prediction on desktop and mobile.
- One real private GPU rehearsal through the actual new queue and API: upload labelled authorized images, train, evaluate, activate only if policy passes, predict on fresh images, verify a second account cannot access the assets/model, and verify text service identity and requests remain healthy. A negative candidate remains a valid test result; never relax its policy after seeing test results to force activation. Test the accepted path with a fixture plus an independently justified real qualifying run before claiming real successful activation.

## Evidence and design boundaries

The prior isolated Imajev research trial established native image input, trainable LoRA/head weights, saved artifact reload and held-out improvement over the published checkpoint on one product category. It did not establish hosted image-job plumbing, all-hardware support, production concurrency or general customer accuracy.

Model source: [Imajev repository](https://github.com/mohit67890/imajev). Use its pinned revision, not a floating download. Customer training uses authorized client images. Research-only benchmark images are not a shared production fine-tuning dataset.

This document approves no production cutover, new paid compute, live-model interruption or irreversible migration. Those changes require a concrete staged result and the applicable deployment authorization. Implementation work after spec and plan review remains within the approved image-vertical scope.
