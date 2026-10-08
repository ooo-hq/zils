# Image training and rollout

The approved profile is `imajev-4b-v1`. Every job freezes its model profile, task,
answer order, canonical image hashes, item groups, split membership and acceptance
policy. A worker may train only profiles explicitly qualified by an operator.
Self-reported hardware or a worker's list of supported profiles is not authorization.

## Pinned runtime and recipe

| Component | Immutable revision |
| --- | --- |
| Base | `Qwen/Qwen3.5-4B`, `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Published adapter/head | `mohit67890/imajev-4b`, `f8d8234cebc6c99065c07731e59716dc0a6e27ab` |
| Upstream runtime | `ccf586d43d2a580319b6535c893668904d909eb9` |
| Preprocessor / prompt | `zils-image-rgb-png/v1` / `imajev-readout/v1` |

`requirements/imajev.txt` pins the Python runtime dependencies. `zils/imajev-pins.json`
records every required upstream file hash; `zils/imajev-schema.json` records config
and tensor names/shapes/dtypes for CPU publication validation. Include both JSON files
when packaging the application.

In a new isolated Python environment, install the pinned requirements and application
requirements, then run `python -m scripts.download_imajev --out PRIVATE_NEW_DIRECTORY`.
It downloads only pinned files and writes a verified `release.json`. Offline runtime
loads verify hashes and imported upstream module location. Create a fresh training
reference with `zils.imajev.create_reference(release, destination)`; never initialize
new training from a prior experiment or a client's adapter. The fresh-download path
requires a separate clean-host installation check before deployment; rehearsals use
existing verified caches.

The recipe uses a BF16 frozen base/vision tower, FP32 language LoRA of rank 64, and an
FP32 decision head: **122,552,320 trainable parameters**. It runs one epoch, batch 1,
accumulation 4 (including correctly scaled partial batches), AdamW at `2e-5`, zero
weight decay and gradient norm clipping at 1. Reload validation checks saved LoRA and
head together. Candidates contain both safetensors files, both JSON configurations,
and `model.json`; an accepted download also includes `release.json`.

Up to 1,024 training, 256 calibration and 512 test photos may be submitted, with a
1 GiB canonical-image total. Workers receive training-only image grants bound to the
current lease/profile. Calibration and test labels/images remain validator-only.
The validator fits one temperature on calibration logits over the full distribution,
then freezes it for test scoring. It checks accuracy, positive skill and required
Brier improvement. Binary tasks additionally require an explicit positive class,
minimum recall and maximum false-alarm rate; multiclass tasks may require per-class
recall. Missing denominators cannot satisfy a quality gate. Incumbents are compared
without re-training or re-calibrating them. A late upgrade cannot replace a newer
accepted version.

## Worker and service setup

1. Apply migrations through `202610080004_image_processing_profiles.sql` in order,
   after the existing queue/API/early-access migrations. Verify private bucket RLS,
   server-only RPCs and retention cleanup in staging. No migration was applied to
   production by this implementation.
2. Qualify each image worker against the exact `models.profile_identity(IMAJEV)`
   hashes. Store an operator-owned `zils_worker_profiles` record with `verified_by`,
   measured `min_free_mib` and evidence: at least four examples, configured maximum
   400,000 processor pixels and 4,096 input tokens, decoder/reload/finite-gradient
   checks, peak reserved bytes, probe/trainer SHA-256 and a 1–3,600 second deadline.
   Memory admission must be at least 12,288 MiB and peak reserved memory + 512 MiB.
   A small-image probe is not maximum-context qualification.
   The existing worker-registration command automatically grants only the two
   existing text profiles, including workers registered after migration. Image
   qualification still requires the operator's measured evidence.
3. Supply workers with the pinned reference using repeatable `miner.queue --reference`
   arguments and set `ZILS_IMAGE_RUNTIME_PYTHON` and `ZILS_IMAGE_REFERENCE`. Use one
   shared compute lock across training/evaluation services. Image subprocesses inherit
   only runtime paths, cache, locale and GPU settings; account/signing credentials
   are excluded. The processor's `--additional-reference` enables image validation
   alongside the unchanged primary text profile.
4. Run a separate `python -m zils.image_server --reference PRIVATE_RELEASE --releases
   PRIVATE_CLIENT_RELEASE_ROOT --image-store-origin https://YOUR_STORAGE_ORIGIN --port
   8931`. Keep it private. Configure gateway entries and the workflow's `image` block
   with its separate endpoint/token/release root. Accepted activation verifies the
   exact fingerprint before atomically registering an owner-only model and task alias.
   One GPU slot serially replaces adapter, decision head and temperature. Returning
   to stock reloads the published LoRA and head; it never disables LoRA as a shortcut.
5. Enable `ZILS_IMAGES_ENABLED=1` only after owned stock prediction works. Enable
   `ZILS_IMAGE_TRAINING_ENABLED=1` only after qualified workers, maximum-size admission,
   evaluation and activation are verified. Keep `python -m zils.image_cleanup` running.

[Placeholder environment](../examples/image-vertical/service.env.example) and
[workflow configuration](../examples/image-vertical/workflow.json) keep credentials
out of checked-in files. The existing text routing and API keys remain in place.

## Rehearsal and evidence

`python -m scripts.rehearse_image_vertical --config PRIVATE_CONFIG --dataset
PRIVATE_DATASET --out NEW_PRIVATE_OUTPUT` requires two authorized owners, explicit
isolated endpoints, expected runtime fingerprints, environment references for secrets,
and a frozen policy. Dataset files are `train.jsonl`, `calibration.jsonl`, `test.jsonl`
and `fresh.jsonl`; rows contain `id`, `group_id`, `family`, `state`, `question`, `label`
and a relative `image` path. The driver snapshots and hashes source bytes/policy
before creating a job, refuses an existing output directory, records each phase and
retains redacted failures. It never cleans up unrelated assets/jobs or controls live
services. Its own private evidence remains for inspection and normal retention.

The integration harness in `tests/image_vertical_fixture.py` uses the production
coordinator, signed worker protocol, validator, publisher, image runtime and gateway
against isolated HTTP account/storage doubles. Actual queue transitions and owner
policies are additionally exercised in disposable PostgreSQL. Fixture model scores
prove contracts, not accuracy. The real-mode harness executes pinned CUDA training
and evaluation and explicitly labels its storage doubles in the output.

The measured release evidence and operational limits are recorded in
[image rollout evidence](image-rollout.md). The existing 24 GB host may run sequential
private GPU phases while its text runtime stays available. This is not proof that
persistent image serving, text serving and adapter training fit simultaneously.

## Deployment and rollback

Keep both image flags off until maximum-context worker qualification and GPU allocation
are resolved. Stage the migrations, application packages, separate image runtime,
cleanup service and private registry configuration. Check text PID/health/fingerprint
before and after image stock prediction, one client training job, another-account
denial and an accepted activation. Confirm queue deferral when memory is insufficient.
Only then approve production admission and any additional GPU spending.

To stop new image work, disable the image training flag and then the image prediction
flag as needed. Preserve artifacts, private records, ongoing lease recovery and cleanup;
leave all text entries and services running. Do not roll back by dropping shared tables
or replacing text model identities. Registry rollback preserves immutable releases and
changes a task alias only through the existing owner/version checks.
