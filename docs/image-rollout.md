# Image vertical rollout evidence — 2026-10-08

The image vertical is merged; worker memory and upload-recovery follow-ups are staged.
**Production image admissions remain off.** No production migration, registry update, model deployment,
worker qualification, paid GPU allocation or live text-service interruption occurred.

## Verified behavior

- Client image upload, label review, grouped splits, training submission, aggregate
  results, activation states and owned private prediction are implemented alongside
  text. Account changes clear private state and abort pending requests.
- Local HTTP integration covers accepted publication/prediction, a rejected candidate,
  activation failure, other-account denial, text-worker denial and withheld holdouts.
  These fixture scores are contract evidence, not accuracy measurements.
- Disposable PostgreSQL checks cover private storage/RPCs, quotas, upload/finalization
  races, retention, worker qualifications, revoked/stale leases, concurrent claims,
  and profile-aware processing that skips image work without available capacity.
- GPU state switching restores stock probabilities exactly (maximum difference 0.0,
  tolerance 1e-6). The 31.84-second check used real pinned weights and a modified head
  with synthetic acceptance evidence; it proves isolation, not trained model quality.
  Cold client activation took 0.925 seconds; two warm predictions took 0.283 seconds.

## Actual pipeline rehearsal

The new signed queue, trainer, validator and gateway paths ran against isolated HTTP
account/storage doubles using pinned Imajev CUDA execution. Actual database semantics
were separately tested in PostgreSQL. The research dataset was an authorized Real-IAD
audio-jack subset with 32 training photos, 12 calibration photos, 20 held-out test photos
and 2 reserved fresh photos. Groups were kept disjoint. This is a small functional
rehearsal, not a replacement for the earlier 200-case research benchmark.

Before any inference, the run froze accuracy >=80%, defect recall >=80%, false alarms
<=20%, positive skill and absolute Brier improvement >=0.01. No target was relaxed.

| Test result | Calibrated stock | Trained candidate |
| --- | ---: | ---: |
| Accuracy | 75% (15/20) | 80% (16/20) |
| Defect recall | 50% (5/10) | 60% (6/10) |
| False alarms | 0% (0/10) | 0% (0/10) |
| Unknown decisions | 0/20 | 0/20 |
| Full-distribution Brier loss | 0.351057 | 0.297833 |
| Full-distribution log loss | 0.548583 | 0.449148 |

The candidate correctly received **no qualifying model**, because defect recall missed
the frozen minimum. It was not published or registered. Consequently, successful
activation of a genuinely qualifying trained GPU candidate remains unverified; the
accepted integration and GPU state-switching checks above do not erase that limit.

The training loop took 20.67 seconds; trainer startup/save/reload included took
37.86 seconds. Upload-to-evaluated-result took 142.10 seconds. Total private harness
setup, pipeline and post-run image capacity check took 173.86 seconds. Training peaked
at 11,741,954,048 reserved GPU bytes. The trained LoRA/head reloaded with zero maximum
logit difference. These times apply to 32 photos at 442 tokens each, not the maximum
customer dataset or context.

These wall times were recorded before final review added a disposable download process
to enforce network deadlines. That change adds transfer startup overhead; the recorded
training-loop measurement and quality scores are unaffected.

The live text PID, invocation, release identity and fingerprint were unchanged before
and after. Once the private image runtime was resident alongside text, image training
admission returned false. Sequential phases fit; concurrent persistent image serving
plus text serving plus training was not demonstrated on this 24 GB host.

A separate bounded capacity probe decoded a 16-million-pixel source, prepared the
400,000-pixel processor budget and exactly 4,096 input tokens with 16 answers. Training
then hit `OutOfMemoryError` under the isolated half-GPU allocation, at 12,058,624,000
peak reserved bytes. The 36.32-second probe registered no worker and left live text
health, process and model identity unchanged. That allocation did not qualify for the
advertised maximum training envelope; the small rehearsal did not establish capacity.

A later maximum-envelope probe used CPU activation offload and a 55% GPU allocation.
Four examples reached 4,096 tokens each at the same 400,000-pixel processor budget,
with a decoded 16-million-pixel source, finite gradients and exact save/reload logits.
Peak reserved memory was 13,279,166,464 bytes; one optimizer step took 35.65 seconds
and the complete probe took 76.86 seconds. The live text identity remained unchanged.
This passed the isolated capacity test; no production worker was registered. The
qualified configuration must include the tested offload and allocation settings.

Private evidence is retained in the implementation workspace under
`.private/image-e2e/last-status.json`, `.private/image-switching/last-status.json` and
`.private/image-capacity/last-status.json`. Failed setup/launcher runs are retained too;
the successful rehearsal kept the same dataset and policy. Raw photos, credentials,
private prediction reports and host details are excluded from this document.

## Hosted Supabase verification — 2026-10-09

An isolated hosted project received the image migrations, including request-owned
upload recovery. A real Supabase account uploaded and verified all 64 dataset photos;
a temporary finalization failure recovered on retry. A separate signed GPU worker
downloaded the 32 training photos and trained the pinned recipe for eight optimizer
steps. Training took 19.36 seconds (37.44 seconds including setup and save/reload),
reserved 11,653,873,664 GPU bytes at peak, and reloaded with zero logit difference.

The trained adapter did **not** reach evaluation. Its largest file was 487,648,432
bytes, while the project's global upload limit was 50 MiB. The private model bucket's
512 MiB setting does not override that global limit. Increasing the global setting
was rejected with HTTP 402 because the project requires a paid plan. All three
assignment attempts failed, leaving the evaluator's candidate status `missing`.
The resulting `no_qualifying_model` is an infrastructure failure, not a measurement
of this candidate's quality. The earlier measured recall failure above is a separate
run and must not be substituted for these missing results.

Real stock image predictions passed with both a Supabase session and the existing
API key. A second account could not use the first account's image. Test-mode credit
deductions matched 738 billable input tokens exactly, and no reservations remained.
The included training run was consumed after the completed evaluation despite the
missing candidate; this run does not establish refund behavior for failed delivery.
The four live text services retained their process/invocation identities and model
fingerprint. Isolated image services stopped after the 762.25-second run.

No paid plan was purchased. Hosted candidate evaluation and accepted-model activation
remain unverified. Storage must accommodate the full pinned adapter before retrying;
check both [global and bucket file limits](https://supabase.com/docs/guides/storage/uploads/file-limits).

## Billing integration

The image branch now includes the deployed prepaid billing and Google/email account
flows. The restored early-access email signup remains unchanged. Disposable database
checks cover the upgrade from the billing schema to image tables, account/key access,
inference settlement and failure release, included and paid training runs, concurrent
submission and insufficient-credit rollback. These checks do not use live payments.

## Launch gates

1. Configure and register a worker using the successful maximum-envelope offload
   configuration. The isolated capacity probe alone does not reserve serving capacity.
2. Verify a fresh pinned installation and complete hosted candidate delivery/evaluation
   after resolving the global storage limit. Hosted uploads, signed GPU training and
   stock inference have passed; the full path has not.
3. Obtain a qualifying real candidate under a policy frozen before evaluation; verify
   its owned API prediction and second-account denial without changing thresholds.
4. Review migrations, private service configuration, cleanup and rollback described in
   [image training](image-training.md). Authorize production and any additional compute
   only after these checks. Enable prediction and training flags separately.
