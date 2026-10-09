# DigitalOcean Spaces storage for Zils

Status: approved design; implementation plan pending review. No Spaces resources
or production changes have been made for this migration.

## Outcome and scope

Use DigitalOcean Spaces for Zils file storage while retaining Supabase Auth,
Postgres job state, billing, balances, API keys and early-access records. Cover
training datasets, image uploads, candidate/accepted adapters and bulk-job inputs.
Do not change model weights, training recipes, acceptance thresholds or customer
pricing. GPU machines continue to execute training and inference.

The immediate success criterion is delivery of the existing approximately 490 MB
image checkpoint through private Spaces storage and evaluation by the hosted test
queue, with unchanged account isolation and exact billing. A missing candidate
cannot count as a successful quality evaluation.

## Selected approach

Add a storage boundary beneath the existing `signed`, `upload`, `download`, `exists`
and `remove` calls. Keep the database/auth client separate from provider-specific
file operations. Use the supported S3 SDK for Spaces; do not implement request
signing cryptography in the application.

One private bucket per environment contains the four existing logical namespaces:
training data, training models, images and bulk inputs. Test and production use
different scoped keys. Browsers and miners receive expiring grants, never Spaces
credentials. No public ACL or CDN is needed for these private files.

A model-only move would unblock the current large upload but leave two storage
systems in normal use. A direct URL replacement would break expiry parsing,
browser/runtime allowlists and immutable-upload assumptions. The selected approach
migrates the shared file boundary, with legacy reads during transition.

## Upload integrity and recovery

Preserve the existing rule that a completed upload cannot be replaced using an
old grant. Do not assume that Spaces implements every AWS conditional-write
header: its documented PutObject headers do not establish that guarantee.

Use service-owned multipart completion. A new service-only object catalog in
Postgres binds each logical bucket/path to a unique physical key, upload ID,
lease token, grant expiry, committed size and state. Grant callers permission to
upload part 1 only; current file limits fit a single part. The trusted service
lists and completes that exact part before marking the object available. Once
completed, the old part grant cannot write the final object. Retain the existing
download size limits and SHA-256 verification at data/model acceptance boundaries.

Catalog transitions are fenced and idempotent. A lost completion reply is
reconciled against the unique physical key. Concurrent claim/completion attempts
cannot replace the committed locator. Do not return a read URL while completion
is uncertain. Abort abandoned multipart uploads after their grants and leases
expire. Test this behavior against real Spaces before enabling new writes.

The catalog is an additive database migration, prepared and tested locally before
requesting permission to apply it to the hosted test project. Production application
of that migration is a separate rollout step.

## Callers and private access

- Return explicit grant expiry; image finalization must no longer decode a
  Supabase-specific token to obtain it.
- Keep PUT upload descriptors usable by existing worker flows. Completion occurs
  through the provider's availability/download path, so object existence means
  completed bytes rather than merely an allocated upload.
- Configure exact permitted storage origins and path families separately from the
  Supabase auth URL in the website and image runtime. Preserve URL, credential,
  redirect, asset-ID and content-hash validation. Permit the legacy origin only
  during migration.
- Change bulk-input cleanup to use the storage boundary rather than directly
  calling a Supabase Storage HTTP route.
- Exclude Spaces secrets from GPU training subprocess environments. Keep private
  browser uploads free of session headers and cookies.

## Existing files and rollback

New records explicitly identify their storage provider. An unregistered legacy
path remains readable from Supabase. A recorded deletion leaves a tombstone so
fallback cannot resurrect a removed legacy object.

The transfer tool first produces an inventory and dry-run report. It copies
objects without removing their source, checks byte length and SHA-256, then
switches each catalog entry to Spaces. New uploads and in-flight jobs must not
race a bulk cutover. Source deletion requires a separate verified cleanup step;
this design does not authorize deleting existing customer files.

Rollback stops new Spaces uploads but preserves reads of already committed Spaces
objects. Changing a global provider flag must not make those objects disappear.
Keep Supabase sources until the corresponding copy and access checks pass.

## Retention and cost

Retain current photo and bulk-input expiry rules, adapted to the new provider.
Abort abandoned multipart uploads and remove orphaned temporary objects safely.
Keep active/accepted adapters available until explicitly retired; do not apply
a blanket bucket-expiry rule to them. Automatic retirement of older customer
models and revised customer storage quotas are separate product decisions.

Spaces currently starts at $5/month with 250 GiB storage and 1 TiB outbound
transfer; additional storage and transfer are metered. The approximate 0.5 GB
checkpoint size does not shrink by changing providers. Copies, failed candidates
and incomplete uploads must be included in storage accounting. Avoid enabling
bucket versioning as an unbounded duplicate-retention workaround.

## Verification and rollout

1. Add failing tests for the provider boundary, fenced object state, lost replies,
   stale grants, duplicate uploads and cleanup. Run real disposable PostgreSQL
   checks for catalog access and concurrency.
2. Exercise website uploads/downloads with exact Spaces origins and validate that
   other origins, objects and accounts are denied. Preserve Supabase mode in the
   existing text, billing and key regression tests.
3. Provision private test storage, verify no anonymous access, stale-part rejection,
   CORS and credential isolation; upload/download/hash the real large checkpoint.
4. Apply the approved test migration and run hosted upload → signed GPU miner →
   actual candidate evaluation → stock/owned prediction as applicable. Preserve a
   genuine quality rejection and report accepted activation as unverified unless
   a real qualifying candidate is served.
5. Review the copy inventory and production migration before cutover. Verify live
   text identities, auth, balances and API keys before and after. Keep production
   image admission off until its independent launch requirements pass.

## References

- [Spaces S3 compatibility](https://docs.digitalocean.com/products/spaces/reference/s3-compatibility/)
- [Spaces API: multipart upload and access control](https://docs.digitalocean.com/reference/api/spaces/)
- [Spaces pricing](https://www.digitalocean.com/pricing/spaces-object-storage)
