# Private file storage with DigitalOcean Spaces

The optional Spaces provider stores training datasets, model artifacts, image files
and bulk inputs. Supabase remains responsible for authentication, Postgres, billing,
API keys and early-access records. Installing this code does not enable Spaces or
production image training.

## Setup

From a fresh clone, install the API dependencies with Python 3.13:

```sh
python3.13 -m venv .venv
.venv/bin/pip install -r requirements/api.txt
```

Prepare a private bucket per environment and bucket-scoped read/write credentials.
Keep public access, CDN and bucket versioning disabled. Apply
`supabase/migrations/202610090002_spaces_storage.sql` through your approved database
migration process before enabling the catalog. The catalog is service-only; browsers
and miners must never receive its credentials or Spaces keys.

Configure the trusted API and file-processing services using protected environment
variables (see `.env.example`):

```dotenv
ZILS_STORAGE_CATALOG=on
ZILS_STORAGE_WRITE_PROVIDER=spaces
ZILS_SPACES_ENDPOINT=https://nyc3.digitaloceanspaces.com
ZILS_SPACES_REGION=nyc3
ZILS_SPACES_BUCKET=your-private-test-bucket
ZILS_SPACES_ACCESS_KEY_ID=your-bucket-scoped-key
ZILS_SPACES_SECRET_ACCESS_KEY=your-bucket-scoped-secret
```

The regional endpoint and bucket must match the provisioned location. Configure
`NEXT_PUBLIC_ZILS_SPACES_URL=https://your-private-test-bucket.nyc3.digitaloceanspaces.com`
on the website. Add the same exact origin to the image runtime using
`--spaces-store-origin`; retain its explicit legacy `--image-store-origin` during
migration. Authentication continues to use the Supabase URL.

Use exact CORS origins for each website environment. Allow `PUT`, `GET`, `HEAD` and
request header `Content-Type`. The server reads `ETag`; browser uploads do not need
it exposed. Do not allow `*` origins or account
credentials. The browser sends neither authorization headers nor cookies to Spaces.
CORS is not access control: private objects always require a signed grant.

Run the provider preflight before admitting jobs. A Spaces grant lasts 600 seconds;
legacy upload expiry is read from the trusted provider response and can be longer.
Clients may upload only part 1. A trusted service checks and completes that part.
Old part grants cannot overwrite the completed object. Recovery uses its unique
physical key and a database lease; uncertainty fails closed.

Set `ZILS_WEB_ORIGIN` to the exact permitted HTTPS test frontend origin, then run:

```sh
.venv/bin/python -m scripts.check_spaces --source /path/to/checkpoint.safetensors --report .private/spaces-check.json
```

This uses real provider requests for anonymous denial, CORS, consumed multipart
replay, a lost catalog reply and a full checkpoint download/hash. It creates unique
test objects and schedules their deletion through the catalog. Run cleanup after
the grants and deletion grace expire. Local protocol fixtures do not establish
real-provider compatibility; retain the real report before enabling writes.

The preflight checks each CORS method separately because Spaces returns the requested
method rather than the whole configured list. Repeating an already completed multipart
request may return the original object idempotently. The check requires its original
ETag, rejects stale part writes and verifies unchanged bytes; a provider error alone
does not establish immutability.

## Copy existing files

Export `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` through your protected service
environment. Inventory and copy dry-runs do not require Spaces credentials or write
catalog rows. Reports are atomically written with mode `0600`; keep them private.

```sh
.venv/bin/python -m scripts.migrate_storage inventory --output .private/inventory.json
.venv/bin/python -m scripts.migrate_storage copy --inventory .private/inventory.json --dry-run --output .private/dry-run.json
.venv/bin/python -m scripts.migrate_storage copy --inventory .private/inventory.json --apply --output .private/copy.json
.venv/bin/python -m scripts.migrate_storage verify --inventory .private/copy.json --output .private/verified.json
```

Review eligibility before applying. Active jobs and unrecognized owners are deferred.
Pause new admissions for the final inventory: paginated listing is not a snapshot.
Reinventory after active jobs finish to capture later uploads. Each apply rechecks the
owner state and source metadata, streams to private disk, fingerprints the source,
then copies to a unique generation. Reads continue from Supabase until an independent
Spaces download matches both byte length and SHA-256. Migration leases renew during
large transfers. A stale lease, deletion or mismatch prevents publication.

A lost reply may leave a retry entry. Preserve the report and rerun the copy command;
committed copies reconcile without duplicates. Ambiguous unfinished attempts wait
for their lease to expire before a fresh generation can replace them. Neither copy
nor verification removes legacy source files. Retirement of retained sources is a
separate operational decision.

## Retention and capacity

```sh
.venv/bin/python -m scripts.migrate_storage cleanup --dry-run --output .private/cleanup.json
.venv/bin/python -m scripts.migrate_storage cleanup --apply --output .private/cleanup-applied.json
```

Schedule cleanup using the existing service runner after reviewing a dry-run. It
finishes previously authorized deletions, abandons unfinished transfers only after
24 hours and expired leases/grants plus a grace period, and scans old unreferenced
objects and multipart uploads. Unknown/active owners are not abandoned. Completed
current objects and active/accepted adapters are preserved. A failed catalog lookup
is an error, never permission to delete. Late writes to retired generations remain
eligible for later orphan scans.

Bucket-scoped application keys cannot configure lifecycle rules. An administrator can
set an additional one-day incomplete-multipart abort rule using the
[Spaces lifecycle API](https://docs.digitalocean.com/products/spaces/how-to/configure-lifecycle-rules/).
Keep that administrative credential out of application services. Without a custom
rule, [Spaces expires incomplete uploads after 30 days](https://docs.digitalocean.com/products/spaces/details/limits/);
the application cleanup schedule remains necessary for its shorter retention policy.

Existing image and bulk retention workers continue to decide which logical files
expire. Explicit deletion hides a file immediately with a durable tombstone, then
physical cleanup waits for outstanding grants. Those normal deletion paths also
remove retained legacy copies. The bulk migration command itself does not delete
legacy source copies. Do not add a blanket bucket-expiry policy for model artifacts.

Budget for both providers during migration, incomplete multipart parts, failed
candidates, private transfer and local staging disk. Each concurrent copy needs space
for both source and verification download. Moving a checkpoint does not compress it.
Use current [Spaces pricing](https://www.digitalocean.com/pricing/spaces-object-storage)
and measured object totals rather than estimating from job counts alone.

## Rollback

1. Stop new training/upload admissions and let active copies settle.
2. Set `ZILS_STORAGE_WRITE_PROVIDER=supabase` on trusted services.
3. Keep `ZILS_STORAGE_CATALOG=on`, Spaces read credentials and both permitted origins.
4. Verify old and new objects, authentication and billing before reopening admission.

Catalog-off is **not** rollback once Spaces objects exist. Each ready object's catalog
record selects its provider, so changing the write default preserves existing reads.
Pending uploads continue on their recorded provider. Rollback cannot make Supabase
accept objects above its configured size limit; keep image admission off if that
limit would block model delivery.

## Production rollout gates

An isolated hosted preflight transferred a 487,648,432-byte pinned image adapter
through real Spaces and independently downloaded identical bytes and SHA-256.
Anonymous reads and stale multipart part writes were denied, exact-origin CORS
passed for all three configured methods, and a lost catalog commit reconciled.
One existing dataset, model and image object also passed verified copying and
rollback reads, with each Supabase source retained. These are storage checks;
hosted signed-miner delivery and evaluation remain a separate gate.

Ship the compatible website first. Apply the separately approved production catalog
migration and deploy backend services with legacy writes. Verify sign-in, early
access, key access, balances and live model identities. After isolated Spaces and
GPU verification passes, review production inventory/dry-run and provision a separate
private production bucket and scoped keys. Enabling production writes and copying
files requires the production cutover approval. Update the website privacy statement
when the deployed processor actually changes. Verify old/new object access and the
same account/model checks after cutover. Keep image admission disabled until its
independent launch gates pass; storage verification does not demonstrate model quality.
