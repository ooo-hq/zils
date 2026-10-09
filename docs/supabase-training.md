# Supabase training workflow

The training service connects a customer dashboard to a shared pool of approved
miners. Supabase provides Auth and Postgres job state. Private file storage defaults
to Supabase; the optional [DigitalOcean Spaces provider](spaces-storage.md) keeps
the same logical files and supports verified migration with legacy reads.
The Python coordinator handles authorization and issues storage URLs; dataset
and checkpoint bytes transfer directly to Storage. A separate processor validates
data and runs the existing validator. Miners claim assigned jobs over outbound
HTTPS, so they no longer need an inbound artifact server or a new bundle per job.

This is an implemented pilot workflow, tested with fixture miners,
a local Supabase HTTP/storage double, and a disposable real PostgreSQL database.
A deployed Supabase smoke test also verified authentication, customer isolation,
private signed uploads, processor validation, and cancellation using synthetic
accounts and two-row dataset splits. Real training through this queue has not
yet been validated. It does not deploy an inference endpoint, publish chain
weights, calculate cross-job emissions, or establish model-quality improvements.
The original local/testnet fleet commands remain supported.

For the JevK5 4B hosted base, use the [JevK5 queue setup](jevk5-queue.md), including
its CUDA requirement, model reference and artifact format. The commands below
describe the legacy Kev runtime; use `ZILS_TRAINING_MODEL=kev-0.8b-v1` with those
commands. Existing jobs retain their pinned model when the active base changes.

## Prerequisites and isolated project resources

Use macOS, Linux, or WSL 2, Python 3.13, and the [repository setup](../README.md#repository-setup).
Download the pinned model before starting the processor or miners. Use an
existing Supabase project with Auth and Storage enabled. For a first private
pilot, configure Auth for the intended customer accounts and redirect URLs.
When using Supabase Storage, choose a plan/global file limit compatible with the configured limits:
128 MiB per uploaded dataset split and 512 MiB per artifact, with a 512 MiB
aggregate checkpoint limit. Project-wide limits may be lower than bucket limits.

Review and apply `supabase/migrations/202609300001_training_jobs.sql` with the
Supabase SQL editor or a privileged PostgreSQL connection:

```bash
psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 \
  -f supabase/migrations/202609300001_training_jobs.sql
```

`SUPABASE_DB_URL` is a server-side connection string supplied by the operator;
never put it in the web app or a miner configuration. Apply this migration once.
It creates only the `fez_training_*` tables/functions, private `fez-training-data`
and `fez-training-models` buckets, and policies protecting those resources. It
neither resets the project nor changes unrelated data. A restrictive Storage
policy prevents existing broad client policies from exposing the training
buckets. Service-role access remains privileged.

Customers can select only their own job records and cannot directly change job
state. Miner registrations, assignments, replay nonces, and queue RPCs are
server-only. The coordinator verifies each customer session against Supabase
Auth. Miners instead sign requests with Bittensor hotkeys; they receive no
Supabase project credentials.

## Run the API and processor

```bash
mkdir -p .private
cp .env.example .private/training.env
chmod 600 .private/training.env
```

Edit the copied file with the project's URL and server service-role key. Set
`ZILS_TRAINING_API_URL` to the coordinator's reachable URL and `ZILS_WEB_ORIGIN` to
the exact browser origin. The example uses loopback for local development.
Load it in each coordinator/processor terminal:

```bash
set -a
. .private/training.env
set +a
.venv-kev/bin/python -m zils.coordinator serve
```

In another terminal with those same environment variables:

```bash
.venv-kev/bin/python -m zils.coordinator process \
  --state .private/queue-processor --reference models/reference --device cpu
```

Use `--device cuda` or `--device mps` on configured hardware. `--once` processes
at most one pending stage and exits. The API itself needs no GPU or checkpoint;
the processor does. They can run on separate machines with the same Supabase
project and API URL. Processors can recover abandoned database leases; each
processor directory has an exclusive process lock.

The API listens on `127.0.0.1:8910` by default. For remote operation, run it behind
an HTTPS reverse proxy with request/concurrency limits and configure the exact
public URL on both coordinator and miners. The built-in threaded HTTP server
is a development server; it is not an internet edge server. A persistent host,
such as a DigitalOcean host managed using `doctl`, can run the API under a service
supervisor. The processor belongs on hardware able to evaluate the pinned model.
No DigitalOcean resources are required or created by this repository.

## Connect the web app

In the independent [Zils website](https://github.com/ooo-hq/zils-web), configure:

```dotenv
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=replace-with-public-publishable-key
NEXT_PUBLIC_ZILS_TRAINING_API_URL=https://your-training-api.example
```

The web app also supports `NEXT_PUBLIC_SUPABASE_ANON_KEY` as a legacy fallback.
Only public keys belong in `NEXT_PUBLIC_*`. Configure the Supabase Auth redirect
allowlist to include the actual `/train` URL, and set the coordinator's
`ZILS_WEB_ORIGIN` to that site's origin. Redeploy/rebuild the web app after setting
public environment variables. The `/train` dashboard is implemented in the web
app repository, not this repository's standalone benchmark preview.

The current form accepts three JSONL files, already split by related source
records: training, calibration, and test. It does not automatically split an
arbitrary spreadsheet. Follow the [customer case format](customer-jobs.md#prepare-data).
It validates locally, asks for training-export permission, creates a job, uploads
each split directly with an immutable signed URL, and submits for server-side
validation. Interrupted uploads can resume missing objects; files already
uploaded cannot be replaced. Cancel and create a new job to change input data.

## Approve miners and run a queued miner

A job enters `awaiting_approval` after validation. It does not expose data to all
registered miners. First register an approved miner's public hotkey and a stable
UID using the coordinator environment:

```bash
.venv-kev/bin/python -m zils.coordinator worker --hotkey "$WORKER_HOTKEY" --uid 1
.venv-kev/bin/python -m zils.coordinator approve \
  --job "$JOB_ID" --hotkeys "$WORKER_HOTKEY"
```

`WORKER_HOTKEY` is the miner's public SS58 address; `JOB_ID` is the UUID displayed
in the dashboard. Supply up to sixteen distinct approved hotkeys. UIDs are local
queue identities, not a claim of chain registration. IDs are frozen in each
assignment. `worker --disable` disables future authenticated miner requests;
previously issued download URLs remain valid until expiry and downloaded data
cannot be recalled.

On the miner, create an ignored, mode-600 JSON configuration using its existing
wallet (the wallet must already exist on that machine):

Wallet configurations require the optional SDK installed with
`uv pip install --python .venv-kev/bin/python -r requirements/testnet.txt`.

```json
{
  "coordinator": "https://your-training-api.example",
  "hotkey": "REPLACE_WITH_PUBLIC_SS58_ADDRESS",
  "wallet": {"name": "REPLACE_WITH_WALLET_NAME", "hotkey": "REPLACE_WITH_HOTKEY_NAME"}
}
```

Save it as `.private/queue-miner.json` and run:

```bash
chmod 600 .private/queue-miner.json
.venv-kev/bin/python -m miner.queue \
  --config .private/queue-miner.json --state .private/queue-miner \
  --reference models/reference --device cpu
```

For a disposable local test identity, a configuration may use `seed` instead of
`wallet` and `hotkey`; generate it with `Keypair.create_from_seed` from a securely
generated 32-byte seed, keep it private, and register the resulting public address.
Never give miners the Supabase service-role key. Each miner downloads and hashes
its assigned training export, reuses the cached base model, trains a candidate,
and uploads only the three checkpoint files. Saved candidates survive miner
restarts. The same running miner can subsequently claim another customer's job;
no new fleet bundle is required.

## Completion, failures, and limits

Jobs transition through `uploading → validating → awaiting_approval → queued →
running → evaluating → completed`, or `failed`. There are at most five active
jobs per customer. Miner leases last twenty minutes and renew every minute;
requests are signed for the exact API URL, route, body, nonce, and timestamp.
Expired claims cannot submit with an old token. Each assignment permits up to
three attempts. A job is evaluated when all assignments finish/fail, or after
its twenty-four-hour deadline. There is no additional score reward for accepting
more jobs or signing repeated requests.

The processor compares candidates against the calibrated starting checkpoint,
using the [existing acceptance gate](customer-jobs.md#inspect-the-result).
For an explicit [version upgrade](version-selection.md), the comparison instead
uses the pinned previous customer model and its existing serving temperature on
the new test data. No qualifying candidate leaves the current API version intact.
It uploads accepted artifacts and a release manifest to private Storage. The
customer receives aggregate results and short-lived download URLs, never other
customers' records or raw validator predictions. The release still requires the
pinned base model to run. `completed` can mean `no_qualifying_model`; this is a
valid experimental result. Job weight vectors are diagnostic within-job scores
and are not sent to Bittensor by the queued workflow.

Storage/network interruptions can be retried after lease recovery. Invalid data
or model/runtime failures produce a failed job for operator inspection. Local
private processor state contains the evaluation reports/logs. Cancellation
stops new claims and fences processing completion; it cannot erase data already
downloaded or immediately terminate remote compute. For a transient failed job,
inspect the cause before creating a new one. Do not reset database states by hand
without understanding lease ownership.

Training data is readable by approved miners, who may retain it. Signed URLs
limit access, not the use of downloaded bytes. Files and raw local runs are
retained until operator cleanup; automatic retention/deletion, billing, resumable
multipart uploads, confidential compute, and automatic serving of training artifacts are not implemented.
The separate [Zils decision API](decision-api.md) provides shared-model inference
and bulk processing with its own credentials, queue, and retention controls. The
initial upload path uses direct PUT; retry restarts a failed file transfer.

## Verification

```bash
make check
make check-queue-db
```

The second command requires PostgreSQL 16+ binaries (`initdb`, `pg_ctl`, `psql`).
Set `ZILS_PG_BIN` to their directory if they are not on `PATH`. It creates and
deletes a separate temporary database cluster, never connects to your existing
Supabase database, and checks migration execution, tenant/storage isolation,
service-only functions, replay rejection, lease recovery, and bounded attempts.
CI runs both suites. HTTP tests exercise real signatures and file transfers with
fixture miners; they do not contact Supabase or send chain transactions.

Implementation references: [Supabase database functions](https://supabase.com/docs/guides/database/functions),
[row-level security](https://supabase.com/docs/guides/database/postgres/row-level-security),
and [signed uploads](https://supabase.com/docs/reference/javascript/storage-from-createsigneduploadurl).
