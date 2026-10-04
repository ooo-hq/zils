# Zils decision API

Zils is the decision protocol/subnet; Fez remains the chat application. This
repository retains its existing package and training names. The new gateway
implements a TypeSafe-style decision interface and a separate durable bulk API.
It starts with shared JevK5 weights. Miners train candidate adapters, validators
evaluate them, and trusted Zils servers run inference. The existing training
queue does not automatically deploy an adapter into this API.

The implementation has local HTTP, real PostgreSQL, official SDK, and RTX 4090
runtime checks. A hosted staging check also verified key creation/revocation,
real inference, private Supabase uploads, retry recovery, and account isolation.
Its four-record synthetic bulk job completed three valid records and returned
one expected validation error. This verifies the workflow, not customer model
quality or production capacity. There is no billing, measured service-level
guarantee, automatic model promotion, or open-miner inference network. Onboarding
is a separate service.

## Request and credential contract

| Operation | Credential | Result |
| --- | --- | --- |
| `POST /v1/keys` with `{"name":"production"}` | Supabase user access token | Key metadata and a secret shown once |
| `GET /v1/keys` | Supabase user access token | Key metadata |
| `POST /v1/keys/{id}/revoke` with `{}` | Supabase user access token | Revoked key metadata |
| `GET /v1/models` | Zils API key | Accessible model names and aliases |
| `POST /v1/systemone` | Zils API key | Typed answers and input/output token counts |

Send credentials as `Authorization: Bearer ...`. Multiple keys belong to one
account; there is no fixed key-count cap. Keys contain a random identifier and
256 bits of random secret material. Only SHA-256 digests and display metadata
are stored. Authentication uses constant-time digest comparison and checks
revocation/account status for every request. Key-management sessions are verified
with Supabase Auth. Database tables and privileged RPCs are service-role-only.

A key authenticates the account; it does not select or train a model. An
operator-managed registry maps an alias such as `zils-shared` to an immutable
release and fingerprint. `owners: null` makes that release shared; a UUID list
restricts it to those accounts. Unknown and unauthorized names return the same
404. Both preflight and execution verify the runtime's actual release identity.

An [example commerce request](../examples/zils-api/request.json) asks Choice,
Noul, and Score questions about one state. Responses retain question IDs:

```json
{"model":"zils-jevk5-v0.3-r1","answers":{"compatible":{"type":"noul","noul":0.9}},"usage":{"input_tokens":120,"output_tokens":0}}
```

This is an illustrative response, not a recorded quality result. `noul` is the
probability of true; Choice includes the selected name, probabilities, and
confidence; Score includes an expected zero-based score, probabilities,
confidence, and the original structured legend. Instructions can be absent/null.
Choice supports 2–255 options and Score supports 2–10 levels. There is no fixed
question-count cap, but body, context, and execution budgets apply.

The wire contract follows the [TypeSafe API](https://docs.typesafe.ai/api) and
[confidence formulas](https://docs.typesafe.ai/confidence). Python SDK **0.7.2**
and JavaScript SDK **0.6.0** passed local compatibility checks. Zils' model names,
capacity, validation boundaries, storage, and bulk endpoints are its own. The
TypeSafe OpenAPI allows one Score level while its prose describes 2–10; Zils
uses 2–10. This is compatibility with the tested operations, not a claim of
identical service behavior or TypeSafe's advertised throughput.

## Start from a fresh clone

Use Python 3.13 and Git on macOS, Linux, or WSL 2. The gateway/worker need no GPU.
The private model runtime requires Linux or WSL 2, CUDA, and enough GPU memory
for BF16 JevK5. The hardware check used a 24 GB RTX 4090. Run from the repository
root; keep credentials and customer data in ignored `.private/` storage.

```bash
python3.13 -m venv .venv-api
.venv-api/bin/python -m pip install -r requirements/api.txt
mkdir -p .private/api
cp examples/zils-api/service.env.example .private/api/service.env
chmod 600 .private/api/service.env
```

Replace the environment placeholders with your Supabase project URL, server
service-role key, and a newly generated runtime secret. Enable Supabase Auth and
Storage. Use the same strong runtime secret on the gateway, bulk worker, and GPU
runtime; never give it or the service-role key to customers or miners.

Review and apply these additive migrations once to the intended project. Use a
privileged connection supplied by the operator as `SUPABASE_DB_URL`:

```bash
psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f supabase/migrations/202610040001_decision_api.sql
psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f supabase/migrations/202610040002_decision_batches.sql
```

These create only the `zils_api_*` resources and private `zils-api-batches`
bucket. They do not alter the existing training queue. Supabase Auth/Storage
schemas must already exist. The bucket accepts at most 25 MiB per upload;
project-wide Storage limits must permit that size. A restrictive policy blocks
client access even when other Storage policies are broadly permissive.

On the GPU host, use a separate environment and explicitly download the pinned
model revision (roughly 8.4 GB of weights):

```bash
python3.13 -m venv .venv-jev
.venv-jev/bin/python -m pip install -r requirements/jevk5.txt
.venv-jev/bin/python -m scripts.download_jevk5 --out models/jevk5
set -a
. .private/api/service.env
set +a
.venv-jev/bin/python -m fez.jev_server --model-dir models/jevk5
```

The downloader creates `models/jevk5/release.json`. Startup verifies all recorded
file hashes and the installed JevK5 runtime commit, then loads local weights with
network model downloads disabled. CUDA graphs are disabled for this release.
Keep the runtime on its loopback port **8921**. If the gateway runs elsewhere,
use an authenticated private tunnel or an HTTPS proxy with restricted access;
remote plaintext HTTP URLs are rejected.

Create the gateway registry from that release manifest. For separate hosts,
copy just the manifest to `models/jevk5/release.json` on the gateway first; model
weights are needed only by the runtime. Replace the runtime URL below when using
a private tunnel/proxy:

```bash
.venv-api/bin/python - <<'PY'
import json
from pathlib import Path
release = json.loads(Path("models/jevk5/release.json").read_text())
entry = {
    "id": release["release_id"], "fingerprint": release["fingerprint"],
    "aliases": ["zils-shared"], "owners": None,
    "url": "http://127.0.0.1:8921", "token_env": "ZILS_RUNTIME_TOKEN",
    "release_date": "2026-10-04", "description": "Shared BF16 JevK5 decision model"
}
Path(".private/api/models.json").write_text(json.dumps({"models": [entry]}, indent=2))
PY
set -a
. .private/api/service.env
set +a
.venv-api/bin/python -m fez.api --registry .private/api/models.json
```

In a second terminal with the same environment loaded:

```bash
.venv-api/bin/python -m fez.batches --registry .private/api/models.json
```

The gateway listens on loopback **8920**. These bounded threaded HTTP servers
are service processes, not internet edge servers. Put the gateway behind an HTTPS
reverse proxy with connection/body/time limits. Supply `--origin https://YOUR_APP`
for an exact browser origin when connecting the separate onboarding app. Run the
worker under a supervisor; it owns deadline expiry, retries, and retention cleanup.
Do not log Authorization headers, real-time bodies, or signed upload URLs at the
proxy. The Python handlers do not log bodies or private exception details.

## Create a key and call the model

`SUPABASE_ACCESS_TOKEN` below is the signed-in user's session access token,
obtained by your Auth client. It is different from the project's service-role key.
The response contains a secret: store it privately before closing the session.

```bash
export ZILS_URL=http://127.0.0.1:8920
curl --fail-with-body "$ZILS_URL/v1/keys" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  -H 'Content-Type: application/json' -d '{"name":"development"}'
```

Set `ZILS_API_KEY` to the returned `key`, then:

```bash
curl --fail-with-body "$ZILS_URL/v1/systemone" \
  -H "Authorization: Bearer $ZILS_API_KEY" -H 'Content-Type: application/json' \
  --data-binary @examples/zils-api/request.json
```

For Python, install `typesafe-sdk==0.7.2` in your client environment:

```python
import os
from typesafe_sdk import RetryPolicy, TypeSafeClient

with TypeSafeClient(api_key=os.environ["ZILS_API_KEY"],
                    base_url=os.environ["ZILS_URL"], model="zils-shared",
                    retry=RetryPolicy(max_retries=0)) as client:
    result = client.system_one(
        state={"query": "USB-C cable", "product": "USB-C to USB-C cable"},
        questions={"match": {"type": "noul", "instructions": "Does this product match?"}},
    )
    print(result.nouls["match"].noul)
```

For Node.js 20+, install `@typesafe-ai/sdk@0.6.0` in your client project:

```javascript
import { TypeSafeClient } from "@typesafe-ai/sdk";
const client = new TypeSafeClient({
  apiKey: process.env.ZILS_API_KEY, baseURL: process.env.ZILS_URL,
  defaultModel: "zils-shared", retry: { maxRetries: 0 }
});
const result = await client.systemOne({
  state: { query: "USB-C cable", product: "USB-C to USB-C cable" },
  questions: { match: { type: "noul", instructions: "Does this product match?" } }
});
console.log(result.answers.match.noul);
```

Retries are disabled in these minimal examples to make execution explicit.
A real client should back off for 429/529 and honor `Retry-After`. A transport
failure or deadline does not prove the GPU did no work; retrying can recompute.
The gateway does not retry inference automatically.

## Bulk jobs

Each JSONL record has a unique `custom_id` and a `body` with the same fields as a
System One request. Generate a one-record example:

```bash
.venv-api/bin/python - <<'PY'
import json
from pathlib import Path
body = json.loads(Path("examples/zils-api/request.json").read_text())
Path(".private/api/input.jsonl").write_text(json.dumps({"custom_id": "product-001", "body": body}) + "\n")
PY
```

| Operation | Behavior |
| --- | --- |
| `POST /v1/batches` with `{"idempotency_key":"my-import-v1"}` | Returns a batch ID, limits, and an immutable signed PUT upload URL |
| PUT the JSONL bytes to the returned `upload.url` | Use returned headers; do not send the Zils API key to Storage |
| `POST /v1/batches/{id}/submit` with `{}` | Requires a completed upload, freezes available aliases/releases, queues the job |
| `GET /v1/batches/{id}` | Returns status, timestamps, and total/completed/failed counts |
| `POST /v1/batches/{id}/cancel` with `{}` | Stops new work and preserves committed results |
| `GET /v1/batches/{id}/results?after=0` | For terminal jobs, returns up to 20 committed records and `next_cursor` |

All API operations use the current account API key. The signed PUT uses the
returned Storage authorization URL. It cannot replace an uploaded object. Repeat
create with the same idempotency key to recover the batch ID. If the input has
already arrived, the response omits `upload`; submit that batch without uploading
again. Otherwise, `upload` contains a fresh signed destination. Use a new
idempotency key for changed inputs. Repeated submit does not change
versions or duplicate jobs. Signed uploads expire after two hours under
[Supabase Storage's documented behavior](https://supabase.com/docs/reference/javascript/storage-from-createsigneduploadurl); create again with the same key for a fresh destination.
Upload completion is checked at submit; dataset/schema validation happens in the
worker. Empty, malformed JSONL, duplicate IDs, or invalid envelope IDs fail the
file before any inference. Invalid individual request bodies produce per-record
errors. Bulk rejects U+0000 because PostgreSQL cannot persist that character.

Jobs move through `uploading → queued → validating → running → completed`, or
`failed`, `cancelled`, or `expired`. `completed` can contain failed records.
A result is either `{"custom_id":"...","response":{...normal response...}}` or
`{"custom_id":"...","error":{"status":422,"code":"...","message":"..."}}`.
Results remain in input order among committed records. Cancelled jobs omit
unfinished records; counts and status distinguish this from successful completion.

Worker claims expire after 120 seconds and renew every 30 seconds. Each worker
processes one record before releasing a batch, so a large job does not monopolize
the queue. Transient model failures permit three attempts; 429/529 wait for capacity
until the job deadline. Interrupted database/storage work recovers after lease
expiry. Result publication and successful usage commit atomically. A crash after
inference but before commit can recompute; exactly-once GPU execution is not
promised. Retain old approved releases in the registry while submitted batches
need them. Removing access or a release can cause those records to fail.

Revoking the submitting key does not cancel a job; another valid key on the same
account can poll/cancel/download it. Disabling an account prevents new worker
claims and fences writes. Already-running GPU work may finish. A key cannot read
another account's job, and miners receive none of these bulk records.

To export results, set `BATCH_ID` to the returned UUID:

```bash
.venv-api/bin/python -m scripts.download_batch --url "$ZILS_URL" \
  --batch "$BATCH_ID" --out .private/api/results.jsonl
```

Use a new output filename. Interrupted exports remove their incomplete file;
custom integrations can page using the returned cursor. The API rechecks the
current key on every page.

## Capacity, usage, and retention

Default operational boundaries are **1 MiB per decision request**, JSON depth 32,
**4,096 tokens per rendered model pass**, **65,536 reserved input tokens across a
request**, **25 MiB / 10,000 records per bulk file**, and **24 hours per batch**.
Large Choice requests use multiple passes. The adaptive final pass uses a
conservative UTF-8 byte bound for admission, so long descriptions can be rejected
even when a particular outcome would fit. No truncation is performed.

Runtime flags can change the token boundaries; operators must publish their
configured values to clients. These are execution bounds, not trained context
quality claims. One GPU executes serially, with at most one pending real-time
request and one pending bulk request. When both lanes wait, at most four
real-time executions precede the waiting bulk execution. The 30-second execution
wait can return 504 while the GPU remains occupied; it never opens a second
execution slot prematurely. Full lanes return 529. HTTP connection concurrency
is bounded to 32 per service; the edge proxy should enforce its own limits.

Account settings live in `zils_api_accounts`: `requests_per_second`,
`tokens_per_second`, `max_active_batches`, and `max_batch_storage_bytes`.
They default to SQL NULL (no configured account quota) for local evaluation.
**Set them before exposing the service to customers**, using measured capacity
and your storage budget. For example, with operator-chosen psql variables:

```sql
update public.zils_api_accounts
set requests_per_second = :'rps'::integer,
    tokens_per_second = :'tps'::bigint,
    max_active_batches = :'active_batches'::integer,
    max_batch_storage_bytes = :'retained_bytes'::bigint
where owner_id = :'account_id'::uuid;
```

The account row is created when its first API key is issued. Positive configured
values apply to all its keys. RPS/TPS use atomic fixed UTC-second windows; each
admission reserves its worst-case token work. Unused reservations are not refunded.
Configure TPS to accommodate the largest single-request reservation: a smaller
ceiling continually throttles that request until it is split or the budget is raised.
Actual successful input usage is recorded separately; unknown/failed work is not
invented as zero usage or treated as a bill. This prototype implements no charges.
Each non-purged batch reserves 25 MiB of input Storage capacity, including failed
and cancelled jobs, until retention cleanup. Database body/result storage is
additional; this byte budget is an input-object reservation, not a whole-database
size guarantee. Create admission is atomic across keys. There is no hardcoded
60/minute policy or claim that one GPU provides TypeSafe's service capacity.

Real-time input/output bodies are not persisted by these services. Bulk inputs,
request records, and committed results are retained for seven days after a
terminal state, then deleted by the worker. Result access returns 410 after that
period even if cleanup is delayed. Batch metadata and usage records are retained
for 30 days. Cleanup runs at most once a minute during worker processing; outages
can delay physical deletion. Run and monitor the worker even with no queued jobs.
Storage backups and copies already downloaded are outside this deletion mechanism.

## Verification and measured scope

```bash
.venv-api/bin/python -m pip install -r requirements/api-test.txt
.venv-api/bin/python -m pip install --no-deps -r requirements/jevk5-source.txt
make check-api PYTHON=.venv-api/bin/python
FEZ_PG_BIN=/PATH/TO/POSTGRESQL16/bin make check-queue-db PYTHON=.venv-api/bin/python
npm install --prefix .private/api-sdk --no-audit --no-fund @typesafe-ai/sdk@0.6.0
.venv-api/bin/python -m scripts.check_api_sdks \
  --js-module .private/api-sdk/node_modules/@typesafe-ai/sdk/dist/index.mjs
```

Replace `FEZ_PG_BIN` with the directory containing `initdb`, `pg_ctl`, and `psql`,
or omit it when they are on PATH. The database checker creates a disposable
cluster; it never connects to the configured Supabase project. The SDK check
uses fixture model probabilities on loopback and does not call TypeSafe's API.
For the repository-wide suite, follow [development setup](development.md) and
install `requirements/jevk5-source.txt` with `--no-deps` into that environment too.

The 2026-10-04 hardware smoke used the pinned BF16 JevK5 release, eager execution,
and synthetic commerce inputs on a 24 GB RTX 4090. The private runtime's HTTP
preflight and inference endpoints were exercised; elapsed times below measure the
inference HTTP call, excluding startup, hashing, and the separate preflight.
Another model was resident but idle, so this was not an isolated performance run.

| Synthetic request | Actual input tokens | Reserved tokens | One observed call |
| --- | ---: | ---: | ---: |
| Choice + Noul + Score | 430 | 430 | 0.655 s |
| 13 Noul questions | 1,638 | 1,638 | 1.137 s |
| 17 Choice options | 820 | 1,634 | 0.328 s |
| 50 Choice options | 1,600 | 2,422 | 0.496 s |
| 255 Choice options | 6,680 | 7,510 | 1.929 s |

All returned distributions and accounting passed the gateway response validator.
These single requests establish runtime plumbing, not quality improvement,
concurrency capacity, p95 latency, or a service guarantee. Prior adapter comparisons
are separate experiments; this implementation does not promote an adapter.
