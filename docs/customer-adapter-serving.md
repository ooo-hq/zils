# Customer adapters behind the existing API

An accepted JevK5 training result can become a private model in the existing
[decision API](decision-api.md). The publisher reads the completed training job
and its immutable storage objects, verifies acceptance and provenance, and builds
a local release plus a compatible API registry entry. The adapter runtime loads
one shared JevK5 base and switches customer LoRA weights serially.

This integration adds no tables, migrations, key formats, or authentication
endpoints. The gateway continues to authenticate existing API keys and enforce
model ownership. The publisher does not write to Supabase. Its existing
server-side service-role credential is privileged; **read-only describes this
program's operations, not that credential's permissions**. The inference process
needs no Supabase credential.

The [automatic workflow](automatic-training.md) connects assignment, capacity,
publication, verified serving, and registry reloads. The standalone publication
command below only prepares files; it does not itself make a model live.
The updated gateway, bulk worker, and adapter runtime discover verified catalog
additions without restart. The workflow service invokes publication; the training
coordinator remains responsible for validation and evaluation.

## Acceptance and identity

Only a `completed` job whose delivery is `accepted` can be published. Incomplete,
failed, and `no_qualifying_model` jobs produce no release or registry change.
An existing private adapter or shared base remains available under its existing
model name. The publisher never relabels a rejected adapter as an accepted one.

Verification binds the job's owner, frozen model and acceptance policy, recorded
candidate selection, manifest hash, calibrated checkpoint hash, completion time,
and accepted release report. This trusts the training coordinator's stored
results; it does not rerun evaluation or independently establish model quality.
No training examples or customer API keys are downloaded into the serving bundle.
The bundle contains private job provenance and must remain access-controlled.

Each release has an immutable ID containing the job UUID and checkpoint hash.
Its fingerprint also covers the serving contract, pinned runtime, calibrated
temperature, owner, and provenance files. Retrying the same publication is
idempotent. A changed checkpoint or provenance fails verification. Downloads
are staged before an atomic rename; an interrupted download cannot replace an
existing release. Updating the prepared registry is also atomic and retryable.

Registry entries use `owners: [job.owner_id]`. An optional alias can move between
that owner's releases; it cannot take over another account's alias, a shared
alias, or an immutable release ID. Registry names are globally unique, so use
account-specific aliases such as `ACCOUNT-support`. Prior immutable entries and
release directories are retained for already-submitted bulk requests.

## Prepare from a fresh clone

Use Python 3.13 and Git on Linux, WSL 2, or macOS for publication. GPU serving
requires Linux or WSL 2, a BF16-capable NVIDIA GPU, a compatible CUDA driver, and
the pinned model dependencies. CPU and MPS serving are unsupported. Allow at
least 25 GB of disk for the base and dependencies, plus storage for each adapter.
Do not run another model or training workload on the same GPU without planning
its memory capacity. Request serialization protects this runtime only.

```sh
git clone https://github.com/ooo-hq/zils.git
cd zils
python3.13 -m venv .venv-publish
.venv-publish/bin/python -m pip install -r requirements/api.txt -r requirements/rehearsal.txt
mkdir -p .private/adapters
chmod 700 .private/adapters
```

Use the already-configured training project's `SUPABASE_URL` and
`SUPABASE_SERVICE_ROLE_KEY` in the publisher's protected environment. Never pass
these credentials to customers, model workers, or the private inference process.
The existing [training setup](supabase-training.md) must already contain a
completed, accepted JevK5 job and its accepted release objects.

Copy the current API registry to a preparation file. Replace the placeholder
path with the existing registry's location:

```sh
cp /path/to/current/models.json .private/adapters/models.next.json
chmod 600 .private/adapters/models.next.json
```

For a new isolated installation with no existing models, create that input
instead using `printf '{"models": []}\n' > .private/adapters/models.next.json`.
Never initialize an empty registry over the current service configuration.

Replace `ACCEPTED_JOB_UUID` and `ACCOUNT-support` below. The URL is the runtime
as reachable by the gateway and bulk worker. Loopback works on the same host;
use an authenticated private tunnel or restricted HTTPS proxy between hosts.
Remote plaintext HTTP is rejected.

```sh
.venv-publish/bin/python -m fez.adapter_releases \
  --job ACCEPTED_JOB_UUID \
  --out .private/adapters/releases \
  --registry .private/adapters/models.next.json \
  --runtime-url http://127.0.0.1:8931 \
  --token-env ZILS_ADAPTER_RUNTIME_TOKEN \
  --alias ACCOUNT-support
```

The command prints `published` with the immutable model ID and fingerprint, or
`not_accepted`. A `published` response means the files are ready, **not that
customer requests are using them**. The publisher requires no runtime secret;
the registry records only its environment-variable name.

## Start the separate adapter runtime

On the GPU host, check out the same revision and prepare the model environment:

```sh
python3.13 -m venv .venv-adapters
.venv-adapters/bin/python -m pip install uv==0.12.19
.venv-adapters/bin/uv pip install --python .venv-adapters/bin/python --torch-backend=cu128 \
  -r requirements/model.txt -r requirements/rehearsal.txt
.venv-adapters/bin/python -m fez.jevk5 reference --out models/jevk5-reference
```

This downloads and checksum-verifies the pinned JevK5 base. Alternatively set
`FEZ_JEVK5_BASE_DIR` to the already-verified base directory and pass `--no-download`
to the reference command. The runtime uses local weights only. Copy the complete
private `releases` directory to this host if publication happened elsewhere,
preserving the directory names and permissions. The runtime identity needs read
access; unrelated users and model workers must not have access.

Generate a new private gateway-to-runtime credential in a protected file:

```sh
umask 077
mkdir -p .private/adapters
.venv-adapters/bin/python - <<'PY'
import secrets
from pathlib import Path
with Path('.private/adapters/runtime.env').open('x') as output:
    output.write('ZILS_ADAPTER_RUNTIME_TOKEN=' + secrets.token_hex(32) + '\n')
PY
set -a
. .private/adapters/runtime.env
set +a
.venv-adapters/bin/python -m fez.adapter_server \
  --releases .private/adapters/releases --port 8931
```

Keep this credential separate from customer API keys. The runtime listens only
on loopback and requires it even for `GET /health`. The health response lists
loaded release IDs and fingerprints. Files alone are not a health check: verify
that these match the prepared registry before routing customer traffic.

The runtime preserves the training prompt, answer-letter order, fixed LoRA
architecture, and each release's calibrated temperature. It supports Choice,
Noul and Score with at most 16 outcomes, 2,048 prompt tokens per question, and
65,536 tokens per request by default. Score retains the API's 2–10 level limit.
Instructions must be text; absent/null instructions use the training fallback
from `state.decision`. Inputs are never silently truncated. A load failure
returns an error; it cannot use another customer's previously active adapter.

One request executes at a time, with one pending slot per realtime/bulk lane.
Saturation returns 529. A timed-out running request keeps its execution slot
until GPU work ends, preventing concurrent weight switches. This design favors
simple isolation over throughput; many alternating customer adapters require
weight reloads and integrity checks.

## Deployment handoff and rollback

1. Publish into a prepared registry copied from the active one. Keep a backup of
   the active configuration and retain every older immutable release directory.
2. Start the private adapter runtime against the release directory if necessary.
   Verify its authenticated health identities match the prepared entries.
3. Give the existing gateway and bulk worker the same private runtime credential
   through their protected environments. Atomically install the prepared registry;
   the updated gateway and bulk worker reload it. No key or Supabase changes are required.
4. With an existing key belonging to the job owner, verify `GET /v1/models` and
   call `POST /v1/systemone` using the new immutable ID or account-specific alias.
   A different owner's key must get the same 404 as an unknown model.
5. To roll back the default selection, move its alias to a previously verified
   release in an atomically installed registry.
   Preserve newer immutable entries while bulk work still references them.

Restarting can interrupt requests; use normal service draining and deployment
controls for the initial rollout. Hot catalog reloads and automatic activation are
available through the workflow guide. It does not acquire GPU hardware or stop
unrelated workloads. Shared-base models retain their existing registry entries;
the combined runtime can serve them and private adapters using one base instance.

## Verification

Run the publisher, owner-isolation, HTTP, and serialization checks without a GPU:

```sh
.venv-publish/bin/python -m unittest tests.test_adapter_releases tests.test_adapter_server -v
```

These use synthetic accepted-result records and deterministic model doubles to
verify release integrity, failure recovery, preserved registry entries, customer
ownership, calibrated-temperature forwarding, and the unchanged gateway's actual
HTTP contract. They do not establish customer model accuracy.

An opt-in real GPU check is included. Set `ZILS_TEST_ADAPTER` to a trusted trained
JevK5 adapter directory with nonzero LoRA updates, then run:

```sh
ZILS_TEST_ADAPTER=/path/to/trusted/adapter \
  .venv-adapters/bin/python -m unittest tests.test_adapter_gpu -v
```

The check copies that adapter, constructs a second adapter with zeroed LoRA B
weights, and publishes both using explicitly synthetic acceptance records. It
sends A → B → A through the existing HTTP gateway, compares each result with the
training prediction path, and checks that switching back reproduces A with one
base instance. It reports memory and request duration. It creates no hosted
jobs, keys, database rows, or production releases. A passing check demonstrates
runtime compatibility, not that either fixture satisfies a real customer's
acceptance policy.
