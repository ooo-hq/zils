# Automatic customer training

The workflow service connects the existing private upload queue, approved training
miner, held-out evaluation, and customer model registry. Customer API keys, Auth,
and database schemas are unchanged. The operator explicitly configures the miner
that may receive training data. Assignment requires a validated JevK5 job with
recorded miner-export consent, an enabled miner, and available GPU capacity.

For an approved pool selected by measured quality, reliability and turnaround,
see [graded miner routing](miner-grading.md). It is opt-in; the fixed-worker
configuration below remains the default.

## Job lifecycle

1. The existing processor validates the uploaded examples and freezes the job.
2. The workflow waits for an idle approved miner and the configured free-memory
   requirement, then uses the existing transactional approval RPC to assign it.
3. The miner trains the adapter. The evaluator checks the baseline and candidate
   against the original acceptance criteria. Neither the criteria nor held-out
   examples are sent to the miner.
4. An accepted result becomes an immutable release. The workflow verifies that
   the prediction runtime reports that release's identity before registering it
   for the job owner. Failed activation retries; rejected candidates never publish.
5. The customer uses the resulting model ID with an existing API key. Other
   accounts cannot list or call the private model.

[Version selection](version-selection.md) compares first versions with the base
and explicit upgrades with the customer's previous accepted version. Versioned
releases also expose a stable task alias. Its target changes only if the named
predecessor is still active; a stale run requires review instead of overwriting
a newer model.

The dashboard receives a separate `workflow` field: `waiting_capacity`,
`waiting_worker`, `needs_review`, `activating`, `activation_failed`, or `ready`.
Only `ready` includes a confirmed API model ID. Queue status remains authoritative
for training/evaluation; a prior capacity message must not override a running job.
Completed evaluation results and their completion timestamp remain immutable.

## Deployment requirements

Complete the [JevK5 training setup](jevk5-queue.md) and
[decision API setup](decision-api.md) first. Use Linux or WSL 2, systemd for the
optional service-health checks, Python 3.13, and a BF16-capable NVIDIA GPU. Install
the pinned training and API dependencies described in those guides. Configure a
single workflow service per approved miner.

Training and prediction may share a GPU only when both fit. The workflow does not
stop prediction servers or unrelated workloads. A dedicated training GPU is the
deployment option for continuous prediction availability. The memory check is a
capacity gate, not a reservation against unrelated applications; operators must
coordinate all GPU workloads. Miner and evaluator share `ZILS_COMPUTE_LOCK`.

Set these variables on both training services, using measured requirements for
the largest supported examples on the chosen hardware:

```sh
export ZILS_GPU_MIN_FREE_MIB=12288
export ZILS_NVIDIA_SMI=/usr/bin/nvidia-smi
```

On WSL 2, the GPU utility is commonly `/usr/lib/wsl/lib/nvidia-smi`; verify the path
on the host. Capacity is checked before claiming work and again inside the shared
compute lock. A capacity timeout releases the miner lease without consuming a
training attempt. Existing job deadlines still apply.

## Serve shared and customer models with one base

Create a private release directory readable by the prediction service, and keep
the existing shared model directory and its verified `release.json`:

```sh
mkdir -p .private/customer-releases
python -m zils.adapter_server \
  --releases .private/customer-releases \
  --shared-model-dir /path/to/verified-shared-model \
  --reference /path/to/jevk5-reference \
  --port 8921 --token-env ZILS_RUNTIME_TOKEN
```

The service starts with no customer releases. It uses one frozen base and one
adapter slot. Shared-model requests disable the adapter; all prediction execution
and adapter switching use the same serial queue. New immutable releases are
verified and discovered without restarting. Shared and customer prompt contracts
remain distinct and retain their existing fingerprints.

Use the updated gateway and bulk-worker entry points. Their file registry reloads
atomically: existing immutable releases and model names must remain present, and
aliases cannot change owners. An invalid replacement leaves the last valid catalog
usable. A one-time controlled service rollout is required to install this code;
later customer releases do not require a restart.

## Configure the workflow

Create `.private/workflow.json` with operator-provided values:

```json
{
  "hotkey": "APPROVED_WORKER_SS58_ADDRESS",
  "releases": "/path/to/private/customer-releases",
  "release_group": "prediction-service-group",
  "min_free_mib": 12288,
  "nvidia_smi": "/usr/bin/nvidia-smi",
  "training_services": ["training-worker.service", "training-processor.service"],
  "runtime_url": "http://127.0.0.1:8921",
  "gateway_runtime_url": "http://127.0.0.1:8921",
  "token_env": "ZILS_RUNTIME_TOKEN",
  "registry": "/path/to/api/models.json"
}
```

The example paths, miner address, service names, and group are placeholders.
Create the directory and group explicitly, grant the workflow write access and
the prediction service read/traverse access, and provision the existing runtime
secret through a protected environment. Omit `release_group` when both services
use the same identity. Omit `training_services` when systemd health checks are not
appropriate, and provide equivalent external service supervision.

Run using the processor's existing protected Supabase environment and the existing
runtime secret; miners must never receive the service-role credential:

```sh
python -m zils.workflow run --config .private/workflow.json
```

For a gateway on another host, replace `registry` with `register_command`, an argv
array for an operator-controlled secure transport. Legacy releases send a registry
entry as JSON on stdin. Versioned releases send `{"entry":ENTRY,"selection":SELECTION}`
to preserve the atomic predecessor check. The command must print
`{"model_id":"THE_REGISTERED_ID"}` after
successful atomic registration. Exit code 3 identifies a stale predecessor;
other failures are retryable activation errors. It must fail on transport errors.
The workflow strips Supabase credentials from the child environment. A fixed,
restricted SSH command can invoke:

```sh
python -m zils.workflow register --registry /path/to/api/models.json
```

Restrict that credential to this command and registry; do not grant arbitrary
shell access. Both hosts must use compatible code. Keep all previous immutable
registry entries so in-flight bulk jobs retain their original model identities.

## Verification and limits

`make test` covers queue behavior, retryable capacity deferral, publication,
owner access, catalog reloads, and shared/private dispatch using fixture models.
The opt-in GPU check can exercise the combined runtime:

```sh
ZILS_TEST_ADAPTER=/path/to/trained-test-adapter \
ZILS_TEST_SHARED_MODEL=/path/to/verified-shared-model \
ZILS_TEST_REFERENCE=/path/to/jevk5-reference \
ZILS_JEVK5_BASE_DIR=/path/to/verified-shared-model \
python -m unittest tests.test_adapter_gpu -v
```

This check uses explicitly synthetic acceptance records and never registers them
in a production catalog. It verifies serving compatibility, not customer model
quality. A real run may complete with `no_qualifying_model`; automatic orchestration
does not weaken acceptance criteria to produce a release.
