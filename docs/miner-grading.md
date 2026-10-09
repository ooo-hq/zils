# Grade miners and assign training jobs

The optional `zils-miner-routing/v1` policy selects an approved JevK5 training
miner by **quality, then reliability, then expected completion time**. Eligibility
comes first: a high score does not bypass export consent, the operator pool,
verified workload capacity, a current profile, a fresh heartbeat, or a free host.

Fixed-worker routing remains the default. This feature does not change customer
acceptance criteria, serve predictions, publish Bittensor weights, or calculate
cross-job rewards. Apple silicon and CUDA identities use the same grading policy;
a working, separately qualified training runtime is still required on each host.
This scheduler alone does not add an MPS training implementation.

## Local verification

Complete the Python 3.13 repository setup, including model, rehearsal, testnet,
API and development requirements. Install PostgreSQL 16+ locally; put its commands
on `PATH`, or set `ZILS_PG_BIN` to the installation's `bin` directory. From the
repository root:

```sh
make check
make check-queue-db
```

The database command creates and removes its own PostgreSQL cluster and uses
local HTTP/storage/model fixtures. It does not use a production database, create
production accounts, or make chain transactions. The signed rehearsal verifies
three qualification rounds, customer selection, a capacity deferral to a second
host, artifact submission, and evaluator completion. Other database tests cover
shared-resource races, old-token rejection, cancellation, deadlines, duplicate
observations, private-table access, and existing billing/image behavior.

These are orchestration tests. Fixture probabilities, tokens, certificates and
memory declarations are synthetic; they do not establish hardware capacity or
model quality. Real certificates require operator-observed training and reload
measurements on the intended workload.

## What a grade means

| Criterion | Evidence and rule |
| --- | --- |
| Context | Same pinned profile, runtime, trainer source hash, benchmark binding, rubric version and token band. Different customer tasks never contribute comparable quality scores. |
| Quality | Median normalized Brier improvement, `(baseline − candidate) / uniform`, over the latest five valid benchmark runs in 30 days. At least three runs are required. Median improvement and accuracy must meet the benchmark floors. Poor valid runs remain in the median. The integer band is `floor(100 × improvement)`, allowing for floating-point rounding. |
| Reliability | The latest 50 finalized attempts in 30 days for the same model/runtime/trainer and workload band. Score is `(valid completions + 1) / (valid completions + attributable failures + 2)`. A technically valid adapter counts as success even when customer acceptance fails. |
| Speed | Server-observed lease start to validated upload. Use the slower of the verified benchmark rate and recent p90 rate per encoded training token. Evaluator `median_ms` and `p95_ms` do not measure miner training speed. |
| Availability | Signed receipt younger than 45 seconds, readiness, no reservation or cooldown, and an estimate plus 60 seconds inside the original deadline. Estimated training cannot exceed the existing one-hour child limit. |

Workload bands are maximum encoded training prompts of 1–512, 513–1024, and
1025–2048 tokens. Example count, total tokens and maximum tokens must all fit a
verified envelope. A fast result on a tiny workload does not authorize a larger
job. Workload metadata is frozen during validation. Old manifests without it keep
the explicitly configured fixed-worker route.

Infrastructure and validator problems, cancellations, and capacity deferrals
are neutral. Unclear failures remain `pending_review`. Independently detected
invalid artifacts and abandoned claimed leases count as failures. A miner's own
failure message cannot overwrite an already finalized observation.

## Operator setup

Use a separate test deployment first. Apply all earlier migrations, then
`supabase/migrations/202610090001_graded_miner_routing.sql`. Deploy the matching
coordinator, processor and workflow code before enabling presence on miners.
The new tables and RPCs are service-role only. Keep the existing protected
Supabase environment on operator services; never copy it to miners.

Create a private directory and assign one UUID per physical training resource:

```sh
mkdir -p .private
chmod 700 .private
python -c 'import uuid; print(uuid.uuid4())'
python -m zils.graded_scheduler bind-resource \
  --hotkey "$WORKER_HOTKEY" --resource "$RESOURCE_UUID"
```

Set `WORKER_HOTKEY` to an already registered public hotkey and `RESOURCE_UUID` to
the generated UUID. Identities sharing one GPU or unified-memory host must share
that UUID. Mapping a worker with active work is refused. Fixed/manual assignments
on mapped hosts also reserve capacity, so they cannot bypass the one-job limit.
A resource mapping is an operator assertion, not remote hardware attestation.

Prepare a synthetic benchmark job using the ordinary upload/validation flow in
your test deployment. Preserve its training, calibration and test splits, base
checkpoint and workload. Before assignment, create a private descriptor from that
validated job. Set `QUALIFICATION_JOB_ID` to its UUID, then run:

```sh
python - <<'PY'
import json, os
from pathlib import Path
from zils import miner_grading, models
from zils.cloud import Supabase
from zils.graded_scheduler import benchmark_binding

job = Supabase().rows('fez_training_jobs', 'id=eq.' + os.environ['QUALIFICATION_JOB_ID'])[0]
binding = benchmark_binding(job)
context = {'model': models.JEVK5, **models.profile_identity(models.JEVK5),
           'trainer_sha256': miner_grading.trainer_identity(),
           'benchmark_sha256': binding, 'rubric': 'operator-benchmark/v1',
           'band': job['manifest']['workload']['band']}
descriptor = {'context': context, 'benchmark_sha256': binding,
              'min_accuracy': 0.8, 'quality_floor': 0.0}
path = Path('.private/qualification-benchmark.json')
with path.open('x') as out:
    out.write(json.dumps(descriptor, indent=2) + '\n')
path.chmod(0o600)
PY
```

Choose accuracy and improvement floors for that benchmark before any run.
Changing them requires a new rubric context. The binding includes frozen data
hashes, model, workload and initial checkpoint, excluding the per-run job UUID.
Qualification cannot use a customer's predecessor model. Calibration/test labels
stay with the validator. Repeated runs reuse the held-out set; their median is
an operational grade, not a claim of statistical significance or generalization.

Create `.private/routing.json` with actual values. Copy `context` from the
descriptor under its matching band key:

```json
{
  "mode": "graded",
  "policy_version": "zils-miner-routing/v1",
  "pool": ["WORKER_1_PUBLIC_SS58", "WORKER_2_PUBLIC_SS58"],
  "qualification_slots": 1,
  "contexts": {"tokens-512": "REPLACE_WITH_DESCRIPTOR_CONTEXT_OBJECT"}
}
```

The context placeholder must become an object, not a string. Enable only bands
with measured benchmarks. The pool supports up to 256 distinct registered
identities. Install the policy and put the same object in the existing workflow
configuration's `routing` field:

```sh
chmod 600 .private/routing.json
python -m zils.graded_scheduler configure --config .private/routing.json
```

Mismatched workflow/database configurations stop assignment instead of falling
back silently. Existing fixed-worker fields remain necessary for legacy jobs;
image routing retains its separate configuration.

## Qualify a new resource

Before queue qualification, import a capacity-only certificate from a trusted
local probe. The probe must use the exact context/trainer revision, complete
training within the intended envelope, reload its artifact successfully, and
record elapsed training/upload time and memory headroom. Do not infer capacity
from a GPU name or from a miner's self-report.

The private JSON report needs these fields:

```json
{
  "kind": "capacity",
  "context": "REPLACE_WITH_DESCRIPTOR_CONTEXT_OBJECT",
  "artifact_valid": true,
  "verified_at": "REPLACE_WITH_OBSERVED_UTC_TIME",
  "expires_at": "REPLACE_WITH_EXPIRY_UTC_TIME",
  "capacity": {"examples": 0, "total_tokens": 0, "max_tokens": 0},
  "seconds_per_token": 0
}
```

Replace zeroes with observed positive quantities; the template intentionally
cannot qualify a host. Retain original logs, artifact hashes and reload evidence
privately for the named verifier. Reports are bound to the worker's current
resource UUID, recorded append-only, and rejected on duplicate evidence hashes.
A capacity-only certificate confers no quality grade.

```sh
python -m zils.graded_scheduler import-qualification \
  --hotkey "$WORKER_HOTKEY" --report .private/capacity-report.json \
  --verified-by "$VERIFIER_ID"
python -m zils.graded_scheduler authorize-qualification \
  --job "$QUALIFICATION_JOB_ID" --benchmark .private/qualification-benchmark.json
```

`VERIFIER_ID` identifies the operator who checked the evidence. Prepare three
separate jobs with the same benchmark inputs and authorize each. Successful
independent evaluations create quality reports automatically using server timing
and immutable job/artifact bindings. Direct quality-report imports are privileged
attestations and require the same verified measurements; they are not proof of
work generated by this CLI.

With `qualification_slots: 1`, each scheduler tick examines qualification jobs
oldest first, skipping jobs without an eligible provisional resource. It offers
the first schedulable job to its least recently offered capable resource before
customer assignment. At most one qualification training reservation is active. Already
qualified resources do not consume this lane. Other free hosts remain available
for customers. On a single host, qualification can delay customer work. Set the
value to `0` to suspend this lane. Expired contexts need a current capacity probe
before requalification. Qualification results cannot become served releases,
even if their model exceeds customer acceptance thresholds.

## Miner presence and operation

After the coordinator/schema upgrade, add these fields to the existing private
miner configuration, replacing the memory value with its measured requirement:

```json
{"graded_scheduling": true, "graded_min_free_mib": 12288}
```

Presence runs every 15 seconds alongside training. It contains installed hashes
and readiness, never a client-chosen grade, timestamp or resource UUID. Requests
use the existing audience/path/nonce-bound hotkey signature. CUDA uses the GPU
memory probe; MPS uses native free/inactive/speculative page counts. Probe errors
fail closed. These checks are advisory; the shared compute lock checks capacity
again immediately before training. Coordinate unrelated applications separately.
Leaving `graded_scheduling` absent preserves the old miner request flow.

A capacity deferral fences the old token, frees the host, applies a 60-second
host cooldown and leaves the original deadline unchanged. Another eligible host
can take the job. Claimed attempts use the existing renewable 20-minute lease;
ready reservations expire after 60 seconds. New graded jobs have a 24-hour
scheduling deadline from validation. Cancellation and expired attempts release
reservations, and stale workers cannot upload a replacement result through an
old lease. Revoked leases stop supervised text/image children.

## Preview and rollout

```sh
python -m zils.graded_scheduler preview \
  --config .private/routing.json --job "$CUSTOMER_JOB_ID"
```

Preview reads private evidence and prints the selected worker, score components,
evidence IDs and exclusion reasons. It neither assigns a job nor downloads data
nor activates a model. Successful reservations retain their policy, decision time and private input
snapshot for replay; repeated waiting polls do not create audit rows. Inspect qualification
and attempt records privately when investigating a grade.

Keep fixed routing during the initial deployment, verify the schema/coordinator,
qualify resources, enable miner presence, and compare previews before enabling
graded workflow routing. Configuration changes that revoke an active graded
context prevent further lease renewal; drain work before changing contexts.
The pilot uses one queue transaction mutex for consistent lock ordering. It is
not a benchmark of large public-subnet scheduling throughput. Bittensor reward
fairness and cross-job emissions remain separate work.
