# Run a validator

Choose the workflow before starting a validator. Both evaluate miner candidates,
but only the closed Bittensor fleet can publish weights to the chain.

| Workflow | Model | Validator role |
| --- | --- | --- |
| [Hosted training queue](#jevk5-queue-validator) | JevK5 4B; qualified ImaJev 4B | Trusted processor validates datasets, evaluates candidates, and exports accepted models |
| [Closed Bittensor testnet](#bittensor-testnet-validator) | One JevK5, ImaJev, or legacy Kev profile per fleet | Registered validator scores the configured miner roster and explicitly publishes testnet weights |

There is no public, permissionless JevK5 validator onboarding in this repository.
Queue validators need protected operator credentials. Testnet validators need
operator coordination, registered identities, and chain eligibility. Mainnet is
rejected by the fleet.

## JevK5 queue validator

### 1. Prepare the model host

Complete [Install and create the reference](jevk5-queue.md#install-and-create-the-reference)
from a fresh clone. Use Linux or WSL 2, Python 3.13, a BF16-capable NVIDIA GPU,
and at least 25 GB free disk for the base model and packages, plus space for job
data and candidates. The checked GPU is an RTX 4090; input lengths affect memory.

The result must be a verified `models/jevk5-reference`. The processor runs its
own inference and calibration; a miner's reported score is not an evaluation.

### 2. Configure protected access

Complete the resources and server environment in
[shared queue setup](supabase-training.md#run-the-api-and-processor).
For an existing deployment, use its operator-provided configuration; do not
recreate its database resources. From the repository root, if the environment
file does not already exist:

```bash
mkdir -p .private
cp .env.example .private/training.env
chmod 600 .private/training.env
```

Edit this file with the deployment's Supabase URL and server service-role key.
Set `ZILS_TRAINING_MODEL=jevk5-4b-v0.3` and `ZILS_TRAINING_API_URL` to the
coordinator's reachable URL (`https://training.zils.ai` for the hosted service).
The API and processor must use the same model and Supabase project. Access to
the hosted URL alone does not authorize a validator.

Keep service credentials, calibration/test data, and processor state away from
miner accounts. On shared hardware, use separate operating-system identities
and the [shared compute lock](jevk5-queue.md#configure-the-services).

### 3. Start the processor

Load the configured environment and run from the repository root:

```bash
set -a
. .private/training.env
set +a
.venv-kev/bin/python -m zils.coordinator process \
  --state .private/queue-processor \
  --reference models/jevk5-reference --device cuda
```

Use a service supervisor for continuous operation. `--once` handles at most one
pending stage and exits; it does not wait for a full training job to finish.
Each state directory permits one processor process. Database leases coordinate
claimed stages and recover abandoned work.

The API, [approved miners](queue-miners.md), and any automatic assignment service
run separately. Follow [miner approval](supabase-training.md#approve-miners-and-run-a-queued-miner)
or [automatic assignment](automatic-training.md) so validated jobs can reach
miners. A waiting job is not evidence that the evaluator is broken.

### 4. Check evaluation and acceptance

The processor checks dataset boundaries and model identity before assignment.
For submitted candidates it verifies artifact hashes and the allowed adapter
tensors, fits calibration on the calibration split, and scores held-out test
cases. It loads its own pinned architecture, not miner-provided executable code.

A first version must beat the uniform floor, meet the customer's accuracy target,
and improve Brier loss against the calibrated base by the required margin.
[Version upgrades](version-selection.md) compare against the pinned previous
customer model instead. An accepted model gets a release manifest and private
artifacts. A completed job can correctly return `no_qualifying_model`.

Inspect the customer-visible aggregate result and the protected local reports
under `.private/queue-processor/`. Do not expose raw evaluation records, storage
URLs, or server credentials. Queue weight vectors are diagnostic scores; this
processor does not submit them to Bittensor. Serving activation is handled by
the separate [automatic workflow](automatic-training.md).

### 5. Recover without changing the evaluation

Restart with the same protected environment and state directory after resolving
an infrastructure failure. Let expired leases recover; do not manually change
job status to bypass ownership or acceptance. Inspect failed-job logs before
creating a replacement job.

If pending jobs use Kev, retain its verified reference and add
`--additional-reference models/reference`. Follow the
[model transition procedure](jevk5-queue.md#existing-jobs-and-rollback) rather
than rewriting an existing job's model identity.

For image jobs, install the isolated runtime, configure its environment, and
complete [image qualification](image-training.md#worker-and-service-setup). Add
`--additional-reference models/imajev-starting-checkpoint` to the processor.
It selects the matching model for each frozen job; keep image admission off until
evaluation and activation are verified. This does not change the primary text reference.

## Bittensor testnet validator

This path uses a model-pinned fleet and private-network transport. It is a closed
rehearsal, not public miner discovery or an untrusted-checkpoint sandbox.

### 1. Register and check eligibility

Follow [Bittensor registration](bittensor-registration.md) using a dedicated
validator hotkey, distinct from every miner hotkey. Use the operator-confirmed
testnet netuid. Miners and validators both register a hotkey to obtain a UID.

Registration alone does not grant validation rights. Check the live metagraph
and subnet parameters for the validator's permit and stake requirements. The
[Bittensor validator guide](https://www.bittensor.com/docs/guides/validating)
explains eligibility; do not treat a fixed stake amount as a permanent rule.
Zils preflight accepts a validator permit or the subnet owner hotkey, and the
SDK still enforces the chain's transaction requirements.

### 2. Prepare the registered fleet

Complete the [fleet model setup](mining.md) for the operator-selected text or image
profile, including the matching reference and qualified hardware.
Install `requirements/testnet.txt` in `.venv-kev`.
Create the frozen job and registered identity file using
[testnet fleet setup](testnet.md#provision-the-registered-identities).
Keep test/calibration data on the validator and distribute only each miner's
assigned bundle. The validator host needs its hotkey, not the coldkey.

With the generated configuration in place, run the read-only check:

```bash
.venv-kev/bin/python -m zils.testnet preflight \
  --config .private/testnet-fleet/validator/config.json
```

Confirm the testnet/netuid, validator UID, every miner UID-to-hotkey mapping,
permit, weight limits, required version, and publication cooldown. A changed
registration requires a new roster, not reassignment of an old score.

### 3. Evaluate one round before publication

Start each configured miner with `./start-miner --rounds 1`. On the validator:

```bash
.venv-kev/bin/python -m zils.fleet validator \
  --config .private/testnet-fleet/validator/config.json --rounds 1
```

This scores submissions and previews chain publication without sending weights.
Inspect the round's `report.json` under
`.private/testnet-fleet/validator/state/rounds/`. Check failures, calibrated
scores, and registered identities. The normalized skill weights and customer
model-acceptance decision are different outputs: a miner can earn a proposed
weight without producing an accepted customer model.

### 4. Publish the reviewed round

Replace `ROUND` below with the completed round directory's name. This command
**submits a testnet weight transaction** using the validator hotkey:

```bash
.venv-kev/bin/python -m zils.testnet publish \
  --config .private/testnet-fleet/validator/config.json \
  --round .private/testnet-fleet/validator/state/rounds/ROUND
```

Only one process should publish for this hotkey and subnet. For subsequent
continuous rounds, run `zils.fleet validator` with `--publish-weights` and omit
`--rounds 1`. Zils uses mechanism 0 and weight version 1. It stops if the chain
requires a newer version. `no_weights` and `rate_limited` submit nothing.

### 5. Verify the revealed weights

An included commit is not yet a verified reveal. After the chain's reveal period:

```bash
.venv-kev/bin/python -m zils.testnet verify \
  --config .private/testnet-fleet/validator/config.json \
  --round .private/testnet-fleet/validator/state/rounds/ROUND
```

Keep `report.json`, `chain-attempt.json`, and `chain-receipt.json` together.
Verification checks the current identities, a later chain update, and expected
weight proportions. If a receipt is `unknown`, inspect the chain before manual
recovery; deleting the attempt marker and retrying can duplicate a submission.
See [testnet outcomes](testnet.md#read-the-outcome) for restart behavior.

The [first verified testnet round](testnet-round-001.md) demonstrates this closed
path. It does not establish current validator uptime, open access, mainnet
support, or JevK5 training-to-chain integration.
