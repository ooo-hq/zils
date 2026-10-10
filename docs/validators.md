# Run a validator

Validators independently verify submitted artifacts, calibrate probabilities,
and score held-out examples. Miner-reported metrics do not determine rewards.
The closed Bittensor fleet can publish testnet weights only after explicit
configuration and review. Public discovery and mainnet are not implemented.

## JevK5 queue validator

The model evaluation and acceptance code lives in this repository. The hosted
queue processor connects that evaluator to customer jobs and protected storage;
its setup moved to
[zils-platform's queue validator guide](https://github.com/ooo-hq/zils-platform/blob/main/docs/validators.md#jevk5-queue-validator).
Only approved operators should receive its service credentials. Miners need no
platform installation or database credential.

For image jobs, install the isolated runtime, configure its environment, and
complete [image qualification](https://github.com/ooo-hq/zils-platform/blob/main/docs/image-training.md#worker-and-service-setup). Add
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
