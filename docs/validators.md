# Run a validator

Validators independently verify submitted artifacts, calibrate probabilities,
and score held-out examples. Miner-reported metrics do not determine rewards.
The closed Bittensor fleet can publish testnet weights only after explicit
configuration and review. Public discovery and mainnet are not implemented.

## Queue validator

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

## Open admission (base tier; not yet enabled)

The default validator admits only the static `members` allowlist and pins each
miner's artifact endpoint to a private or Tailscale address. Two settings relax
this for an open competition, and both must be set together with a `chain`
section:

- `"admission": "chain"` derives the member list at each round start from the
  subnet's registered hotkeys at the preflight block, excluding the validator's
  own hotkey, and freezes it in the round's `chain-snapshot.json`.
- `"public_endpoints": true` accepts globally routable IPv4 artifact endpoints.
  The validator still pins the download to the submitting connection's source
  address; only the port comes from the signed claim. Private, loopback,
  link-local, multicast and reserved addresses are refused in this mode, and
  submissions are throttled per source address (30 per minute).

Registration on the subnet is the admission check; there is no further vetting.
Every submission is still hash-verified, size-bounded, and scored in the
isolated runner described in [evaluation](evaluation.md).

### What opening the door means today

Open admission is a property of the fleet validator, and the fleet validator
runs **adapter rounds**: one frozen customer job per fleet, whose training split
is shipped to every admitted miner. Opening it therefore hands customer training
data to anyone who pays the registration burn. The base tier needs no customer
data (candidates come in, the sealed holdout never leaves the validator), and a
validator configured with `base_round_dir` collects and scores base candidates
itself (section 6 below). So, in order:

1. **Policy (owner, decided 18 October 2026).** Outside miners may train customer
   jobs as well as base candidates, under miner terms still to be written. Until
   the terms exist and a customer has accepted them, a public validator carries a
   Zils-authored job (public data), not a customer's.
2. **Base-tier intake (merged).** With `base_round_dir` set, every round also
   collects base candidates (9.5 GiB each, downloaded once per distinct hash and
   kept across epochs so a resubmitted hash costs nothing), scores them with
   `base_round` on the configured draws inside the sandbox, and writes the decision
   sidecar itself. The per-hash store is the cross-round submission cap.
3. **Runbook below**, once 1 holds.

### Runbook: a public validator for the base tier

Host: a GPU node of node-1's class (RTX 4000 Ada 20 GB; scoring needs about
10 GB; about $0.76/h on DigitalOcean), provisioned with
`scripts/provision_node.sh` and the `fleet-validator` role, plus 10 GiB of disk
per distinct candidate hash ever accepted (the store under
`validator/base-tier/store/` keeps them across epochs; prune by hand). Keep the
validator hotkey on the host, never a coldkey.

Network: expose only the announcement port (8900 by default) to the internet;
artifacts are fetched *outbound* from each miner's announced endpoint, pinned to
the announcing connection's source address. Plaintext HTTP is accepted for round
one because claims are signed and artifacts hash-verified; put a TLS terminator in
front when the operator wants transport privacy.

Config (`/etc/zils-node/fleet/validator/config.json`; the fields the closed
testnet fleet already uses, plus admission and the base tier):

```json
{
  "validator": "http://VALIDATOR_PUBLIC_IPV4:8900",
  "chain": {"network": "test", "netuid": 579},
  "validator_hotkey": "VALIDATOR_SS58",
  "admission": "chain",
  "public_endpoints": true,
  "base_tier": {
    "share": 0.7,
    "incumbent_sha256": "CHECKPOINT_HASH_OF_THE_REFERENCE",
    "draws": [{"manifest_sha256": "…", "dataset_sha256": "…"}],
    "policy": {"min_brier_improvement": 0.01}
  },
  "base_round_dir": "draws",
  "base_reference": "reference"
}
```

`validator` is the validator's own public IPv4 and announcement port; with
`public_endpoints` the fleet accepts a global address here (and in a miner's
config that announces to it) instead of the private pin.
`members` is omitted: preflight derives it from the chain and freezes it per
round. `model` is the pinned d3 profile as `zils.fleet init` writes it (the
intake scores d3-family candidates only). `base_round_dir` holds one sealed draw per subdirectory (copy each
sealed round directory there) and `base_reference` is the incumbent reference
(`zils.d3 reference --out validator/reference`), both relative to the config.
Run `python -m zils.testnet preflight --config …` and check that the
member list is what the metagraph shows minus the validator. Start the unit; it
refuses to start when a listed draw has no directory or the reference hash is
not `incumbent_sha256`. Then watch `state/rounds/<round>/chain-snapshot.json`,
`submissions/` and `base-submissions/` fill. Miners announce with the same
signed claim as a fleet miner plus `"kind": "candidate"`, against the
validator's public URL, with their own public artifact port in the claim.

Limits that stay: HTTPS, and a miner may hold one adapter and one candidate
announcement per round. Rollback: set `"admission": "members"` and remove
`public_endpoints`; the next round uses the static roster again. Removing
`base_round_dir` returns to operator-scored decisions.

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

### 6. Pay the base tier from the same vector

A testnet validator config may carry a `base_tier` track. Each publication then
blends base-tier skill with the round's adapter skill into the one weight vector
sent to the subnet; see [the miner view](mining.md#two-tracks-on-the-testnet-the-base-tier-pays-first)
for the payment rule. Add the track to `validator/config.json` by hand:

```json
"base_tier": {
  "share": 0.7,
  "incumbent_sha256": "CHECKPOINT_HASH_OF_THE_PINNED_BASE_REFERENCE",
  "incumbent_uid": 1,
  "draws": [{"manifest_sha256": "…", "dataset_sha256": "…"}],
  "policy": {"min_brier_improvement": 0.01}
}
```

`share` is the base tier's fraction of the vector and must be explicit.
`incumbent_sha256` is the `zils submit` checkpoint hash of the incumbent
reference the scorer compares against. `incumbent_uid`, optional, is the roster
UID of the miner whose promoted candidate is the incumbent: king of the hill,
it keeps the whole base share in every epoch without an eligible challenger,
including epochs with no decision file. Leave it unset while the incumbent is
the vendor base, which nobody owns. `draws` lists every sealed draw whose
decision may pay in this epoch; print an entry with
`python -m zils.base_round --round SEALED_DIR --describe-draw`, which audits the
directory first. Remove a draw to rotate the holdout. `policy` may only override
the `zils.base_round` defaults and is optional. The fleet refuses to start with
a track on a non-testnet config or with an invalid track.

The validator collects and scores base candidates itself when the config also
names `base_round_dir` (a directory with one sealed draw per subdirectory; each
listed draw must be present, extra directories are ignored) and
`base_reference` (the incumbent reference whose checkpoint hash is
`incumbent_sha256`). Both paths are relative to the config; the fleet refuses to
start if either check fails or the profile is not a d3-family base. In every
round, admitted miners may then announce a candidate with the adapter claim plus
`"kind": "candidate"` (one adapter and one candidate announcement per miner per
round; the incumbent's own hash and a hash another miner announced are refused).
After the adapter evaluation the round:

1. downloads each distinct checkpoint hash once into
   `validator/base-tier/store/<sha256>/checkpoint/` (9.5 GiB budget, the
   d3 candidate contract checked by `validate_candidate`; a hash seen in an
   earlier round is neither fetched nor rescored, which is the cross-round cap);
2. runs `zils evaluate` in the sandbox for the incumbent (uid 65535) and every
   candidate without a stored score, on the calibration and test split of each
   listed draw, and keeps each row under the hash
   (`<manifest_sha256>-<split>.json`); a failed evaluation is kept too, so a
   checkpoint that crashes the runner is not retried (delete its entry to force
   a rescore);
3. calls `base_round.decide` per draw over this round's announced candidates,
   records refused announcements under `rejected`, and, with two or more draws,
   adds `confirmed_winner` from `base_round.confirm(first, last)`;
4. writes the last draw's decision to `validator/base-tier/decision.json` and a
   copy of every draw's decision into the round directory. The sidecar is
   removed at the start of the epoch, so a failed epoch pays nobody.

List the confirmation draw only once the required interval since the decision
draw has passed; the validator scores every listed draw in the same epoch.
Miners receive aggregate metrics only (`base_tier` in the signed result);
per-case predictions stay in the store. Without `base_round_dir` the validator
still accepts no candidates and the operator scores them with
`python -m zils.base_round … --out validator/base-tier/decision.json`
(re-score when a new candidate arrives; the confirmation draw's decision
replaces the first draw's; the incumbent row must use a UID outside the roster,
such as 65535). Either way, each round freezes the current decision into its
directory as `base-tier.json` before the first publish attempt and embeds it in
the report, so the report hash binds it and a restart republishes the same
weights. A missing decision file means nothing was scored this epoch: the round
publishes adapter weights alone.

The publisher refuses a decision for a draw not in the track, scored with a
different policy or against a different incumbent, paying a UID outside the
roster, with duplicate or incumbent hashes among the eligible candidates, whose
pending winner is not an eligible candidate, or whose recorded weights differ
from that winner (the winner takes the whole base share); it also refuses a
report carrying a decision when the config has no track. The attempt and
receipt record `tracks` (`base_share`, per-track normalized weights) next to
`requested_weights`, so the on-chain proportions can be traced to each track.

The [first verified testnet round](testnet-round-001.md) demonstrates this closed
path. It does not establish current validator uptime, open access, mainnet
support, or queue training-to-chain integration.
