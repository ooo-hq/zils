# Run a validator

Validators independently verify submitted artifacts, calibrate probabilities,
and score held-out examples. Miner-reported metrics do not determine rewards.
A testnet fleet validator publishes weights only after explicit configuration
and review (`--publish-weights`). Base-tier miners join by registering on testnet
subnet 579; mainnet is not implemented.

## Queue validator

The model evaluation and acceptance code lives in this repository. The hosted
queue processor connects that evaluator to customer jobs and protected storage;
its setup moved to
[zils-platform's queue validator guide](https://github.com/ooo-hq/zils-platform/blob/main/docs/validators.md#jevk5-queue-validator).
Only approved operators should receive its service credentials. Miners need no
platform installation or database credential.

d3-mini serves both text and image jobs, so one d3 reference covers both. The
retired H2O and Imajev profiles are no longer evaluated.

## Open admission (base tier)

Off by default. The default validator admits only the static `members` allowlist and pins each
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
qualified validator configured with `base_pools` collects and scores base candidates
itself (section 6 below). So, in order:

1. **Policy (owner, decided 18 October 2026).** Outside miners may train customer
   jobs as well as base candidates, under the [miner terms](mining-terms.md)
   (published 18 October 2026; their hash travels in every signed queue claim).
   The hosted queue's chain-derived admission is built but off, so a public
   validator carries a Zils-authored job (public data), not a customer's.
2. **Qualified pool intake.** Configure `base_pools` plus the private paths for
   each qualified pool in `base_pool_rounds`. Every round freezes chain
   commitments, routes each candidate by its pinned parent, rejects incumbent
   copies and evaluates eligible candidates in the isolated runner. Per-pool
   stores retain checkpoint bytes and actual evaluation times across epochs.
3. **Preparation and rollout.** Follow the [base-election runbook](base-election.md).
   New intake requires v2 private exams, public family history, a pinned adapter
   suite and a passing independent pilot. The old single-base Brier configuration
   is archival and cannot start new-policy intake.

### Network and operator setup

Provision the `fleet-validator` role with `scripts/provision_node.sh` on hardware
qualified for the selected pool and exam. Keep the validator hotkey on the host,
never a coldkey. Reserve disk space for each distinct accepted checkpoint; the
per-pool store keeps downloaded weights across epochs.

Base candidates require outbound HTTPS to Hugging Face and its allowed CDN
redirects. The announcement port serves adapter rounds and signed results.
Adapter artifacts are fetched from each miner's announced endpoint, pinned to
the announcing connection's source address. Claims are signed and artifacts
hash-verified; a TLS terminator provides transport privacy.

Run `python -m zils.testnet preflight --config CONFIG` before starting the unit.
Inspect the frozen `chain-snapshot.json` member/commitment maps and each pool's
intake and decision reports. Miners use
[`zils commit-candidate`](base-candidates.md#commit); they need neither the
validator's URL nor its hotkey. One commitment per hotkey remains the submission
contract. A registration change cannot inherit another hotkey's evidence.

Settle or explicitly handle pending rounds before changing configuration. Use
`admission: "members"` with a static roster and remove `public_endpoints` to
close public admission. Disabling pool intake does not authorize returning to
old-policy crowns; preserve the reviewed incumbent and existing customer pins.

## Bittensor testnet validator

This path uses a model-pinned fleet, a static roster and private-network
transport. For the base tier open to every registered hotkey, see
[open admission](#open-admission-base-tier) above.

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

New text-base elections use `base_pools` and the
[decision-quality runbook](base-election.md). Stage old configurations using
`migrate_model_pools`; every pool remains disabled until its own exam pilot,
private paths and v2 draws are installed. The previous single-base Brier policy
and untimed caches cannot qualify new candidates. Settle pending old-publication
attempts with the old release before cutover; retain uncertain chain receipts.

The existing base budget is split by pool allocation. Within a pool, the last
five distinct registered king hotkeys split its share. Exam winners and ready
candidates earn nothing until an approved promotion changes the incumbent and
king history. Disabled/unowned allocations burn under the existing subnet-owner
rule; failures in one pool do not transfer its allocation to another.

Each pool's private `base_pool_rounds` entry names `base_reference`,
`base_round_dir`, `family_history`, `suite_manifest` and `pilot_report`. The
validator audits current history, score policy, pool pilot and draw independence
before inference. Candidate and incumbent cache files retain their actual start
and completion times. A second draw cannot begin for a candidate until its own
first evidence and the incumbent's evidence are at least 12 hours old.

Private evidence is retained under `base-pools/<pool>/base-tier/decisions` and
`store`. The public decision sidecar contains aggregate scores and evidence hashes.
Each pending round freezes pool configuration and public results before a publish
attempt. Changing configuration cannot rewrite that already-frozen result.

At scheduled 14–28 day windows, `base_release` verifies every ready receipt,
freezes the complete roster and evaluates a common fresh draw. `promote_base`
rechecks the window against the active configuration and proposes the king/profile
change. Upload, live config installation and chain publication remain separate
operator actions. Stable migration cannot be inferred from an election result.

### 7. Pay the adapter track from hosted-queue work

Off by default. With an `adapter_evidence` section the adapter share pays real
customer work from the hosted queue instead of the frozen-job round, whose rows
are then ignored for weights:

```json
"adapter_evidence": {
  "url": "https://TRAINING_API/v1/network/adapter-evidence",
  "signer": "SS58_OF_THE_PLATFORM_EVIDENCE_HOTKEY",
  "window_hours": 72
}
```

Each round, before its first publish attempt, the validator requests the window
of `window_hours` ending at the current whole UTC hour, verifies the payload is
signed by `signer` (a dedicated platform key, not a miner's) for exactly that
window and schema `zils-adapter-evidence/v1`, and freezes the signed document in
the round as `adapter-evidence.json`. The report embeds it, so the report hash
binds it and a restart republishes the same vector. A failed fetch stops the
round; it retries after a restart rather than burning the share.

The platform exports facts per finished attempt (hotkey, purpose, outcome,
tokens, artifact hash, delivered or not, held-out skill, whether the job is
operator-owned, an owner hash); no job ids, names or owners. `zils.queue_evidence`
applies the rule per hotkey over the window: score = Σ `total_tokens × skill`
over customer attempts that are valid, delivered to the customer with that
attempt's artifact, and not operator-owned; weight = score × reliability, where
reliability is `(valid + 1) / (valid + failures + 2)` over all its attempts
(failures: `invalid_artifact`, `worker_failure`, `abandoned`, as graded routing
counts them). Qualification runs never earn. A hotkey is paid at the uid it
holds in the round's chain snapshot (the queue's own UID is unrelated); an
unregistered hotkey earns nothing. No earner: the adapter share burns at
`base_tier.burn_uid`, as in §6. The publisher refuses evidence from another
signer or window, an invalid row, one artifact hash earning twice, a report
without evidence when the section is set, and evidence when it is not.
Receipts record `"adapter_source": "queue"`. The fleet refuses the section on a
non-testnet config.

Trust: the platform's processor evaluated the work; the signature proves the
export came from the platform, not that the evaluation was right. Customer test
data stays private, so other validators can audit the export but not re-score it.

The [first verified testnet round](testnet-round-001.md) demonstrates this closed
path. It does not establish current validator uptime, open access, mainnet
support, or queue training-to-chain integration.
