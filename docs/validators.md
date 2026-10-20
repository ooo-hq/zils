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
Imajev image profile is retired; a processor that still evaluates frozen Imajev
jobs keeps its isolated runtime, [image qualification](https://github.com/ooo-hq/zils-platform/blob/main/docs/image-training.md#worker-and-service-setup)
and `--additional-reference models/imajev-starting-checkpoint`, which selects the
matching model for each frozen job without changing the primary reference.

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
validator configured with `base_round_dir` collects and scores base candidates
itself (section 6 below). So, in order:

1. **Policy (owner, decided 18 October 2026).** Outside miners may train customer
   jobs as well as base candidates, under the [miner terms](mining-terms.md)
   (published 18 October 2026; their hash travels in every signed queue claim).
   The hosted queue's chain-derived admission is built but off, so a public
   validator carries a Zils-authored job (public data), not a customer's.
2. **Base-tier intake (merged; chain commitments since the 18 October owner
   decision).** With `base_round_dir` set, every round also reads each member's
   candidate commitment at the preflight block, pulls new hashes from Hugging Face
   (9.5 GiB each, downloaded once per distinct hash and kept across epochs so an
   unchanged commitment costs nothing), scores them with
   `base_round` on the configured draws inside the sandbox, and writes the decision
   sidecar itself. The per-hash store is the cross-round submission cap.
3. **Runbook below.**

### Runbook: a public validator for the base tier

Host: a GPU node of node-1's class (RTX 4000 Ada 20 GB; scoring needs about
10 GB; about $0.76/h on DigitalOcean), provisioned with
`scripts/provision_node.sh` and the `fleet-validator` role, plus 10 GiB of disk
per distinct candidate hash ever accepted (the store under
`validator/base-tier/store/` keeps them across epochs; prune by hand). Keep the
validator hotkey on the host, never a coldkey.

Network: base candidates need no inbound port at all. The validator reads the
commitments from the chain and downloads outbound over HTTPS from
`huggingface.co` (allow its CDN redirects). The announcement port (8900 by
default) serves only the adapter track and the signed round results; adapter
artifacts are fetched *outbound* from each miner's announced endpoint, pinned to
the announcing connection's source address. Plaintext HTTP is accepted there
because claims are signed and artifacts hash-verified; put a TLS terminator in
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
    "kings": [],
    "burn_uid": 0,
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
`kings` starts empty (stock d3 has no owner, so the base share burns) and
`burn_uid` must be the subnet owner hotkey's uid; see §6.
`members` is omitted: preflight derives it from the chain and freezes it per
round. `model` is the pinned d3 profile as `zils.fleet init` writes it (the
intake scores d3-family candidates only). `base_round_dir` holds one sealed draw per subdirectory (copy each
sealed round directory there) and `base_reference` is the incumbent reference
(`zils.d3 reference --out validator/reference`), both relative to the config.
Run `python -m zils.testnet preflight --config …` and check that the
member list is what the metagraph shows minus the validator. Start the unit; it
refuses to start when a listed draw has no directory or the reference hash is
not `incumbent_sha256`; it also refuses a base tier without a `chain` section,
since candidates are chain commitments. Then watch
`state/rounds/<round>/chain-snapshot.json` (its `commitments` map, uid to block
and `zils1:<hash>:<revision>:<repo>`) and `submissions/` fill. Miners commit with
`zils.miner commit-candidate` ([base candidates: commit](base-candidates.md#commit));
they need neither the validator's URL nor its hotkey.

Limits that stay: HTTPS for the adapter track, one adapter announcement per
miner per round, one candidate commitment per hotkey. Rollback: set `"admission": "members"` and remove
`public_endpoints`; the next round uses the static roster again. Removing
`base_round_dir` returns to operator-scored decisions.

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

A testnet validator config may carry a `base_tier` track. Each publication then
blends base-tier skill with the round's adapter skill into the one weight vector
sent to the subnet; see [the miner view](mining.md#two-tracks-on-the-testnet-the-base-tier-pays-first)
for the payment rule. Add the track to `validator/config.json` by hand:

```json
"base_tier": {
  "share": 0.7,
  "incumbent_sha256": "CHECKPOINT_HASH_OF_THE_PINNED_BASE_REFERENCE",
  "kings": ["HOTKEY_OF_THE_CURRENT_KING", "HOTKEY_OF_THE_PREVIOUS_KING"],
  "burn_uid": 0,
  "draws": [{"manifest_sha256": "…", "dataset_sha256": "…"}],
  "policy": {"min_brier_improvement": 0.01}
}
```

`share` is the base tier's fraction of the vector and must be explicit.
`incumbent_sha256` is the `zils submit` checkpoint hash of the incumbent
reference the scorer compares against. `kings`, optional, is the king history:
hotkeys, newest first, the first being the miner whose promoted candidate is
the incumbent. King of the hill (Teutonic's rule): the base share of every
publication is split equally over the first five distinct hotkeys of the
history that are registered at the round's preflight block, wherever their uid
is now, with or without a decision file; a deregistered king's slice goes to
the rest, and exam and confirmed winners are paid nothing. Leave it empty
while the incumbent is the vendor base, which nobody owns. `burn_uid`
(default 0) receives the share of a track with no recipient: no registered
king, or no adapter earner; with both empty it receives the whole vector, so
the validator keeps setting weights. Publication refuses a burn slice unless
`burn_uid` is the uid of the subnet owner hotkey (subtensor burns the owner
hotkey's incentive and keeps only the owner's self-weight; another
validator's weight on itself is masked by Yuma). Crowning is a config change in
the owner's promotion PR: set `incumbent_sha256` to the promoted candidate's
hash and prepend its miner's hotkey to `kings` (`scripts/promote_base.py
--king-hotkey HOTKEY --previous-kings CURRENT,KINGS` prints both as
`proposed_base_tier`), point `base_reference` at the promoted checkpoint, and
restart the validator. `incumbent_uid` is gone; remove it from older configs.
`draws` lists every sealed draw whose
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
start if either check fails, the profile is not a d3-family base, or the config
has no `chain` section. Candidates are chain commitments, not announcements:
preflight reads every member's `Commitments.CommitmentOf` at the round's
preflight block and freezes them in `chain-snapshot.json` under `commitments`.
After the adapter evaluation the round:

1. turns the snapshot into candidates with `zils.commitments.candidates`:
   a commitment that is not `zils1:<sha256>:<40-hex revision>:<owner/repo>`, that
   names the incumbent's hash, or that carries a hash another hotkey committed at
   an earlier block (or in the same earliest block) is recorded under `rejected`;
2. downloads each distinct new checkpoint hash once from
   `https://huggingface.co/<repo>/resolve/<revision>/<file>` into
   `validator/base-tier/store/<sha256>/checkpoint/`: only
   `models.D3_CANDIDATE_FILES`, after the repo listing at that exact commit shows
   them within the 9.5 GiB budget, which is enforced again on every byte; then the
   checkpoint hash and `validate_candidate`. A hash seen in an earlier round is
   neither fetched nor rescored, which is the cross-round cap;
3. runs `zils evaluate` in the sandbox for the incumbent (uid 65535) and every
   candidate without a stored score, on the calibration and test split of each
   listed draw, and keeps each row under the hash
   (`<manifest_sha256>-<split>.json`); a failed evaluation is kept too, so a
   checkpoint that crashes the runner is not retried (delete its entry to force
   a rescore);
4. calls `base_round.decide` per draw over this round's committed candidates,
   records refused commitments and failed downloads under `rejected`, and, with
   two or more draws,
   adds `confirmed_winner` from `base_round.confirm(first, last)`;
5. writes the last draw's decision to `validator/base-tier/decision.json` and a
   copy of every draw's decision into the round directory. The sidecar is
   removed at the start of the epoch, so a failed epoch records no challenger
   (the kings, if any, are still paid).

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
weights. A missing decision file means nothing was scored this epoch: the
kings, if any, are still paid their share.

The publisher refuses a decision for a draw not in the track, scored with a
different policy or against a different incumbent, paying a UID outside the
roster, with duplicate or incumbent hashes among the eligible candidates, or
whose pending winner is not an eligible candidate; it also refuses a report
carrying a decision when the config has no track. An accepted decision pays
nobody by itself: the base track is always the kings (`kings`) or empty.
A decision written before 19 October may still carry `weights`; they are
ignored. The attempt and
receipt record `tracks` (`base_share`, `burn_uid`, the `burned` fraction,
per-track normalized weights) next to
`requested_weights`, so the on-chain proportions can be traced to each track.

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
