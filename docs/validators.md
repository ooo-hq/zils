# Run a validator

Validators independently verify submitted artifacts, calibrate probabilities,
and score held-out examples. Miner-reported metrics do not determine rewards.
A fleet validator publishes weights only after explicit configuration
and review (`--publish-weights`). Base-tier miners join by registering on Bittensor
mainnet subnet 127 (network `finney`). A validator config names its network and
netuid explicitly; mainnet is never a default.

## Queue validator

The model evaluation and acceptance code lives in this repository. The hosted
queue processor connects that evaluator to customer jobs and protected storage;
its setup moved to
[zils-platform's queue validator guide](https://github.com/ooo-hq/zils-platform/blob/main/docs/validators.md#jevk5-queue-validator).
Only approved operators should receive its service credentials. Miners need no
platform installation or database credential.

d3-mini serves both text and image jobs, so one d3 reference covers both. The
retired H2O and Imajev profiles are no longer evaluated.

To grade miners that serve z3 checkpoints, see [grading serving miners](serving-validators.md).

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

A validator that takes adapter weights from signed queue evidence (`adapter_evidence`)
can use `"admission": "chain"` without `public_endpoints`. It needs no inbound
address: base candidates come from chain commitments and adapter work from the
evidence export, so it refuses adapter announcements from chain members.

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
   suite and a pool-specific admission document (a passing independent pilot or an
   explicit incumbent-challenge contract). The old single-base Brier configuration
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
[`zils-miner commit-candidate`](base-candidates.md#commit); they need neither the
validator's URL nor its hotkey. One commitment per hotkey remains the submission
contract. A registration change cannot inherit another hotkey's evidence.

Settle or explicitly handle pending rounds before changing configuration. Use
`admission: "members"` with a static roster and remove `public_endpoints` to
close public admission. Disabling pool intake does not authorize returning to
old-policy crowns; preserve the reviewed incumbent and existing customer pins.

## Bittensor fleet validator

This path uses a model-pinned fleet, a static roster and private-network
transport. For the base tier open to every registered hotkey, see
[open admission](#open-admission-base-tier) above.

### 1. Register and check eligibility

Follow [Bittensor registration](bittensor-registration.md) using a dedicated
validator hotkey, distinct from every miner hotkey. Use the operator-confirmed
network and netuid (`finney` and 127 for the public subnet). Miners and validators both register a hotkey to obtain a UID.

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
[chain fleet setup](testnet.md#provision-the-registered-identities).
Keep test/calibration data on the validator and distribute only each miner's
assigned bundle. The validator host needs its hotkey, not the coldkey.

With the generated configuration in place, run the read-only check:

```bash
.venv-kev/bin/python -m zils.testnet preflight \
  --config .private/testnet-fleet/validator/config.json
```

Confirm the network and netuid, validator UID, every miner UID-to-hotkey mapping,
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
**submits a weight transaction on the configured network** using the validator hotkey:

```bash
.venv-kev/bin/python -m zils.testnet publish \
  --config .private/testnet-fleet/validator/config.json \
  --round .private/testnet-fleet/validator/state/rounds/ROUND
```

Only one process should publish for this hotkey and subnet. For subsequent
continuous rounds, run `zils.fleet validator` with `--publish-weights` and omit
`--rounds 1`. Zils uses mechanism 0 and weight version 1. It stops if the chain
requires a newer version. `no_weights` and `rate_limited` submit nothing. A
frozen member whose UID holds another hotkey (or none) by publication time is
dropped from the vector and listed under `replaced`; its share burns at the
fleet's burn uid, so a replacement never inherits the evaluated score.

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
`migrate_model_pools`; every pool remains disabled until its own admission document,
private paths and v2 draws are installed. The previous single-base Brier policy
and untimed caches cannot qualify new candidates. Settle pending old-publication
attempts with the old release before cutover; retain uncertain chain receipts.

The existing base budget is split by pool allocation. Within a pool, the last
five distinct registered king hotkeys split its share. Exam winners and ready
candidates earn nothing until an approved promotion changes the incumbent and
king history. Disabled/unowned allocations burn under the existing subnet-owner
rule; failures in one pool do not transfer its allocation to another.

For bounded asynchronous candidate evaluation, enable the
[budgeted base-validation queue](budgeted-base-validation.md) and its separate
worker. Pending candidate work does not replace the existing king history.

Each pool's private `base_pool_rounds` entry names `base_reference`,
`base_round_dir`, `family_history`, `suite_manifest` and `pilot_report`. The
validator audits current history, score policy, pool admission and draw independence
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

#### Soup pools: hardware, disk and the emission schedule

Every base pool at launch is a [soup pool](model-pools.md#soup-pools): `d3-lite`,
`d3-mini`, `d3-flash` and `d3` (z3). `zils/data/base-pools-launch.json` is the
launch `base_pools` block: shares, daily commit deadlines, intake
caps and the [emission schedule](model-pools.md#emission-schedule). Every
validator must run the same `base_pools` block, including the same opening
blocks, or its weights disagree with the others'. The pools' house and known-public lists are not
part of the block: they ship with the code, so updating to a release updates
them ([model pools](model-pools.md#soup-pools)).

A soup round loads one model at a time on the GPU. Approximate memory for one
grading in bf16, and disk for the pool's state at its launch caps:

| Pool | GPU memory | Body size | Bodies kept (pool cap + house) | Disk for bodies |
|---|---|---|---|---|
| `d3-lite` | 4 GiB | 1.7 GB | 20 + 2 | about 40 GB |
| `d3-mini` | 12 GiB (9.5 GB measured) | 9.1 GB | 20 + 5 | about 230 GB |
| `d3-flash` | 20 GiB | 17 GB | 10 + 1 | about 190 GB |
| `d3` (z3) | 56 GiB | 52 GB | 5 + 1 | about 320 GB |

Add the pool's chassis and Qwen parent (each about one body) and the round's
scratch soup (one more body). Only the mini figure is measured; the others are
estimates from the weights' size. FP8 grading is not supported, so z3 needs a
card with at least 56 GiB free (an 80 GB card). Rounds of different pools share
the validator's GPU lock and run one after another; the launch deadlines are
staggered (mini 00:00, lite 06:00, flash 12:00, z3 18:00 UTC). A round has a
6-hour limit; a z3 round's duration has not been measured.

Declare what this validator can grade with a local `soup_grading` setting (it
is not part of `base_pools` and is never published):

```json
"soup_grading": {"pools": ["d3-lite", "d3-mini", "d3-flash"], "gpu_mib": 24564}
```

`pools` lists the soup pools to grade (default: all); `gpu_mib` is the card's
memory, and a pool needing more is skipped. A skipped pool runs no rounds, needs
no `base_pool_rounds` entry, and is logged in the round's `soup-pools.json` with
its reason. Its share burns in this validator's weights, while the pools it
grades pay as usual. A validator that skips a pool other validators grade
disagrees with them on that pool's share, so run the pools your hardware fits
and add the others when you can. Without `soup_grading`, every enabled soup pool
is graded.

Each graded pool's `base_pool_rounds` entry names `round_bank` and `base`, the
folder of the pool's Qwen parent at its pinned revision (`Qwen/Qwen3.5-0.8B`,
`Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B` or `Qwen/Qwen3.8-27B`; revisions in
[mining](mining.md#four-pools)). Round items longer than a pool's input cap
(4,096 tokens for lite, flash and z3; 8,192 for mini) are left out of that pool's
rounds and counted in the round log. The cap is checked with the chassis
tokenizer on the fixed round bank; Zils Outcomes items are filtered by an
estimated token count ([current](current.md)).

### 7. Pay the adapter track from paid adapter use

Off by default. With an `adapter_evidence` section the adapter share pays miners
whose delivered customer adapters earn paid API use, instead of the frozen-job
round, whose rows are then ignored for weights:

```json
"adapter_evidence": {
  "url": "https://TRAINING_API/v1/network/adapter-evidence",
  "signer": "SS58_OF_THE_PLATFORM_EVIDENCE_HOTKEY",
  "window_hours": 72,
  "revenue_cap": 0.5
}
```

Each round, before its first publish attempt, the validator requests the window
of `window_hours` ending at the current whole UTC hour, verifies the payload is
signed by `signer` (a dedicated platform key, not a miner's) for exactly that
window and schema `zils-adapter-evidence/v2`, and freezes the signed document in
the round as `adapter-evidence.json`. The report embeds it, so the report hash
binds it and a restart republishes the same vector. A failed fetch stops the
round; it retries after a restart rather than burning the share.

The platform exports, per hotkey, the live API spend on open-tier adapters it
delivered, credited for 30 days after delivery, and the window's TAO/USD price;
no job ids, hashes or owners. `zils.queue_evidence` applies the rule:

- **Cap.** `paid = min(1, revenue_cap × revenue / value)`, where `value` is the
  adapter share of the miners' 41% of the subnet's alpha emission over the
  window, at the subnet's moving alpha price (`SubnetMovingPrice`,
  `SubnetAlphaOutEmission`, read at the round block) times `tao_usd`. A miner
  posing as a customer pays at least `1 / revenue_cap` of what it earns back.
- **Split.** Each registered hotkey gets `paid × spend / revenue` of the share.
  The rest, including spend credited to unregistered hotkeys, burns at the burn
  uid (`base_pools.burn_uid` or `base_tier.burn_uid`; the section needs one).

No paid spend: the adapter share burns. A hotkey is paid at the uid it holds in
the round's chain snapshot. The publisher refuses evidence from another signer
or window, an invalid row, a hotkey listed twice, a report without evidence when
the section is set, and evidence when it is not. Receipts record
`"adapter_source": "queue"`, `adapter_paid` and the chain price inputs. The fleet
refuses the section on a config without a `chain` section.

Trust: the platform's processor evaluated the work and its billing ledger
recorded the spend; the signature proves the export came from the platform, not
that either was right. Customer data and usage stay private, so other validators
can check the export's signature and the chain price but not re-count the spend.

The [first verified testnet round](testnet-round-001.md) demonstrates this closed
path. It does not establish current validator uptime, open access,
or queue training-to-chain integration.

### 8. Pay the serving share

Off by default. With a `serving_share` section the validator pays a share of the
vector to miners that serve z3 checkpoints, from the scores its
[serving grader](serving-validators.md#pay) writes. The share burns until enough
miners earn it.
