# Run a validator

Validators independently verify submitted artifacts, calibrate probabilities,
and score held-out examples. Miner-reported metrics do not determine rewards.
A fleet validator publishes weights only after explicit configuration
and review (`--publish-weights`). Miners join by registering on Bittensor
mainnet subnet 127 (network `finney`). A validator config names its network and
netuid explicitly; mainnet is never a default.

## Queue validator

The model evaluation and acceptance code lives in this repository. The hosted
queue processor connects that evaluator to customer jobs and protected storage;
its setup moved to
[zils-platform's queue validator guide](https://github.com/ooo-hq/zils-platform/blob/main/docs/validators.md#queue-validator).
Only approved operators should receive its service credentials. Miners need no
platform installation or database credential.

Customer jobs train on z3-mini (`d3-mini-4.5b-v1`, 8,192-token inputs), so one
z3-mini reference covers them.

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
address: soup bodies come from chain commitments and adapter work from the
evidence export, so it refuses adapter announcements from chain members.

Registration on the subnet is the admission check; there is no further vetting.
Every submission is still hash-verified and size-bounded. Frozen-job adapters
are scored in the isolated runner described in [evaluation](evaluation.md); soup
bodies are graded in soup rounds ([model pools](model-pools.md#soup-pools)).

### What open admission exposes

The fleet validator still runs a **frozen-job round** each round, and with
`public_endpoints` it ships that job's training split to every admitted miner
that announces. Use a job built from public data, never a customer's. With
`adapter_evidence` (section 7) the frozen-job rows do not set weights, and with
`"admission": "chain"` and no `public_endpoints` the validator takes no
announcements at all: soup bodies come from chain commitments and adapter pay
from the signed evidence.

Customer data reaches miners only through the hosted queue, under the
[miner terms](mining-terms.md) (their hash travels in every signed queue claim)
and [chain admission](miner-admission.md). The base pools need no customer data:
bodies come in, and the private round bank never leaves the validator. Configure
`base_pools` and each graded pool's `base_pool_rounds` entry as in section 6.

### Network and operator setup

Provision the `fleet-validator` role with `scripts/provision_node.sh` on hardware
qualified for the selected pool and exam. Keep the validator hotkey on the host,
never a coldkey. Reserve disk space for each distinct accepted checkpoint; the
per-pool store keeps downloaded weights across epochs.

Soup bodies require outbound HTTPS to Hugging Face and its allowed CDN
redirects. The announcement port serves adapter rounds and signed results.
Adapter artifacts are fetched from each miner's announced endpoint, pinned to
the announcing connection's source address. Claims are signed and artifacts
hash-verified; a TLS terminator provides transport privacy.

Run `python -m zils.testnet preflight --config CONFIG` before starting the unit.
Inspect the frozen `chain-snapshot.json` member/commitment maps and each pool's
round records. Miners use
[`zils-miner commit-body`](mining.md#commit-a-body); they need neither the
validator's URL nor its hotkey. One commitment per hotkey remains the submission
contract. A registration change cannot inherit another hotkey's evidence.

Settle or explicitly handle pending rounds before changing configuration. Use
`admission: "members"` with a static roster and remove `public_endpoints` to
close public admission.

## Bittensor fleet validator

This path uses a model-pinned fleet, a static roster and private-network
transport. For pools open to every registered hotkey, see
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

Complete the [fleet model setup](mining.md) for the operator-selected profile,
including the matching reference and qualified hardware.
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

`base_pools` pays the base pools from the same weight vector. At launch it holds
80% of emissions (`share` 0.8) across four soup pools; the adapter track
(section 7) gets the other 20%. Lite and mini get 25% of all emissions each and
are open; flash and z3 get 15% each and stay closed, burning their share, until
their opening blocks are set. Anything a pool does not pay burns at `burn_uid`,
the subnet owner's uid; one pool's failure never moves its share to another.

A pool entry without `"type": "soup"` is a king pool, paid from a king history
through [base elections](base-election.md). The code still supports king pools,
but no launch pool is one.

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
round, whose rows are then ignored for weights. The adapter share is what
`base_pools`, `feature_tracks` and `serving_share` leave: 20% in the launch
configuration.

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

The [first verified testnet round](testnet-round-001.md) is a historical record
of the earlier frozen-job path, before adapter pay came from spend.

### 8. Pay the serving share

Off by default, and not part of the launch configuration. With a
`serving_share` section the validator pays a share of the vector to miners that
serve z3 checkpoints, from the scores its
[serving grader](serving-validators.md#pay) writes. The share burns until enough
miners earn it.
