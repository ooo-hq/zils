# Miner admission

**Draft, 18 October 2026. Documented only; nothing here is built or enabled.**

Registration on Bittensor subnet 579 is the admission. There is no application,
no form and no operator approval. This matches the fleet validator's
`admission: chain` mode (built, off) and extends it to the hosted customer-job
queue, which today only talks to hotkeys an operator inserted by hand.

## 1. What a miner does

1. Register a hotkey on subnet 579 ([registration](bittensor-registration.md)).
2. Read the [miner terms](mining-terms.md). Mining under a registered hotkey is
   acceptance; the first signed claim records it.
3. Install the runtime and the d3 reference ([queue miners](queue-miners.md)),
   run `scripts.qualify_node --model d3-mini-4.5b-v1`, start the worker.
4. Work arrives: qualification jobs first (synthetic, grade the host), then
   customer jobs once three reports pass in the band, and base-tier rounds
   from the validator.

## 2. What the coordinator does (code, not built)

- Admit any hotkey registered on 579 at the current chain snapshot: create the
  `fez_training_workers` row on first contact with the next free queue UID,
  `terms_sha256` from the claim, `terms_accepted_at` now. Deregistered hotkeys
  are disabled at the next snapshot.
- Read the profile and runtime hashes from the miner's own qualification report
  on first presence, so the `zils_worker_profiles` row is written by the
  coordinator, not the operator; the database guard on `min_free_mib` still
  applies.
- Queue qualification jobs automatically for a new hotkey (today an operator
  submits and authorizes them, about $12 of credit for five on 17 October 2026).
  Graded routing is unchanged: three passing reports before customer work.
- Route a customer job to outside hotkeys only when the job's `outside_miners`
  flag allows it ([customer jobs](customer-jobs.md#prepare-data)); Zils-run
  hotkeys are marked `operated_by_zils` and take everything.

The base-tier path is the
[open-admission runbook](validators.md#open-admission-base-tier-not-yet-enabled)
(ooo-hq/zils PR #51); it needs the validator's base-tier intake.

## 3. Operator duties left

Revocation (`worker --disable` plus a notice if a contact is known), audits,
and the routing pool. Nothing per-applicant.

## 4. Qualification credit

Zils pays it. Registration already costs the miner a burn, which is the
spam gate Bittensor provides; a second gate here would be a form by another
name. Cap it: one qualification run per hotkey per 30 days.

## 5. Miners page

Replace "All miners are currently run by Zils; outside miners can join once
the subnet moves to mainnet" with:

> Subnet 579 on the Bittensor testnet. Register a hotkey on 579, run the miner,
> and you are in under the [miner terms](mining-terms.md). Testnet rewards have
> no value and no earnings are promised.

zils-web change, out of scope here.

## 6. Code still needed

- Coordinator: admit chain-registered hotkeys, write worker and profile rows on
  first contact, auto-queue qualification jobs.
- `terms_sha256` in the signed claim and in `qualification.json`;
  `terms_sha256`, `terms_accepted_at`, `operated_by_zils` on
  `fez_training_workers`.
- `outside_miners` on `fez_training_jobs` and the create-job body, honoured by
  the graded scheduler.
- Validator base-tier intake (separate session).
