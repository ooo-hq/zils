# Miner admission

**18 October 2026. Built in zils-platform (`zils_platform/admission.py`, migration
`202610180001_chain_admission.sql`) and on the miner side (this repository); off in
production until the owner sets `ZILS_CHAIN_ADMISSION=on` on the coordinator.**

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
   customer jobs once three reports pass in the band. The base tier is separate:
   upload a candidate to Hugging Face and commit it with
   `python -m zils.miner commit-candidate`
   ([compete from outside](mining.md#compete-in-the-base-tier-from-outside)).

## 2. What the coordinator does (built, off by default)

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
- Route customer jobs to the whole pool. Zils' own hotkeys are ordinary rows
  with no special path ([customer jobs](customer-jobs.md#prepare-data)).

The base-tier path is the
[open-admission runbook](validators.md#open-admission-base-tier)
(merged); the validator's base-tier intake reads candidates from chain
commitments made with `zils.miner commit-candidate` (ooo-hq/zils#59), so the
base tier needs no coordinator admission.

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

## 6. What exists and how to enable it

Everything in section 2 is implemented behind one switch. Nothing changes while
it is off: `fez_worker_nonce` still refuses hotkeys without an operator row.

**Miner side (this repository).** `.private/queue-miner.json` takes two optional
keys: `terms_sha256` (the SHA-256 of `docs/mining-terms.md` you accept) and
`qualification` (path to the `qualification.json` written by
`scripts.qualify_node --terms-sha256 <hash>`). At startup `miner.queue` POSTs
both to `/v1/workers/admit`, signed by the hotkey; every later claim carries
`terms_sha256`. A coordinator with admission off answers 409 and the miner
continues (operator-registered nodes need neither key).

**Coordinator side (zils-platform).**

- `POST /v1/workers/admit`: signature checked, hotkey looked up in the subnet's
  registration set (read from the testnet chain, cached five minutes; a hotkey
  that drops off the chain is refused at the next refresh), terms hash compared
  with the current one, report checked against the pinned profile. Then
  `zils_admit_worker` writes the `fez_training_workers` row (next free queue UID,
  `admission='chain'`, `terms_sha256`, `terms_accepted_at`, its own resource
  UUID) and the `zils_worker_profiles` row from the report (`min_free_mib`
  12288, `verified_by='chain-admission'`; a row an operator revoked is never
  revived).
- Every other signed route: a chain-admitted hotkey must still be registered and
  on the current terms; any claim presenting a `terms_sha256` must present the
  current one. Operator rows are untouched.
- Qualification: one job per hotkey per 30 days, cloned from the operator's
  validated benchmark job under the internal account (`ZILS_INTERNAL_OWNER_ID`,
  `ZILS_QUALIFICATION_SOURCE_JOB`), credit debited from that account. When the
  processor validates it, the job authorizes itself against the benchmark already
  installed for its model/band and enters the graded qualification lane. Chain
  rows are graded-pool members without being listed in `routing.json`; customer
  work still needs three passing graded reports, exactly as for Zils' own nodes.
- `GET /v1/config` publishes `admission` (`netuid`, `terms_sha256`) when on.
- Kill switch: `python -m zils_platform.coordinator worker --hotkey <ss58>
  --disable` (no `--uid` needed) disables the row; `ZILS_CHAIN_ADMISSION=off`
  restores operator-only behaviour for everyone.

**To enable (owner).** Apply the migration; set on the coordinator
`ZILS_CHAIN_ADMISSION=on`, `ZILS_INTERNAL_OWNER_ID` (the internal test
account's UUID, credited), `ZILS_QUALIFICATION_SOURCE_JOB` (one of the
validated `qual-d3-2048-*` jobs; its three input splits must still exist);
install `requirements/testnet.txt` in the API's environment for the chain read;
restart `zils-training-api`. When these terms change, the new hash goes in
`ZILS_MINER_TERMS_SHA256` (or the platform constant) and miners re-admit by
restarting with the new `terms_sha256`.

Still open: the miners-page wording (section 5) and HTTPS on the public
validator. The base-tier submission command exists (`zils.miner commit-candidate`,
ooo-hq/zils#59, which replaced the earlier announce command); `zils.miner doctor`
checks a miner's setup step by step.
