# Miner admission (self-serve apply flow)

**Draft, 18 October 2026. Documented only; nothing here is built or enabled.**
Admission to the hosted training queue is operator-only today: the coordinator
rejects any hotkey without an enabled `fez_training_workers` row, claims need an
operator-written `zils_worker_profiles` row, and graded routing needs three
qualification reports in the band. This page turns those manual steps into a
flow an outsider can start. It assumes the [miner terms](mining-terms.md) are
published. The chain-side path for base candidates is the
[open-admission runbook](validators.md#open-admission-base-tier-not-yet-enabled)
(ooo-hq/zils PR #51); it stays off until the validator has a base-tier intake.

## 1. Application form

One form (web or a signed email template; the web form is the later step).
Fields:

| Field | Check |
|---|---|
| Hotkey SS58 address | Valid SS58; not already a worker row. This hotkey signs claims and is the accepting identity for the terms. |
| Hardware | GPU model, VRAM, free disk, OS. d3 needs a BF16 NVIDIA card with 12,288 MiB free at start; see [queue miners](queue-miners.md#1-confirm-access-and-hardware). |
| Region | Country or cloud region of the host. Used for routing and for the lawyer's export questions. |
| Contact | Email that Zils can reach for revocation and audit notices. |
| Terms version accepted | The `terms_sha256` from [mining-terms.md](mining-terms.md#8-acceptance) and its date. Must match the current published version. |
| Qualification report | `qualification.json` from `scripts.qualify_node --model d3-mini-4.5b-v1` run on the actual host, with `terms_sha256` added. |

Everything else (wallet name, host address, coordinator URL) stays off the form.

## 2. Operator checklist

Today's manual path, in order. One applicant at a time until the steps are
scripted.

1. **Review.** Check the form, the report's profile and runtime hashes against
   the current d3 pins, and that `terms_sha256` matches the published terms.
   Reject on mismatch; nothing is created.
2. **Stake.** Confirm the qualification credit is paid (section 3).
3. **Worker row.** `python -m zils_platform.coordinator worker --hotkey <SS58> --uid <next free uid>`
   (enabled). Record the terms hash and application date with the row (new
   columns `terms_sha256`, `terms_accepted_at`; until they exist, in the
   operator's admission log).
4. **Profile row.** Insert the `zils_worker_profiles` row from the report:
   profile and runtime hashes, measured `min_free_mib` (the database refuses a
   d3 value below 12,288), `verified_by` the operator, evidence = the report.
5. **Capacity certificate.** Import it with
   `graded_scheduler import-qualification` as in
   [miner grading](miner-grading.md).
6. **Qualification jobs.** Submit the qualification benchmark jobs and authorize
   them (`graded_scheduler authorize-qualification`); five jobs cost about $12 of
   operator credit on 17 October 2026. Three passing reports in the band are
   needed before customer d3 jobs route to the hotkey.
7. **Graded routing.** Add the hotkey to the routing pool. Watch the first
   customer assignment complete.
8. **Tell the applicant** the queue UID and that work depends on job volume.
   Settle the stake.

Revocation is `worker --disable` plus a deletion notice to the contact; the
profile row stays as history.

## 3. Who pays the qualification credit

Proposal: **the applicant stakes it, refunded on passing.** The applicant
prepays a fixed amount of credit on a zils.ai account (the prepaid billing path
customers already use) equal to the cost of the qualification jobs, about $15
today. Zils submits the jobs. If three reports pass, the credit is refunded in
full; if the host fails or the applicant abandons the run, the credit covers the
jobs. This keeps spam out without charging a fee, and it needs no new billing
code beyond an operator refund. Alternatives: Zils pays (today's reality, does
not scale); a non-refundable fee (simplest, but a fee for testnet work with no
earnings reads badly). Owner decision.

## 4. What the miners page should say

Replace the current line ("All miners are currently run by Zils; outside
miners can join once the subnet moves to mainnet") with, once the terms are
published:

> Subnet 579 on the Bittensor testnet. Miners are admitted by application under
> the [miner terms](mining-terms.md): apply with a hotkey and a qualification
> report, stake the qualification credit, and your hotkey appears here once it
> passes. Testnet rewards have no value and no earnings are promised.

Until then the page keeps the current line. The page change is a zils-web
change, out of scope here.

## 5. What has to change in code (not done)

- Read `terms_sha256` from `.private/queue-miner.json` in `miner.queue` and
  write it into `qualification.json` in `scripts.qualify_node`.
- `terms_sha256` and `terms_accepted_at` on `fez_training_workers`.
- `outside_miners` on `fez_training_jobs` and the create-job body, honoured by
  the graded scheduler ([customer jobs](customer-jobs.md#prepare-data)).
- An apply endpoint or form that files the application; until then, email.
