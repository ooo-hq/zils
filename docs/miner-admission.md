# Miner admission

Registration on Bittensor mainnet subnet 127 is the admission. There is no
application, no form and no operator approval. This holds for both ways to earn:

- **Soup pools.** Validators read body commitments from the chain; any
  registered hotkey can commit ([mining](mining.md#soup-pool)).
- **Customer jobs.** The hosted queue at `https://training.zils.ai` runs chain
  admission: `GET /v1/config` shows `admission.mode` `chain`, the network and
  netuid, the current `terms_sha256` and `qualification_days`. This page covers
  that queue.

## 1. What a miner does

1. Register a hotkey on subnet 127 ([registration](bittensor-registration.md)).
2. Read the [miner terms](mining-terms.md) and put their SHA-256 in the worker
   config as `terms_sha256`. Every signed claim carries it.
3. Install the stable release and the d3 reference, then run
   `scripts.qualify_node --model d3-mini-4.5b-v1 --terms-sha256 <hash>`
   ([customer-job miners](queue-miners.md)).
4. Start `python -m miner.queue` with `terms_sha256` and `qualification` (the
   path to `qualification.json`) in its config.
5. Work arrives: qualification jobs first (synthetic, they grade the host), then
   customer jobs once three reports pass in the band.

## 2. What the coordinator does

- **Admit.** At startup the worker POSTs the terms hash and its qualification
  report to `/v1/workers/admit`, signed by the hotkey. The coordinator checks the
  signature, looks the hotkey up in the subnet's registration set (read from the
  chain, cached five minutes), compares the terms hash with the current one, and
  checks the report against the pinned z3-mini profile. Then it writes the
  worker row (next free queue UID, `admission='chain'`, the terms hash and the
  acceptance time) and the profile row from the report (`min_free_mib` 12288).
  A row an operator revoked is never revived.
- **Keep checking.** On every signed route, a chain-admitted hotkey must still
  be registered and on the current terms. A hotkey that drops off the chain is
  refused at the next refresh.
- **Qualify.** For a new hotkey the coordinator queues one graded qualification
  job, cloned from a validated benchmark job and paid by Zils. Customer work
  needs three passing graded reports, the same rule as for every other miner.
- **Route.** Chain-admitted miners are offered open-tier customer jobs and their
  own qualification runs. Private-tier jobs stay on Zils hardware.
- **Check the release.** Customer jobs go only to miners on an accepted
  release ([releases](releases.md)).

The soup pools need no coordinator: validators read body commitments made
with `zils-miner commit-body` from the chain themselves.

## 3. Qualification credit

Zils pays it. Registration already costs the miner a burn, which is the spam
gate Bittensor provides; a second gate here would be a form by another name.
The cap is three qualification runs trained by the hotkey per 30 days, queued
one at a time and only while its worker is online; customer work needs three
passing reports. As reports expire, older runs
leave the window and the coordinator queues new ones.

## 4. Operator duties

Revocation, audits and the routing pool. Nothing per applicant.

- **Revoke one hotkey:** `python -m zils_platform.coordinator worker --hotkey <ss58> --disable`
  (no `--uid` needed).
- **Turn chain admission off for everyone:** `ZILS_CHAIN_ADMISSION=off` on the
  coordinator restores operator-only behaviour.
- **Change the terms:** the new hash goes in `ZILS_MINER_TERMS_SHA256` (or the
  platform constant). Miners re-admit by qualifying and restarting with the new
  `terms_sha256`.

A coordinator with admission off answers the admit call with 404 or 409; the
worker logs `Chain admission not applied` and continues, which is how
operator-registered nodes (no `terms_sha256` in their config) run.
