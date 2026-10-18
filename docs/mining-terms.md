# Zils miner terms

Version 18 October 2026. In force once [admission](miner-admission.md) opens;
until then no outside miner is admitted.

These terms apply to anyone who runs a miner against the Zils hosted training
queue or a Zils validator ("you") and are made with Zils, the operator of
Bittensor subnet 579 ("Zils").

## 1. What you receive

- **Customer training data.** For each customer job assigned to you, the job's
  training split only: states, questions and labels. Calibration and test
  splits, acceptance thresholds and customer identity stay on the Zils
  processor. Each job carries a job ID and a manifest hash.
- **Base weights.** The pinned base model (today `d3-mini-4.5b-v1`, a public
  release under its own vendor licence) and Zils' training runtime from the
  public repository.
- **Qualification material.** Synthetic benchmark jobs used to grade your host.
  They are not customer data.

## 2. What you may do with it

Train the assigned adapter for the assigned job, on the host you qualified, and
submit the signed artifact. That is all.

## 3. What you may not do with it

- **No retention beyond the job.** Delete the job's training data, intermediate
  checkpoints and logs that contain examples within 7 days of the job finishing
  (accepted, rejected, cancelled or expired), and at once on revocation.
- **No redistribution.** Do not copy customer data off the qualified host, share
  it, publish it, or let any other person or service read it.
- **No other training.** Do not train, fine-tune, evaluate or prompt any other
  model with customer data, and do not use it to build datasets, benchmarks or
  derived data.
- **No reconstruction.** Do not try to identify the customer, the people in the
  data, or the calibration/test cases.
- **No tampering.** Do not alter model IDs, hashes or reports to pass a check
  that would otherwise fail.

The submitted adapter belongs to the customer who ordered the job once Zils
accepts it. You keep no rights in it.

## 4. Security duties

- Run the miner under a dedicated operating-system account that owns nothing
  else; keep job data under that account's private state directory.
- Keep the mining hotkey on the host and the coldkey and recovery phrase off it.
  Never put wallet files, the miner configuration or coordinator addresses in a
  repository, chat or log.
- Expose no public sidecar services from the miner host (no open artifact ports
  beyond what the validator contract requires, no remote shells on the mining
  account, no shared GPU tenants that can read the state directory).
- Keep the host patched. Report a suspected leak or compromise to Zils within
  24 hours and stop mining until Zils clears you.

## 5. What Zils may do

- **Revoke** your admission at any time, with or without a reason, by disabling
  your worker row. Downloads already issued cannot be recalled; the deletion
  duty above still applies.
- **Audit artifacts.** Verify hashes, re-run your adapter, inspect training
  metrics and reports you submit, and ask for evidence of deletion.
- **Publish aggregate scores.** Your hotkey, queue UID, online status, models,
  job counts and scores may appear on zils.ai and in the public network API.
  Customer data and job contents are never published.
- **Change these terms.** A new version has a new hash; you accept it by
  continuing to mine after notice, and your worker config must carry it.

## 6. Payment reality

Zils runs on Bittensor **testnet** subnet 579. Testnet tokens have no value.
Queue jobs publish no chain weights. **No earnings are promised**, now or when a
mainnet subnet exists. Your costs (hardware, power, qualification credit) are
yours. Nothing here is investment advice or an offer of securities.

## 7. Liability, briefly

You are responsible for your host and for any use of customer data from it.
Zils provides the queue, weights and runtime as they are, without warranty of
uptime, work volume or acceptance, and is not liable for your costs or losses
from mining.

## 8. Acceptance

- **Registering a hotkey on subnet 579 and mining with it is acceptance.** The
  hotkey that signs your claims is the accepting identity; the first signed
  claim records it. One person may run several hotkeys; each accepts separately.
- **Terms version** = the SHA-256 of this file at the commit you accepted, plus
  the date in the heading. Your worker config (`.private/queue-miner.json`,
  `terms_sha256`) carries it into every claim and into the qualification report.
  (Documented only today; see [admission](miner-admission.md) for the code.)
- Zils records your hotkey, the terms version and the first-claim time with your
  worker row.

Every customer job goes to the whole miner pool; there is no per-job opt-out.
A mainnet subnet, when it exists, gets a new version of these terms.
