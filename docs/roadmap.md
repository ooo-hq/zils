# Subnet milestones

Mainnet subnet 127 went live on 3 November 2026. This page lists what is merged
on `main` and what remains, as of 9 November 2026. It does not record what a
given validator runs; check its signed round results for that.

## Merged

1. **Admission.** Registration on subnet 127 is the admission. Validators read
   soup-body commitments from the chain themselves, and the hosted queue admits
   any registered hotkey that accepts the [miner terms](mining-terms.md) and
   qualifies ([miner admission](miner-admission.md)).
2. **Soup pools.** Four pools, one per z3 size, each a uniform average of up to
   five miner bodies with five paid seats and a crown
   ([soup pool](mining.md#soup-pool)). Lite and mini are open in the launch
   configuration; flash and z3 open at a block set in the validator config.
3. **Adapter track.** Validators pay customer-job miners from paid API spend on
   the open-tier adapters they delivered, from platform-signed evidence
   (`adapter_evidence`, [validators §7](validators.md#7-pay-the-adapter-track-from-paid-adapter-use)).
   The launch split is 80% base pools, 20% adapter track.
4. **Releases.** Miners install the commit in `release/stable`; customer-job
   claims name it, and customer jobs go only to accepted releases
   ([releases](releases.md)).
5. **Isolated evaluation.** Submissions are evaluated in a sandboxed subprocess.
6. **Soup releases.** A soup is served only after it passes the release gate:
   sealed release draws, a confirmation draw and the adapter regression suite
   ([base election](base-election.md)).

## Remaining

- Outside re-grading of soup rounds: the round bank is private, so a round can
  be audited against hashes but not re-scored by others.
- Deleting a job's candidate, raw output and logs within 7 days, as the miner
  terms require; the queue worker deletes only the training data today.
- Public health and chain reporting beyond the per-validator round results.

## History

The [first registered training-to-chain round](testnet-round-001.md) ran on
testnet subnet 579 on 25 September 2026.

### d3-mini

On 17 October 2026 d3-mini
(`d3-mini-4.5b-v1`) became the only base by the owner's decision, after its
confirmation draw tied (an override of the promotion rule, not a win under it);
the evidence is in [model research](model-research.md#d3-mini-evidence-16-october-2026).
The z3 names replaced d3 in public text later that month. King-of-the-hill base
pools ([king pools](mining.md#king-pools)) were the first base-tier design; the
launch configuration replaced them with soup pools.

Customer-job data contracts, calibration, and acceptance are shared with the
hosted platform. Hosted account management, APIs, billing, storage, and serving
belong to [zils-platform](https://github.com/ooo-hq/zils-platform); SDKs and agent
interfaces belong to [zils-sdk](https://github.com/ooo-hq/zils-sdk).
