# Next milestones

The local training loop and guarded testnet integration are implemented. The
[first registered training-to-chain round](testnet-round-001.md) completed on
testnet subnet 579, including verification after commit–reveal.
Public discovery, isolated untrusted-model evaluation, benchmark refresh, and
model release/promotion rules remain work for an open competition.

## Customer jobs

The experimental [customer job workflow](customer-jobs.md) freezes authorized
data and acceptance criteria, binds submissions to a job, compares candidates
against a calibrated starting checkpoint, and exports qualifying artifacts
locally. Each fleet configuration currently pins one job.

The [Supabase training queue](supabase-training.md) now connects private uploads,
approved-worker assignments, expiring claims, and accepted-model downloads.
It has been tested locally with fixture models and a disposable PostgreSQL
database. A live Supabase smoke test verified authentication, customer isolation,
private uploads, processor validation, and cancellation. Real queued training
remains unverified.

The separate [Zils decision API](decision-api.md) now implements authenticated
shared JevK5 inference and durable bulk jobs. Local database, SDK, and real GPU
checks passed; hosted deployment and automatic adapter promotion remain unverified
or unimplemented. Miners continue to train candidates, not serve bulk requests.

Next steps include real queued-training validation, rewards across different jobs,
independent final evaluation, confidential compute, retention controls, billing,
and authenticated inference deployment with release approval and rollback.

## Dashboard plans

Planned dashboard capabilities:

- Current winning Zils checkpoint, version/hash, download, and winner history.
- Decision accuracy, Brier probability score, and highly confident mistakes;
  show dataset/rubric versions and comparable evaluation settings.
- Median and p95 response latency, with the measured hardware and timing scope.
- Submission queue, evaluation progress, and per-round candidate comparisons.
- Miner/validator health and published chain weights/reward allocation, with
  testnet status clearly labeled.

Use Teutonic's visibility into model progress as inspiration. Zils's dashboard
should report its decision-model results; percentages from different benchmark
suites must not be presented as directly comparable.

The [public model page](https://zils.ai/model) presents recorded benchmark results
and the verified testnet round. Live subnet views require an explicit public
aggregate feed; unavailable data stays labeled.
