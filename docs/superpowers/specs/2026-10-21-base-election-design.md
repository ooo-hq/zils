# Base election: what "better" means

Status: design, 21 October 2026. Owner conversation; not implemented. Builds on
`zils/base_round.py` and [base candidates](../../base-candidates.md) at main `d44ace0`.
Scoring/readiness revision: 21 October 2026. Core main rechecked at `0b56003`;
its base-round implementation is unchanged from that baseline. Choices explicitly
marked below are builder proposals, not claims of implemented or measured behavior.

## Intent

The base tier elects the shared base every customer adapter trains on. The owner's
definition of a better base, in their words:

1. **Higher than the top models** on the public decision leaderboards: the
   [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
   and [JevBench](https://benchmarkheaven.com/jev-models).
2. **More accurate customer adapters** when customers train on it.
3. **Accurate in choosing the correct results.**

Releases should ship **every two weeks to a month**. When nothing has earned a
release by a window, **skip that release** (owner).

Why now: today's rule elects on calibration (Brier) first, while the owner and the
Decision Index care first about right answers. A partial S1MB run on 21 October
(30 of 137 English benchmarks, r5 vs stock d3-mini, same runtime) found r5 picked the
same answers but was less well calibrated on 9 of 10 yes/no benchmarks. The current
exam had still crowned r5 on the testnet validator. A narrow, Brier-first exam can
crown a base that is not better in the ways the owner means.

## Goals

- Elect on **right answers on decisions the base has not seen**, across many task
  types, with a margin large enough to show up on public leaderboards.
- Never ship a base that makes calibration or customer adapters worse.
- A predictable release rhythm: at most one base release per window of 2–4 weeks.
- Public, third-party evidence for every release.

## Bootstrap and network scope

Owner clarification, 21 October 2026:

- The two bootstrap miner hosts both use **d3-mini** and must exercise the
  base-improvement competition: train candidate bases, commit them, and have
  validators evaluate them. Running only a customer-adapter queue worker does
  not satisfy this bootstrap requirement.
- The intended network has **hundreds of miners and multiple validators**.
  The two-host bootstrap rehearses that protocol; host-specific assignments
  must not define the base election. Customer adapter job routing is a separate
  operation from submitting a base challenger.
- Validators grade base challengers using the broad decision-capability exam
  and safeguards described here. The S1MB investigation motivated replacing
  the narrow old objective; Decision Index and JevBench are the external
  performance targets. A pilot validates the exam, not the complete network.
- A winner's release channel and the machine serving it are separate choices.
  `latest` and `stable` support the adapter migration lifecycle: a newly elected
  base can be released before becoming the default base for new customer jobs.
  Whether a dedicated second serving host is needed for `latest` remains
  **undecided**; do not require a particular host for a candidate to win.

The multi-validator agreement and release-authority rules still need to be
specified before claiming the election is ready at network scale. The existing
owner-reviewed promotion step remains the bootstrap mechanism; this document
does not establish how independent validator results resolve to one network
incumbent. A successful two-host rehearsal alone does not establish that scale.

### Optional miner inference

Proposed direction from the owner, 21 October 2026: miners may also offer
inference, but serving is **opt-in**. A miner does not need to operate an inference
endpoint to compete in the base election or participate in customer adapter
training. Electing a base and choosing a server for an inference request remain
separate decisions.

Zils may also offer model hosting itself. The bootstrap hosts therefore have
an intended role as operator-owned serving capacity as well as miners.
Optional inference from other miners could supplement that capacity and help
subsidize offering the models. This is the owner's product/economic rationale,
not a measured cost saving or a selected reward mechanism. Operator hosting and
miner-provided inference can coexist; neither requires every miner to serve.

Optional serving preserves `stable`/`latest` and each adapter's pinned base;
it does not change the migration lifecycle. Whether a dedicated second serving
host is useful remains undecided. Inference qualification, routing and payment
rules need a separate design; no new inference reward or election gate is
specified here, and this proposal does not claim a deployed miner-serving network.

## Non-goals

- Changing how candidates are committed, pulled, loaded or audited (chain
  commitments, contract files, sandboxing stay as they are).
- Changing customer adapter training, upgrades or migration rules.
- Vision or bounding-box tracks. This covers the text base tier only.
- Choosing which base family (d3-mini, d3-lite) competes.
- Implementing optional inference payments or automatic multi-validator release
  agreement in this grading change. Both remain separate network-design work;
  the bootstrap retains the existing reviewed promotion mechanism.

## Anti-goals

- **Electing on public leaderboard questions.** S1MB, Decision Index and JevBench are
  external checks, not the routine election set. Exposed test questions can be
  trained on. Decision Index's current Full score instead uses maintainer-private
  tests; those are not an exam service available to our validators.
- **Calibration-only wins.** A challenger that only sounds better calibrated, with
  the same categorical decisions, must not win those task scores. Numeric Score
  tasks measure the predicted expected value, so a more accurate numeric answer
  can count even if its most-probable bucket stays unchanged.
- **Churn.** Customers should not face a migration every few days.

## Current state (main `d44ace0`)

- `decide()`: temperature fitted per model on the sealed calibration split; eligible
  when Brier skill > 0, accuracy not below the incumbent, paired Brier improvement
  ≥ 0.01 with bootstrap lower 95% > 0. Winner = lowest Brier.
- `confirm_promotion()`: same checkpoint eligible on a second, disjoint draw at least
  12 hours later, same policy and incumbent.
- Adapter suite: reference adapters retrained on incumbent and candidate; candidate
  no worse on every job (0.005 Brier tolerance) and better on at least one.
- Exam: teacher-authored realistic decision families, sealed per round, published
  after the round. Kings paid per round from the base share (last five kings).
- Promotion is an owner-reviewed PR; `latest`, then `stable` after a window.

## Cutover (owner decision, 21 October 2026)

- **The king resets to stock `d3-mini-4.5b-v1`.** r5 was crowned on the testnet
  validator under the Brier-first rule this design replaces. It may re-qualify under
  the new rule like any challenger.
- The base-tier `kings` history is cleared, so the base share is burned until the
  first crowning under the new rule (as before the first promotion today).
- Pending and confirmed winners decided under the old policy are void. The new policy
  keys replace the old ones in one change; `confirm_promotion` already refuses a
  confirmation whose policy differs from the first draw's.
- Promotion must additionally compare both reports with the **currently installed**
  pool, incumbent, model and policy. Two mutually consistent historical reports
  are insufficient. Policy/incumbent changes invalidate retained ready candidates.

## How bootstrap miners produce a challenger

Each bootstrap miner starts with the pinned d3-mini profile and its own registered
hotkey. It trains on permitted public or miner-owned development data, tests on
separate development cases, and submits an immutable full-base checkpoint through
the existing upload and on-chain commitment path. The initial recipe is the
existing [LoRA training and merge recipe](../../base-candidates.md#reference-recipe-train-a-lora-merge-it).
Other training methods remain allowed within the candidate contract.

Customer adapters do not automatically become base candidates. Private customer
examples and sealed validator cases are not inputs to this bootstrap recipe.
The bootstrap must demonstrate two distinct miners submitting candidates that
validators can discover, evaluate and reject or qualify. Registration, a running
adapter queue worker, and successful inference do not demonstrate that loop.
A baseline copy is a useful rejection control, never an earned improvement.

Training is miner work; the exam pilot is validator work. Run them as separate
jobs using the existing host GPU coordination. An operator runbook can drive the
first rehearsal; an unattended training/search daemon is not required to test
the election protocol. A failed candidate is a valid rehearsal outcome.

## The new election rule

Three gates, in order. All use the round's sealed draw, scored exactly as today
except where stated.

### 1. Win on decision quality (replaces the Brier win)

- **Score:** the base-only, fixed 0–100 score defined below. It combines correctness
  and numeric error above explicit trivial baselines. It is a Zils metric inspired
  by the external benchmarks, not an official Decision Index, S1MB or JevBench score.
- **Win:** paired improvement over the king of at least **+1.0 point** (on the
  0–100 scale), with paired bootstrap lower 95% bound above zero (resampling cases
  within families, as today). The winner among eligible challengers is the highest
  score. Exact score ties use the earliest accepted commitment block, then
  lexicographic checkpoint hash; ties are not evidence of quality separation.
- **Confirmation:** unchanged mechanism: the same checkpoint must stay eligible on a
  disjoint draw at least 12 hours later.

### Exact score contract (builder proposal)

Keep this scorer separate from shared `zils.score`; customer acceptance and
feature-track metrics retain their current meaning. First version uses hard
labels only: reject soft-target base-exam cases rather than silently ignoring
their probabilities. Each family has one question type, category and block.
Use the model's held-out-temperature-calibrated probabilities; retain the fitted
temperature with its predictions. Ties for categorical predictions use the
existing `zils.score` lexicographic option-key rule.

Compute the following per family on its complete test cases:

| Type | Raw metric | Baseline-adjusted skill |
|---|---|---|
| Choice | Fraction of correct choices, `a` | `(a - b) / (1 - b)`, where `b` is the greatest accuracy of uniform random choice, a fixed option ID, or a fixed original option position. Missing fixed IDs/positions fall back to uniform choice. |
| Yes/no | Balanced accuracy `BA = (true recall + false recall) / 2` | `2 * BA - 1`; both target classes must be present. |
| Score | Mean absolute error `MAE` between predicted expected value and target | `1 - MAE / baseline_MAE`; the constant baseline is the median normalized target value across the family. |

For Score, the existing ordered criteria list supplies values `j/(K-1)` for
indices `j=0..K-1`. Thus prediction `sum_j p_j*j/(K-1)` and the hard-label target
share a 0–1 scale. Example: a five-level scale with target index 3 has target
0.75. A predicted expectation 0.60 has absolute error 0.15. An arbitrary criteria
list that does not describe an ordered, equally spaced scale is invalid for this
first version. No per-case division by a midpoint error is used.

Clamp each family's skill to `[0,1]`; also report unclipped skill, raw metric and
baseline so losses below zero remain visible. Average families equally within
each category/block. Combine its known/new block means with 0.6/0.4, then apply
the five fixed category weights below and multiply by 100. All families and
their type mix are sealed before candidate predictions are inspected. Reject a
draw if a category/block is missing, either yes/no class is absent, a Choice
baseline is at least `1 - 1e-12`, or a Score baseline MAE is at most `1e-12`.
Do not remove problematic families separately for each model.

The S1MB reference is its [scoring specification](https://github.com/hotchpotch/S1MB/blob/main/evaluator/SCORING.md),
rechecked 21 October 2026. Zils retains its own tie rule, hard-label contract,
category weighting and 60/40 split. This is neither S1MB Task Avg (equal weights
across three task types) nor its roster-dependent Borda ranking. Freeze these
definitions in the policy; changes require a new policy and requalification.

Bootstrap uncertainty is paired: apply the same sampled case indices to every
model within each family and recompute the complete weighted score. Keep the
family baselines from the original sealed test set fixed across resamples and
keep fitted temperatures fixed. For yes/no, stratify within family by true/false
label so both classes remain present. Use independent source cases: retain at
most one case per source-lineage group across a qualification's calibration/test
splits and draws; keep related variants in development data rather than count
them as independent election trials. The interval is conditional on this frozen family mix and
these baselines; it does not prove generalization to arbitrary new families.
Record 2,000 resamples and a shared deterministic seed derived from the draw
identity, not the candidate's UID. Missing or malformed prediction rows make a
candidate ineligible; an incomplete incumbent invalidates the comparison.

### 2. Do not lose on calibration (guard)

- Calibrated Brier (after each model's own temperature, as today) may be worse than
  the king's by at most **0.005**, on both draws.
- Preserve the existing equal-family **full probability-vector** Brier definition,
  `sum_k (p_k - 1[k=label])^2`. The binary form is twice the single-probability
  Brier used by some external suites; the tolerance is in Zils's existing units.

### 3. Do not lose on customer adapters (guard)

- The existing reference-adapter suite, with accuracy added: on every reference
  job, the challenger's adapters must have accuracy no lower than the king's by more
  than **1.0 point** and calibrated Brier no worse by more than **0.005**, using the
  **mean over three seeds (553, 771, 991)** per job.
- The "better on at least one job" requirement is dropped: gate 1 already requires
  a real gain.
- The suite producer and promotion verifier must enforce the same rule, require
  all three exact seeds, and recompute means from per-seed evidence. Preserve
  the separate stable migration requirement that every reference job and every
  seed passes; a passing mean cannot override a failed migration comparison.

## The exam

Same sealing and publication rules as today; content widens.

- **Categories guided by the Decision Index.** The proposed fixed weights below
  approximate its current area weights; matching coverage alone does not establish
  that score gains transfer to its leaderboard:

  | Category | Weight | What it covers |
  |---|---|---|
  | Knowledge & reasoning | 25.85% | academic knowledge, commonsense, maths, code |
  | Language understanding | 25.85% | entailment, sentiment, sarcasm, stance |
  | Retrieval & classification | 20.02% | rank documents/products, route intents, spam/phishing |
  | Tools | 18.28% | choose and call functions for a request |
  | Arts | 10.00% | the Index's arts category (check its benchmark list) |

- **Task types:** each category mixes Choice, yes/no and Score questions.
- **Two blocks inside every category**, mirroring the Decision Index's split: the
  category score is 60% known skills + 40% new domains.
  - **Known skills**: families already used in a published draw.
  - **New domains**: families never used in any published draw. Every draw (first,
    confirmation, window re-score) gets its own new-domain families; a family used
    only in an unpublished draw stays new. After publication it moves to known skills.
  - Cases carry `category` and `block` tags next to today's `family`.
  - A sealed family registry records type/category, prior publication and source
    lineage. Validate tags and block membership before scoring. Keep an unpublished
    pilot/validation family separate from miner development data.
  - **Bootstrap history (builder proposal):** map existing published families into
    the registry where they cover the required categories. For gaps, prepare and
    publish a versioned seed/development draw before freezing the first qualifying
    pilot. Record the actual publication time and artifact hash; never label a
    private family "known" merely to satisfy coverage. Use new source cases for
    the sealed pilot, and keep its new-domain families unpublished. A private
    rehearsal without that history is a dry run, not a qualifying pilot.
- **Sources:** teacher-authored families as today. Public decision datasets may be
  used only as seeds that the teacher rewrites into new cases, never copied
  verbatim, because miners can train on any public split. Never seed from the
  S1MB, Decision Index or JevBench test sets.
- **Size:** use a development pilot with predeclared, distinct model pairs that
  actually disagree, including at least one within-d3-mini pair. For each pair
  compute the paired 95% score interval; require both distances from the point
  estimate to its endpoints to be at most 0.5 points. Grow the sealed case count
  and repeat sizing on development data until all predeclared nondegenerate pairs
  meet that target. A pair with identical per-case outcomes provides no sizing
  evidence; report it as such. Likewise, differing predictions whose paired score
  distribution has zero variance after clipping provide no sizing evidence.
  Require at least one within-d3-mini pair with nonzero paired score variance;
  if none is available, obtain a distinct development checkpoint before sizing
  can pass. Freeze the resulting family mix and case counts
  before the independent validation draw. This is a precision criterion, not a
  claim of 95% power or guaranteed detection of a true one-point gain.
- **Disjointness:** distinct dataset hashes alone are insufficient. Compare case
  IDs, normalized input/question fingerprints and source-lineage identifiers across
  calibration/test and all first/confirmation/closing draws. Reject collisions;
  track related variants during dataset construction and audit their grouping.

## Proving the exam tracks the leaderboard (before first use)

The pilot has development and independently sealed validation draws. Before
running models, pin at least five open reference checkpoints, their native
inference settings and one dated Decision Index edition/snapshot spanning at
least 20 Full-score points. Use untuned references where possible. Verify model
access, runtime compatibility and GPU fit rather than promising that every
reference fits one 24 GB card. Run sequentially under the existing GPU lock.

Development sets family coverage and case counts. Freeze the scoring policy and
model roster before validation. Validation passes with Spearman rank correlation
at least 0.9 and no inverted model pair separated by at least five external Full
points. Publish all pairwise scores and uncertainty, not only the correlation.
Also report the predeclared within-d3-mini pair; good ranking of unrelated models
does not establish sensitivity to improvements within one lineage. The validation
draw must also satisfy the frozen precision criterion and include a score-informative
within-d3-mini comparison; saturation at the clipping floor cannot certify it.

If validation fails, revise the exam on development data and use a new, untouched
validation draw. Do not tune repeatedly on a held-out draw and still call it
independent. The pilot never crowns a miner. A passing result supports this
bootstrap experiment, not an equivalence to an official private leaderboard run.

Before unrestricted hundreds-of-miners election, separately validate the false-win
rate under repeated candidate submissions and selection of a best-of-many model.
The current per-comparison 95% interval is not a network-wide or lifetime error
bound. Trial limits/multiple-comparison policy and multi-validator release
agreement remain explicit scale-readiness work; do not claim this pilot proves them.

## Checking it keeps tracking

After every release, compare the exam gain (release vs previous king) with the
paired S1MB `english-v1` gain from the public-proof run. After three releases, if
exam gains have not come with S1MB gains in the same direction, stop electing and
revise the exam.

## Release windows

- Rounds keep running. Challengers are scored, confirmed and suite-tested between
  windows exactly as above; a challenger that has passed all three gates is
  **release-ready**.
- A window closes every 2–4 weeks. If one or more release-ready challengers exist,
  re-score them all on one fresh draw against the king; the best that still passes
  gates 1 and 2 is promoted (owner-reviewed PR, unchanged) and crowned. If none
  exists, **skip the release**; the king and payouts continue unchanged.
- Before a closing draw, freeze the eligible checkpoint list. Retain evidence for
  every release-ready candidate, not only the most recent pending winner. No new
  checkpoint can enter after closing scores are observed. Revalidate the active
  incumbent/policy and all confirmation/suite bindings at the promotion boundary.
- **Channels (owner decision):** a release goes to `latest` at its window. At the next
  window it moves to `stable` (the default for new customer jobs) unless a problem was
  found, so the customer default changes at most once per window. The existing
  `stable` migration check still applies.
- **Adapter continuity (owner clarification):** these channels exist because
  customer adapters are bound to particular base releases and need migration
  when that base changes. Moving a channel does not replace the base underneath
  an existing adapter. Existing adapters retain their pinned base until their
  own migration passes the existing acceptance checks. Stable promotion requires
  the existing migration qualification as well as the release window; elapsed
  time alone does not establish adapter compatibility. This does not introduce
  an automatic customer-migration scheduler or require every customer adapter to
  migrate before the channel advances.
- Kings are still paid every round from the base share; only crowning and release
  follow the window.

## Public proof for each release

- Run the new release and the previous one on S1MB `english-v1` on the same
  hardware and runtime, paired. Publish the comparison in the release notes.
- Prepare submissions to S1MB, the Decision Index and JevBench. Each submission is
  public and needs the owner's approval.
- These results never select the current winner. Aggregate transfer checks may stop
  future elections and motivate a new exam version under the rule above; no public
  test questions are incorporated into the private election set.

## Bars to beat (21 October 2026)

| Leaderboard | Headline | Stock d3-mini | Best ~4B | Jev 1.13 |
|---|---|---|---|---|
| Decision Index | balanced skill (accuracy-type, private tests) | 49.97 (#76) | StartLux-Decision 4B 54.80 | 59.22 |
| JevBench | Capability = mean(accuracy above chance, calibration) | not listed | H2O-Lightning-4B 75.0 | 77.1 |

## Left to the builder

- Internal function names and storage layout for the base-only scorer and paired
  bootstrap; shared `zils.score` semantics must remain unchanged.
- Policy names and defaults, recorded in each round result so confirmation can
  check them as it does today.
- How window dates are configured (validator track field or release script), as
  long as a skipped window leaves the king and payouts unchanged.
- Choice of public datasets for fresh cases, within the licence and test-set rules
  above.
- Exact models for the ranking test, as long as at least five have Decision Index
  scores spread over at least 20 points and their native runtime/memory needs are
  verified before execution. Include the separate within-d3-mini sizing comparison.

## Evidence and limits

- S1MB partial run: 30 paired benchmarks on one RTX 4090, stopped by the owner;
  results kept privately.
- The +1.0 point margin is a proposal sized to leaderboard noise, not measured on our
  exam yet. The 0.005 and 1.0-point guard tolerances follow today's suite tolerance.
