# Base election: what "better" means

Status: design, 21 October 2026. Owner conversation; not implemented. Builds on
`zils/base_round.py` and [base candidates](../../base-candidates.md) at main `d44ace0`.

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

## Non-goals

- Changing how candidates are committed, pulled, loaded or audited (chain
  commitments, contract files, sandboxing stay as they are).
- Changing customer adapter training, upgrades or migration rules.
- Vision or bounding-box tracks. This covers the text base tier only.
- Choosing which base family (d3-mini, d3-lite) competes.

## Anti-goals

- **Electing on public leaderboard questions.** S1MB, Decision Index and JevBench are
  reported, never used to pick a winner; their test items are public or semi-public
  and would be trained on.
- **Calibration-only wins.** A challenger that only sounds better calibrated, with
  the same answers, must not be crowned.
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

## The new election rule

Three gates, in order. All use the round's sealed draw, scored exactly as today
except where stated.

### 1. Win on accuracy (replaces the Brier win)

- **Score:** chance-corrected accuracy, `(accuracy − chance) / (1 − chance)` per
  case, where chance is 1 / number of options. Score questions are not scored by
  exact match: they use `1 − normalised expected-score error`, chance-corrected
  against always predicting the middle of the scale, as S1MB scores them
  (builder's call made in this document). Average within each family, then families equally within each category,
  then categories by the Decision Index weights (see the exam below). This mirrors
  how both leaderboards normalise.
- **Win:** paired improvement over the king of at least **+1.0 point** (on the
  0–100 scale), with paired bootstrap lower 95% bound above zero (resampling cases
  within families, as today). The winner among eligible challengers is the highest
  score.
- **Confirmation:** unchanged mechanism: the same checkpoint must stay eligible on a
  disjoint draw at least 12 hours later.

### 2. Do not lose on calibration (guard)

- Calibrated Brier (after each model's own temperature, as today) may be worse than
  the king's by at most **0.005**, on both draws.

### 3. Do not lose on customer adapters (guard)

- The existing reference-adapter suite, with accuracy added: on every reference
  job, the challenger's adapters must have accuracy no lower than the king's by more
  than **1.0 point** and calibrated Brier no worse by more than **0.005**, using the
  **mean over three seeds (553, 771, 991)** per job.
- The "better on at least one job" requirement is dropped: gate 1 already requires
  a real gain.

## The exam

Same sealing and publication rules as today; content widens.

- **Categories, matched to the Decision Index** so that exam gains track its headline.
  Weights are the Index's public formula:

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
- **Sources:** teacher-authored families as today. Public decision datasets may be
  used only as seeds that the teacher rewrites into new cases, never copied
  verbatim, because miners can train on any public split. Never seed from the
  S1MB, Decision Index or JevBench test sets.
- **Size:** set by sampling noise, not rerun noise (inference is close to
  deterministic). Size each draw so the paired 95% bootstrap interval for a
  challenger identical to the king is no wider than ±0.5 points, measured on the
  pilot draw.

## Proving the exam tracks the leaderboard (before first use)

The exam only elects bases once it ranks known models the way the Decision Index
does. On the first pilot draw, score at least five open models that already have
Decision Index results with a spread of scores, for example (balanced skill,
21 October): d3-lite 26.2, d3-nano 34.3, OneJev1.6-4B 48.9, d3-mini 50.0, d3-flash
54.3. Pass: Spearman rank correlation of at least 0.9 with the Index, and no pair
that the Index separates by 5 points or more in the wrong order. Fail: change the
exam's families or mix, not the target, and rerun. The pilot does not elect
anything. It fits on a single 24 GB GPU and should take the same shared GPU lock as
other validator work.

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
- **Channels (owner decision):** a release goes to `latest` at its window. At the next
  window it moves to `stable` (the default for new customer jobs) unless a problem was
  found, so the customer default changes at most once per window. The existing
  `stable` migration check still applies.
- Kings are still paid every round from the base share; only crowning and release
  follow the window.

## Public proof for each release

- Run the new release and the previous one on S1MB `english-v1` on the same
  hardware and runtime, paired. Publish the comparison in the release notes.
- Prepare submissions to S1MB, the Decision Index and JevBench. Each submission is
  public and needs the owner's approval.
- These results never feed back into an election.

## Bars to beat (21 October 2026)

| Leaderboard | Headline | Stock d3-mini | Best ~4B | Jev 1.13 |
|---|---|---|---|---|
| Decision Index | balanced skill (accuracy-type, private tests) | 49.97 (#76) | StartLux-Decision 4B 54.80 | 59.22 |
| JevBench | Capability = mean(accuracy above chance, calibration) | not listed | H2O-Lightning-4B 75.0 | 77.1 |

## Left to the builder

- Where the accuracy score lives (`zils.score` or `base_round`) and its exact
  bootstrap code, keeping the current paired, within-family resampling.
- Policy names and defaults, recorded in each round result so confirmation can
  check them as it does today.
- How window dates are configured (validator track field or release script), as
  long as a skipped window leaves the king and payouts unchanged.
- Choice of public datasets for fresh cases, within the licence and test-set rules
  above.
- Exact models for the ranking test, as long as at least five have Decision Index
  scores spread over at least 20 points and they fit on a 24 GB GPU.

## Evidence and limits

- S1MB partial run: 30 paired benchmarks on one RTX 4090, stopped by the owner;
  results kept privately.
- The +1.0 point margin is a proposal sized to leaderboard noise, not measured on our
  exam yet. The 0.005 and 1.0-point guard tolerances follow today's suite tolerance.
