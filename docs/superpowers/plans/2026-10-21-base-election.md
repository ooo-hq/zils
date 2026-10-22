# Decision-quality base election Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Brier-first base elections with the approved decision-quality rule across all five text-base pools, qualify mini first, and produce verifiable, windowed release proposals.

**Architecture:** Keep the existing candidate contracts, validator intake, prediction cache and king payouts. Add base-specific metric, exam, qualification and release-evidence modules; share them across pools while preserving pool identities and failure isolation. Reuse the platform's existing per-pool channels through reviewed configuration, without adding a training daemon or serving router.

**Tech Stack:** Python 3.13+, standard library, existing `kev` temperature fitting, `unittest`, existing d3 runtime and Bittensor/Hugging Face tooling. No new dependencies.

**Spec:** [Base election: what "better" means](../specs/2026-10-21-base-election-design.md).

## Global Constraints

- Pools: `d3-lite`, `d3-nano`, `d3-mini`, `d3-flash`, `d3`; independent kings, evidence, allocation and channels. Mini is the first bootstrap, not evidence that another pool qualifies.
- Decision gain `>= 1.0` on the 0–100 scale; paired lower 95% bound `> 0`; 2,000 resamples; confirmation at least 12 hours later. Calibration guard: candidate full-vector Brier minus incumbent Brier `<= 0.005`.
- Adapter gate: every job's mean accuracy loss `<= 0.01` and mean calibrated Brier increase `<= 0.005`, with exactly seeds `553, 771, 991`. No better-on-one-job requirement. Preserve every-job/every-seed stable migration checks.
- Category weights: reasoning `0.2585`, language understanding `0.2585`, retrieval/classification `0.2002`, tools `0.1828`, arts `0.10`; each category combines known/new blocks `0.6/0.4`. Hard labels only. Preserve shared customer scoring and feature-track semantics.
- Release interval: 14–28 days, at most one crown per pool/window. No qualified winner means unchanged king/payouts. Candidate publication and official benchmark submissions remain explicit operator actions. Optional serving, automatic multi-validator agreement and unrestricted best-of-hundreds selection are outside this implementation.

## Review Focus

- A late candidate must not inherit another candidate's earlier timestamp or use prematurely cached confirmation predictions; Task 3 tests its own evidence interval.
- A renamed or reordered copy of a disclosed case must not become a sealed case in another pool; Task 2 tests normalized fingerprints, source lineage and shared publication history.
- A plausible adapter report can omit the hardest job, duplicate a seed or forge an average; Task 4 binds the suite manifest and recomputes every required mean.
- A restart, hotkey deregistration or policy/incumbent change must not crown an obsolete candidate or move a channel twice; Task 5 tests active bindings, membership and idempotent window receipts.
- Different raw predictions can collapse to identical clipped scores, making precision look perfect; Task 6 rejects that as pilot evidence and tests each pool's qualification independently.

---

## Starting point and ownership

Source reviewed against core `main` **`0b56003abb1518a2a73089046b39cdb9af8e1931`** on 21 October 2026. The design checkout is two design commits ahead and two commits behind that default before this plan. The implementation difference is the evaluation-timeout fix in `zils/validator.py` and `tests/test_evaluation_timeout.py`; preserve it. At execution, start from freshly verified core default with this design/plan carried forward; preserve unrelated edits.

Platform source was read at its current default **`af266c0e39d31ff784efb30b4b88dec0157401ee`**, not its older primary checkout. `zils_platform/model_pools.py` already supports independent stable/latest channels and `select()` returns the exact profile frozen into a new job. Existing accepted adapters keep their profile. No platform schema, SDK, web or public HTTP contract change is required for the scorer; release rollout still needs platform configuration and profile qualification.

### File responsibilities

| Files | Responsibility |
|---|---|
| New `zils/base_metrics.py`, `tests/test_base_metrics.py` | Pure fixed decision score and paired uncertainty, independent of customer scoring. |
| New `zils/base_exam.py`, `tests/test_base_exam.py` | Sealed exam metadata, publication history, provenance and cross-draw audit; preparation CLI. |
| Existing `zils/base_round.py`, `zils/validator.py`, `zils/model_pools.py` and their tests | New policy, intake integration, exact candidate evidence, all-pool qualification checks. Preserve `zils/feature_tracks.py` callers. |
| Existing `zils/regression_suite.py`, `scripts/promote_base.py` and their tests | Three-seed adapter gate, manifest-bound verification and promotion enforcement. |
| New `zils/base_release.py`, `tests/test_base_release.py` | Retained qualification receipts, frozen release-window roster, closing decision and channel-change proposal. |
| New `zils/base_pilot.py`, `tests/test_base_pilot.py` | Offline pilot checks over real prediction exports; no universal model runner. |
| Existing operator docs plus new `docs/base-election.md` | All-pool rollout, two-miner rehearsal, public proof, channel continuity and rollback procedures. |

Use current `unittest` patterns and fixture runners, not a new test framework. Commands below use `python` to mean the repository's Python 3.13+ test environment. CPU tests must not download models, contact the chain or require CUDA.

## Task 1: Implement the base-only score and paired interval

**Files:** Create `zils/base_metrics.py`, `tests/test_base_metrics.py`. Read `zils/core.py`, `zils/base_round.py`; do not change their shared scoring helpers in this task.

**Interfaces:**
- `baselines(cases: list[dict]) -> dict`: validate hard-label family/cell contracts and return each family's fixed trivial baseline; reusable by the exam audit before predictions exist.
- `score(cases: list[dict], predictions: list[dict]) -> dict`: return `score` (0–100), `families`, `categories`. Each family records `raw_metric`, `baseline`, `unclipped_skill`, `skill`, `category`, `block`, `type`, `count`.
- `paired_improvement(cases: list[dict], incumbent: list[dict], candidate: list[dict], *, resamples: int, seed: int) -> dict`: return `improvement`, `lower_95`, `upper_95`, `bootstrap_variance`, `resamples`, `seed`.
- Constants: `VERSION = "zils-base-decision/v1"`, `CATEGORY_WEIGHTS` using the five identifiers `reasoning`, `language`, `retrieval`, `tools`, `arts`; blocks `known`, `new`.

- [ ] **Step 1: Write deterministic metric and interval tests.** Build a fixture with all five categories and both blocks, independently identified families/cases, balanced labels and complete normalized probability rows. Repeat the same family construction across cells to make its final score analytically checkable. Tests assert:

```python
# test_choice_chance_correction: balanced binary choice, accuracy 3/4.
self.assertAlmostEqual(result["score"], 50.0)
# test_noul_imbalance: always false, even with 99 false targets and one true.
self.assertEqual(result["score"], 0.0)
# test_score_median_baseline: targets [0, .25, .75, 1], expectations [.1, .35, .65, .9].
self.assertAlmostEqual(result["score"], 100 * (1 - .1 / .375))
# test_identical_predictions: fixture passed to paired_improvement on both sides.
self.assertEqual((pair["improvement"], pair["lower_95"], pair["upper_95"]), (0, 0, 0))
```

Also test a fixed-ID/position baseline stronger than uniform with variable choice sets, lexicographic categorical ties, fixed baselines across bootstrap samples, yes/no class stratification, category and block weights, score endpoints, clipping, and same seed under different UID/order. Reject soft targets, missing cells, single-class yes/no, baseline degeneracy (`1e-12`), malformed predictions, nonfinite values and invalid resample counts. Keep a shared `zils.score` fixture showing its old exact-match/Brier values unchanged.

- [ ] **Step 2: Run `python -m unittest tests.test_base_metrics -v`.** Expect failure because the new module does not exist.
- [ ] **Step 3: Implement the three functions.** Validate original rows through existing `zils.score`, then compute per-family metrics from indexed rows. Use normalized expected Score value and target-median MAE. Resample indices internally rather than constructing duplicate case IDs; use the same draws for both models. Freeze original family baselines and calibrated probabilities. Use `random.Random(seed)` and the existing percentile indexing convention; publish variance to distinguish clipping degeneracy. Raw losses remain visible even when family skill clips to zero.
- [ ] **Step 4: Run `python -m unittest tests.test_base_metrics -v`.** Expect all assertions to pass without inference or temperature fitting.
- [ ] **Step 5: Commit only these files:** `git commit -m "Add base-only decision quality metrics"` after staging them.

## Task 2: Seal and audit independent exam draws

**Files:** Create `zils/base_exam.py`, `tests/test_base_exam.py`; modify `zils/base_round.py`, `tests/test_base_round.py` and fixture builders that create base holdouts.

**Interfaces:**
- `seal(calibration: list[dict], test: list[dict], registry: dict, blueprint: dict, *, pool_id: str, history: dict, sealed_at: float, out: Path) -> dict`: create a new private directory, write files/manifest exclusively and return the existing two-hash draw identity.
- `audit(root: Path, *, history: dict | None = None, as_of: float | None = None) -> tuple[dict, dict]`: verify immutable files, metadata, coverage and provenance; return manifest and split cases. With current history, reject exposure by `as_of` (default now); historical verification uses the bound evaluation completion time. Without current history, verify only the sealed history snapshot and make no current-freshness claim.
- `disjoint(draws: list[tuple[dict, dict]]) -> None`: reject case/fingerprint/source collisions across every split/draw and reused new-domain families. Each draw must also pass `audit` for its applicable evaluation time.
- `blueprint(cases: dict) -> dict`: deterministic category/block/type family counts and per-family calibration/test counts; no model predictions or private family names in this structural description.
- `base_round.audit(root)` delegates new election manifests to this audit. Preserve `dataset_hash`, `rows_for`, `fit_temperature`, `apply_temperature` signatures for their shared callers.

- [ ] **Step 1: Add manifest and contamination tests.** New schema `zils-base-holdout/v2` pins `calibration.jsonl`, `test.jsonl`, `registry.json`, `blueprint.json`, `history.json`; manifest includes `pool_id`, `sealed_at` and those file hashes. Cases add `category`, `block`, `source_id`; a family registry supplies `type`, `category`, `ordered_equally_spaced` for Score, and publication receipts. The history snapshot has version `zils-base-family-history/v1`, family publication receipts and disclosed case fingerprints/source IDs from every pool. Test these boundaries:

```python
# test_different_dataset_hash_is_not_disjoint:
with self.assertRaisesRegex(ValueError, "overlap"):
    base_exam.disjoint([first, renamed_duplicate])
# test_publication_history_cannot_be_invented:
with self.assertRaisesRegex(ValueError, "publication"):
    base_exam.audit(draw_with_known_family_but_no_receipt)
# test_cross_pool_disclosed_case_is_rejected:
with self.assertRaisesRegex(ValueError, "disclosed"):
    base_exam.audit(lite_draw_reusing_published_mini_case, history=history)
```

Fixture values above are local sealed directories/decoded draws created in test setup. Also cover reordered JSON keys, normalized Unicode/whitespace, same lineage with different wording, calibration/test collisions, reuse of new-domain families across qualification draws, soft targets, mixed family tags/types, invalid ordinal declarations, mutated files, malformed hashes/times, missing category/block/type coverage, symlinks, and disagreement with the frozen blueprint. Every category must contain all three task types across its two blocks. New unpublished families stay new until actual publication; evaluation alone never changes history. Test that first seed publication can start from empty history; later disclosure preserves already-completed evidence but rejects new evaluation of the same draw, even if it was sealed earlier.

- [ ] **Step 2: Run `python -m unittest tests.test_base_exam tests.test_base_round -v`.** Expect new-schema/audit tests to fail.
- [ ] **Step 3: Implement preparation/audit.** Normalize state/question strings with Unicode NFC and whitespace collapse, canonicalize mapping keys while explicitly preserving Choice option order, and hash the result excluding case IDs/labels. Also compare a choice-order-insensitive fingerprint so reordered copies collide. Source IDs identify independent source cases, not entire broad datasets. Require at most one case per source group across qualification draws, and different new-domain families for each draw. Call Task 1's `baselines()` before inference. Publication receipts record actual artifact SHA-256, public location and publication time; audit verifies their local bindings, while the operator verifies that publication actually occurred. Do not claim a local timestamp proves public availability.

Provide `python -m zils.base_exam seal --calibration FILE --test FILE --registry FILE --blueprint FILE --history FILE --pool POOL --out NEW_DIR` and `audit --round DIR --history FILE`. A blueprint is authored during development and frozen before validation. Add `seed --cases FILE --registry FILE --out NEW_DIR` to prepare a `zils-base-seed/v1` public development bundle: valid hard-label cases and typed/category-tagged families, with no known/new coverage prerequisite and no qualification status. This is how empty history starts.

Expose `publish-history --source DIR --artifact FILE --url URL --published-at UTC --history FILE --out NEW_FILE` to record actual publication of a seed bundle or completed v2 draw; this command does not publish anything. History stores first disclosure times with case fingerprints/source IDs and family receipts. Known/new tags are determined at sealing time. New inference requires a current-history audit before starting and a repeat check at completion; historical confirmation/promotion audits use the original bound completion time, not today's date. Never relabel a cached prediction's timestamp to pass this check. Public seeds establish missing known-family coverage before a qualifying pilot. Existing v1 holdout records remain archival input, never qualifying new-policy evidence.
- [ ] **Step 4: Run the Task 2 command plus `python -m unittest tests.test_feature_round tests.test_migration_regression -v`.** Expect base audits to enforce v2 without changing unrelated data contracts.
- [ ] **Step 5: Commit the audit and updated fixtures:** `git commit -m "Audit sealed election coverage and case provenance"`.

## Task 3: Apply the new policy through all pool intake paths

**Files:** Modify `zils/base_round.py`, `zils/validator.py`, `zils/model_pools.py`, `tests/test_base_round.py`, `tests/test_base_tier_intake.py`, `tests/test_pool_intake.py`, `tests/test_model_pools.py`, `tests/test_pool_rewards.py`; update affected fixtures in `tests/test_testnet.py`. Verify `zils/fleet.py`, `zils/testnet.py`, `zils/feature_tracks.py` callers.

**Interfaces:**
- Replace base `DEFAULT_POLICY` with numeric keys `score_version: 1`, `min_score_improvement: 1.0`, `max_brier_increase: 0.005`, `bootstrap_resamples: 2000`. `score_version` selects the immutable metric definitions; do not add string values to the currently numeric pool-policy contract.
- Extend `decide(cases, incumbent_uid, calibration_rows, test_rows, policy=None, *, draw_identity: dict, commitments: dict) -> dict`. Commitment metadata is keyed by UID and contains `sha256`, `block`, `hotkey`, derived from the frozen chain snapshot.
- Result adds `metric_version`, full score diagnostics and private evidence bindings. Each candidate records `commitment_block`, `hotkey`, `evaluation` (calibration/test report hashes and their `started_at`/`completed_at` times), plus existing identity/eligibility fields. The incumbent records the same evaluation evidence. `pending_winner` remains a display summary, not the only candidate permitted to qualify.
- Add `confirm_candidates(first: dict, second: dict, *, active: dict) -> list[dict]` in `base_round`; return all candidates eligible on both audited, disjoint draws with valid timing. Active context contains `pool_id`, full `model`, `incumbent_sha256`, normalized `policy`, currently allowed draw identities and private `draw_roots` keyed by manifest hash, so confirmation reopens/audits the actual cases.

- [ ] **Step 1: Add policy, timing and isolation tests.** Test +1.0 inclusive, positive lower bound strict, Brier +0.005 inclusive, calibration-only categorical improvement rejected, numeric expected-value improvement allowed, and tie order score/commitment block/hash. Use mocked metric results for exact boundary tests and one actual-score integration fixture. Parameterize all five pools. Preserve feature king-weight tests.

```python
# test_gain_and_calibration_boundaries
self.assertTrue(exact_one_point_gain_and_005_brier["eligible"])
self.assertFalse(zero_lower_bound["eligible"])
# test_late_candidate_cannot_inherit_confirmation_delay
self.assertEqual(base_round.confirm_candidates(first, premature_second, active=active), [])
# test_wrong_current_incumbent_invalidates_matching_historical_pair
with self.assertRaisesRegex(ValueError, "active"):
    base_round.confirm_candidates(first, second, active=changed_incumbent)
```

Test valid UID reassignment to the same hotkey versus a different hotkey reusing that UID; malformed candidate predictions reject that candidate only, malformed incumbent predictions fail the comparison. Test a pool failure leaves another pool's decision and rewards intact. Test public results contain no private case IDs, family names, local paths or prediction arrays. Reject old policy keys for new evaluations, noninteger/bool/zero resample counts, unknown score versions and nonfinite policy values. Version 1 production policy must equal its fixed defaults; lower-level metric tests may request fewer resamples, but a configured election cannot silently lower the thresholds or resample count.
- [ ] **Step 2: Run `python -m unittest tests.test_base_round tests.test_base_tier_intake tests.test_pool_intake tests.test_model_pools tests.test_pool_rewards -v`.** Expect policy/timing tests to fail.
- [ ] **Step 3: Integrate Task 1/2.** Keep each model's calibration fit and diagnostic full-vector Brier. Derive the shared bootstrap seed from the manifest hash, not UID. Before scoring, audit all configured qualification draws for actual disjointness. Capture each cached prediction's actual `started_at` and `completed_at` around its evaluation; retain original times on cache hits. Candidate confirmation may start only after both its own first-draw and the incumbent's first-draw evidence completed plus 12 hours. Reject old caches without timing for qualification; never invent times at restart. Bind timing to the checkpoint, split and draw, and do not let another candidate's timestamp establish eligibility. New inference must pass current disclosure checks; reuse/confirmation audits each cached result at its bound completion time. Publishing a draw retires it for new inference without erasing earlier legitimate evidence.

Retain immutable per-draw decisions for every eligible candidate under the pool store, with private prediction/report paths and hashes. Public signed results expose aggregate metrics and evidence hashes, not sealed case IDs, family names or raw predictions. Catch malformed candidate rows individually before `rows_for`; preserve its shared behavior. Keep `king_weights` and reward allocation independent of winning scores. Feature tracks currently use `validate_track` and `king_weights` for common fields: leave those common inputs valid without requiring base-only pilot paths. Active new evaluations fail closed on old policies; existing signed receipts remain unchanged on disk. Document settling pending old-publication attempts with the old release before cutover.

Update `base_round.main()` with the same required draw/commitment context: new scoring takes `--config`, `--pool`, `--chain-snapshot` plus existing report paths; `--describe-draw` remains audit-only. Remove `--min-brier-improvement` from new election scoring. Route every new-policy confirmation, including the legacy single-base intake entry, through active-context checks; use the existing pool-config migration before production cutover rather than leave an unguarded fallback. Historical `confirm` functions may remain for archival tests but cannot authorize new-policy staging.
- [ ] **Step 4: Run Task 3 tests plus `python -m unittest tests.test_feature_round tests.test_feature_rewards tests.test_testnet tests.test_evaluation_timeout -v`.** Expect pool/feature/reward compatibility and the current-main timeout fix to pass.
- [ ] **Step 5: Commit:** `git commit -m "Elect all text pools on decision quality with bound confirmation"`.

## Task 4: Verify the adapter gate from complete three-seed evidence

**Files:** Modify `zils/regression_suite.py`, `scripts/promote_base.py`, `tests/test_regression_suite.py`, `tests/test_pool_intake.py`, `tests/test_base_candidate.py`, `tests/test_migration_regression.py`; preserve `zils/migration_regression.py` every-seed logic.

**Interfaces:**
- `election_verdict(runs: dict, *, expected_jobs: list[str]) -> dict` in `regression_suite`: consume existing `runs[job][str(seed)][side]` records, require exact job/seed sets, validate finite accuracy/Brier values and compute both means. Return `passed`, `jobs`, `seeds`, `max_accuracy_drop: 0.01`, `max_brier_increase: 0.005`.
- `suite_manifest(suite: Path) -> dict`: deterministic hashes for each job's train/calibration/test files, job names and version `zils-base-adapter-suite/v1`.
- Extend `verify_regression(report, candidate, incumbent, model, *, expected_suite: dict) -> dict`; return the recomputed election verdict, rejecting incorrect profile/checkpoint/suite bindings and mismatched reported aggregates.
- Keep existing `verdict(results, tolerance)` for historical/cross-profile studies and its use in the stricter stable migration path; do not silently alter that path when dropping better-on-one-job from the election gate.

- [ ] **Step 1: Add producer/verifier parity tests.** Build per-seed fixtures with three exact seeds and two jobs. The accuracy mean is a fraction, so one point is `0.01`, not `1.0`.

```python
# test_flat_adapters_pass_election: equal sides for every required seed/job.
self.assertTrue(regression_suite.election_verdict(runs, expected_jobs=names)["passed"])
# test_forged_average_or_omitted_job_fails
with self.assertRaises(ValueError):
    promote_base.verify_regression(forged, candidate, incumbent, model, expected_suite=suite)
# test_mean_does_not_override_stable_migration: one failed seed, passing mean.
self.assertFalse(migration_regression.summary(outcomes)["stable_eligible"])
```

Also test duplicate/extra/missing seeds, wrong hashes/profile, substituted suite, invalid bounds, exactly allowed drops and slightly larger failures. A suite with no jobs fails. Preserve cross-profile study and H2O historical tests.
- [ ] **Step 2: Run `python -m unittest tests.test_regression_suite tests.test_pool_intake tests.test_base_candidate tests.test_migration_regression -v`.** Expect new verifier tests to fail.
- [ ] **Step 3: Implement the shared election gate.** Add `--election` to the existing suite CLI: require same-profile evaluation, fixed tolerance values and exact seeds; reject conflicting flags instead of silently accepting them. Pin the suite manifest before training, write its hash and full per-seed evidence to `suite-result.json`, and reject any input mutation during the run. Use the same verifier at qualification and final promotion. Existing non-election study mode and migration evidence remain readable. For stable qualification retain both the existing legacy regression requirement and every-job/every-seed migration result; passing the new mean gate alone is insufficient.
- [ ] **Step 4: Rerun Task 4 tests.** Expect identical producer/verifier verdicts and unchanged stricter migration behavior.
- [ ] **Step 5: Commit:** `git commit -m "Bind election adapter qualification to three-seed suite evidence"`.

## Task 5: Retain candidates and enforce a reviewable release window

**Files:** Create `zils/base_release.py`, `tests/test_base_release.py`; modify `scripts/promote_base.py`, `zils/validator.py`, `tests/test_base_candidate.py`, `tests/test_pool_intake.py`.

**Interfaces:**
- `active_context(config: dict, pool_id: str, *, directory: Path) -> dict`: select the installed pool, resolve private paths relative to the config directory, and normalize its model/policy/evidence references; no caller-supplied replacement incumbent or king list.
- `qualify(first: dict, second: dict, suite_report: dict, *, candidate_sha256: str, active: dict, expected_suite: dict, evidence: dict) -> dict`: select that candidate from `confirm_candidates`, verify all hashes, disjointness, ownership and Task 4; return receipt version `zils-base-ready/v1`, addressed by its canonical content hash.
- `freeze(ready: list[dict], *, active: dict, schedule: dict, window_id: str, frozen_at: float) -> dict`: require complete active receipt inventory; freeze sorted checkpoint/hotkey/commitment/evidence identities before closing inference.
- `close(frozen: dict, closing: dict, *, active: dict, members: dict, now: float) -> dict`: verify complete common closing draw, retain only surviving gates 1/2 and choose score/block/hash winner; return `status` (`skipped` or `qualified`), `winner`, `uploaded: false`, immutable evidence bindings and proposed king update. Never upload, edit live config or set weights.
- `channel_proposal(opening_channels: dict, *, new_profile_id: str | None, migration: dict | None, release_review: dict | None) -> dict`: return proposed stable/latest exact profile IDs. Consume a frozen window-opening channel snapshot and verified migration/review evidence bound to that opening latest. Missing migration/review evidence leaves stable unchanged; a newly registered winner may still enter latest. This helper never installs platform configuration.

- [ ] **Step 1: Add lifecycle tests.** Use two qualified candidate receipts, a frozen roster and mocked closing results. Hash-bound fixture reports must be recomputed/verified by the real helper, not trusted `passed: true` flags.

```python
# test_window_scores_every_ready_candidate_and_selects_best_survivor
self.assertEqual(closed["winner"]["sha256"], better_ready_hash)
# test_no_survivor_skips_without_changing_king
self.assertEqual(skipped["status"], "skipped")
self.assertEqual(skipped["proposed_base_tier"], current_track)
# test_old_policy_receipts_and_post_freeze_entries_fail
with self.assertRaises(ValueError):
    base_release.close(frozen, altered_closing, active=active, members=members, now=now)
```

Cover both candidates retained across restart; first-draw runners-up can qualify; stale incumbent/policy/suite/pilot receipts; missing closing candidates or incumbent; timestamp before roster freeze; draw overlap; UID reuse/deregistered winning hotkey; second call with identical inputs returns the same receipt; conflicting reuse of a pool/window ID fails. Missing candidate inference invalidates that candidate with a recorded reason; missing evidence for a roster member cannot silently remove it from the comparison. Later candidates may qualify for a future window but cannot enter the frozen closing roster. Test channel proposals for a passing prior latest, a failed/missing migration, a recorded problem, a skipped crown and a simultaneous new latest; the new latest never becomes stable in the same window.
- [ ] **Step 2: Run `python -m unittest tests.test_base_release tests.test_base_candidate -v`.** Expect release module tests to fail.
- [ ] **Step 3: Implement the operator workflow.** Ready receipts live in a per-pool immutable directory. Read the complete directory at freeze; take a digest of the inventory so omitted candidates cannot be chosen by a supplied shortlist. The schedule file has version `zils-base-windows/v1`, `anchor_utc` and `interval_days` (integer 14–28); window ID derives from the interval, not a free-form label. Closing waits for the configured window, freezes the roster before inference, and uses one new sealed draw for all candidates. Reuse `base_tier_epoch`/existing runner through a frozen-registry input path; no second evaluator implementation. Closing uses Tasks 1–4 verifiers and checks active bindings again immediately before staging.

Expose `python -m zils.base_release qualify`, `freeze`, `close`, each taking `--config`, `--pool`, explicit evidence paths and exclusive `--out`. `freeze` takes `--ready-dir`, `--schedule`; `close` takes `--frozen`, `--closing-round` and the frozen chain snapshot. Extend `promote_base.py` with required `--config`, `--pool`, `--window-report` for new-policy promotions. Recompute the closed window from bound evidence; resolve winner hotkey and previous kings from the verified snapshot/config, rejecting conflicting legacy flags. Keep upload opt-in. Reports remain local attestations under the bootstrap validator's trust, not multi-validator consensus.

Create a reviewable channel proposal after profile registration: the new winner becomes pool latest; the latest present at window opening may become stable only with its migration evidence and no unresolved recorded problem. A skipped crown may still advance that prior latest to stable; it never advances the just-elected base immediately. Require an explicit operator release review for problem/migration status rather than guessing from elapsed time. Keep channel proposals separate from platform availability and existing adapters' pinned models. Persist the accepted window receipt so reruns cannot crown or advance twice.
- [ ] **Step 4: Rerun Task 5 tests plus `python -m unittest tests.test_pool_rewards tests.test_feature_rewards -v`.** Expect unchanged reward budgets and no side effects from dry-run commands.
- [ ] **Step 5: Commit:** `git commit -m "Gate base promotions on retained evidence and release windows"`.

## Task 6: Qualify the exam before enabling a pool

**Files:** Create `zils/base_pilot.py`, `tests/test_base_pilot.py`; modify `zils/validator.py`, `tests/test_pool_intake.py`, `tests/test_model_pools.py`.

**Interfaces:**
- `check(bundle: dict, *, root: Path) -> dict` in `base_pilot`: read pinned real prediction reports, audit their draws and recompute scores/intervals; return `passed`, `reasons`, rank correlation, external-score span, all pairwise metrics, within-pool comparisons, policy/blueprint/runtime identities and source artifact hashes.
- `verify_qualification(report: dict, *, pool_id: str, policy: dict, blueprint: dict, runtime_identity: dict) -> None`: reject a failed, mismatched or incomplete qualification receipt.
- `runtime_contract(model: str) -> dict`: return the pool's immutable architecture/tokenizer/decision-contract hashes and runtime code/dependency/settings hashes, excluding trainable weight bytes and profile release names. Reuse pinned file inventories. Do not reuse `models.profile_identity()["runtime_sha256"]` unchanged: its current digest includes all model pins and changes with released weights.
- Extend private `base_pool_rounds[pool]` with `pilot_report`, `suite_manifest`, `family_history` paths alongside existing reference/draw paths. Task 5's active context reads these; keep private paths out of public payout contracts.

- [ ] **Step 1: Add ranking, precision and admission tests.** Fixture roster contains five immutable external reference identities with one dated Decision Index snapshot, Full-score span 20, native runtime/settings hashes, and a predeclared within-pool pair. Local fixture prediction exports simulate outcomes; they never constitute real qualification.

```python
# test_perfect_order_but_clipping_degenerate_pair_fails
self.assertFalse(clipping_degenerate["passed"])
# test_precision_boundary_is_half_a_point_on_each_side
self.assertTrue(exact_half_point["passed"])
self.assertFalse(wider_than_half_point["passed"])
# test_mini_receipt_cannot_enable_lite
with self.assertRaisesRegex(ValueError, "pool"):
    base_pilot.verify_qualification(mini, pool_id="d3-lite", policy=policy,
                                    blueprint=blueprint, runtime_identity=runtime)
```

Also test fewer than five references, mixed leaderboard editions, span below 20, average ranks with ties, constant-score correlation, Spearman below 0.9, an inverted pair with external gap exactly five, missing predeclared comparisons, mutated roster/policy after validation began, reused development cases, missing measured runtime/capacity evidence and validation without published seed history. Test enabled new-policy pool admission requires its own passing receipt; disabled pools can remain unqualified.
- [ ] **Step 2: Run `python -m unittest tests.test_base_pilot tests.test_pool_intake tests.test_model_pools -v`.** Expect missing qualification checks to fail.
- [ ] **Step 3: Implement offline qualification.** Use standard-library average ranks and correlation; no new statistics dependency. Bundle version `zils-base-pilot/v1` records `phase`, target pool, fixed policy/blueprint, development and validation draw identities, roster/reference scores, predeclared pairs, normalized report paths/hashes, native runtime/settings and measured capacity receipts. Bind the frozen design/roster hash and freeze time before any validation evaluation starts. Recompute metrics from predictions, not supplied summary scores. Require both interval endpoint distances `<= 0.5` for every nondegenerate predeclared pair and a nonzero-variance within-pool pair, in development and validation. Report excluded pairs explicitly; a required within-pool pair cannot be excluded to pass.

CLI: `python -m zils.base_pilot --bundle FILE --out NEW_REPORT`. It does not download weights or run arbitrary foreign model code. Existing d3 evaluation produces its native reports; other references use their own pinned upstream runtimes with an explicitly reviewed conversion into the existing per-case probability format. Before real execution, inventory access/licences, metric compatibility and GPU fit for the chosen roster. If a runtime cannot provide required probabilities, replace that reference before freezing the roster. Do not substitute fabricated probabilities or treat published Full scores as local measurements.

In `validator.pool_rounds`, verify the receipt against pool, frozen policy, blueprint and runtime contract before new-policy inference; incumbent weight changes alone do not change pool identity. Changed score semantics, exam blueprint or runtime contract require requalification. The pilot never produces a ready candidate or crown. Retuning after failed validation requires a new untouched validation draw. Publication history is shared across pool audits to prevent cross-pool disclosure reuse.
- [ ] **Step 4: Rerun Task 6 tests and Task 3 intake tests.** Expect fixture qualification to gate all five pools independently, with no network/GPU work.
- [ ] **Step 5: Commit:** `git commit -m "Require independently validated exam qualification per pool"`.

## Task 7: Verify the complete bootstrap path and document rollout

**Files:** Create `docs/base-election.md`; update `docs/base-candidates.md`, `docs/model-pools.md`, `docs/validators.md`, `docs/mining.md` and the existing pool-intake fixture tests. Add an integration fixture in `tests/test_base_release.py`; use synthetic weights through existing fake runner/HF server, not a new harness.

**Interfaces:** The runbook uses Task 2 preparation, Task 6 qualification, existing `zils commit-candidate`, Task 3 validator intake, Task 4 suite and Task 5 release proposals. It distinguishes software verification, GPU qualification, two-miner rehearsal and live cutover; a passing earlier stage does not mark later stages complete.

- [ ] **Step 1: Add one end-to-end CPU fixture.** Two distinct hotkeys submit immutable mini candidates; the validator discovers and scores both, retains their evidence, confirms after 12 hours, applies the adapter suite and selects on the common closing draw. Include an incumbent-copy rejection and an isolated failing second pool. Assert:

```python
self.assertEqual(set(receipts_by_hotkey), {first_hotkey, second_hotkey})
self.assertEqual(window["winner"]["sha256"], expected_checkpoint)
self.assertEqual(before_crown_weights, existing_king_weights)
self.assertFalse(window["uploaded"])
```

The fixture writes its own clock values; operational commands never accept them as actual GPU measurements. Also confirm old-cache/missing-pilot/old-policy inputs cannot reach staging and that the proposed stable movement excludes the new same-window latest.
- [ ] **Step 2: Run `python -m unittest tests.test_base_release tests.test_pool_intake -v`.** Expect any missing cross-task integration to fail explicitly.
- [ ] **Step 3: Complete integrations and write the runbook.** Include the bounded sequence below, with commands from the preceding tasks and existing training/commit documentation. Do not embed personal hostnames, private paths, credentials or customer examples.

  1. Stage current-default code and all five pool policies without activating unqualified pools; preserve allocations, other pool incumbents, feature tracks and existing adapters. Verify stock mini reset from installed settings instead of repeating it blindly. Settle or explicitly handle pending rounds before config replacement; never discard an uncertain chain publication.
  2. Inventory public seed families, teacher/source provenance, suite inputs and actual reference-model capacity. Publish missing seed history, develop the exam, freeze it and run independent validation using the existing per-host GPU coordination. Failed validation leaves election disabled under the new policy; do not silently resume old-policy crowns.
  3. Exercise two real mini miners: separate hotkeys, permitted development data, existing LoRA training/merge recipe, pinned full checkpoint and public revision, existing commit-candidate dry run, then authorized submission/readback. Keep customer-adapter jobs and optional inference as separate capabilities. Capture discovery and rejection/qualification evidence; no winner is required for a successful protocol rehearsal.
  4. Stage the active config and new prediction stores with rollback copies, install only after qualification and the operational rollout is authorized, and observe one complete validation cycle. No automated search daemon, new serving topology, paid capacity, or forced customer migration is required.
  5. At release windows, generate the reviewable winner/profile/channel changes, provision exact-profile training/serving qualifications in the platform and verify old adapter pinning before exposure. Run paired release-versus-previous S1MB `english-v1` on the same runtime/hardware, publish aggregate evidence and unfavorable results, and prepare Decision Index/JevBench submissions for owner approval. Keep a transfer ledger of exam and S1MB gains; after three releases review the spec's transfer rule and pause future election if it fails. This does not retroactively choose a different winner.

  Record what each test establishes. A mini rehearsal does not establish another pool's quality, hundreds-of-miners false-win control, multi-validator release agreement, optional inference economics or official leaderboard placement. Document those remaining network-scale gates without inventing their policies.
- [ ] **Step 4: Run full repository verification once the focused tests pass.** Run `make lint PYTHON=python` and `make test PYTHON=python`, matching CI's existing dependencies and offline settings. Expect zero failures. Then, from a fresh current-default platform checkout with the candidate core dependency, run `python -m unittest tests.test_base_channels tests.test_model_pools tests.test_pool_rehearsal -v`; expect unchanged channel selection and exact model pinning. Add a narrow platform regression test only if that existing coverage misses an observed integration change. No production database or service is used by these checks.
- [ ] **Step 5: Commit:** `git commit -m "Document and rehearse decision-quality election across pools"`. Prepare the source change for review with test results and explicitly pending GPU/operational evidence; do not describe a source-only result as a deployed election.

## Completion boundaries

**Software ready:** all seven tasks, focused and full checks pass; reviewed promotion inputs cannot bypass policy, suite, pilot, timing or closing-draw checks. Current contracts and feature/customer behavior remain intact.

**Mini exam qualified:** actual inputs and reference runtimes are available, measured, pinned, and the independent validation passes. This requires GPU measurements; it cannot be concluded from CPU fixtures.

**Bootstrap operating:** both real miners train/submit, validators discover and reject/qualify under the new installed policy, and any eventual release follows the existing reviewed promotion/channel process. A failed challenger remains a useful result. Automatic multi-validator consensus and unrestricted open-network statistical control remain separate work.
