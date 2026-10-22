# Decision-quality base elections

This is the text-base election contract for `d3-lite`, `d3-nano`, `d3-mini`,
`d3-flash` and `d3`. Every pool has its own incumbent, candidates, qualification,
king history and stable/latest channels. Mini is the first bootstrap target.
Vision and box competitions keep their own metrics. CPU fixtures establish
software behavior; they do not qualify an exam, measure model improvement or
establish an official leaderboard result.

## What miners improve

Miners train and merge full-weight candidates under the existing
[base candidate contract](base-candidates.md#reference-recipe-train-a-lora-merge-it).
They may choose their training method and permitted development data. Validators
measure unseen decision quality against the pool's incumbent:

| Question | Family metric | Trivial baseline |
| --- | --- | --- |
| Choice | Exact accuracy | Strongest of uniform guessing, fixed option ID and fixed original position |
| Yes/no (`noul`) | Balanced accuracy | 0.5; both classes are required |
| Score | Expected numeric outcome's normalized MAE | Best constant target-median prediction |

Convert each family to skill relative to its baseline and clip to [0,1]. Average
families equally within each category/block; weight known/new blocks 60/40.
Category weights are reasoning 25.85%, language 25.85%, retrieval 20.02%, tools
18.28%, arts 10%. The result is a 0–100 score. Score outcomes must be ordered and
equally spaced. Every category must cover all three question types. Degenerate
baselines, soft labels and incomplete coverage are rejected before inference.

The fixed production policy is:

```json
{"score_version": 1, "min_score_improvement": 3.0,
 "max_brier_increase": 0.005, "bootstrap_resamples": 2000}
```

Each candidate needs at least +3 points, a strictly positive paired bootstrap
lower 95% bound, and calibrated full-vector Brier no more than 0.005 worse.
Temperature is fitted only on calibration. Resampling is paired within families,
stratified by class for yes/no, with original baselines held fixed. All candidates
use the same manifest-derived seed. Score ties prefer the earlier commitment,
then checkpoint hash. Shared customer `zils.score` and customer acceptance rules
are unchanged.

Every eligible candidate can qualify, including a first-draw runner-up. A second
independent draw must start at least 12 hours after both that candidate's and the
incumbent's first-draw evidence finished. Cached evidence retains its actual times.
Late entrants get a fresh incumbent confirmation comparison when the earlier
cache is too old; earlier report files remain immutable. A new UID cannot inherit
another hotkey's result. Public results contain aggregate
metrics and evidence hashes; private files retain predictions and family details.
Winning an exam pays nothing. Only the existing crowned kings receive the existing
base allocation until an approved promotion changes the incumbent/history.

## Prepare public history and private exams

Use teacher-authored cases. Public decision datasets may seed newly rewritten
cases; never copy their public cases into the sealed exam or seed from S1MB,
Decision Index or JevBench test sets. Never use customer examples. Audit semantic
relatedness as well as exact duplicates. `source_id` identifies an independent
source case, not a broad dataset. The audit checks IDs, normalized prompt copies,
reordered Choice copies, source lineage and repeated new-domain families across
draws. It cannot discover semantic copies with falsely unrelated source IDs.

Author hard-label case JSONL files and `registry.json`, mapping each family to
`type` and `category`; Score families additionally declare
`"ordered_equally_spaced": true`. Start a shared history file with
`{"version":"zils-base-family-history/v1","families":{},"cases":[]}`.
Prepare initial public development material without a known/new prerequisite:

```sh
python -m zils.base_exam seed --cases public-seed.jsonl \
  --registry public-registry.json --out seed-bundle
```

Publish the actual seed artifact through the operator's approved publication
process. Record its verified public URL, file and UTC publication time:

```sh
python -m zils.base_exam publish-history --source seed-bundle \
  --artifact published-seed.jsonl --url https://example.org/actual-public-artifact \
  --published-at 2026-10-21T12:00:00Z --history history-empty.json \
  --out family-history.json
```

Replace the example URL/time with the real publication receipt. This command
records publication; it never uploads or proves a URL is publicly accessible.
History is shared across pools. Known families require matching first-publication
receipts in history and the registry; unpublished families remain new. Evaluation
alone does not make a family known. Publication retires exposed cases for new
inference but preserves evidence completed before disclosure.

For each private draw, author distinct calibration/test cases with `category`,
`block` (`known`/`new`) and `source_id`. Generate the structural blueprint with
`zils.base_exam.blueprint({"calibration": cases, "test": cases})` using the actual
split lists. Freeze it during development; the serialized blueprint contains
counts by category/block/type, without private family names. Seal each draw:

```sh
python -m zils.base_exam seal --calibration calibration.jsonl --test test.jsonl \
  --registry registry.json --blueprint blueprint.json --history family-history.json \
  --pool d3-mini --out private-draw
python -m zils.base_exam audit --round private-draw --history family-history.json
python -m zils.base_round --round private-draw --describe-draw
```

Use new output paths. Keep the private directory private. V2 manifests pin both
splits, registry, blueprint and history. V1 remains archival and cannot qualify.
New inference audits current history before and after execution; historical
verification uses the original completion times. Never edit cache times.

## Qualify the exam, independently for every pool

Inventory access/licences, upstream runtimes, supported probability outputs and
measured capacity before choosing references. Use at least five immutable
external model identities from one dated Decision Index Full-score snapshot,
spanning at least 20 points. Freeze the roster, policy, blueprint, settings and
predeclared pairs after development and before untouched validation. Include a
pair of distinct candidates within the target pool under the same execution
contract. If a reference cannot export the required probabilities, replace it
before freezing; published leaderboard scores are not local predictions.

`zils-base-pilot/v1` bundles contain `phase: "validation"`, `pool_id`, `policy`,
`blueprint`, `runtime_identity`, `roster`, `pairs`, `within_pool_pair`, `draws`,
`reports`, `capacity`, `history`, `frozen_at` and `design_sha256`. The design hash
is `base_round.dataset_hash(base_pilot.design(bundle))`. Development reports
must finish before that freeze time; validation must begin after it.

Each roster entry pins `model_sha256`, `runtime_sha256`, `settings_sha256` and
`kind` (`reference` or `pool`). References add `edition` (date, URL, SHA256) and
`full_score`; pool entries add `pool_id`. `draws` maps development/validation to
path and the two draw hashes. `reports[phase][model][split]`, `capacity[model]`
and `history` are `{path, sha256}` file references. Normalized prediction exports
contain those three model/runtime/settings hashes, dataset hash, actual start/
completion times and complete `predictions` in the existing probability format.
Capacity receipts bind the same identities, device, measured peak/available GPU
MiB, completion time and an explicit `probability_export_reviewed: true` review.
Foreign runtime execution and probability conversion remain operator work.

```sh
python -m zils.base_pilot --bundle pilot-bundle.json --out pilot-report.json
```

The verifier recomputes ranking and every pairwise interval from the exports.
Both phases require Spearman >=0.9 and no inversion for an external gap >=5.
The external ranking scores Choice and yes/no families only: Decision Index references do not
answer Score questions, and on 22 Oct 2026 Score families alone moved the five-model Spearman
from 1.0 to 0.7 by favouring the d3 runtime. Score families stay in every election, where all
candidates share the pool's runtime, and in the within-pool precision comparisons.
Every nondegenerate predeclared interval must be within 0.5 point of the estimate
on each side. Identical or clipping-degenerate pairs provide no sizing evidence;
the required within-pool pair must have nonzero paired variance. Failed validation
requires a new untouched validation draw after any retuning. A pilot never crowns.

## Stage and rehearse before cutover

1. Back up the installed config and pending-round state. Settle pending old-policy
   publications with the old release; preserve uncertain chain attempts. Verify
   the installed mini incumbent and stock reset rather than repeating it blindly.
   `python -m scripts.migrate_model_pools --config CONFIG --out STAGED_CONFIG`
   preserves mini's incumbent, kings and allocation,
   installs the fixed policy, and leaves every pool disabled with no active draws.
2. Supply passing per-pool evidence before enabling intake. Each enabled
   `base_pool_rounds[pool]` requires `base_reference`, `base_round_dir`,
   `family_history`, `suite_manifest`, `pilot_report`, resolved relative to config.
   Pin the suite using `regression_suite.suite_manifest(path)`. The public pool
   contract carries no private paths. Keep independent pool stores; old untimed
   caches cannot qualify. Disabling intake is separate from hosted availability.
3. Exercise two real bootstrap miners with separate hotkeys, d3-mini on both,
   permitted development data, the existing LoRA/merge recipe and immutable public
   candidates. Run `zils commit-candidate` as a dry run, inspect it, and submit with
   `--yes` only when operational submission is authorized. Capture commitment
   readback, discovery, copy rejection, inference and qualification evidence.
4. Install the reviewed config only after GPU qualification and rollout approval;
   observe a full cycle. Keep a rollback copy of the prior binary/config/store,
   but never resume old-policy crowns or replay an uncertain chain transaction.
   A protocol rehearsal can succeed without a winning candidate.

Customer adapter jobs remain separate from base competition. Miner inference is
optional and may supplement operator hosting; it is not an election gate. No
new serving topology, automated training daemon or forced adapter migration is
introduced. A per-host GPU lock coordinates local work; miners still compete.

## Qualify candidates and close a release window

Run the adapter suite for each candidate under the same profile:

```sh
python -m zils.regression_suite --election --model d3-mini-4.5b-v1 \
  --suite SUITE --incumbent REFERENCE --candidate CANDIDATE --out ADAPTER_RESULT
python -m zils.base_release qualify --config validator/config.json --pool d3-mini \
  --first FIRST_PRIVATE_DECISION --second SECOND_PRIVATE_DECISION \
  --snapshot-first FIRST_CHAIN_SNAPSHOT --snapshot-second SECOND_CHAIN_SNAPSHOT \
  --suite ADAPTER_RESULT/suite-result.json --candidate-sha256 CHECKPOINT_SHA256 \
  --out ready-receipt.json
```

The suite pins every job's train/calibration/test bytes and requires exactly
seeds 553, 771, 991. For every job, mean accuracy may fall by at most 0.01 and
mean Brier rise by at most 0.005. Flat results pass; omitted jobs/seeds, changed
inputs or forged averages fail. The receipt binds raw decisions, reports, chain
ownership, active incumbent/policy, suite and pilot. Private decisions are under
`base-pools/<pool>/base-tier/decisions`; the public sidecar is insufficient.

Author a schedule `{ "version": "zils-base-windows/v1", "anchor_utc":
"ACTUAL_UTC_ANCHOR", "interval_days": 14 }` with a 14–28 day interval. At a
scheduled window, freeze the entire active ready directory before any closing
inference. Window IDs derive from the schedule; existing receipts are immutable.
Save the actual opening profile IDs in `opening-channels.json` as
`{"stable": "EXACT_STABLE_PROFILE", "latest": "EXACT_LATEST_PROFILE"}`. The freeze
pins this snapshot. Requalification retains all attestations while choosing one
verified receipt per checkpoint/owner; UID changes do not create another entrant.

```sh
python -m zils.base_release freeze --config validator/config.json --pool d3-mini \
  --ready-dir validator/base-pools/d3-mini/ready --schedule schedule.json \
  --channels opening-channels.json --out frozen-window.json
python -m zils.base_release close --config validator/config.json --pool d3-mini \
  --frozen frozen-window.json --closing-round FRESH_SEALED_DRAW \
  --chain-snapshot CLOSING_SNAPSHOT --device cuda --out closed-window.json
python scripts/promote_base.py --config validator/config.json --pool d3-mini \
  --window-report closed-window.json --candidate WINNER_DIRECTORY \
  --round RELEASE_NUMBER --staging NEW_STAGING_DIRECTORY
```

`close` uses the existing evaluator for the incumbent and every frozen candidate
on a common independent draw. It selects the highest-scoring survivor, then
commitment block/hash; no survivor means no new king. Later entrants wait for a
future window. Staging recomputes the evidence and derives kings from installed
configuration and verified snapshots. Upload remains opt-in (`--upload`). None
of these commands installs live config, publishes chain weights or changes a
hosted channel. Local receipts are bootstrap-validator attestations, not
multi-validator consensus. If a process stops after saving its internal receipt
but before creating `--out`, rerun with the same inputs: it recovers the recorded
times/result. A closing retry does not run inference again. Conflicting inputs
are rejected; a frozen roster never admits later entrants.

After exact-profile registration and platform training/serving qualification,
propose the new profile as `latest`. `channel_proposal` requires the verified closed-window report and active context;
it uses the opening channels saved by `freeze`. Its explicit release review must
bind migration evidence for that opening latest. Stable requires the existing legacy regression test and
every-job/every-seed migration success, with no unresolved recorded problems.
The just-elected latest never becomes stable in the same window. A skipped crown
can still advance the prior latest. The helper persists one content-addressed proposal per pool/window, containing
its input evidence and output `channels`. Generate it once when review inputs
are final; identical retries recover it and conflicting proposals fail. Review
the saved window and channel proposal before installing it. Existing adapters retain their exact base until their own
migration is accepted.

## Evidence after releases

Run paired release-versus-previous S1MB `english-v1` under identical runtime and
hardware; publish aggregates, uncertainty and unfavorable outcomes. Prepare
Decision Index/JevBench submissions for owner approval. They are external transfer
checks, not per-crown approval gates. Maintain the exam/S1MB transfer ledger; after
three releases apply the [design's transfer review](superpowers/specs/2026-10-21-base-election-design.md)
and pause future election if its transfer rule fails, without choosing a different
past winner.

`make lint` and `make test` cover software contracts and CPU fixtures. Real exam
qualification, both real miner submissions, live cutover, external placement,
hundreds-of-miners false-win control, multi-validator release agreement and
optional-inference economics require separate evidence.
