# Model pools

Zils registers five d3 lineages: `d3-lite`, `d3-nano`, `d3-mini`, `d3-flash`
and `d3`. Each improves against its own incumbent, with its own sealed draws,
king history, allocation and exact model profiles. Registration does not make a
pool available for training or serving. New pools start disabled and unallocated;
qualification, channel configuration and healthy capacity are separate requirements.

The `d3` pool's pinned base is `vllm-sr/d3-ultra` (profile `d3-ultra-26b-v1`,
`zils/d3-ultra-pins.json`), which replaced `vllm-sr/d3` (`d3-26b-v1`). It uses
`prefix_bidir` attention: the input block attends only to itself and the question
attends to every token. Validators score it with the vendor's block mask; this
profile accepts text prompts only. A configuration that still names `d3-26b-v1`
for the `d3` pool fails validation; set that entry's `model` to the new profile
before upgrading.

## Stage an existing mini validator

From a checkout installed with the normal validator dependencies, write a **new**
configuration for review. `validator/config.json` must be your existing working
mini configuration with `base_tier`, `base_round_dir` and `base_reference`:

```sh
python -m scripts.migrate_model_pools --config validator/config.json \
  --out validator/pools-staged.json
```

This preserves mini's exact incumbent, king list and the outer
base/adapter budget. Mini receives all of the existing base allocation. Every pool starts disabled with the fixed decision-quality policy and no active
draws. The four other profiles have no king and zero allocation.
The command does not install configuration, restart a service or publish weights.
Old receipts remain archival; new single-base intake requires this migration. Never specify both
`base_tier` and `base_pools`.

`base_pools` contains `share` (the outer base budget), `burn_uid`, and `pools`.
Each pool entry contains its immutable `model`, `enabled`, `share` (fraction of
that base budget), `incumbent_sha256`, `kings`, `draws` and `policy`.
An optional `cooldown_seconds` sets the minimum spacing between one hotkey's
distinct submissions to that pool, measured between commitment blocks (12 s per
block). With the budgeted queue, a commitment inside the cooldown is refused and
recorded on that hotkey's queue row; the miner commits again after it has passed.
A retry or the confirmation run of an accepted submission is never refused. Fractions must
sum to one. Disabled or unowned pools burn their fraction; another pool cannot
inherit it. Last-five-distinct registered kings split each earned allocation.
Burn publication checks that `burn_uid` is the subnet owner's uid.

`base_pool_rounds` maps each enabled pool to `base_reference`, `base_round_dir`,
`family_history`, `suite_manifest` and `pilot_report`. Set these to separately audited, pool-specific inputs before
enabling a new competition. Never reuse a disclosed test set as another pool's
sealed draw. Runtime state goes under `base-pools/<pool>/base-tier`; prior legacy
state is preserved. Each pending validator round freezes the pool config and
results. A changed config cannot replay that pending round.

## Soup pools

A pool entry with `"type": "soup"` pays from daily soup rounds instead of a king
history (omit `type` for a king pool). Each round admits `zilsb2` body
commitments, grades the pool's bodies alone and as uniform averages ("soups") on
a seeded subset of a private round bank, and pays the soup's seats and the best
solo body (the crown). The mechanism and its constants live in
`zils/soup_round.py`; copy and twin checks in `zils/soup_copies.py`; pay rates and
the house registry in `zils/soup_rates.py`.

> **Chain endpoint requirement.** A soup round reads the roster's commitments as
> of its seed block (the first block after the day's commit deadline), so every
> validator sees the same set. Use an archive-capable chain endpoint. A pruned
> node keeps only about 256 blocks (under an hour) of state, so a round started
> later than that fails to read its commitments; it is retried, and until it
> succeeds the previous decision keeps paying.

A soup entry contains `type`, `model`, `enabled`, `share`, `house` and
`schedule`, and optionally `subset_size` (default 3,000), `screen_size` (300),
`max_pool` (20, at least 5), `max_new_per_round` (6), `max_claims_per_round`
(12), `max_download_bytes_per_round` (60 GiB), `failed_claim_backoff_rounds` (3),
`others_max` (0.3), `house_mix_max` (0.05), `max_decision_age_days` (3) and
`sketch_seed`:

```json
"d3-mini": {
  "type": "soup",
  "model": "d3-mini-4.5b-v1",
  "enabled": true,
  "share": 0.4,
  "house": "zils/data/soup-house.json",
  "schedule": {"deadline_utc": "00:00"}
}
```

- `house` is the house registry inline (a list of `name`, `repo`, `revision`,
  `body_sha256`) or a path to it relative to the validator directory. A
  `body_sha256` of `null` is identified by the hash of the body at its revision.
- `schedule.deadline_utc` is the daily commit deadline. The round for a day uses
  the first block after that deadline as its seed and admits only commitments
  made before that block.
- `max_new_per_round` caps the new bodies admitted in one round, in commitment
  block order, then hotkey; later commitments wait for the next round.
  `max_claims_per_round` and `max_download_bytes_per_round` cap the commitments
  whose files pass the header and layout check and the bytes downloaded, failed
  ones included. A hotkey whose claim fails validation or download sits out the
  next `failed_claim_backoff_rounds` rounds (0 to 30). A
  commitment whose repo cannot be downloaded at all does not count toward the
  claims cap; after three rounds of failed downloads it is retired (that
  commitment only, never the body hash). Each claim's outcome
  is saved as it finishes, so a round cut off by its deadline resumes without
  downloading a claim twice.
- `sketch_seed` (64 hex characters) seeds the weight sketches used by the copy
  checks and the scout rate. It is a private, per-validator secret: sketches are
  compared only on the validator that made them, and a known seed would let a
  copier disguise a copy. Omit it and the validator generates one on the first
  round and keeps it in `base-pools/<pool>/soup/sketch-seed`. It never appears in
  a report. **It must never change once set:** sketches of pruned bodies cannot be
  remade, so a round whose seed differs from the pool's recorded seed stops with
  an error until the original seed is restored.

`base_pool_rounds` names a soup pool with `round_bank` (the private bank, JSON
list or JSONL of exam cases) and `base` (the vendor parent folder the body change
is measured against). The chassis is the pool model's pinned parent.

### Round items from Zils Current

By default every round draws its subset from `round_bank`. The optional `exam`
setting points the pool at [Zils Current](current.md) dated exports instead:

```json
"exam": {
  "source": "current",
  "current_bank": "/var/lib/zils-current/bank",
  "max_age_days": 14,
  "max_uses": 3,
  "current_items": 1000,
  "min_items": 300,
  "min_streams": 3,
  "stream_max": 0.4
}
```

`current_bank` is the `--bank` directory of `python -m zils.current nightly` (or
`export --bank`). A relative path is read from under the validator directory.
Only `source` and `current_bank` are required. The rest are shown with their
defaults. `{"source": "bank"}`, or no `exam` at all, keeps the fixed bank.

Each round's subset (`subset_size` items, 3,000 by default) has two parts:

- **The Current part:** `current_items` items from the dated folders.
  - A folder is used only if its `manifest.json` checks out. The SHA-256 and row
    count of `items.jsonl` must match the manifest, the overlap check must have
    run, and the mode must be `local` or `publishable`. Any other folder is
    skipped and logged, and the round goes on.
  - An item is eligible when all of these hold:
    - it resolved in the `max_age_days` before the round's commit deadline;
    - earlier rounds drew it fewer than `max_uses` times (the use ledger);
    - with the question, it fits the pool model's input cap (8,192 tokens for
      d3-mini). The estimate is the same one Zils Current uses.
  - Each stream's labels are balanced. No stream may hold more than `stream_max`
    of the part, and at least `min_streams` streams must contribute.
  - Newer resolution days are drawn first. Within a day, items are ranked under
    the validator's private `sketch_seed`, because anyone can rebuild Current
    items from the public sources.
- **The bank part:** the rest of the subset, drawn from `round_bank` as before.
  Each Current stream is a family in its category and block cell of the base
  score. In the cells the Current part covers, its streams replace the bank's
  families. The bank fills every other cell, so the score keeps all ten cells.

If the Current part cannot meet these rules (too few streams, fewer than
`min_items` items, or an unreadable bank), the whole round draws from
`round_bank` exactly as it would without `exam`. The round record says so.

The private round record (`rounds/<round id>/private.json`) holds `bank_source`:
- the dated folders used, with each manifest's and items file's SHA-256;
- the folders skipped, and why;
- the items excluded, by reason;
- each stream's drawn, available and per-label counts;
- the cells Current replaced;
- the fallback reason, if there was one.

The same record is saved as `bank-source.json`, next to `subset.json` and
`current-items.jsonl` (the Current items drawn). A resumed round reuses all
three. The Current items of the last eight rounds stay loadable after they leave
the freshness window, so a round can still score earlier rounds' items.

Payout: the latest round's public record pays each seated hotkey its seat
weight and the crown its crown weight, as fractions of the pool's share. The
crown (20%) goes to the best non-house solo body only if its solo score beats
every house body's solo score and the current soup's score on the same subset
by at least the greedy margin (0.3 points); otherwise it burns. If the current
soup cannot be graded that round, the crown burns too. Before the first
soup only the house bodies set the bar. Whatever a round does not pay goes to
`burn_uid`; before the first round the whole share burns. A failed round keeps
the previous round's record for at most `max_decision_age_days` days after that
round's commit deadline; after that the pool burns its whole share until a round
succeeds.

Operators should alert when soup rounds fail: the per-pool `error` in the
validator's `soup-pools.json` and its log (a changed sketch seed, a full disk, a
GPU or driver fault, a chain endpoint that cannot serve the seed block). A pool
whose rounds keep failing pays nothing once its decision is too old.

The validator loop starts at most one round per pool per day and never waits
for it: the round runs detached (`python -m zils.soup_pool --run <request>`),
takes the shared GPU lock and runs in the GPU runtime under a 6-hour deadline.
Later loops report it running, restart a round that died (at most three times a
day) and otherwise leave the previous decision paying. The round reads the
roster's commitments as of its seed block (see the chain endpoint requirement
above). State lives under `base-pools/<pool>/soup`: known bodies, the
prediction cache, the item use ledger and `rounds/<round id>/` with the private
and public records. A crashed round resumes from its cache and is counted once in
the ledger. Each body is built once per round and graded on the whole round
subset, with rows cached every 200 items. A GPU, disk or out-of-memory error,
or a chassis that cannot grade either, fails the round (it is retried); any
other grading failure makes the body sit the round out, and a body that fails
grading in three separate rounds is banned. A body out of the pool for three
rounds loses its folder (its hash and sketch stay); cached soup predictions
unused for three rounds are deleted.
When the round bank has too few unretired items, the round is skipped and logged.
Only the public record (members, rates, rounded contributions, category scores and
hashes of the subset, predictions and soup) enters the published report.

### Submitting a body

A body's hash is refused for good only when weights that hash to it fail a check
(shape, a NaN or infinite value, the change cap) or are a near-exact re-upload.
Tensors may be stored as bf16 or f32; each is rounded to bf16 for the hash, the
change check, the copy checks and the soup.
Any other failure (owner file, licence, a hash mismatch, unreadable files, the
repo) fails only that hotkey's commitment of that repo revision; a download that
times out is retried next round. The first valid commitment of a hash wins.

Fine-tunes are welcome, including fine-tunes of house models, the current soup
or the release; they are paid at the trainer rate. A house mix is paid at the
scout rate (0.25): a body whose change from the base, fitted on the house
bodies' changes (non-negative least squares on the sketches), leaves a residual
share under `house_mix_max` (default 0.05, settable from 0 to 0.5). The dry
run's two real fine-tunes of a house body left about 0.10 to 0.12. The copy check rejects a
body that is a near-exact copy (cosine at least 0.999 between weight-change
sketches) of an earlier body by another hotkey, a house body, the current soup
or the release. The body is also fitted on earlier other-hotkey bodies and
house bodies together; if the other hotkeys' bodies add
`others_max` (default 0.3, settable from 0.1 to 0.9) or more of the body's
squared norm beyond what the house bodies explain alone, it is a copy. So
merging another hotkey's body with a house body (or averaging other hotkeys'
bodies) is rejected, while a fine-tune of a house body, two independent
fine-tunes of the same house body, or a merge with your own earlier bodies are
not. A behavioural twin of a pool body is also refused.

Only a near-exact re-upload (or a body with no change from the base) bans the
body hash. "Explained by earlier" and behavioural twins judge the current pool:
they refuse only that commitment (explained shares count only earlier bodies
still in the pool), and the commitment is evaluated again when it is
re-committed or when a body that explained it, or its twin partner, leaves the
pool. A house entry that cannot be downloaded or hashed sits that round out and
is recorded in the private round record.

Commitments are public on chain the moment they are made, and a public repo can
be copied and committed by someone else before the deadline. To avoid
front-running, upload the body to a **private** Hugging Face repo, commit
`zilsb2:<body sha256>:<revision>:<repo>` while the repo is still private, then make
the repo public before the round's commit deadline. Validators download at the
round, after the deadline; a repo that is still private then fails the download
and is retried in a later round.

## Submit, evaluate and promote

Run `zils setup` to select the exact base profile to beat in your tier, or pass
`--model <profile-id>` to `setup`, `doctor` or `commit-candidate`. `doctor` prints
the matching reference/training recipe. The CLI reads older candidates' profiles
from their metadata and checks uploads against each profile's own file list.
It does not discover or enable live competitions; verify the accepted incumbent
and allocation with the validator before training.

The existing `zils1` commitment format is unchanged: one active candidate per
hotkey. A hotkey's one commitment slot is shared with `zilsb2` soup bodies
([soup pools](#soup-pools)). The validator reads at most 16 KiB of metadata at the committed revision,
resolves its exact parent, and refuses a stale or wrong-pool profile before weight
downloads. Full download still verifies the committed checkpoint hash, files and
tensor contract. Work uses the existing shared GPU lock and separate pool stores.
A pool failure is recorded in the local round's `pool-intake.json`; another pool's
result and king history remain intact. Exam challengers earn nothing until crowned.

The [decision-quality election runbook](base-election.md) defines the fixed score,
independent exam qualification, candidate-specific 12-hour confirmation delay,
three-seed adapter gate and common closing draw. Every eligible candidate can
qualify. Each pool needs its own admission document: a passing pilot receipt or an explicit
incumbent-challenge contract. A mini document cannot enable lite, nano, flash or d3. A weight-only release preserves the execution
contract, while changed runtime, blueprint or score semantics require requalification.

Promotion uses `promote_base.py --config ... --pool ... --window-report ...`.
It rechecks the closed window against installed state and produces a reviewable
profile/king update. Stable/latest movement remains separate from election and
existing adapters' exact model pins.

### Soup releases

A soup pool (see [soup pools](#soup-pools)) has no committing king. The soup changes the pool's served model only through the release gate. At a
release window (every 14 to 28 days) the soup must beat the incumbent on two
sealed release draws, pass the confirmation draw and the adapter suite, and be
promoted by the owner's promotion PR. Members are paid through seats, never
added to `kings`. Eligibility, the round steps, rates and the published record
are in [mining: soup pool](mining.md#soup-pool). The greedy margin, contribution
floor, member cap, pool size, change bound, 80/20 split and rates are launch
policy, not tuned results.

## Qualify each profile and role

Start with the pinned CUDA runtime from the normal [mining setup](mining.md).
The following is a command template, **not evidence of a qualified lite pool**:

```sh
python -m scripts.qualify_node --model d3-lite-0.8b-v1 \
  --runtime-python /path/to/d3-runtime/bin/python --out qualification/d3-lite
```

Qualification pins the selected model in the runtime check, reference download,
sizing, training and reload/parity probes. It writes measured memory and prompt
length with exact profile, runtime and trainer identities. Image/video quality,
base-candidate merge and serving parity still need their own evidence before those
capabilities are advertised. The probe can download weights and consume GPU time;
it is not run by staging a configuration.

Mini keeps its existing measured memory curve. Other lineages need a local
`qualifications` mapping in the queue-worker config, from exact profile ID to that
host's `qualification.json`. The fit check requires matching profile/runtime/trainer
identities and a measured prompt ceiling covering the job. It reserves the measured
peak plus the larger of 1,024 MiB or 15% as headroom. A result measured below a
band's ceiling (for example 8,192 tokens for mini) does not qualify that band. This local fit check does not
replace coordinator-issued capacity/quality qualifications. Image jobs additionally
require `training.modalities` to include `images`, a measured ceiling covering the
profile cap and a peak covering those image probes. The text qualification command
records only `text`; it cannot certify image memory.

Set each runtime's measured free-memory gate separately: `ZILS_D3_LITE_MIN_FREE_MIB`,
`ZILS_D3_NANO_MIN_FREE_MIB`, `ZILS_D3_FLASH_MIN_FREE_MIB`, or
`ZILS_D3_FULL_MIN_FREE_MIB`. Mini retains `ZILS_D3_MIN_FREE_MIB`. New profiles fail
closed when neither a configured gate nor an explicit measured requirement is
present. Training certificates do not certify serving; admission remains bound to
an exact profile, recipe, context and role. No automatic substitution with a larger
or smaller model occurs when a pool has no qualified capacity.
