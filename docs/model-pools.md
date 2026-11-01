# Model pools

Zils runs four base pools, one per model size: `d3-lite` (0.8B), `d3-mini`
(4.5B), `d3-flash` (9B) and `d3` (z3, 27B). Each is a [soup pool](#soup-pools)
with its own bodies, rounds, house list, allocation and exact model profile.
(The `d3-nano` lineage stays registered so older records stay readable; it has no
pool.) Registration does not make a pool available for training or serving;
qualification, channel configuration and healthy capacity are separate
requirements. `zils/data/base-pools-launch.json` is the launch `base_pools`
block for all four pools.

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
total at most one; an unassigned remainder burns. Disabled or unowned pools burn
their fraction; another pool cannot inherit it. Last-five-distinct registered kings split each earned allocation.
Burn publication checks that `burn_uid` is the subnet owner's uid.

`base_pool_rounds` maps each enabled pool to `base_reference`, `base_round_dir`,
`family_history`, `suite_manifest` and `pilot_report`. Set these to separately audited, pool-specific inputs before
enabling a new competition. Never reuse a disclosed test set as another pool's
sealed draw. Runtime state goes under `base-pools/<pool>/base-tier`; prior legacy
state is preserved. Each pending validator round freezes the pool config and
results. A changed config cannot replay that pending round.

## Emission schedule

A soup pool entry may add an opening block and a boost:

```json
"d3-flash": {
  "type": "soup",
  "model": "d3-flash-8.4b-v1",
  "enabled": true,
  "share": 0.20,
  "opens_block": 7300000,
  "boost": [{"share": 0.30, "blocks": 100800}, {"share": 0.25, "blocks": 100800}],
  "house": "zils/data/soup-house-flash.json",
  "schedule": {"deadline_utc": "12:00"}
}
```

- `opens_block` is the chain block at which the pool opens, or `null` for a pool
  that stays closed until a block is set. Before it the pool runs no rounds,
  accepts no decision and pays nothing; its share burns. A pool without
  `opens_block` is open.
- `boost` (it needs `opens_block`) is a list of 1 to 8 steps paid in order from
  the opening block: each pays `share` for `blocks` blocks (7,200 blocks is about
  a day at 12 s per block), then the pool pays its own `share`.
- The pools' own shares must total at most one, and so must the shares paid at
  every block: a boost is paid out of budget that closed pools and the
  unassigned remainder leave unpaid, never out of another pool's share. A
  configuration that would overfill the budget at any block is refused at
  startup, naming the block.
- Shares depend only on the chain block. Each validator round freezes the block
  of its chain snapshot into its `base-pools.json` (as `block`), and the published
  weights use the shares at that block, so validators running the same
  configuration agree on every pool's share. A configuration without any
  `opens_block` or `boost` pays exactly as before and freezes no block.
- Only soup pools take a schedule.

The launch configuration gives lite 35% and mini 30% from the start, and flash
20% and z3 10% once they open; 5% is unassigned and burns. Flash is boosted to
30% for two weeks and 25% for two more; z3 to 15% for two weeks. Both opening
blocks are `null` in `zils/data/base-pools-launch.json` and must be set to the
same values on every validator before they open. z3 cannot open while flash is
still boosted to 30%: 35 + 30 + 30 + 15 would exceed 100%.

## Soup pools

A pool entry with `"type": "soup"` pays from daily soup rounds instead of a king
history (omit `type` for a king pool). Each round admits the pool's body
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
`others_max` (0.3), `house_mix_max` (0.05), `max_decision_age_days` (3),
`decision_rounds` (3, from 1 to 7), `known_public` (empty),
`max_known_public_bytes_per_round` (16 GiB), `max_known_public_seconds_per_round`
(7,200), `scoring` (`calibrated_brier`) and `sketch_seed`:

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
- `known_public` lists public models of the pool's size that are not house
  ingredients, in the same format and forms as `house` (inline or a path). An
  unchanged or near re-upload of one is paid at the scout rate (see below).
  `zils/data/soup-known-lite.json` (lite) and `zils/data/soup-known-mini.json`
  (mini) are the launch lists; flash and z3 have none.
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

- `scoring` sets how every graded body and soup is scored. Under the default,
  `calibrated_brier`, the validator holds apart 20% of each family's round items
  (and of each yes/no label), chosen by a private secret that changes every
  round. On those items it fits one temperature per body or soup: the value on a
  grid from 0.1 to 10 that minimises the mean Brier loss. It then scores the
  other 80% at that temperature: Brier skill for yes/no and Choice questions,
  ranked probability score skill for Score questions, each against the best
  constant answer for the family, in the same cells and weights as before.
  Confidence therefore counts, while a pure change of confidence (sharper or
  softer probabilities with the same answers) is undone by the fit and earns
  nothing. `calibrated_argmax` uses the same fit but scores top answers (ranked
  probability score for Score questions). `argmax` is the earlier exam: top
  answers, and the distance of the expected value from the label for Score
  questions, with no fit; that distance moves when probabilities are only
  sharpened, so it rewards confidence changes that change no answer.
  The released soup's `model.json` carries the temperature fitted for it on the
  round's held-apart items, and the public round record publishes it. Bodies are
  always graded as temperature-1 candidates built by the validator; a
  temperature in a body's own files is never read. Greedy margins, the
  contribution floor and the crown bar stay at 0.3 points on whichever scale the
  pool uses.
- `decision_rounds` sets how many rounds' items each seat and crown decision is
  scored on. A body's measured gain moves by about 0.4 to 0.6 points from one
  3,000-item subset to the next, more than the 0.3-point margin, so a single
  subset seats a body with no real gain on a large share of nights. With the
  default of 3, the round scores every pool body, every soup it tries, each
  leave-one-out soup, the current soup (the crown bar) and the chassis baseline
  on the union of its own subset and the subsets of the validator's last two
  finished rounds that drew items. The greedy margin, the contribution floor,
  the house-mix rule and the crown bar all use those pooled scores. The earlier
  items are read from each round's own record of the cases it drew
  (`rounds/<round id>/subset-cases.json`), not from the bank, so an item that has
  since left the bank still counts. An earlier item is dropped, for every body
  alike, if the use ledger has retired it since or if the current bank holds a
  different case under the same id. A round that failed, was skipped or has no
  record of its cases is not pooled. Predictions are cached per body or soup and
  item, so only missing rows are graded: in practice the soups the round tries,
  and new bodies on the earlier items. The use ledger still counts only each
  round's own subset. If the estimated grading would not finish within the
  round's 6-hour deadline (less 30 minutes), the round pools fewer rounds, down
  to its own subset alone. The private round record names the pooled rounds, the
  item count, any rounds that could not be pooled and any fallback. The public
  record carries the pooled round ids and item count. With `decision_rounds` 1 a
  round behaves exactly as a single-subset round. Cached soup predictions are
  kept for at least `decision_rounds` rounds.

`base_pool_rounds` names a soup pool with `round_bank` (the private bank, JSON
list or JSONL of exam cases) and `base` (the vendor parent folder the body change
is measured against). The chassis is the pool model's pinned parent. Bank items
longer than the pool's input cap are left out of its rounds.

Each pool reads only the body commitments addressed to it: `zilsb2:…` for
`d3-mini` and `zilsb3:<pool>:<body sha256>:<revision>:<repo>` for `d3-lite`,
`d3-flash` and `d3`. Body parents, byte caps and download deadlines per pool are
in `zils/bodies.py` (`BODY_POOLS`) and [mining](mining.md#four-pools). A
validator grades the pools its `soup_grading` setting allows
([validators](validators.md#soup-pools-hardware-disk-and-the-emission-schedule));
a pool it does not grade burns its share in that validator's weights.

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
  - Yes/no streams keep their labels balanced. No stream may hold more than `stream_max`
    of the part, and at least `min_streams` streams must contribute.
  - Newer resolution days are drawn first. Within a day, items are ranked under
    the validator's private `sketch_seed`, because anyone can rebuild Current
    items from the public sources.
- **The bank part:** the rest of the subset, drawn from `round_bank` as before.
  Each Current stream is a family in its category and block cell of the base
  score. In the cells the Current part covers, its streams replace the bank's
  families. The bank fills every other cell, so the score keeps all ten cells.
  When the bank's remaining cells cannot fill the rest of the subset, the
  Current part grows to take it. If Current streams cover every cell, the
  whole subset comes from Current.

If the Current part cannot meet these rules (too few streams, fewer than
`min_items` items, or an unreadable bank), the whole round draws from
`round_bank` exactly as it would without `exam`. The round record says so.

#### Quality settings

Every category and block cell of the base score has a fixed weight, and the
families in a cell count equally. A Current stream that contributes a few dozen
items can therefore move the whole score, and a stream whose labels no body can
predict adds only noise. Three optional settings guard against both:

```json
"exam": {
  "source": "current",
  "current_bank": "/var/lib/zils-current/bank",
  "current_share": 0.33,
  "min_family_items": 100,
  "exclude_streams": ["stackexchange-score-*"]
}
```

- **`min_family_items`** (default 100): families are chosen before the Current
  budget is spread, so no kept family falls below this size.
  - A stream's supply is what it can give: yes/no streams balanced, and at most
    `stream_max` of the budget. A stream whose supply is below the minimum joins
    the other small streams of its cell and question type in one family,
    `current-<category>-<block>-<type>-small` (yes/no, choice and score streams
    never share a family). Each merged item keeps its stream name in
    `current_stream`.
  - Only families whose supply reaches the minimum qualify. While the budget
    has room for another family's minimum, families are kept in this order:
    first the largest family of each category, block and question type, then
    the family whose cell has the fewest kept families (larger supply first).
    Families of equal supply are ordered by a hash of their name under the
    validator's private `sketch_seed` and the round, so rounds rotate which of
    them are kept and miners cannot predict it.
  - Each kept family gets the minimum, and the rest of the budget is spread
    over the kept families as evenly as their supply allows.
  - A cell with no kept Current family is filled from `round_bank`. If fewer
    than `min_items` items or `min_streams` streams remain, the round falls
    back to `round_bank`.
- **`exclude_streams`** (default `["stackexchange-score-*"]`): glob patterns of
  stream names that are never drawn. Stack Exchange score streams show close to
  zero skill for every body measured, so they are excluded by default. `[]`
  draws every stream.
- **`current_share`** (default 1, in (0, 1]): below 1 the round **blends**.
  `round_bank` keeps its families in every cell, and Current families are added
  next to them, up to `current_share` of the subset (and at most
  `current_items`). At 1, Current streams replace the bank's families in the
  cells they cover, as described above.

A pool that sets none of the three draws exactly as before they existed. A pool
that sets any of them gets the defaults above for the others; set
`"min_family_items": 1` or `"exclude_streams": []` to turn one off.

With any quality setting the round record also holds the excluded streams and
their item counts (`excluded_streams`), every Current family with its streams,
supply, items drawn and whether it was kept (`current_families`), the merged
families among them (`merged_families`), the cells a blend
added Current families to (`added_cells`), the items, families and Current
items per cell (`cells`), and every cell resting on fewer than
`min_family_items` items (`thin_cells`, also logged). A thin cell is still
scored; the warning says its weight rests on few items.

The private round record (`rounds/<round id>/private.json`) holds `bank_source`:
- the dated folders used, with each manifest's and items file's SHA-256;
- the folders skipped, and why;
- the items excluded, by reason;
- each stream's drawn, available and per-label counts;
- the cells Current replaced;
- with the quality settings, the excluded streams, merged families, blended
  cells and per-cell item counts;
- the fallback reason, if there was one.

The same record is saved as `bank-source.json`, next to `subset.json` and
`current-items.jsonl` (the Current items drawn). A resumed round reuses all
three. The Current items of the last eight rounds stay loadable after they leave
the freshness window, so a round can still score earlier rounds' items.

Pooled decisions (`decision_rounds`) work the same way on a Current exam: each
earlier round's items come from its `subset-cases.json`, item by item, so a bank
that changes every day still pools. Cached predictions are scoped to the round
bank's hash, though. On a day the bank changes, each pool body and soup is graded
again on the earlier rounds' items, not only on the new subset. That is about
twice the grading of a fixed bank. The deadline fallback above still bounds it.

Payout: the latest round's public record pays each seated hotkey its seat
weight and the crown its crown weight, as fractions of the pool's share. The
crown (20%) goes to the best non-house solo body only if its solo score beats
every house body's solo score and the current soup's score on the same pooled
items by at least the greedy margin (0.3 points); otherwise it burns. If the current
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
hashes of the subset, predictions and soup, all on the round's own subset, plus
the pooled round ids and item count, the pool's scoring and the soup's fitted
temperature) enters the published report.

### Submitting a body

A body's hash is refused for good only when weights that hash to it fail a check
(shape, a NaN or infinite value, the change cap) or are a near-exact re-upload.
Tensors may be stored as bf16 or f32; each is rounded to bf16 for the hash, the
change check, the copy checks and the soup.
Any other failure (owner file, licence, a hash mismatch, unreadable files, the
repo) fails only that hotkey's commitment of that repo revision; a download that
times out is retried next round. The first valid commitment of a hash wins.

Fine-tunes are welcome, including fine-tunes of house models, the current soup
or the release (the pool's served model, whose body is the chassis body); they
are paid at the trainer rate. These three are the allowed bases: what a body
changes beyond them is its own work.

The copy check rejects a body that is a near-exact copy (cosine at least 0.999
between weight-change sketches) of an earlier body by another hotkey, a house
body, the current soup or the release. The body is also fitted (non-negative
least squares on the sketches) on the allowed bases alone, and on the allowed
bases plus earlier other-hotkey bodies. If those other bodies explain
`others_max` (default 0.3, settable from 0.1 to 0.9) or more of the part the
allowed bases leave unexplained, it is a copy. The soup is mostly seated miners'
bodies, so a fine-tune of the soup is still not refused because those miners'
bodies resemble it. A near-copy of another hotkey's body, a merge of it with a
house body, and a re-weighted mix of the seated bodies are all rejected. A
fine-tune of a house body, two independent fine-tunes of the same house body,
and a merge with your own earlier bodies are not. On the weight changes of four
public 4B decision models and two fine-tunes of one of them, independent pairs
scored at most 0.10 on this measure, and copies and re-weighted mixes scored
0.36 or more. A behavioural twin of a pool body is also refused.

A house mix is paid at the scout rate (0.25): a body whose change from the base,
fitted on the house bodies' and listed public models' changes, leaves a residual
share under `house_mix_max` (default 0.05, settable from 0 to 0.5). So is a body
that, when admitted, was almost all allowed bases: fitted on the house bodies,
the listed models, the current soup and the release, it left a residual under
the same bar (a near-pure re-weighting of the soup, or a mix of the soup with a
listed model). The dry run's two real fine-tunes of a house body
left about 0.07 to 0.12, so a real fine-tune of the soup or release keeps the
trainer rate.

A known public model's re-upload is also paid at the scout rate: a body whose
weight-change sketch has cosine at least 0.9 to the sketch of a model on the
pool's `known_public` list. The check looks at the body as committed, so an
unchanged or lightly changed re-upload is a scout's, while a fine-tune that
moves well away from the public model (cosine under 0.9) keeps the trainer rate.
Listed models are also allowed bases in the copy check, like the house bodies,
the soup and the release: a real fine-tune of one is never refused as a copy of
another hotkey's re-upload of it. Unlike the other bases, re-uploading a listed
model is allowed, since that is the scout's work. The first commitment that is a
near re-upload (cosine at least 0.9) of a listed model wins: a later near
re-upload of the same model by another hotkey is refused as explained by that
earlier body while it stays in the pool, and a near-exact copy of any earlier
body is refused as before. Listed models are part of the house-mix fit, so a mix of
several listed models (whose cosine to each one is under 0.9) is still a scout's,
not a trainer's. House bodies stay at 0.

The validator resolves each list entry once: it downloads the pinned revision
through the same fetch path as a body (within twice the pool's body size cap,
since public repos may carry an untied `lm_head`), checks `body_sha256` when it
is pinned, sketches the language layers with its own `sketch_seed`, keeps only
the sketch and deletes the files. A text-only checkpoint (no vision tower, with
`layers.*`, `model.layers.*` or `model.language_model.layers.*` names) is
sketched on its language layers alone; it has no body hash, so its entry must
leave `body_sha256` `null`. The list is resolved before intake. No new download
for it starts once `max_known_public_bytes_per_round` is spent (separate from the
miners' `max_download_bytes_per_round`), after `max_known_public_seconds_per_round`
(default 7,200 s, at most 21,600), or within four hours of the round's deadline;
the remaining entries wait for a later round, and that round's copy and scout
checks use the entries resolved so far (each skipped entry is logged). The launch
byte budgets (32 GiB for lite, 80 GiB for mini) cover each packaged list: about
26 GB for lite and 73 GB for mini, which takes about 1.6 hours at 100 Mbit/s, so
the mini list resolves in the first round on a link of roughly that speed or
faster. A pool's first round also downloads its house bodies before the list, and
on a slower link the rest of the list resolves in later rounds.
An entry that fails for its contents (a malformed file, a wrong pinned hash) is
skipped until its pin changes; a transport error is retried next round. Neither
stops the round; both are recorded in the private round record. A list file that
cannot be read runs the round with no list and is reported in the epoch summary.
Public models are never graded, seated or paid themselves.

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
the body (`zilsb2:…` or `zilsb3:<pool>:…`) while the repo is still private, then make
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
hotkey. A hotkey's one commitment slot is shared with soup bodies of every pool
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
