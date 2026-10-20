# Model pools

Zils registers five d3 lineages: `d3-lite`, `d3-nano`, `d3-mini`, `d3-flash`
and `d3`. Each improves against its own incumbent, with its own sealed draws,
king history, allocation and exact model profiles. Registration does not make a
pool available for training or serving. New pools start disabled and unallocated;
qualification, channel configuration and healthy capacity are separate requirements.

## Stage an existing mini validator

From a checkout installed with the normal validator dependencies, write a **new**
configuration for review. `validator/config.json` must be your existing working
mini configuration with `base_tier`, `base_round_dir` and `base_reference`:

```sh
python -m scripts.migrate_model_pools --config validator/config.json \
  --out validator/pools-staged.json
```

This preserves mini's exact incumbent, king list, draw policy and the outer
base/adapter budget. Mini receives all of the existing base allocation. The four
other profiles have no king, no draws, zero allocation and disabled intake.
The command does not install configuration, restart a service or publish weights.
Old single-base configs and receipts remain readable. Never specify both
`base_tier` and `base_pools`.

`base_pools` contains `share` (the outer base budget), `burn_uid`, and `pools`.
Each pool entry contains its immutable `model`, `enabled`, `share` (fraction of
that base budget), `incumbent_sha256`, `kings`, `draws` and `policy`. Fractions must
sum to one. Disabled or unowned pools burn their fraction; another pool cannot
inherit it. Last-five-distinct registered kings split each earned allocation.
Burn publication checks that `burn_uid` is the subnet owner's uid.

`base_pool_rounds` maps each enabled pool to its `base_reference` and
`base_round_dir`. Set these to separately audited, pool-specific inputs before
enabling a new competition. Never reuse a disclosed test set as another pool's
sealed draw. Runtime state goes under `base-pools/<pool>/base-tier`; prior legacy
state is preserved. Each pending validator round freezes the pool config and
results. A changed config cannot replay that pending round.

## Submit, evaluate and promote

The existing `zils1` commitment format is unchanged: one active candidate per
hotkey. The validator reads at most 16 KiB of metadata at the committed revision,
resolves its exact parent, and refuses a stale or wrong-pool profile before weight
downloads. Full download still verifies the committed checkpoint hash, files and
tensor contract. Work uses the existing shared GPU lock and separate pool stores.
A pool failure is recorded in the local round's `pool-intake.json`; another pool's
result and king history remain intact. Exam challengers earn nothing until crowned.

The first draw records evaluation time. A second draw is not evaluated until at
least 12 hours later. The decision and cached predictions retain those timestamps
across restarts. Promotion requires the two recorded decisions, disjoint test
hashes and the same incumbent/profile/policy. Existing historical decisions are
not rewritten or assigned invented timestamps. New promotion attempts need timed
evidence. Minimum Brier gain remains 0.01 with positive paired 95% lower bound,
positive skill and no accuracy loss.

Run the existing adapter regression suite for the selected profile and candidate.
`promote_base.py` now requires `--first-decision` and `--regression-report` as well
as the confirming `--decision`. Its regression gate binds both checkpoint hashes
and the exact profile, recomputes the verdict, and requires no job worse by more
than 0.005 Brier and at least one improvement. It only proposes a release and king
update for the selected pool; applying that update remains a reviewed promotion.
See [base candidates](base-candidates.md) for the training and staging commands.

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
peak plus the larger of 1,024 MiB or 15% as headroom. A result measured below 4,096
tokens does not qualify the full 4,096-token band. This local fit check does not
replace coordinator-issued capacity/quality qualifications.

Set each runtime's measured free-memory gate separately: `ZILS_D3_LITE_MIN_FREE_MIB`,
`ZILS_D3_NANO_MIN_FREE_MIB`, `ZILS_D3_FLASH_MIN_FREE_MIB`, or
`ZILS_D3_FULL_MIN_FREE_MIB`. Mini retains `ZILS_D3_MIN_FREE_MIB`. New profiles fail
closed when neither a configured gate nor an explicit measured requirement is
present. Training certificates do not certify serving; admission remains bound to
an exact profile, recipe, context and role. No automatic substitution with a larger
or smaller model occurs when a pool has no qualified capacity.
