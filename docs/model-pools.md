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

## Submit, evaluate and promote

Run `zils setup` to select the exact base profile to beat in your tier, or pass
`--model <profile-id>` to `setup`, `doctor` or `commit-candidate`. `doctor` prints
the matching reference/training recipe. The CLI reads older candidates' profiles
from their metadata and checks uploads against each profile's own file list.
It does not discover or enable live competitions; verify the accepted incumbent
and allocation with the validator before training.

The existing `zils1` commitment format is unchanged: one active candidate per
hotkey. The validator reads at most 16 KiB of metadata at the committed revision,
resolves its exact parent, and refuses a stale or wrong-pool profile before weight
downloads. Full download still verifies the committed checkpoint hash, files and
tensor contract. Work uses the existing shared GPU lock and separate pool stores.
A pool failure is recorded in the local round's `pool-intake.json`; another pool's
result and king history remain intact. Exam challengers earn nothing until crowned.

The [decision-quality election runbook](base-election.md) defines the fixed score,
independent exam qualification, candidate-specific 12-hour confirmation delay,
three-seed adapter gate and common closing draw. Every eligible candidate can
qualify. Each pool needs its own passing pilot receipt; a mini receipt cannot
enable lite, nano, flash or d3. A weight-only release preserves the execution
contract, while changed runtime, blueprint or score semantics require requalification.

Promotion uses `promote_base.py --config ... --pool ... --window-report ...`.
It rechecks the closed window against installed state and produces a reviewable
profile/king update. Stable/latest movement remains separate from election and
existing adapters' exact model pins.

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
