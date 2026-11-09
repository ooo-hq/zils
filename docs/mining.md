# Mining on Zils

Zils runs on Bittensor mainnet subnet 127 (network `finney`). Registering a
hotkey on 127 is the admission: there is no application and no operator
approval ([registration](bittensor-registration.md)). Read the
[miner terms](mining-terms.md) first; mining under a registered hotkey means
you accept them.

Under the launch configuration (`zils/data/base-pools-launch.json`) miners earn
in two ways:

- **Soup pools, 80% of emissions.** Upload a fine-tune (a *body*) to Hugging
  Face and commit it on chain. Validators average the best bodies into each
  pool's soup and pay its seats. See [soup pool](#soup-pool) below.
- **Customer jobs, 20% of emissions (the adapter track).** Train customer
  adapters claimed from the hosted queue. You are paid from paid API use of the
  open-tier adapters you delivered. See
  [customer-job miners](queue-miners.md).

Serving z3 checkpoints to validators is described in
[serving miners](serving-miners.md); the launch configuration gives it no
share. The [closed development fleet](#closed-development-fleet) at the end of
this page is a local test loop, not a way to earn.

## Install

Start with the installer on the mining host. It clones Zils into `~/.zils/zils`,
builds the miner runtime and, on Linux x86_64 with an NVIDIA GPU, the d3 training
runtime, links `zils-miner` and `btcli` into `~/.local/bin` and starts
`zils-miner setup`, which asks for
your wallet, base profile, candidate folder and Hugging Face repo, saves them to
`~/.zils/miner.json` and runs the doctor. It installs the stable release, the
commit named in [`release/stable`](../release/stable); `zils-miner update` moves an
install to a newer one, and the doctor warns when yours is behind
([releases](releases.md)).

```bash
curl -fsSL https://zils.ai/install.sh | sh
```

The miner command is `zils-miner` (`setup`, `doctor`, `update`, `commit-body`,
plus `commit-candidate`, `announce-serving` and `serve`). The `zils` command
belongs to the customer CLI (`npm install -g @zils/cli`: sign-in, decisions,
training); it also runs miner commands, so `zils doctor` and `zils-miner doctor`
do the same thing. Installs from before `zils-miner` linked the miner as `zils`;
the installer (and `zils-miner update`) removes that old link.

`zils-miner` never assumes mainnet. Run
`zils-miner setup --network finney --netuid 127` once: it saves the chain you
named, and every later chain command (`doctor`, `commit-body`,
`commit-candidate`) uses it unless you pass `--network` again. From then on
`zils-miner doctor` checks the requirements, the d3 training runtime and GPU, the
hotkey file and its registration, and prints the command that fixes the first
failing step. Its later candidate and commitment steps are for
[king-pool candidates](#king-pools); check a soup body with
`zils-miner commit-body --check` instead.

## Soup pool

Every pool in the launch configuration is a soup pool: its validator config sets
`"type": "soup"` ([model pools](model-pools.md#soup-pools)). The design rests on
one internal experiment on public open models, in which plain uniform averages of
two or three public Qwen3.5-4B decision fine-tunes scored above stock z3-mini on
two sealed exam draws, and adding a fifth model lowered the score. Treat the
numbers in this section as launch policy that may change, not as measured
results.

Each round builds the best **soup**: a uniform average of
up to five bodies submitted by different miners. The average covers the language
weights; the soup keeps the release's vision tower unchanged, since no exam grades
images yet. A body still includes a vision tower (any one; usually its parent's). You earn by holding a seat in
the round's soup, or by submitting the round's best single body.

### Four pools

There is one soup pool per model size, each with its own bodies, rounds, seats
and share of the base emission. A body competes only in the pool it is committed
to.

| Pool | Chassis (pinned) | Body parent (pinned revision) | Chassis tensors | Body size cap | Input cap | Commitment |
|---|---|---|---|---|---|---|
| `d3-lite` (z3-lite, 0.8B) | `vllm-sr/d3-lite` | `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17` | 473 | 2 GiB | 4,096 tokens | `zilsb3:d3-lite:…` |
| `d3-mini` (z3-mini, 4.5B) | `zils-ai/d3-mini-4.5b-v1` | `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 723 | 10 GiB | 8,192 tokens | `zilsb2:…` |
| `d3-flash` (z3-flash, 8.4B) | `vllm-sr/d3-flash` | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 759 | 24 GiB | 4,096 tokens | `zilsb3:d3-flash:…` |
| `d3` (z3, 26B) | `vllm-sr/d3-ultra` | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 1,183 | 64 GiB | 4,096 tokens | `zilsb3:d3:…` |

The chassis revisions are in `zils/d3-lite-pins.json`, `zils/d3-pins.json`,
`zils/d3-flash-pins.json` and `zils/d3-ultra-pins.json`. The z3 chassis uses
`prefix_bidir` attention and the `header-v1` prompt layout; both are fixed by the
chassis, so a z3 body is an ordinary Qwen3.8-27B-layout fine-tune. Round items
longer than a pool's input cap are left out of that pool's rounds.

Lite and mini run rounds from the start. Flash and z3 open later, each at a
block fixed in the validator configuration. A pool pays nothing before it opens
(its share burns); see [the emission schedule](model-pools.md#emission-schedule). A pool takes no
submissions until then: validators ignore a body committed before the pool's
opening block, even after it opens. Commit flash and z3 bodies only once the
pool is open.

### What a body is

A body is the full set of language and vision weights of a fine-tune of the
pool's body parent at its pinned revision (table above). Every soup runs on the
pool's pinned chassis: its answer layer, prompt and runtime are fixed and are not
part of a submission. A body is eligible when:

- **Layout:** its safetensors contain every chassis tensor under
  `language_model.*` and `visual.*` (the pool's tensor count) with the chassis shape, stored
  as `bfloat16` or `float32` (Qwen's own format keeps 48 `linear_attn` tensors in
  `float32`). Every tensor is rounded to `bfloat16` for the body hash, the change
  check, the copy checks and the soup, so a tensor saved as `float32` hashes like
  its `bfloat16` cast. One leading `model.` is stripped from names; `lm_head*` and `mtp.*`
  are ignored; any other tensor rejects the body. Sharded (`model.safetensors.index.json`)
  and single-file checkpoints are both accepted, and the body hash does not depend
  on the shard layout.
- **Change:** the relative change of its language layers from the pool's Qwen
  parent, ‖body − Qwen‖ / ‖Qwen‖, is at most **0.15**. You may start from plain
  Qwen, from the pool's chassis, from a release or from any open fine-tune of
  that Qwen revision.
- **Licence:** the repo's `README.md` front matter declares `license:` as
  `apache-2.0`, `mit`, `bsd-3-clause` or `bsd-2-clause`.
- **Size:** the safetensors total at most the pool's body size cap.
- **Ownership:** the repo holds `zils-owner.json` containing exactly
  `{"hotkey": "<your ss58 hotkey>"}`, the hotkey that commits it. Scouts who find a
  good open model re-upload it to their own repo with their own ownership file.
  The first valid commitment of a body hash wins.

### Commit a body

A body must be the **full** Qwen3.5 model: the language weights under
`language_model.` and the vision tower under `visual.` (an optional leading
`model.` is fine). Many public fine-tunes, including most listed cousins, are
text-only exports (`model.layers.*`, `model.embed_tokens.*`): to submit one,
add the vision tower of the pool's Qwen parent, unchanged. Its language layers
still match the listed model, so a re-upload pays the scout rate (see
[Model pools](model-pools.md)). `zils merge` keeps the first model's layout, so a
merge that starts from a full checkpoint (the pool's release or its Qwen parent)
already has a vision tower; a mix of listed models also pays the scout rate.
Check any folder first, with no wallet or repo needed:

```bash
zils-miner commit-body --pool <pool> --path ./my-body --check
```

Register a hotkey ([registration](bittensor-registration.md)) and
[install](#install) the miner. Then:

1. Put the body's safetensors and `README.md` (with the licence) in one folder and
   let the CLI write the ownership file for your hotkey. It also checks the body
   locally and prints its hash. The body hash does not cover the ownership file,
   so anyone who copies a public body could commit its hash first: upload to a
   **private** repo, commit while it is private, and make it public before the
   round's commit deadline (validators download after the deadline, and a repo
   that is still private then fails):

   ```bash
   zils-miner commit-body --pool <pool> --repo <you>/<body-repo> --path ./my-body --write-owner
   ```

   `--pool` is `d3-lite`, `d3-mini` (the default), `d3-flash` or `d3` (z3). A commitment to a
   pool made before its opening block is ignored; see [four pools](#four-pools).

2. Upload the folder to your Hugging Face model repo and note the commit sha the
   upload prints (`hf auth login` once with a write token):

   ```bash
   hf upload <you>/<body-repo> ./my-body .
   ```

3. Check and commit. Without `--yes` the command is a dry run: it validates the
   body at that revision against the pool's chassis and parent, prints the body
   hash and the exact commitment (`zilsb2:<body sha256>:<revision>:<repo>` for
   d3-mini, `zilsb3:<pool>:<body sha256>:<revision>:<repo>` for the other pools),
   and writes nothing. With
   `--yes` it also confirms the public repo matches your local folder, writes the
   commitment with your hotkey and reads it back:

   ```bash
   zils-miner commit-body --pool <pool> --repo <you>/<body-repo> --revision <upload commit> --path ./my-body
   zils-miner commit-body --pool <pool> --repo <you>/<body-repo> --revision <upload commit> --path ./my-body --yes
   ```

   Without `--path` the command downloads the body at the revision instead. It
   also downloads the pool's pinned chassis and Qwen parent (about 2 GB each for
   lite, 9 GB for mini, 17 to 19 GB for flash and 52 to 56 GB for z3) unless you
   pass local copies with `--chassis` and `--base`. Wallet flags,
   `--network` and `--netuid` work as for the other `zils-miner` commands.

A hotkey has **one commitment slot, shared by `zils1` base candidates and soup
bodies of every pool**: committing a body replaces the hotkey's base candidate or
its body in another pool, and the reverse. Use a separate hotkey for each pool. Validators key seats by hotkey and body
hash, so a body already in the pool keeps its seat until it leaves the pool even
after you commit a new one.

### A round

Validators run one soup round per day for each open soup pool:

1. **Seed.** The hash of the first block after the commit deadline picks the
   round's private exam subset and breaks ties.
2. **Admit** new bodies with the checks above, in commitment order: at most 6
   new bodies and 12 claims per round in lite and mini (flash 4 and 6, z3 2 and
   3, in the launch configuration). A hotkey whose claim fails a check or a
   download sits out the next 3 rounds.
3. **Reject copies:** a re-upload, whose change from Qwen is nearly identical
   (cosine ≥ 0.999) to an earlier commitment's, a house ingredient's, the current
   soup's or the release's; a body built largely from other miners' bodies (other
   hotkeys' bodies explain at least 30% of the part of its change that the house
   ingredients, the current soup and the release leave unexplained; two
   fine-tunes of the same house ingredient share that change and are not copies of
   each other); and a behavioural twin of a pool body (near-identical answers,
   temperature-scaled copies included). **You may fine-tune the current soup or
   the release**, as well as a house ingredient: the change you add is measured
   from them, so a fine-tune is not refused because the soup contains other
   miners' bodies, and it is paid as a trainer body. Re-weighting the seated
   bodies, or re-uploading one with small changes, is refused.
   Only a near-exact re-upload or a broken body (shape, non-finite weights, change
   bound) bans its hash. A body built from other miners' bodies, or a behavioural
   twin, refuses only that commitment: it is evaluated again when you commit a new
   revision or when the pool body it matched leaves the pool. Other failures also
   apply to that commitment only.
4. **Pool:** new bodies are screened on a few hundred items; the pool is the
   current members, the house ingredients, the best screened new bodies and the
   best past single bodies, at most 20 (10 in flash, 5 in z3).
5. **Solo:** each pool body is graded alone on the chassis. Every score from here
   on (solo, greedy, contribution, crown) is on the items of this round and the
   validator's last two finished rounds pooled together (`decision_rounds`, default
   3), because one subset's noise is larger than the greedy margin. A new body is
   graded on the earlier rounds' items too. Under the default `calibrated_brier`
   scoring, each body and soup first gets one temperature fitted on a private 20%
   of those items; the rest are scored on Brier skill (ranked probability score
   for Score questions) at that temperature. Confidence counts, but sharpening or
   softening a body's probabilities without changing its answers earns nothing
   ([model pools: scoring](model-pools.md#soup-pools)).
6. **Greedy soup:** starting from the best single body, add the next best only if
   the soup's score rises by at least **0.5** points; stop at **5** members. The
   current soup stays unless the new one beats it by at least **0.5** points too,
   so a light re-tune of the soup cannot take its members' seats for a gain
   within noise. A current soup with a member that was not graded this round is
   not kept.
7. **Contribution:** each member's leave-one-out effect on the soup's score.

### Pay

Each pool's emission share pays **five seats of 16% each** (80% in all) and
**20% to the best single body** (the crown). A seat's value is fixed: it does not
grow when fewer members are seated. Seats are re-earned every round; a displaced
member stops earning at once.

- **Seats:** each member whose contribution is at least **0.8** points earns one
  seat, 16% of the pool's share multiplied by its rate. Empty seats, house seats
  and seats below the floor burn.
- **Fine-tunes (launch rule):** if at least half of your body's change comes from
  house ingredients, the soup or the release, your seat is paid only if your
  body alone beats the best of those bases by at least **1.2** points. If part
  of it (10% or more) comes from a listed public model, it is paid at the scout
  rate. Train enough to beat what you started from.
- **Crown:** the round's best single body that is not a house ingredient takes the
  20%, multiplied by its rate, only if it beats every house ingredient alone and
  the current soup by at least **1.2** points on the pooled items; otherwise the
  crown burns. Since a soup usually beats its members, expect the crown to burn in
  most rounds; it pays for a body that is better than the soup by itself.
- **Rates:** **1.0** for a trainer's body; **0.25** for a scout's body, an
  unchanged or near re-upload (sketch cosine at least 0.9) of a public model on
  the pool's known-public list, and for a house mix, a body whose change from Qwen is almost
  entirely a mix of house ingredients' and listed models' changes (less than 5% of
  it is left over; real fine-tunes of a house ingredient leave about 10%), or, when
  it was admitted, almost entirely a mix of house ingredients, listed models, the
  soup and the release (a near-pure re-weighting of the soup); **0** for house
  ingredients, the open Qwen3.5-4B decision models already public at launch, which
  the pool includes unpaid.

Each pool has its own house ingredients, the open decision models of its size
already public at launch, listed in `zils/data/`: `soup-house.json` (mini),
`soup-house-lite.json` (`vllm-sr/d3-lite`, `internlm/Intern-Decision-0.8B`,
`Maincode/Matilda-Decision-v1-Edge` and `zils-ai/jiwo-0.8b-full`),
`soup-house-flash.json` (`vllm-sr/d3-flash`) and `soup-house-z3.json`
(`vllm-sr/d3-ultra`).

The known-public lists name other public decision models of a pool's size:
`soup-known-lite.json` and `soup-known-mini.json`, each entry a repository at a
pinned revision. Finding one of them and committing it, unchanged or lightly
changed, is a scout's work and earns 0.25. Full pay is for real training: a
fine-tune that moves well away from every listed model (sketch cosine under 0.9)
earns the trainer rate, whichever model it started from. A listed model is an
allowed base: your fine-tune of it is not a copy of someone else's re-upload of
it. The first near re-upload of each listed model wins; another hotkey's later
near re-upload of the same model is refused while the first stays in the pool,
and a near-copy of any other hotkey's earlier body is refused as before.

**Scouting.** The known-public list is not the limit of what a scout can
commit. Any open fine-tune of the pool's Qwen parent that passes the eligibility
checks above can be committed by the first hotkey to find it, listed or not. The
list only names the public models validators already know about, and those pay
0.25. A find that is not on the list, and whose weight change is far from every
listed model (sketch cosine under 0.9), pays the trainer rate even though you did
not train it: the pool cannot tell a find from a fine-tune and does not need to,
because every body earns its seat the same way, by making the soup better on the
round's items. If the find is itself a fine-tune of a house ingredient, the soup
or the release, its seat pays only if it beats that base by 1.2 points, like any
other fine-tune. Hunting Hugging Face for strong open decision models of the
pool's size, adding the Qwen vision tower when the export is text-only, and
committing them first is real work the pool pays for. Check a candidate before
you upload it:

```bash
zils-miner commit-body --pool <pool> --path ./found-body --check
```

Every unearned part burns to the subnet owner's uid, as in every track. Until
miners submit new bodies, most of the pool's share burns.

**Pool shares.** The launch configuration (`zils/data/base-pools-launch.json`)
gives the base pools 80% of all emissions and the adapter track the other 20%.
Lite and mini get 25% of all emissions each, so a trainer's seat there is 4% and
a crown 5%; flash and z3 get 15% each (a seat 2.4%, a crown 3%). A pool has five
seats and one crown, so the four pools have 24 paid spots. Flash and z3 burn
their shares until their opening blocks, which are in each validator's
configuration and its published report.

### What validators publish

Each round's public record lists the soup members (hotkey, body hash, rate), their
rounded contributions, per-category soup scores, the pool's scoring and the
soup's fitted temperature, the rounds whose items the decisions were scored on
and their item count, and sha256 hashes of the item
subset, the per-item predictions and the soup weights, so a round can be audited
against the validator's private record. Exam items, labels and per-family scores
are never published. The round bank is private, so outside re-grading of a round
is not possible yet.

### Releases

A soup is not served until it is released. Every 14 to 28 days, at a release
window, the current soup must beat the current release by at least +1 point with a
positive lower bound on each of two sealed release draws that rounds never use,
pass the 12-hour confirmation draw and the pinned three-seed adapter suite (the
[release gate](base-election.md)); the owner's promotion PR then makes it the next
z3 version. Members are paid through seats, not as kings.

The member cap (5), pool sizes, change bound (0.15), seat/crown split (80/20), rates
(1.0 / 0.25 / 0) and release cadence are policy choices made for the launch, not
tuned results. The greedy margin (0.5), seat floor (0.8) and the crown and
fine-tune bars (1.2) are about 2.5 standard errors of the differences they test,
measured by resampling the items of a 3,000-item dry run and scaled to the pooled
three-round window; a smaller real gain is not paid until the pool has more items.

## Customer jobs

The adapter track pays for customer adapters trained from the hosted queue at
`https://training.zils.ai`. The path is: register a hotkey on 127, install the
current stable release, accept the miner terms, qualify the host, start
`python -m miner.queue`, pass the qualification jobs, then take customer jobs.
It needs Linux with an NVIDIA GPU and 12 GB of free GPU memory. The full setup
is in [customer-job miners](queue-miners.md); payment is in
[validators §7](validators.md#7-pay-the-adapter-track-from-paid-adapter-use).

## How a validator splits the weight vector

Every track pays from one weight vector. With `base_pools` in its config, the
validator gives the base pools `base_pools.share` of the vector (**0.8** in the
launch configuration; the validator records the share in every chain attempt and
receipt) and the adapter track the rest. Each track is normalized over its own
earners first, so a miner with no seat still earns its adapter share and a miner
that trains no customer adapter still earns its pool share. A track nobody
earned on is not handed to another track: its share goes to
`base_pools.burn_uid` (default 0), as Teutonic burns to the subnet owner's uid.
If nobody earned on any track, the whole vector goes to the burn uid, so the
validator keeps setting weights. Subtensor burns incentive paid to the subnet
owner's hotkey and keeps the owner's own self-weight, so the validator refuses to
publish a burn slice unless `burn_uid` is the subnet owner hotkey's uid at that
block.

On the adapter track ([validators §7](validators.md#7-pay-the-adapter-track-from-paid-adapter-use),
`adapter_evidence`), you earn when an open-tier customer adapter you trained is
delivered and customers pay to use it through the API: your share of that paid
use over each 72-hour window, for 30 days after delivery, capped by the
validator's `revenue_cap` (0.5: miners as a group earn at most half of that
spend; the rest burns). Free, test-mode and operator use earn nothing, and your
queue hotkey must be registered on the subnet.

With a `serving_share` section, serving is a third track: miners that serve
z3 checkpoints and pass the validator's [serving grader](serving-validators.md)
earn `serving_share.share` of the vector, and the other tracks shrink to make
room. The serving share burns, like any empty track, until `switch_on_miners`
different miners earn it, and whenever the grader's scores are missing or older
than `max_age_seconds`. With `receipts` configured, 30% of the serving share
(`volume_share`) pays by the input tokens a miner answered through the gateway,
and the rest by grading score. A miner the gateway catches failing or
mismatching a check in the window is struck: it earns nothing from either part.
The launch configuration has no `serving_share`.

## King pools

A validator can also run a pool as a **king pool** instead of a soup. The launch
configuration has none: every pool in it is a soup pool, and a soup pool refuses
`zils1` candidates. Because a hotkey has one commitment slot, committing a
`zils1` candidate also replaces the soup body that hotkey committed. Commit one
only to a validator whose config runs a king pool for your profile.

In a king pool, a miner beats the pinned incumbent in its
[model tier](model-pools.md) with a full-weight candidate under the
[base candidate contract](base-candidates.md). The scorer follows the
[decision-quality election](base-election.md): at least +1 point, a positive
paired lower 95% bound and Brier regression at most 0.005, followed by
candidate-specific confirmation, the three-seed adapter suite and a common
release-window closing exam. Winning the exam makes you a challenger and pays
nothing. You are crowned only after you also pass the confirmation draw and the
adapter regression suite and a promotion makes your candidate the incumbent and
your hotkey the newest entry in the track's king history. The pool's share is
split equally over the last five distinct kings still registered, so a dethroned
king keeps earning until five newer kings have reigned; deregister and your slice
goes to the others. The vendor base has no owner, so until the first promotion
the share is burned; a resubmitted stock base or a copy of the king is never
eligible.

### Submit a king-pool candidate

You do not need a bundle, an operator, an invitation or any validator's address:
candidates are chain commitments that every validator reads itself. Validators
are on the metagraph (`btcli subnets metagraph 127 --network finney`).

`zils-miner setup` lists the installed d3 profiles: lite, nano, mini, flash and
d3, including registered promoted profiles. Choose the exact incumbent profile
accepted by your validator. A listed profile does not mean that its competition
is enabled or funded; see [model pools](model-pools.md). `--model` overrides the
saved profile for one run:

```bash
zils-miner setup --model d3-lite-0.8b-v1
zils-miner doctor --model d3-lite-0.8b-v1
```

The doctor prints a reference download command with that exact `--model` and
separate working folders per profile. Training and merging follow the reference's
metadata. A candidate for a different tier or parent profile is rejected before
upload.

1. **Register a hotkey on mainnet subnet 127** following the
   [registration guide](bittensor-registration.md); keep the hotkey file on the
   mining host and the coldkey elsewhere.
2. **Train and merge a candidate** with the
   [reference recipe](base-candidates.md#reference-recipe-train-a-lora-merge-it):
   `zils.d3 reference`, `zils.d3 train` on data of your choosing, `zils.d3 candidate`
   to merge the adapter into the parent's shards. On 20–24 GB cards a d3-mini LoRA
   trains in about 10.5 GB of GPU memory; the merge needs about 10 GB of RAM and
   10 GB of disk, and the upload is at most 9.5 GiB.
3. **Upload and commit.** Log in to Hugging Face once (`hf auth login`, a write
   token), then let `zils-miner commit-candidate` upload the candidate files to
   your public model repo and commit `zils1:<hash>:<revision>:<repo>` on chain
   with your hotkey (details in [base candidates: commit](base-candidates.md#commit)).
   Without `--yes` it is a dry run that prints the commitment; with `--yes` it
   uploads, checks the public repo at the new commit, writes the commitment and
   reads it back:

   ```bash
   zils-miner commit-candidate
   zils-miner commit-candidate --yes
   ```

   It reads the wallet, base profile, candidate and repo saved by `zils-miner setup`;
   flags such as `--model`, `--candidate` or `--repo` override them for one run.
4. **Let the doctor check** (`zils-miner doctor`): it runs through the candidate,
   the Hugging Face login and repo at the revision, and the commitment on chain.
5. **Read the result.** Your payout is on chain: the validators' weights and your
   incentive (`btcli subnets metagraph 127 --network finney`). The aggregate row
   per round is in each validator's signed round result on its own API
   (`GET /results/<round>`), not on chain. Per-case results are never returned.
   Leave an unchanged commitment alone: it stays in every round at no cost, and
   committing it again moves your block later, which can hand the hash to a copier
   who committed it in between.

Anti-gaming: one base candidate commitment per hotkey on chain; a candidate hash
another hotkey committed at an earlier block is refused; every sealed split is
audited against its manifest before scoring, and payment goes only to registered
UIDs. You receive aggregate metrics, never per-case outcomes.

## Closed development fleet

A closed fleet is a validator and a few miners on a private network, all
started from bundles that one operator generates. It runs frozen customer-job
rounds on **d3-mini 4.5B** text jobs (`d3-mini-4.5b-v1`); each round's calibrated
test-split skill is the miner's score. It is for development and rehearsals: a
local fleet never writes weights, and a [chain fleet](testnet.md) publishes only
for its explicit roster. To earn on subnet 127, use the soup pools or customer
jobs above.

After the one-time setup below, run this in your assigned miner folder:

```bash
./start-miner
```

The miner trains one candidate per round, submits its signed checkpoint, receives
the validator's result, and waits for the next round. Stop with Ctrl+C, or use
`./start-miner --rounds 1` for a single result. The first start downloads the
pinned base model. Later starts reuse the cache.

### Prepare bundles on the validator

For a text fleet, create the d3 reference with the
[customer-job miner setup](queue-miners.md#2-install-the-stable-release).
Prepare authorized, independent training/calibration/test JSONL files using the
[customer data format](customer-jobs.md#prepare-data).
The following freezes those files and an example acceptance policy; choose your
policy before evaluating candidates:

```bash
.venv-kev/bin/python -m zils.jobs --job-id text-v1 \
  --model d3-mini-4.5b-v1 --train .private/train.jsonl \
  --calibration .private/calibration.jsonl --test .private/test.jsonl \
  --min-accuracy 0.80 --min-brier-improvement 0.01 \
  --allow-training-data-export --out .private/jobs/text-v1
.venv-kev/bin/python -m zils.fleet init --out .private/fleet-local \
  --benchmark .private/jobs/text-v1 --checkpoint models/d3-reference \
  --host VALIDATOR_PRIVATE_IPV4
```

Replace `VALIDATOR_PRIVATE_IPV4` with the validator's numeric private address
(RFC 1918 or a Tailscale `100.64.0.0/10` address; public addresses are refused).
Use new output paths or reuse an existing audited benchmark. The default ports
are 8900 for the validator and 8901–8903 for three miners. Assign one
`miner-N.tar.gz` to each machine.

Only training data goes into a miner bundle. The validator keeps calibration
cases, test cases, and full reports. Local bundles contain disposable signing
keys and must stay private. [Chain fleet bundles](testnet.md) contain public wallet
references instead; provision each hotkey separately.

Each model uses its own fleet directory and ports. Never combine text and image
scores into one reward vector. Only one validator may publish for a hotkey/subnet;
do not run competing vertical publishers that overwrite each other's weights.

For a legacy Kev rehearsal, keep `zils.benchmark build` and `models/reference`.
Do not relabel a Kev manifest as a different model.

### Set up each machine once

1. Unpack its assigned archive and enter the `miner-N` folder.
2. Install Git and Python 3.13, then install the pinned dependencies:

   ```bash
   python3.13 -m venv .venv-kev
   .venv-kev/bin/python -m pip install -r requirements/model.txt -r requirements/rehearsal.txt
   ```

   With `uv`, use `uv venv --python 3.13 .venv-kev` and
   `uv pip install --python .venv-kev/bin/python -r requirements/model.txt -r requirements/rehearsal.txt`.

3. Allow the assigned miner TCP port from the validator's private address.
   Allow validator TCP 8900 from the miner machines.
4. Run `./start-miner --rounds 1`. It can start before the validator and will retry.

d3 needs a qualified CUDA host and the isolated runtime below on every miner
and the validator. CPU and Apple MPS fallback apply only to Kev. Use `--device cuda`
explicitly. To reuse an environment or cache, set `ZILS_PYTHON`
to its absolute Python path and `HF_HOME` to the cache directory.

For d3 bundles, install the isolated runtime on each host and export it before
every start (the validator needs the same variable):

```bash
uv venv --python 3.12 .venv-d3
uv pip install --python .venv-d3/bin/python --torch-backend=cu130 -r requirements/d3.txt
export ZILS_D3_RUNTIME_PYTHON="$PWD/.venv-d3/bin/python"
./start-miner --device cuda --rounds 1
```

d3 training waits until at least 12,288 MiB of GPU memory is free
(`ZILS_D3_MIN_FREE_MIB` raises that floor). Neither side silently substitutes
another runtime. The pins are `zils/d3-pins.json`.

#### Windows / NVIDIA GPU

Run the miner inside WSL 2, using the supported NVIDIA driver installed on
Windows. Do not install a Linux NVIDIA display driver inside WSL. Check CUDA
from the miner environment:

```bash
.venv-kev/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA is not ready"; print(torch.cuda.get_device_name(0))'
```

The validator must reach the miner directly at its private IPv4 address.
Default WSL NAT does not support this callback layout without additional
network setup. Use [WSL mirrored networking](https://learn.microsoft.com/en-us/windows/wsl/networking#mirrored-mode-networking)
and limit the Windows/Hyper-V firewall rule to the assigned port and validator.
See [NVIDIA's WSL setup](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
Native Windows service execution and NAT traversal are not implemented.

### Start the validator

From the repository root:

```bash
.venv-kev/bin/python -m zils.fleet validator \
  --config .private/fleet-local/validator/config.json --rounds 1
```

Remove `--rounds 1` for continuous rounds. The validator evaluates after every
miner submits or the collection deadline expires. It verifies signatures and
hashes, calibrates each candidate, and scores the separate evaluation cases.
Local mode never writes weights to Bittensor. Chain mode needs registered
wallets and explicit publication; follow the [chain fleet guide](testnet.md).

### Results and restarts

Miner logs, frozen candidates, and results are in `state/jobs/`. Full validator
reports are in `validator/state/rounds/`. Restarting a miner reuses completed
candidates; an interrupted training epoch starts again. The validator reloads
accepted submissions after restart. Service locks prevent duplicate launches,
and a per-device compute lock serializes work on shared hardware.

When a miner finishes its last round (`--rounds`), it deletes its bundle's
`miner-training.jsonl`, as the [miner terms](mining-terms.md) require; candidates,
results and logs stay. Starting that bundle again stops with a message to build a
new one with `zils.fleet init`. A miner run without `--rounds` keeps the rows until
it is stopped; delete them then.

Keep started bundles at their original location: saved checkpoint paths are
absolute. Archive completed rounds between runs if disk space fills; training
and evaluation stop when less than 2 GiB is free. Regenerate bundles to upgrade
code; existing bundles are standalone snapshots.

This is a closed development fleet with a fixed one-epoch recipe. Every round
starts from the same reference with different seeds; winners are not promoted
automatically. The reused synthetic benchmark measures development progress.
Signed HTTP authenticates messages but does not encrypt them; use a trusted
private LAN. Evaluation runs each submission in a network-less namespace where
the host allows it (`sandbox` in every report says which mode ran); that bounds
network and lifetime, not every hostile-checkpoint behaviour.
