# Run a fleet miner

Local and Bittensor testnet fleets support **d3-mini 4.5B** text jobs
(`d3-mini-4.5b-v1`, the base since 17 October 2026). Existing Kev 0.8B bundles
remain supported; H2O Lightning 4B, ImaJev and JevK5 4B bundles are retired and no
longer run. d3 image jobs run in the [hosted queue](queue-miners.md), where a d3
miner downloads a job's frozen training photos itself. For workers that claim different customer jobs,
use the [queue miner guide](https://github.com/ooo-hq/zils/blob/main/docs/queue-miners.md).
Queue and fleet miners use the same trainers but different credentials and startup commands.
To serve public z3 checkpoints to validators, see [serving miners](serving-miners.md).

For the synthetic bounding-box adapter pilot, see the
[feature competition guide](feature-tracks.md). It covers public training data,
exact-parent adapters, feature commitments, and the planned 50/20/30 allocation;
each validator's signed round records which tracks and reward shares it enables.

After the one-time setup below, run this in your assigned miner folder:

```bash
./start-miner
```

The miner trains one candidate per round, submits its signed checkpoint, receives
the validator's result, and waits for the next round. Stop with Ctrl+C, or use
`./start-miner --rounds 1` for a single result. The first start downloads the
pinned base model. Later starts reuse the cache.

## Prepare bundles on the validator

For a text fleet, create the d3 reference with the
[queue miner setup](https://github.com/ooo-hq/zils/blob/main/docs/queue-miners.md#2-install-and-verify-the-reference)
Prepare authorized, independent training/calibration/test JSONL files using the
[customer data format](https://github.com/ooo-hq/zils/blob/main/docs/customer-jobs.md#prepare-data).
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
keys and must stay private. [Testnet bundles](testnet.md) contain public wallet
references instead; provision each hotkey separately.

Each model uses its own fleet directory and ports. Never combine text and image
scores into one reward vector. Only one validator may publish for a hotkey/subnet;
do not run competing vertical publishers that overwrite each other's weights.

For a legacy Kev rehearsal, keep `zils.benchmark build` and `models/reference`.
Do not relabel a Kev manifest as a different model.

## Set up each machine once

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

### Windows / NVIDIA GPU

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

## Start the validator

From the repository root:

```bash
.venv-kev/bin/python -m zils.fleet validator \
  --config .private/fleet-local/validator/config.json --rounds 1
```

Remove `--rounds 1` for continuous rounds. The validator evaluates after every
miner submits or the collection deadline expires. It verifies signatures and
hashes, calibrates each candidate, and scores the separate evaluation cases.
Local mode never writes weights to Bittensor. Testnet mode needs registered
wallets and explicit publication; follow the [testnet guide](testnet.md).

## Two tracks on the testnet: the base tier pays first

On a testnet fleet whose validator config carries `base_pools`, miners
earn on two tracks, paid from one weight vector on the configured subnet:

- **Base tier (primary, continuous).** Beat the pinned incumbent in your
  [model tier](model-pools.md). Submit a full-weight candidate under the
  [base candidate contract](base-candidates.md). The validator's base-tier
  scorer follows the [decision-quality election](base-election.md): at least +1 point,
  a positive paired lower 95% bound and Brier regression at most 0.005, followed
  by candidate-specific confirmation, the three-seed adapter suite and a common
  release-window closing exam. Only crowned kings are paid.
  Winning the exam makes you a challenger and pays nothing. You are crowned
  only after you also pass the confirmation draw and the adapter regression
  suite and the owner's promotion PR makes your candidate the incumbent and
  your hotkey the newest entry in the track's king history. The base share of
  every publication is split equally over the last five distinct kings still
  registered, so a dethroned king keeps earning until five newer kings have
  reigned; deregister and your slice goes to the others. The vendor base has no
  owner, so until the first promotion the base share is burned; a resubmitted
  stock base or a copy of the king is never eligible.
- **Adapter rounds (alongside).** The frozen customer job rounds above continue
  unchanged; each round's calibrated test-split skill is the adapter weight.
  A validator may instead take the adapter track from real customer work in the
  [hosted queue](queue-miners.md) (`adapter_evidence`, off by default; see
  [validators §7](validators.md#7-pay-the-adapter-track-from-hosted-queue-work)).
  Then you earn for customer jobs the queue routed to your hotkey and delivered
  to the customer, weighted by training tokens × held-out skill × your
  reliability over the window. Qualification runs and operator-owned jobs earn
  nothing, and your queue hotkey must be registered on the subnet.

Split: the base tier takes `base_pools.share` of the vector (for example **0.7**;
the validator records the share in every chain attempt and receipt), adapters
the rest. Each track is normalized over its own earners first, so a miner with
no eligible base candidate still earns its adapter share and a miner that trains
no customer adapter still earns its base share. A track nobody earned on is
not handed to the other track: its share goes to `base_pools.burn_uid` (default
0), as Teutonic burns to the subnet owner's uid. For example, with no king the
vector is 0.7 to the burn uid and 0.3 to adapter earners in their own
proportions. If nobody earned on either track, the whole vector goes to the
burn uid, so the validator keeps setting weights. Subtensor burns incentive
paid to the subnet owner's hotkey and keeps the owner's own self-weight, so
the validator refuses to publish a burn slice unless `burn_uid` is the subnet
owner hotkey's uid at that block.

With a `serving_share` section, serving is a third track: miners that serve
z3 checkpoints and pass the validator's [serving grader](serving-validators.md)
earn `serving_share.share` of the vector (for example **0.10**). To keep base
and adapters in their old proportion, operators lower `base_pools.share` from
0.7 to 0.63; adapters take what is left (0.27). A miner's serving weight is the
sum of its per-model serving scores. The serving share burns, like any empty
track, until `switch_on_miners` different miners earn it, and whenever the
grader's scores are missing or older than `max_age_seconds`. With `receipts`
configured, 30% of the serving share (`volume_share`) pays by the input tokens a
miner answered through the gateway, and the rest by grading score. A miner the
gateway catches failing or mismatching a check in the window is struck: it earns
nothing from either part.

Anti-gaming: one adapter submission per hotkey per round and one base candidate
commitment per hotkey on chain; a claim cannot change within a round; a candidate
hash another hotkey committed at an earlier block is refused; every
sealed split is audited against its manifest before scoring, and payment goes
only to registered roster UIDs. You receive aggregate metrics, never per-case
outcomes; a round's cases are published after it closes and are not reused.

For the mini tier specifically, the 20–24 GB cards in the measured fleet cannot
full-fine-tune the 4.5B backbone, but a d3 LoRA trains in about 10.5 GB of
GPU memory. Train a LoRA on d3 and merge it with `zils.d3 candidate`, which
writes the two bf16 shards in the parent's layout plus the untouched contract
files and validates the result (see
[the reference recipe](base-candidates.md#reference-recipe-train-a-lora-merge-it));
the merge needs about 10 GB of RAM and 10 GB of disk per candidate, and the
upload is at most 9.5 GiB. Customer jobs take priority on a shared card; base
training fills the idle time. What you train on decides everything: the 18
October rehearsal candidate, trained only on synthetic rule cases, tied stock d3
on the sealed exam.

## Compete in the base tier from outside

You do not need a bundle, an operator or an invitation to enter the base tier:
registration on subnet 579 is the admission. About an afternoon end to end, most
of it the training run.

You do not need any validator's address or hotkey to compete: candidates are
chain commitments that every validator reads itself. Validators are run by
whoever operates them; find them on the metagraph
(`btcli subnets metagraph --netuid 579 --network test`).

Start with the installer on the mining host. It clones Zils into `~/.zils/zils`,
builds the miner runtime and, on Linux x86_64 with an NVIDIA GPU, the d3 training
runtime, links `zils` into `~/.local/bin` and starts `zils setup`, which asks for
your wallet, base profile, candidate folder and Hugging Face repo, saves them to
`~/.zils/miner.json` and runs the doctor. It installs the stable release, the
commit named in [`release/stable`](../release/stable); `zils update` moves an
install to a newer one, and the doctor warns when yours is behind.

```bash
curl -fsSL https://zils.ai/install.sh | sh
```

Every tier has a base to beat. `zils setup` lists the installed d3 profiles:
lite, nano, mini, flash and d3, including registered promoted profiles. Choose
the exact incumbent profile accepted by your validator. A listed profile does
not mean that its competition is enabled or funded; see [model pools](model-pools.md).

`--model` overrides the saved profile for one run. For example, if your validator
accepts the stock lite profile:

```bash
zils setup --model d3-lite-0.8b-v1
zils doctor --model d3-lite-0.8b-v1
```

The doctor prints a reference download command with that exact `--model` and
separate working folders per profile. Training and merging follow the reference's
metadata. Existing candidates supply their profile when no selection is saved;
without a candidate or selection, the CLI asks you to choose instead of assuming
mini. A candidate for a different tier or parent profile is rejected before upload.

From then on `zils doctor` names the next step and prints its command; the steps are:

1. **Register a hotkey on testnet subnet 579** following the
   [registration guide](bittensor-registration.md); keep the hotkey file on the
   mining host and the coldkey elsewhere.
2. **Read the [miner terms](mining-terms.md).** Running a miner against a Zils
   validator means you accept them.
3. **Train and merge a candidate** with the
   [reference recipe](base-candidates.md#reference-recipe-train-a-lora-merge-it):
   `zils.d3 reference`, `zils.d3 train` on data of your choosing, `zils.d3 candidate`
   to merge the adapter into the parent's shards. The result is a directory that
   passes the candidate contract; the exam is realistic decisions in families you
   have not seen, so what you train on decides everything.
4. **Upload and commit.** Log in to Hugging Face once (`hf auth login`, a write
   token), then let `zils commit-candidate` upload the candidate files to
   your public model repo and commit `zils1:<hash>:<revision>:<repo>` on chain
   with your hotkey (details in [base candidates: commit](base-candidates.md#commit)):

   ```bash
   zils commit-candidate
   ```

   It reads the wallet, base profile, candidate and repo saved by `zils setup`;
   flags such as `--model`, `--candidate` or `--repo` override them for one run.
   Upload and remote verification use the selected candidate's pinned file list,
   including the correct number of weight shards for its tier.

   Without `--yes` it is a dry run that prints the commitment; with `--yes` it
   uploads, checks the public repo at the new commit, writes the commitment and
   reads it back. No validator URL, no open port, nothing left running: every
   validator reads the commitments at the start of its round and pulls the
   revision from Hugging Face, once per distinct hash.
5. **Let the doctor check.** It runs the checks in order (requirements, d3
   training runtime and GPU, hotkey file, registration, candidate, Hugging Face
   login and repo at the revision, commitment on chain) and stops at the first
   failing one with the exact command that fixes it:

   ```bash
   zils doctor
   ```

   You can run it before step 1 (it tells you to create the wallet), before
   step 3 (it prints the merge recipe), before step 4 (it prints the commit
   command) and again after each fix.
6. **Read the result.** Your payout is on chain: the validators' weights and your
   incentive (`btcli subnets metagraph --netuid 579 --network test`). The
   aggregate row per round (Brier and accuracy against the incumbent, eligibility,
   the pending and confirmed winner) is in each validator's signed round result on
   its own API (`GET /results/<round>`), not on chain. Per-case results are never returned. An ineligible or
   rejected candidate earns nothing; fix it, merge again and commit the new hash.
   Leave an unchanged commitment alone: it stays in every round at no cost, and
   committing it again moves your block later, which can hand the hash to a copier
   who committed it in between.

## Soup pool

The soup pool is **new at the mainnet launch** and replaces the king election in
a pool whose validator config sets `"type": "soup"` ([model pools](model-pools.md#soup-pools)).
Nothing below has run as a live competition yet: no miner-submitted body has been
graded or paid. The design rests on one internal experiment on public open models,
in which plain uniform averages of two or three public Qwen3.5-4B decision
fine-tunes scored above stock z3-mini on two sealed exam draws, and adding a
fifth model lowered the score. Treat the numbers in this section as launch policy
that may change, not as measured results.

Instead of one winner, each round builds the best **soup**: a uniform average of
up to five bodies submitted by different miners. You earn by holding a seat in
the round's soup, or by submitting the round's best single body.

### What a body is

A body is the full set of language and vision weights of a fine-tune of
`Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Every
soup runs on the pinned d3-mini chassis: its answer layer, prompt and runtime are
fixed and are not part of a submission. A body is eligible when:

- **Layout:** its safetensors contain every chassis tensor under
  `language_model.*` and `visual.*` (723 tensors) with the chassis shape, stored
  as `bfloat16` or `float32` (Qwen's own format keeps 48 `linear_attn` tensors in
  `float32`). Every tensor is rounded to `bfloat16` for the body hash, the change
  check, the copy checks and the soup, so a tensor saved as `float32` hashes like
  its `bfloat16` cast. One leading `model.` is stripped from names; `lm_head*` and `mtp.*`
  are ignored; any other tensor rejects the body. Sharded (`model.safetensors.index.json`)
  and single-file checkpoints are both accepted, and the body hash does not depend
  on the shard layout.
- **Change:** the relative change of its language layers from the Qwen parent,
  ‖body − Qwen‖ / ‖Qwen‖, is at most **0.15**. You may start from plain Qwen, from
  a z3 release or from any open fine-tune of that Qwen revision.
- **Licence:** the repo's `README.md` front matter declares `license:` as
  `apache-2.0`, `mit`, `bsd-3-clause` or `bsd-2-clause`.
- **Size:** the safetensors total at most 10 GiB.
- **Ownership:** the repo holds `zils-owner.json` containing exactly
  `{"hotkey": "<your ss58 hotkey>"}`, the hotkey that commits it. Scouts who find a
  good open model re-upload it to their own repo with their own ownership file.
  The first valid commitment of a body hash wins.

### Commit a body

Register a hotkey as in [the base tier](#compete-in-the-base-tier-from-outside) and
install with `curl -fsSL https://zils.ai/install.sh | sh`. Then:

1. Put the body's safetensors and `README.md` (with the licence) in one folder and
   let the CLI write the ownership file for your hotkey. It also checks the body
   locally and prints its hash. The body hash does not cover the ownership file,
   so anyone who copies a public body could commit its hash first: upload to a
   **private** repo, commit while it is private, and make it public before the
   round's commit deadline (validators download after the deadline, and a repo
   that is still private then fails):

   ```bash
   zils commit-body --repo <you>/<body-repo> --path ./my-body --write-owner
   ```

2. Upload the folder to your Hugging Face model repo and note the commit sha the
   upload prints (`hf auth login` once with a write token):

   ```bash
   hf upload <you>/<body-repo> ./my-body .
   ```

3. Check and commit. Without `--yes` the command is a dry run: it validates the
   body at that revision, prints the body hash and the exact
   `zilsb2:<body sha256>:<revision>:<repo>` commitment, and writes nothing. With
   `--yes` it also confirms the public repo matches your local folder, writes the
   commitment with your hotkey and reads it back:

   ```bash
   zils commit-body --repo <you>/<body-repo> --revision <upload commit> --path ./my-body
   zils commit-body --repo <you>/<body-repo> --revision <upload commit> --path ./my-body --yes
   ```

   Without `--path` the command downloads the body at the revision instead. It
   also downloads the pinned d3-mini chassis and the Qwen parent (about 9 GB each)
   unless you pass local copies with `--chassis` and `--base`. Wallet flags,
   `--network` and `--netuid` work as for `commit-candidate`.

A hotkey has **one commitment slot, shared by `zils1` base candidates and
`zilsb2` bodies**: committing a body replaces the hotkey's base candidate, and the
reverse. Use a separate hotkey for each. Validators key seats by hotkey and body
hash, so a body already in the pool keeps its seat until it leaves the pool even
after you commit a new one.

### A round

Validators run one soup round per day for each soup pool:

1. **Seed.** The hash of the first block after the commit deadline picks the
   round's private exam subset and breaks ties.
2. **Admit** new bodies with the checks above, at most 6 new bodies and 12
   claims per round in commitment order. A hotkey whose claim fails a check or a
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
   best past single bodies, at most 20.
5. **Solo:** each pool body is graded alone on the chassis. Every score from here
   on (solo, greedy, contribution, crown) is on the items of this round and the
   validator's last two finished rounds pooled together (`decision_rounds`, default
   3), because one subset's noise is larger than the 0.3-point margin. A new body is
   graded on the earlier rounds' items too.
6. **Greedy soup:** starting from the best single body, add the next best only if
   the soup's score rises by at least **0.3** points; stop at **5** members.
7. **Contribution:** each member's leave-one-out effect on the soup's score.

### Pay

The pool's emission share splits **80% to soup seats and 20% to the best single
body** (the crown). Seats are re-earned every round; a displaced member stops
earning at once.

- **Seats:** members whose contribution is at least **0.3** points share the 80%
  equally, each share multiplied by its rate. If no member clears the floor, the
  seat share burns.
- **Crown:** the round's best single body that is not a house ingredient takes the
  20%, multiplied by its rate, only if it beats every house ingredient alone and
  the current soup by at least **0.3** points on the pooled items; otherwise the
  crown burns. Since a soup usually beats its members, expect the crown to burn in
  most rounds; it pays for a body that is better than the soup by itself.
- **Rates:** **1.0** for a trainer's body; **0.25** for a scout's body, one within a
  similarity threshold of a listed open model released after launch (the list is
  empty at launch), and for a house mix, a body whose change from Qwen is almost
  entirely a mix of house ingredients' changes (less than 5% of it is left over;
  real fine-tunes of a house ingredient leave about 10%), or, when it was
  admitted, almost entirely a mix of house ingredients, the soup and the release
  (a near-pure re-weighting of the soup); **0** for house
  ingredients, the open Qwen3.5-4B decision models already public at launch, which
  the pool includes unpaid.

Every unearned part burns to the subnet owner's uid, as in the base tier. Until
miners submit new bodies, most of the pool's share burns.

### What validators publish

Each round's public record lists the soup members (hotkey, body hash, rate), their
rounded contributions, per-category soup scores, the rounds whose items the
decisions were scored on and their item count, and sha256 hashes of the item
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

The margin (0.3), floor (0.3), member cap (5), pool size (20), change bound (0.15),
80/20 split, rates (1.0 / 0.25 / 0) and release cadence are policy choices made
for the launch, not tuned results.

## Results and restarts

Miner logs, frozen candidates, and results are in `state/jobs/`. Full validator
reports are in `validator/state/rounds/`. Restarting a miner reuses completed
candidates; an interrupted training epoch starts again. The validator reloads
accepted submissions after restart. Service locks prevent duplicate launches,
and a per-device compute lock serializes work on shared hardware.

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
