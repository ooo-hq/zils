# Base candidates

Each [model tier](model-pools.md) competes against its own pinned incumbent.
Use `zils setup` to choose that exact profile; `zils doctor` prints its reference
and training commands. The concrete shard layouts, sizes and hardware figures
below describe the mini recipe, not every tier. Other tiers use their own
`models.candidate_files(profile, "candidate")` manifest and pinned parent.

A base candidate is a full-weight replacement for the pinned text base,
`d3-mini-4.5b-v1` (the base on both channels since 17 October 2026, adopted by
owner decision after it tied the H2O incumbent on the confirmation draw; see
[d3-mini](roadmap.md#d3-mini)). H2O Lightning 4B v1.2.3 is retired: no longer
served, no new jobs; its candidate contract is kept below for the record.
Everything below describes d3 unless it says H2O. Miners may train a candidate
however they like; validators score it on a sealed holdout against the current
base and, for a pending winner, retrain a fixed suite of reference adapters on
both. The 17/18 October rehearsal exercised the earlier Brier election and
is historical evidence only. The current [decision-quality election](base-election.md)
requires independent exam qualification before a pool can accept candidates.
Candidates enter through a public Hugging Face revision and a chain commitment;
qualified validators discover those commitments and score the pinned artifacts.
The channel setting on the platform
is a configuration value (`ZILS_TRAINING_MODEL_LATEST`), not yet a registry
alias; automatic "follow stable" upgrades are not implemented.

## Promoted profiles

A confirmed d3 winner becomes a new profile, not a new revision of `d3-mini-4.5b-v1`:
The [verified-window promotion command](base-election.md#qualify-candidates-and-close-a-release-window) stages the candidate's
files with a Zils `NOTICE` and `README.md`, optionally uploads them to `zils-ai/d3-mini-4.5b-zils-rN`,
and prints the `SPECS` entry (`parent` names the profile it was trained from), the
pins file `d3-mini-4.5b-zils-rN-pins.json` (the parent's runtime pins over the
published files) and the registration the runtime needs: the new id joins
`TEXT_ADAPTER_MODELS`, `TEXT_RUNTIME_MODULES` (`.d3`), `RUNTIME_PYTHON_ENV` (the
same interpreter), `CANDIDATE_BASE_ENV` and `PINS_FILES`. Every d3-family profile
shares the vendor runtime code, which is loaded from the vendor release only; the
profile a checkpoint belongs to comes from its `model.json`, so a reference
written with `zils.d3 reference --model <id>` trains, evaluates and serves
adapters for that base with no new flags. Frozen jobs keep their profile; the
platform registers the row (`zils_model_profiles`) and the `latest` channel names
the new id. `zils.d3 candidate --adapter <dir> --out <dir>` is the reference merge
recipe: the parent's shards with `lora_alpha / r` scaled `B @ A` added to each of
the 152 language targets, contract files copied byte for byte.

## What stays fixed

Only the weights change. A d3 candidate must keep:

- the architecture, layer sizes and tokenizer of the pinned base, so the
  production LoRA recipe attaches to the same 152 language modules;
- the 19 files listed as `models.D3_CONTRACT_FILES` byte-identical to the parent
  (`config.json`, `decision_config.json`, the two preprocessor configs,
  `chat_template.jinja`, tokenizer files, `readout.safetensors`, the seven vendor
  runtime modules, `LICENSE`, `NOTICE`);
- the 4,096-token input limit, the FP32 code readout and the pinned runtime
  (`zils/d3-pins.json`).

Before loading a d3 candidate the validator checks that its index `weight_map`
names exactly the parent index's tensors, maps them only to the two shard file
names, and that each shard holds exactly the tensors the index assigns to it;
transformers then rejects any tensor whose shape differs from the parent. Vendor
code is always loaded from the vendor release, never from a candidate.

The validator refuses a candidate whose contract files differ from the parent
hashes, that contains any file outside the list below, or whose weights do not
load exactly.

## Artifact

A directory containing exactly these files (`models.D3_CANDIDATE_FILES`):

| File | Content |
| --- | --- |
| `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors` | all weights, bf16, in the parent's shard layout |
| `model.safetensors.index.json` | the parent's index, unchanged |
| `model.json` | `{"version": "zils-checkpoint/v1", "model": "d3-mini-4.5b-v1", "kind": "candidate", "temperature": 1.0, "profile": <the parent profile>}`; write it with `zils.models.write_metadata(path, kind="candidate", model=zils.models.D3)` |
| the 19 contract files | copied from the parent |

Nothing else: a `candidate.json`, a README or a stray `__pycache__` is rejected.
Total size must stay under 9.5 GiB (the parent is 9.08 GB; a merged LoRA is the
same size).

For unattended bootstrap attempts using the existing public-data recipe, see
[scheduled base mining](automatic-base-mining.md).

## Commit

A candidate is two things: the artifact in **your own public Hugging Face model
repo**, pinned by commit, and **one chain commitment** made with your registered
miner hotkey on subnet `<NETUID>` (the hotkey file only; the coldkey is never read):

```text
zils1:<checkpoint sha256>:<revision>:<owner/repo>
```

`<checkpoint sha256>` is the `zils.checkpoint_hash` of the directory (it covers
every file in the artifact), `<revision>` the full 40-hex commit sha of the upload
(never a branch or tag, so the bytes cannot change under the validators) and
`<owner/repo>` the model repo id. The string is plaintext, at most 305 bytes,
written as up to three 128-byte Raw fields of the Commitments pallet. Every
validator reads each registered hotkey's commitment at its round's preflight
block, freezes it in that round's chain snapshot and pulls the revision itself.
No validator URL, open port or long-running process is needed.

Install with `scripts/install.sh` ([miner setup](mining.md)); `zils setup` saves
the wallet, base profile, candidate and repo that the commands below default to.
`zils doctor` checks each prerequisite in order (requirements, d3 training
runtime, hotkey file, registration, candidate, Hugging Face login and repo at the
revision, the commitment on chain) and prints the first fix or, when all pass,
the block your commitment landed in. `zils commit-candidate` does the work:

```bash
hf auth login   # once, with a write token for your account
zils commit-candidate   # --model, --candidate and --repo override the saved values
```

From a checkout without the installer, `.venv-kev/bin/python -m zils.miner` takes
the same subcommands with `--wallet-name`, `--wallet-hotkey`, `--candidate` and
`--repo`.

It checks the directory with the validator's own `validate_candidate` (it needs
the pinned parent on disk, as the merge recipe does), hashes it, reads your uid
and current commitment from the chain and prints the commitment it would write.
That is a dry run; add `--yes` to upload only the candidate files to the repo
(created public if absent), check through the public API that the repo at the new
commit holds every file at the local size and LFS digest, write the commitment
and read it back. With an upload you made yourself, pass `--revision <commit sha>`
and the command skips the upload. It refuses an unregistered hotkey, a branch
name for a revision, a revision the repo does not have, and files that differ
from the local candidate; when the chain already holds exactly this commitment it
writes nothing, because committing again moves your block later.

What the validator does with it, every round: refuses a commitment that does not
parse, names the incumbent's hash, or carries a hash another hotkey committed at
an earlier block; downloads a new hash once from
`huggingface.co/<repo>/resolve/<revision>/<file>`, only the candidate files, under
the 9.5 GiB budget declared by the repo listing and enforced on every byte; checks
the checkpoint hash and the candidate contract; and scores it in the sandbox.
A hash it has already pulled costs no download and no rescoring, so leave an
unchanged commitment alone: it stays in every round's decision. Commit again only
when you have a new candidate (new hash). Keep the repo and revision public until
your candidate is decided; a validator that has not pulled it yet can only score
what it can download.

### Copies

A checkpoint hash belongs to the hotkey that committed it at the earliest block.
The same hash from another hotkey at a later block is refused, so replaying a
winner's commitment, or re-uploading its files under your own repo, earns
nothing. A hash two hotkeys committed in that same earliest block belongs to
nobody (no tie-break a front-runner could win); if that ever happens to you,
merge again (a new hash) and commit.

Commitments are plaintext, not timelocked (decided 18 October 2026). A sealed
commitment would hide the pointer only until it reveals, and the repo itself must
be public for every validator to pull, so a copier watching Hugging Face learns
nothing later than one watching the chain; sealing would also delay every
candidate by the reveal period and tie intake to the drand beacon. What timelock
does stop is a same-block replay of a commitment seen in the mempool, which today
earns the replayer nothing (the tie rule above); revisit if griefing that way
appears. Neither scheme stops a near-copy: the same weights re-saved so the hash
changes. Such a copy scores the same as the original, and `base_round.decide`
breaks exact score ties by earliest commitment block, then checkpoint hash.
Near-copy detection remains limited; a copy of the incumbent is never eligible.

### Results

The per-round `base_tier` row (Brier and accuracy against the incumbent,
eligibility, pending and confirmed winner) is published by each validator in its
signed round result over its own HTTP API (`GET /results/<round>`); the chain does
not carry it. What the chain does carry, readable without any URL, is the payout:
the validator's weights and your incentive, for example
`btcli subnets metagraph --netuid <NETUID> --network finney`.

### Reference recipe: train a LoRA, merge it

The fleet's 20–24 GB cards cannot full-fine-tune the 4.5B backbone, and every
full fine-tune Zils has tried lost to the LoRA. The published recipe is the
production adapter merged into the backbone:

```bash
export ZILS_D3_RUNTIME_PYTHON="$PWD/.venv-d3/bin/python"
.venv-d3/bin/python -m zils.d3 reference --model d3-mini-4.5b-v1 --out models/d3-reference
.venv-d3/bin/python -m zils.d3 train --data train.jsonl --reference models/d3-reference \
  --out candidate-adapter --seed 553 --device cuda
.venv-d3/bin/python -m zils.d3 candidate --adapter candidate-adapter --out candidate > provenance.json
```

`train.jsonl` is the customer training export format (one `{"state", "questions":
{"decision": {..., "label"}}}` per line); rank 16, lr 2e-5, one epoch, about
10.5 GB of GPU memory. `candidate` adds `lora_alpha / r` scaled `B @ A` to each
of the 152 language targets in FP32, stores BF16 in the parent's shards, copies
the index and contract files and runs the validator's own `validate_candidate`
before returning; it needs about 10 GB of RAM and 10 GB of disk. Keep
`provenance.json` yourself; it must not go in the artifact. Whatever you train
on, the exam is realistic decisions in families you have not seen: the 18
October rehearsal candidate, trained on 8,400 synthetic rule cases, learned the
format perfectly and still tied stock d3 on the sealed draws.

### Teacher Chain recipe (experimental): learn from a larger model, then soup

`zils.teacher` trains a student on a larger d3-family model's answers instead of
single labels. The teacher answers a question pool with a probability for every
declared option; the student fine-tunes every language-model weight of the
pinned base so its distribution matches the teacher's (KL divergence), scored
exactly as validators score; `soup` then averages the student back toward the
pinned parent. Each step writes an ordinary base candidate.

```bash
# pool.jsonl: one {"state", "question"} per line (labels, if present, are ignored)
.venv-d3/bin/python -m zils.teacher answer --teacher models/d3-27b --pool pool.jsonl --out answers.jsonl
.venv-d3/bin/python -m zils.teacher train --answers answers.jsonl --reference models/d3-reference \
  --out student --lr 1e-6
.venv-d3/bin/python -m zils.teacher soup --out candidate stock:0.5 student:0.5
```

Requirements and settings from the 26 October 2026 pilot (vllm-sr/d3 27B
teacher, d3-mini student, one NVIDIA H200): the teacher needs about 51 GB and
answered about 10 questions per second; the student's full fine-tune with FP32
master weights needs about 100 GB and trained at about 7 rows per second, so it
does not fit 20–24 GB cards. Learning rate 1e-5 diverged; 1e-6 lowered the
held-out KL to the teacher from 0.095 to 0.039 in one epoch over 67,224 answers.
Gradient checkpointing is not supported (its recompute fails torch's metadata
check), and on Hopper GPUs the attention backward pass needs `pip install
tilelang` (Triton 3.6 is refused by flash-linear-attention).

Results of that pilot, unfavorable ones included. The pool covered 17 task
families (68,224 questions). Measured as paired score gaps against stock
d3-mini with 95% bootstrap intervals:

| Candidate | Development exam from the Decision Index's task sources (2 draws) | Current exam's qualification draws (2) |
|---|---:|---:|
| Student alone | +5.45 [+3.11, +7.20], +2.16 [−0.10, +3.74] | **−2.14** [−3.11, −1.29], **−1.67** [−2.71, −0.63] |
| Soup, 0.5 parent / 0.25 + 0.25 two students | +4.03 [+1.95, +5.39], +3.23 [+1.34, +4.45] | −0.33 [−0.96, +0.20], +0.23 [−0.30, +0.92] |

The development exam shares task families (not rows) with the pool, so part of
its gain is in-distribution; on its new-domain half the student moved +2.2 and
−3.8 and the soup +1.4 and −0.9. The student learned the pool's families and
lost general ability; the soup kept most of the gain without the loss but is
level, not better, on the exam that decides elections. A broader pool is the
next experiment. Treat this recipe as a starting point to beat, not a winning
recipe.

## How it is scored

All five text pools use the [decision-quality election](base-election.md).
A candidate needs a +1 point gain, positive paired lower 95% bound and no more
than 0.005 Brier regression on each of two independent draws. Confirmation starts
at least 12 hours after that candidate and the incumbent finish the first draw.
Every eligible candidate can qualify; the best score is only a display summary.

Qualification also requires all pinned adapter jobs under seeds 553, 771, 991:
mean accuracy drop at most 0.01 and mean Brier increase at most 0.005 per job.
Flat adapters pass. At a scheduled release window, every retained ready candidate
faces the same fresh closing draw. The best surviving candidate may be staged
for reviewed promotion. No survivor means the incumbent remains king.

**King of the hill** (owner decisions, 19 October 2026, following Teutonic):
only kings are paid. The king is the promoted base the network actually runs;
the track keeps a history of kings, newest first. The base-tier share of every
publication is split equally over the last five distinct kings (by hotkey),
whether or not anyone challenges; with fewer than five, the ones there are
share it. A king whose hotkey is no longer registered drops out and the others
share its slice. A challenger earns nothing until it is crowned, and then it
is the newest king; to dethrone the current king a challenger must beat it by
the same +1 decision-score margin on the exam and pass the confirmation draw and the suite.
A copy of the king is never eligible. The stock `d3-mini-4.5b-v1` has no owner,
so until the first promotion the history is empty and the base share is
burned (below), not paid to anyone.

Per-case results are never returned; you learn aggregate metrics only. The
round's questions are published after the round closes and are not reused.

On a validator with a `base_tier` track, the kings are paid at every
publication from the base tier's share of the one weight vector, blended with
the adapter round's weights; a share with no recipient goes to the burn uid; see
[the two-track rule](mining.md#two-tracks-the-base-tier-pays-first).

## Validator and publisher tooling

Use the [operator runbook](base-election.md#qualify-candidates-and-close-a-release-window)
for `regression_suite --election`, retained ready receipts, freeze/close and
`promote_base.py --config ... --pool ... --window-report ...`. Staging is local by
default; upload requires `--upload`. Winner hotkey and prior kings come from
verified evidence and installed configuration.

Historical cross-profile studies still accept `--incumbent-model` and
`--candidate-model` in `regression_suite`, with their original regression rule.
They do not authorize a new-policy crown. Stable migration remains a separate,
stricter evaluation described below.

## Starting points

- `python -m zils.d3 reference --out models/d3-reference` downloads and
  verifies the parent (9.1 GB); the merge recipe above is the baseline to beat.
- Full-weight fine-tuning of a 4B-class parent fits on one 80 GB GPU with fp32
  master weights and bf16 autocast (peak observed 85 GB on a 288 GB card) and has
  lost to the LoRA every time Zils tried it: lr 2e-5 diverges, 5e-6 trails the
  production adapter at equal data. Treat it as a research direction, not the
  recipe.
- Use `zils.base_metrics` on published development cases to estimate decision
  quality. Public cases cannot become private qualification evidence.

## What is not promised

The incumbent can win. Challengers are never paid, only kings: if no candidate
is promoted, the base does not change, the existing kings keep their share and,
with no king yet, the base share is burned at `base_tier.burn_uid`. A candidate that clears the exam but makes adapters
worse is rejected regardless of its exam score.

## Frozen-adapter migration coverage

The retraining comparison above does not establish that an adapter already being
served can migrate. Add `--migration-references references.json` to score one
frozen, calibrated reference release per suite job on that job's new test split.
Use only public or Zils-owned benchmark releases and datasets, never customer
releases. The manifest must cover every suite job exactly:

```json
{
  "abcd": {
    "release": "frozen/zils-adapter-JOB_UUID-CHECKPOINT_SHA256",
    "fingerprint": "SHA256_FROM_SERVING_JSON",
    "splits": "original-abcd-splits"
  },
  "flight-delay": {
    "release": "frozen/zils-adapter-JOB_UUID-CHECKPOINT_SHA256",
    "fingerprint": "SHA256_FROM_SERVING_JSON",
    "splits": "original-flight-delay-splits"
  }
}
```

Replace the placeholders with immutable accepted reference releases (including
`serving.json`, `source.json`, `manifest.json` and every listed artifact). Retain
their original `train.jsonl`, `calibration.jsonl`, and `test.jsonl` in each
`splits` directory. Paths are relative to the reference manifest. These are
operator-provisioned inputs; the suite does not create or fetch releases.
Freeze the manifest before evaluating candidate bases. Prepare new suite test
conversations not present in any of those three original splits; training and
calibration may be reused. The tool checks original split hashes, release
fingerprints, current split separation, and new-test IDs, groups and exact
prompts. Operators must still audit semantic overlap and source provenance.

```bash
python -m zils.regression_suite --model d3-mini-4.5b-v1 --suite SUITE \
  --incumbent models/d3-reference --candidate CANDIDATE --out OUT \
  --migration-references references.json --seeds 553 771 991 \
  --min-accuracy 0.8 --min-brier-improvement 0.01
```

The frozen release is verified and evaluated once per job, using its own
incumbent profile/interpreter and stored serving temperature. It is never
retrained or recalibrated. Candidate adapters use the production training recipe
and are calibrated on the current calibration split. Both strict upgrade and
migration outcomes use `zils.jobs.select`, including the accuracy and positive
skill requirements. Migration requires Brier to match or improve on the frozen
release; `--tolerance` controls only the separate retraining regression gate.

Without migration references, the historical default remains seed 553. With
references the default is three distinct seeds: 553, 771 and 991. `runs` records
each seed; `jobs` contains mean Brier/accuracy/skill when multiple seeds run.
The existing base regression verdict uses mean Brier per job. Migration uses
**every job, every seed**, not the mean: `migration.stable_eligible` requires at
least three distinct seeds, all migration comparisons passing, and the base
regression verdict passing. A single-seed diagnostic can never certify stable
eligibility. `strict_upgrade_yield` is the fraction of job/seed comparisons that
would pass the stricter customer upgrade policy, not an estimate for customers.

Migration results are report-only for the `latest` promotion helper; its existing
exam and retraining regression gates remain unchanged. Before a `stable` move,
operators must require `migration.stable_eligible: true` from this bound report.
There is no automatic stable-channel mover or follow-stable migration scheduler
in this repository. This suite does not test hosted publishing or routing; those
require an end-to-end platform migration test as well.
