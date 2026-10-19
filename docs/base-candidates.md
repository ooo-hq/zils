# Base candidates

A base candidate is a full-weight replacement for the pinned text base,
`d3-mini-4.5b-v1` (the base on both channels since 17 October 2026, adopted by
owner decision after it tied the H2O incumbent on the confirmation draw; see
[d3-mini](roadmap.md#d3-mini)). H2O Lightning 4B v1.2.3 is retired: no longer
served, no new jobs; its candidate contract is kept below for the record.
Everything below describes d3 unless it says H2O. Miners may train a candidate
however they like; validators score it on a sealed holdout against the current
base and, for a pending winner, retrain a fixed suite of reference adapters on
both. Status, 18 October 2026: candidate loading, round decisions, sandboxed
evaluation, the regression suite, chain-derived admission, the merge recipe,
promoted-profile support in the runtime and the dry-run promotion script are
merged; the full loop was rehearsed end to end on 17/18 October (a synthetic-only
candidate tied stock d3 on the sealed exams and passed the adapter suite; nothing
was promoted). Any validator running this code on subnet 579 collects and scores
candidates. Candidates enter the
standard Bittensor way: the miner uploads the candidate to its own Hugging Face
repo and commits repo, revision and checkpoint hash on chain with
`zils.miner commit-candidate`; every validator reads the commitments and pulls
(below; replaces the 18 October push announcement). The channel setting on the platform
is a configuration value (`ZILS_TRAINING_MODEL_LATEST`), not yet a registry
alias; automatic "follow stable" upgrades are not implemented.

## Promoted profiles

A confirmed d3 winner becomes a new profile, not a new revision of `d3-mini-4.5b-v1`:
`scripts/promote_base.py --model d3-mini-4.5b-v1 --round N` stages the candidate's
files with a Zils `NOTICE` and `README.md`, uploads them to `zils-ai/d3-mini-4.5b-zils-rN`,
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

(H2O, retired: seven contract files `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja`, `serve_config.json`, `h2o_lightning_shim.py`,
`generation_config.json`, `LICENSE`; one `model.safetensors`; 2,048 tokens;
`zils/h2o-pins.json`.)

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

## Commit

A candidate is two things: the artifact in **your own public Hugging Face model
repo**, pinned by commit, and **one chain commitment** made with your registered
miner hotkey on subnet 579 (the hotkey file only; the coldkey is never read):

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
the wallet, candidate and repo that the commands below default to.
`zils doctor` checks each prerequisite in order (requirements, d3 training
runtime, hotkey file, registration, candidate, Hugging Face login and repo at the
revision, the commitment on chain) and prints the first fix or, when all pass,
the block your commitment landed in. `zils commit-candidate` does the work:

```bash
hf auth login   # once, with a write token for your account
zils commit-candidate   # flags such as --candidate or --repo override the saved values
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
breaks exact ties by uid. The defence there is score-based (prefer the earlier
commitment unless the later one is better by a margin) and is an open owner
decision; until then a promoted winner is protected by king of the hill, since a
copy of the incumbent is never eligible.

### Results

The per-round `base_tier` row (Brier and accuracy against the incumbent,
eligibility, pending and confirmed winner) is published by each validator in its
signed round result over its own HTTP API (`GET /results/<round>`); the chain does
not carry it. What the chain does carry, readable without any URL, is the payout:
the validator's weights and your incentive, for example
`btcli subnets metagraph --netuid 579 --network test`. A validator could publish
its API address on chain (`serve_axon`) so miners find it without an operator
announcement; not done yet.

### Reference recipe: train a LoRA, merge it

The fleet's 20–24 GB cards cannot full-fine-tune the 4.5B backbone, and every
full fine-tune Zils has tried lost to the LoRA. The published recipe is the
production adapter merged into the backbone:

```bash
export ZILS_D3_RUNTIME_PYTHON="$PWD/.venv-d3/bin/python"
.venv-d3/bin/python -m zils.d3 reference --out models/d3-reference
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

## How it is scored

1. The validator fits one temperature for your candidate on the round's
   sealed calibration split, then scores the sealed test split by
   equal-family Brier loss, exactly as it scores the incumbent.
2. You are **eligible** when your skill is positive, your accuracy is not
   below the incumbent's, your Brier improvement is at least 0.01, and the
   paired bootstrap 95% lower bound of that improvement is above zero. The
   best eligible candidate (lowest Brier) is the round's **pending winner**: a
   challenger, recorded for promotion, paid nothing. Eligible runners-up are
   not challengers.
3. The pending winner is re-scored on a second, disjoint draw at
   least 12 hours later and must stay eligible (the confirmed winner, still
   unpaid).
4. Before promotion, reference adapter jobs are retrained on the incumbent and
   on your candidate with the production recipe. Your candidate passes only if
   its adapters are no worse than the incumbent's on every job (0.005 Brier
   tolerance) and better on at least one.
5. A candidate that passes all three gates is promoted in an owner-reviewed
   pull request, which also **crowns** it: the validator track's
   `incumbent_sha256` becomes the candidate's hash and its miner's hotkey is
   prepended to `kings` (`scripts/promote_base.py --king-hotkey … --previous-kings …`
   prints both as `proposed_base_tier`).
   It becomes the `latest` base, and `stable`, the default for new customer
   jobs, after surviving a full window on `latest`.

**King of the hill** (owner decisions, 19 October 2026, following Teutonic):
only kings are paid. The king is the promoted base the network actually runs;
the track keeps a history of kings, newest first. The base-tier share of every
publication is split equally over the last five distinct kings (by hotkey),
whether or not anyone challenges; with fewer than five, the ones there are
share it. A king whose hotkey is no longer registered drops out and the others
share its slice. A challenger earns nothing until it is crowned, and then it
is the newest king; to dethrone the current king a challenger must beat it by
the same 0.01 margin on the exam and pass the confirmation draw and the suite.
A copy of the king is never eligible. The stock `d3-mini-4.5b-v1` has no owner,
so until the first promotion the history is empty and the base share is
burned (below), not paid to anyone.

Per-case results are never returned; you learn aggregate metrics only. The
round's questions are published after the round closes and are not reused.

On a testnet fleet with a `base_tier` track, the kings are paid at every
publication from the base tier's share of the one weight vector, blended with
the adapter round's weights; a share with no recipient goes to the burn uid; see
[the two-track rule](mining.md#two-tracks-on-the-testnet-the-base-tier-pays-first).

## Validator and publisher tooling

The reference-adapter suite and the publication helper take the profile with
`--model` (`d3-mini-4.5b-v1`, the base; `h2o-lightning-4b-v1.2.3` only for the
retired profile's records and still the code default):

```bash
python -m zils.regression_suite --model d3-mini-4.5b-v1 --suite SUITE \
  --incumbent models/d3-reference --candidate CANDIDATE --out OUT

python scripts/promote_base.py --model d3-mini-4.5b-v1 --candidate CANDIDATE \
  --decision decision.json --round N --staging STAGING
```

Each run uses that profile's trainer, interpreter (`ZILS_H2O_RUNTIME_PYTHON` or
`ZILS_D3_RUNTIME_PYTHON`) and candidate override (`ZILS_H2O_CANDIDATE_BASE` or
`ZILS_D3_CANDIDATE_BASE`). To compare a base on a different profile from the
incumbent, give `--incumbent-model`, `--incumbent`, `--candidate-model` and
`--candidate` (a reference directory of the candidate profile) instead of
`--model`; each side then trains every suite job with its own profile and
`--candidate-runtime-python` selects the candidate profile's interpreter. The
verdict rule is the same. `promote_base.py` proposes ids
`h2o-lightning-4b-zils-r<N>` or `d3-mini-4.5b-zils-r<N>` and the matching
`zils-ai/` repository.

## Starting points

- `python -m zils.d3 reference --out models/d3-reference` downloads and
  verifies the parent (9.1 GB); the merge recipe above is the baseline to beat.
- Full-weight fine-tuning of a 4B-class parent fits on one 80 GB GPU with fp32
  master weights and bf16 autocast (peak observed 85 GB on a 288 GB card) and has
  lost to the LoRA every time Zils tried it: lr 2e-5 diverges, 5e-6 trails the
  production adapter at equal data. Treat it as a research direction, not the
  recipe.
- A local scorer with the same Brier, calibration and family weighting is
  `zils.base_round`; run it against a previously published round to estimate
  where you stand before submitting.

## What is not promised

The incumbent can win. Challengers are never paid, only kings: if no candidate
is promoted, the base does not change, the existing kings keep their share and,
with no king yet, the base share is burned at `base_tier.burn_uid`. A candidate that clears the exam but makes adapters
worse is rejected regardless of its exam score.
