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
was promoted). Open admission is off; no public round has been run; the channel
setting on the platform is a configuration value (`ZILS_TRAINING_MODEL_LATEST`),
not yet a registry alias; automatic "follow stable" upgrades are not implemented.

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
same size). Announce it with the same signed claim as an adapter plus
`"kind": "candidate"`; the checkpoint hash covers every file above. One live
candidate per hotkey per round, a claim cannot change within a round, identical
hashes from different miners are rejected, and the incumbent's own hash is
refused. Announce every round: a validator that scores candidates itself keeps
each hash it has fetched, so a repeated hash costs no download and no rescoring.

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
   winner takes all: the pending winner (lowest Brier among eligible
   candidates) takes the whole base-tier share; eligible runners-up earn
   nothing. Once a winner is promoted its miner keeps that share, king of the
   hill, in every epoch without an eligible challenger.
3. The best eligible candidate is re-scored on a second, disjoint draw at
   least 12 hours later and must stay eligible.
4. Before promotion, reference adapter jobs are retrained on the incumbent and
   on your candidate with the production recipe. Your candidate passes only if
   its adapters are no worse than the incumbent's on every job (0.005 Brier
   tolerance) and better on at least one.
5. A promoted candidate becomes the `latest` base; it becomes `stable`, the
   default for new customer jobs, after surviving a full window on `latest`.

Per-case results are never returned; you learn aggregate metrics only. The
round's questions are published after the round closes and are not reused.

On a testnet fleet with a `base_tier` track, the pending winner (or, with no
challenger, the promoted incumbent's miner) is paid at every publication from
the base tier's share of the one weight vector, blended with the adapter round's
weights; see
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

The incumbent can win. If no candidate is eligible, the round pays nobody and
the base does not change. A candidate that clears the exam but makes adapters
worse is rejected regardless of its exam score.
