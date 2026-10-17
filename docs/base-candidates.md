# Base candidates

A base candidate is a full-weight replacement for a pinned text base, per
profile: d3-mini-4.5b-v1, the base for new jobs since 17 October 2026, and H2O
Lightning 4B v1.2.3 while existing H2O adapters are still served. d3-mini did not
pass the confirmation draw on 17 October 2026; it became the base by the owner's
decision, recorded as an override of the promotion rule (see
[d3-mini](roadmap.md#d3-mini)). Everything below describes H2O unless it says d3. Miners may train it however they like; validators
score it on a sealed holdout against the current base and, for a pending
winner, retrain a fixed suite of reference adapters on both. Status, 17 October 2026:
candidate loading, round decisions, sandboxed evaluation, the regression suite,
chain-derived admission and the dry-run promotion script are merged; open
admission is off by default; no public round has been run; the channel setting
on the platform is a configuration value (`ZILS_TRAINING_MODEL_LATEST`), not yet
a registry alias; automatic "follow stable" upgrades are not implemented. Since
17 October 2026 both channels name d3-mini; existing H2O jobs keep their frozen
profile and move to d3 only through `base_change` upgrade jobs.

## What stays fixed

Only the weights change. A candidate must keep:

- the architecture, layer sizes and tokenizer of the pinned base, so the
  production LoRA recipe attaches to the same 152 language modules;
- the seven contract files byte-identical to the parent (`tokenizer.json`,
  `tokenizer_config.json`, `chat_template.jinja`, `serve_config.json`,
  `h2o_lightning_shim.py`, `generation_config.json`, `LICENSE`);
- the 2,048-token input limit, the answer-token readout and the pinned runtime
  (`zils/h2o-pins.json`).

A d3 candidate keeps the files listed as `models.D3_CONTRACT_FILES` byte-identical
(config, tokenizer, readout, vendor runtime modules, `LICENSE`, `NOTICE`) and
the 4,096-token input limit; its weights are the two `model-0000N-of-00002.safetensors`
shards plus `model.safetensors.index.json`, and its `model.json` is written with
`write_metadata(path, kind="candidate", model=zils.models.D3)`. Before loading a
d3 candidate the validator checks that its index `weight_map` names exactly the
parent index's tensors, maps them only to those two shard file names, and that
each shard holds exactly the tensors the index assigns to it; transformers then
rejects any tensor whose shape differs from the parent.

The validator refuses a candidate whose contract files differ from the parent
hashes, whose `architectures` or `text_config` differ structurally from the
parent, or whose weights do not load exactly.

## Artifact

A directory containing:

| File | Content |
| --- | --- |
| `model.safetensors` | all weights, bf16, one file |
| `config.json` | the parent configuration (volatile keys such as `dtype`, `use_cache` and `transformers_version` may differ) |
| `model.json` | `{"version": "zils-checkpoint/v1", "model": "h2o-lightning-4b-v1.2.3", "kind": "candidate", "temperature": 1.0, "profile": <the parent profile>}`; write it with `zils.models.write_metadata(path, kind="candidate", model=zils.models.H2O)` |
| the seven contract files | copied from the parent |

Total size must stay under 10 GiB (the parent is about 9.1 GB). Announce it
with the same signed claim as an adapter; the checkpoint hash covers every
file above. One live submission per hotkey per round; identical hashes from
different miners are rejected.

## How it is scored

1. The validator fits one temperature for your candidate on the round's
   sealed calibration split, then scores the sealed test split by
   equal-family Brier loss, exactly as it scores the incumbent.
2. You are **eligible** when your skill is positive, your accuracy is not
   below the incumbent's, your Brier improvement is at least 0.01, and the
   paired bootstrap 95% lower bound of that improvement is above zero.
   Eligible candidates share the round's weight in proportion to skill.
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

## Validator and publisher tooling

The reference-adapter suite and the publication helper take the profile with
`--model` (`h2o-lightning-4b-v1.2.3`, the default, or `d3-mini-4.5b-v1`):

```bash
python -m zils.regression_suite --model h2o-lightning-4b-v1.2.3 --suite SUITE \
  --incumbent models/h2o-reference --candidate CANDIDATE --out OUT
python -m zils.regression_suite --model d3-mini-4.5b-v1 --suite SUITE \
  --incumbent models/d3-reference --candidate CANDIDATE --out OUT

python scripts/promote_base.py --model h2o-lightning-4b-v1.2.3 --candidate CANDIDATE \
  --decision decision.json --round N --staging STAGING
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

- `python -m zils.h2o reference --out models/h2o-reference` downloads and
  verifies the parent.
- Full-weight fine-tuning of the parent fits on one 80 GB GPU with fp32 master
  weights and bf16 autocast; peak observed 85 GB on a 288 GB card. Learning rate
  2e-5, the LoRA default, diverges; 5e-6 trains but, at equal data, trailed the
  production adapter in the only published attempt. Treat this as the recipe to
  beat, not to copy.
- A local scorer with the same Brier, calibration and family weighting is
  `zils.base_round`; run it against a previously published round to estimate
  where you stand before submitting.

## What is not promised

The incumbent can win. If no candidate is eligible, the round pays nobody and
the base does not change. A candidate that clears the exam but makes adapters
worse is rejected regardless of its exam score.
