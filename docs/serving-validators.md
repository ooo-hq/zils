# Grade serving miners as a validator

A validator can grade miners that serve z3 checkpoints ([serving miners](serving-miners.md)).
The grader checks that each miner answers real requests, and that its answers match
the validator's own run of the same pinned model. It reads the chain. The grader
itself sets no weights and writes nothing to the chain; it only keeps scores and
evidence on disk. The validator pays from those scores when `serving_share` is
configured (see [Pay](#pay)).

## What it does each round

1. Check that the host's GPU lock is free. If it is busy, skip the round: no
   requests are sent and no evidence or score rows are written.
2. Read the chain for miners that announce a serving endpoint, and fetch each one's
   signed manifest. Only a miner that lists a configured model at the configured
   `weights_sha256` is graded, once per model.
3. For each miner and model, make fresh questions from a seed derived from the
   validator hotkey, the round id, the miner hotkey and the model. Send one signed
   request per question.
4. After all of a miner's answers are in, pick a secret subset of the questions to verify. The
   miner cannot know which ones.
5. Take the host's GPU lock once, load the reference model, run every answered
   miner's subset on it, unload it, and release the lock.
6. Compare each miner's probabilities with the reference and record the outcome.

A round is over after `round_seconds` of work: miners not reached get no record that
round. The order is shuffled each round. The grader then waits about `round_seconds`
(20% jitter) and starts the next round.

## Run it

```bash
python -m zils.serving_grader --config serving-grader.json
```

Run it with the same Python that runs `zils serve`. Like `zils serve`, it computes
reference answers in a child process under the pinned d3 runtime (`.venv-d3`, or
`ZILS_D3_RUNTIME_PYTHON` when it is set). The grader needs that runtime, a state directory that already exists, the request
key file, and each reference checkpoint directory (see
[serving miners](serving-miners.md#make-a-checkpoint-directory)). It refuses to
start if one is missing, if the reference worker serves a model at weights other
than `weights_sha256`, or if `serving-rounds.json` is corrupt (it never resets
scores silently). It checks the weights at startup only if it gets the GPU lock in time;
otherwise it starts anyway, and every round checks the weights again and stops the grader on a mismatch.

```json
{
  "chain": {"network": "NETWORK", "netuid": 0},
  "validator_hotkey": "SS58_OF_THE_VALIDATOR_HOTKEY",
  "request_key_file": "/path/to/request-key.json",
  "state": "/path/to/existing/state-dir",
  "models": {
    "MODEL_NAME": {
      "weights_sha256": "HASH_FROM_THE_REFERENCE_model.json",
      "reference": "models/d3-reference"
    }
  },
  "round_seconds": 300,
  "questions": 8,
  "verified": 3,
  "window_hours": 24,
  "floor": 0.9,
  "lock_wait_seconds": 60
}
```

| Key | Default | Meaning |
| --- | --- | --- |
| `chain` | required | `network` and `netuid` of the subnet. |
| `validator_hotkey` | required | Part of the question seed; lets an auditor regenerate questions. |
| `request_key_file` | required | The request key, a wallet-style JSON file with `secretSeed`. |
| `state` | required | Existing directory for the output files. |
| `models` | required | Per model: `weights_sha256` and `reference`, the checkpoint directory. |
| `round_seconds` | 300 | Work budget and spacing of rounds. |
| `questions` | 8 | Requests sent to each miner per round. |
| `verified` | 3 | Secret subset recomputed; between 1 and `questions`. |
| `window_hours` | 24 | How long a result counts, and how long a strike lasts. |
| `floor` | 0.9 | Minimum pass rate; below it the score is 0. |
| `lock_wait_seconds` | 60 | How long to wait for the GPU lock. |

Each request has a 30 second timeout.

## The request key

Requests are signed with a separate Bittensor-format hotkey that is not registered
on the subnet. Never use the validator's reward hotkey for this. It is the same
key a gateway uses to call miners. A miner admits it with `--gateway-hotkey` (its
SS58 address) when running `zils serve`; without that, the miner refuses it.

## GPU lock

The reference model shares the host's GPU with validation and training, so it is
loaded only while the grader holds the host's exclusive compute lock, and unloaded
before the grader releases it. Every round pays the model's load time, and no GPU
memory is held between rounds.

Before sending any request, the grader checks the lock without waiting. If it is
busy, the round is skipped: no requests are sent and no evidence or score rows are written.
After asking the miners, it waits up to `lock_wait_seconds` for the lock. If it
cannot get it, every answered miner gets `lock_timeout` in the evidence, and nobody is scored.

The path is `ZILS_COMPUTE_LOCK` (or the older `FEZ_COMPUTE_LOCK`) if set, otherwise
`fez-compute-<uid>-cuda.lock` in the system temporary directory, which is the same lookup the
validator and training use. Set `ZILS_COMPUTE_LOCK` to the same path as your validator
and training services, for example by loading the same env file. Otherwise the grader
does not share their lock.

## Score rule

Each miner and model gets a rolling summary over the last `window_hours`.

- `pass_rate`: passes divided by answered rounds. `marginal` counts as answered but
  not as a pass, and is not a strike.
- `availability`: answered rounds divided by all rounds. Answered means `pass`,
  `marginal`, `fail` or `mismatch`. Every other outcome counts as not answered:
  `busy`, `unavailable`, `contradicts_manifest`, `refused`, `rejected`, `timeout`,
  `http`, `invalid`, `unverified`.
- Strike: any `fail` or `mismatch` zeroes the score for `window_hours` after it.
  Then the miner recovers as the result ages out.
- `score` is 0 if `pass_rate` is below `floor` or the miner is struck. Otherwise
  it is `pass_rate * availability`.

Our own failures never count against a miner. `reference_unavailable` (the worker
would not start, failed, or refused one request) and `lock_timeout` go to the
evidence file only. If the worker fails partway through, miners already compared
keep their outcome and the rest get `reference_unavailable`.

## Output files

All are in `state`.

- `serving-rounds.json`: per hotkey and model, the rows inside the window. Each row
  has `outcome`, `mean`, `worst`, `latency_s` and `at`. If it is corrupt, the
  grader refuses to start. Fix or remove it on purpose.
- `serving-scores.json`: rewritten each round. Per hotkey and model: `rounds`,
  `pass_rate`, `availability`, `median_latency_s`, `struck_until` (Unix time, 0 if
  never) and `score`.
- `serving-evidence.jsonl`: one line per miner and model per round, including our
  own failures. Fields: `v`, `kind` (`validator`), `at`, `round_id`, `hotkey`,
  `model`, `seed`, `verified` (indexes of the checked questions), `messages` (the
  signed answers), `mean`, `worst`, `latency_s` and `outcome`. It holds miner
  answers and grows without limit. To rotate, rename it. The grader opens the file
  for each write, so a renamed file is safe and a new one starts.

## Pay

Add a `serving_share` section to the validator's fleet config to pay serving
miners from the grader's scores:

```json
"serving_share": {
  "share": 0.10,
  "scores": "/path/to/state-dir/serving-scores.json",
  "switch_on_miners": 3,
  "max_age_seconds": 3600
}
```

- It requires `base_pools`. `base_pools.share` + `feature_tracks.share` +
  `serving_share.share` must not exceed 1; adapters take the rest. The fleet
  refuses a config over budget.
- A miner's serving weight is the sum of its `score` over the models it serves,
  normalized over the serving earners. Only registered hotkeys count.
- The share burns at `base_pools.burn_uid` until `switch_on_miners` different
  miners have a score above 0, and whenever the scores file is missing, corrupt,
  larger than 8 MiB, older than `max_age_seconds`, or dated more than a minute in
  the future. Only each model's `score` is frozen.
- The validator freezes the scores file into the round (`serving-scores.json`
  in the round's work directory) the first time it publishes. The report embeds
  it, so the report hash binds it and a retry pays the same vector.
- Chain attempts record `serving_share`, `adapter_share` and the serving track
  (raw score totals, earners, whether it switched on).
- Edit `serving_share` only when no round is pending: a pending round embeds the
  config installed when it first publishes.
- Volume pay is optional. Add `receipts` and `volume_share` and the share splits:
  `1 - volume_share` (70% at 0.3) pays by grading score as above, and
  `volume_share` (30%) pays by request volume. Both keys come together or not at
  all; without them the whole share pays by score.

```json
"serving_share": {
  "share": 0.10,
  "scores": "/path/to/state-dir/serving-scores.json",
  "switch_on_miners": 3,
  "max_age_seconds": 3600,
  "volume_share": 0.3,
  "receipts": {
    "url": "https://GATEWAY/v1/network/serving-receipts",
    "signer": "<gateway evidence ss58 hotkey>",
    "window_hours": 24
  }
}
```

- The gateway signs one receipts document per whole UTC hour window. It holds
  routing facts only: per miner hotkey, model and weights hash, the requests
  answered, our own count of input tokens, and the failure reasons. No request
  text, account or key is in it. The validator checks the signature against the
  pinned `signer` and the exact window.
- A miner earns volume only when it has a grading score above 0 and no gateway
  strike. Its volume weight is its input tokens summed over all models it served
  in the window, normalized over the volume earners. Only real routed traffic
  counts; hidden test questions never produce receipts.
- A gateway strike is per miner hotkey: any failed or mismatched gateway check in
  the window strikes the miner on every model. A struck miner earns nothing from
  either part and does not count toward `switch_on_miners`.
- With no volume earners the volume part burns at `base_pools.burn_uid`.
- The validator freezes the receipts into the round (`serving-receipts.json`)
  with the scores. If the fetch fails the round raises and retries after a
  restart; it does not burn the share.

## Audit

An auditor can regenerate a miner's questions from the validator hotkey, round id,
miner hotkey and model with `zils.serving_items`, then check the signed answers
in the evidence. That depends on `zils/benchmark.py`; a golden test pins its output,
so changing it changes the questions.

## Limitation

Questions vary a great deal (millions of variants) but come from rule-based
families, so a miner could learn the rules. That is acceptable: a correct answer is
not enough. It must match the validator's own run of the pinned model.
