# Grade serving miners as a validator

A validator can grade miners that serve z3 checkpoints ([serving miners](serving-miners.md)).
The grader checks that each miner answers real requests, and that its answers match
the validator's own run of the same pinned model. It reads the chain. It sets no
weights and writes nothing to the chain; it only keeps scores and evidence on disk.

## What it does each round

1. Read the chain for miners that announce a serving endpoint, and fetch each one's
   signed manifest. Only a miner that lists a configured model at the configured
   `weights_sha256` is graded, once per model.
2. For each miner and model, make fresh questions from a seed derived from the
   validator hotkey, the round id, the miner hotkey and the model. Send one signed
   request per question.
3. After all answers are in, pick a secret subset of the questions to verify. The
   miner cannot know which ones.
4. Run those questions on a local reference worker that serves the same pinned
   model, while holding the host's GPU lock.
5. Compare the miner's probabilities with the reference and record the outcome.

A round is over after `round_seconds` of work: miners not reached get no record that
round. The order is shuffled each round. The grader then waits about `round_seconds`
(20% jitter) and starts the next round.

## Run it

```bash
.venv-d3/bin/python -m zils.serving_grader --config serving-grader.json
```

The grader needs the d3 runtime, a state directory that already exists, the request
key file, and each reference checkpoint directory (see
[serving miners](serving-miners.md#make-a-checkpoint-directory)). It refuses to
start if one is missing, if the reference worker serves a model at weights other
than `weights_sha256`, or if `serving-rounds.json` is corrupt (it never resets
scores silently). If the worker fails to start, it is retried at the next miner.

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

The reference worker shares the host's GPU, so the grader takes an exclusive file
lock only while it computes reference answers. The path is `ZILS_COMPUTE_LOCK` if
set, otherwise `fez-compute-<uid>-cuda.lock` in the system temporary directory. It
waits up to `lock_wait_seconds`. If it cannot get the lock, that miner's round is
recorded as `lock_timeout` in the evidence for it and every miner left in the round, and nobody is scored.

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
would not start or failed) and `lock_timeout` go to the evidence file only, and
the remaining miners that round get the same evidence row.

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

## Audit

An auditor can regenerate a miner's questions from the validator hotkey, round id,
miner hotkey and model with `zils.serving_items`, then check the signed answers
in the evidence. That depends on `zils/benchmark.py`; a golden test pins its output,
so changing it changes the questions.

## Limitation

Questions vary a great deal (millions of variants) but come from rule-based
families, so a miner could learn the rules. That is acceptable: a correct answer is
not enough. It must match the validator's own run of the pinned model.
