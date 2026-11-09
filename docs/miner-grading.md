# Miner grading and assignment

Miners submit candidate artifacts and signed operational evidence. Validators
independently check the artifacts and evaluate held-out examples. A miner's own
quality claim cannot set its grade or bypass acceptance thresholds.

The shared policy is implemented in `zils/miner_grading.py`. It checks comparable
model/trainer/rubric/benchmark contexts, expires stale evidence, and separates
qualification from customer work. Routing contexts are keyed by model and token
band (`context_key`, for example `d3-mini-4.5b-v1/tokens-4096`); a context whose
band exceeds its model's input cap is invalid, and evidence for one model never
grades another. The signed presence client is in `zils/miner_presence.py`; a
queue miner with `graded_scheduling` in its config sends a heartbeat that tells
the router it is online and how much GPU memory it has free.

Qualification, assignment, and model acceptance are different decisions. A
successful qualification is not a customer model, a heartbeat is not proof of
quality, and an accepted assignment does not guarantee a qualifying candidate.
The policy's observation binding uses the uploaded artifact hash before
calibration changes the checkpoint.

Run `make check` to exercise the shared policy and worker contracts. Hosted
assignment, admission, database migrations, resource binding and qualification
administration moved to the
[zils-platform operator guide](https://github.com/ooo-hq/zils-platform/blob/main/docs/miner-grading.md).
Customer-job miners: see [customer-job miners](queue-miners.md) for setup and
[miner admission](miner-admission.md) for how a registered hotkey enters the
graded pool.
