# Flight-delay adapter experiment 001

This experiment asks whether training a JevK5 adapter on public flight outcomes
improves arrival-delay probabilities on later flights. It compares the unchanged
base, a trained adapter, and a simple historical-rate baseline. It is a research
experiment; running it never registers or promotes a customer prediction model.

## Frozen design

The target is arrival at least 15 minutes late, conditional on a completed,
non-diverted flight with a recorded arrival time. Data comes from the US Bureau
of Transportation Statistics [reporting-carrier on-time records](https://transtats.bts.gov/ONTIME/Index.aspx).
The [manifest](data/flight-delay-001-manifest.json) records official archive URLs,
SHA-256 checksums, filtering counts, and hashes of every prepared input. The
[protocol](data/flight-delay-001-protocol.json) fixes the recipe and success rule
before training or model evaluation.

| Purpose | Period | Flights | At least 15 minutes late |
| --- | --- | ---: | ---: |
| Train weights and historical rates | January 2025 | 3,072 | 630 |
| Fit temperatures | February 2025 | 512 | 121 |
| Final unseen evaluation | March 2025 | 1,024 | 201 |

All flights depart JFK, LGA, or EWR. Selection uses the smallest identity hashes
with seed 553, independent of labels and archive ordering. Sampling does not
balance classes. No flight identity or flight date crosses a split. Test labels
are used for final metrics only; preprocessing records counts but does not use
them to select a model, training recipe, or prediction threshold.

The model sees only the scheduled date, weekday, airline, flight number, origin,
destination, local departure and arrival times, scheduled duration, and distance.
Actual departure time, departure delay, actual duration, arrival delay,
cancellation, diversion, and delay-cause fields never enter model inputs.
Historical schedule fields are treated as predeparture information; this archive
is not a timestamped snapshot of what a booking system knew at each moment.

Training uses the repository's pinned JevK5 4B base and existing attention-only
LoRA implementation: one epoch, rank 16, alpha 32, dropout 0.05, learning rate
0.00002, batch size one, accumulation four, gradient clipping 1.0, seed 553.
Base weights and the output head remain frozen. The saved adapter is frozen and
hashed before either model receives final test inputs.

The historical baseline uses exactly the same 3,072 training flights. It estimates
rates per airline, origin, and four-hour departure window, with 20 observations
of smoothing toward the global rate. The global rate uses Beta(1,1) smoothing;
unseen groups use that global rate. It does not use February or March labels.

Both neural models receive a separate temperature fit on February, minimizing
binary log loss over 81 log-spaced temperatures from 0.25 to 4. Raw results are
also reported: base temperature 1.22 and adapter temperature 1.0. Calibration
changes probabilities, not which outcome has the highest score.

The primary metric is **binary Brier score**, the mean of
`(predicted_probability_of_late - observed_late)^2`; lower is better. This is half
the sum-over-two-classes Brier score used elsewhere in the queue. Accuracy,
balanced accuracy, late recall, ROC-AUC, log loss, and confident errors are
secondary metrics. Accuracy alone is misleading when most flights are not late.

Success requires the calibrated adapter to beat **both** the equally calibrated
base and the historical baseline on test Brier, with both paired 95% bootstrap
intervals entirely above zero. Positive improvement means reference Brier minus
adapter Brier. The bootstrap resamples whole flight dates, 2,000 times with seed
553, keeping correlated same-day flights together. There are 31 test-date groups.

## Reproduce

Use Linux or WSL 2, Python 3.13, a BF16-capable NVIDIA GPU and CUDA driver, Git,
and `uv`. Allow at least 25 GB disk for dependencies and the pinned base, plus
the downloaded archives. Follow [JevK5 installation](jevk5-queue.md#install-and-create-the-reference)
from a fresh clone to create `.venv-kev` and `models/jevk5-reference`.

1. Download and freeze the public data from the repository root:

   ```sh
   .venv-kev/bin/python -m scripts.flight_data \
     --raw .private/flight-delay-001/raw \
     --out .private/flight-delay-001/data --download
   ```

   The command checks each archive against its published hash and refuses to
   overwrite prepared data. If BTS revises an archive, preserve the original
   experiment and version a new one rather than silently changing the source.

2. Configure the GPU lock. On a dedicated experimental machine:

   ```sh
   touch .private/flight-delay-001/gpu.lock
   export FEZ_COMPUTE_LOCK="$PWD/.private/flight-delay-001/gpu.lock"
   export FEZ_GPU_MIN_FREE_MIB=12288
   export FEZ_NVIDIA_SMI="$(command -v nvidia-smi)"
   ```

   On a shared machine, use the **existing administrator-provisioned lock** used
   by every training process. A separate new lock would not coordinate them.
   Set `FEZ_NVIDIA_SMI` to the installed executable; WSL commonly uses
   `/usr/lib/wsl/lib/nvidia-smi`. The runner requires at least 12 GiB measured
   free memory and at most 512 input tokens. The memory gate is a minimum, not a
   reservation; unrelated GPU programs must also be coordinated. Keep serving
   health under supervision when sharing a GPU.

3. Run the bounded experiment:

   ```sh
   .venv-kev/bin/python -m scripts.flight_experiment run \
     --data .private/flight-delay-001/data \
     --reference models/jevk5-reference \
     --out .private/flight-delay-001/run-001 --timeout 3600
   ```

   Model files must already be cached: execution is offline. The runner uses
   the shared lock for all GPU work, checks the unchanged base metadata, trains
   one adapter, evaluates sequentially, and writes `results.json`. It terminates
   and reaps child processes on timeout or a termination request. Use a service
   supervisor that kills the full process group on forced termination. Existing
   output directories are preserved; use a new name for an intentional rerun.

4. Inspect `results.json`, `training.log`, `frozen-adapter.json`, and the four
   evaluation logs in the output directory. Raw probabilities, adapter weights,
   and prepared rows stay in ignored storage. Hashes permit integrity checks;
   CUDA numeric behavior may prevent byte-identical retraining across hardware
   or package versions.

## Interpretation limits

This tests one training recipe and seed on a small seasonal sample from three
airports. It cannot establish general flight-prediction quality, airline-specific
reliability, or customer readiness. Thirty-one test dates provide limited
uncertainty information and do not cover all seasonal conditions.

There is no live weather, inbound-aircraft state, congestion feed, or information
about disruptions developing after the prediction. An adapter can learn useful
patterns from schedules and outcomes; it cannot recover facts absent from its
inputs. Cancellation and diversion are excluded using outcomes, so results
describe the completed-flight population rather than all scheduled flights.

Public historical records may have appeared in the base model's pretraining;
that contamination cannot be ruled out. Later calendar splits protect against
leakage from this adapter's training, not against unknown base-model exposure.
Any further recipe changes require a new experiment and a fresh final test.
