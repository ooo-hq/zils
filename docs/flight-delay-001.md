# Flight-delay adapter experiment 001

This experiment asks whether training a JevK5 adapter on public flight outcomes
improves arrival-delay probabilities on later flights. It compares the unchanged
base, a trained adapter, and a simple historical-rate baseline. It is a research
experiment; running it never registers or promotes a customer prediction model.

## Measured result

**Training improved JevK5's probability predictions on the unseen March flights,
but did not demonstrate an advantage over simple historical rates.** The
calibrated adapter reduced Brier error by **6.53% relative to the calibrated
base** and improved ROC-AUC from 0.5165 to 0.6401. The historical baseline had
the lowest measured Brier score and highest AUC.

| Model | Binary Brier ↓ | Accuracy | Late recall at 50% | ROC-AUC ↑ |
| --- | ---: | ---: | ---: | ---: |
| Unchanged JevK5, calibrated | 0.164076 | 80.18% | 0.50% | 0.5165 |
| Trained adapter, calibrated | 0.153357 | 80.37% | 0.00% | 0.6401 |
| Historical-rate baseline | **0.151347** | 80.37% | 0.00% | **0.6503** |
| Unchanged JevK5, raw | 0.170116 | 80.18% | 0.50% | 0.5165 |
| Trained adapter, raw | 0.156231 | 80.37% | 0.00% | 0.6401 |

There were 201 late flights among 1,024 test flights. Always predicting
"not late" would achieve 80.37% accuracy. The adapter and historical baseline
both selected that outcome for every flight at a 50% cutoff, so the small
accuracy change does **not** demonstrate useful delay alerts. Their probability
rankings contain more information than those hard decisions. A different alert
threshold would need selection on separate validation data and evaluation
against the intended cost of missed delays and false alarms.

| Reference minus calibrated adapter | Brier improvement | Paired date-bootstrap 95% interval |
| --- | ---: | --- |
| Calibrated base | +0.010719 | [+0.006059, +0.015117] |
| Historical rates | −0.002010 | [−0.005960, +0.002194] |

Positive values favor the adapter. Improvement over the base is supported by
this experiment's interval. The historical comparison crosses zero and its
point estimate favors historical rates, so the frozen requirement to beat both
references **was not met**. This is evidence that this adapter learned useful
flight-risk patterns beyond the unchanged base, not evidence that it is the
best predictor or ready for deployment. No model was promoted.

February-only calibration selected temperatures 0.933033 for the base and
0.870551 for the adapter. Mean predicted late probabilities on March were
24.42%, 23.28%, and 21.12% for calibrated base, calibrated adapter, and historical
rates respectively, versus an observed 19.63%. Median synchronized forward time
was 49.7 ms for the base and 64.7 ms for the adapter. All raw/calibrated metrics,
latencies, artifact hashes, and provenance are in the
[aggregate result](data/flight-delay-001.json).

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

## Execution and verification

The recorded run started October 5, 2026 at 01:47:17 UTC on Linux/WSL 2 with an
NVIDIA RTX 4090 (23,028 MiB reported memory, driver 595.79). It used Python
3.13.15, PyTorch 2.8.0 with CUDA 12.8, Transformers 5.17.0, PEFT 0.21.0, and the
pinned JevK5 runtime revision documented in the installation guide. The frozen
experiment implementation is commit `33e63fdc9cd36ef44f27342084e02719832150c5`.

The adapter trained on all 3,072 examples for 768 optimizer steps in 1,651.7
seconds (27.5 minutes, including saving and excluding initial model loading).
Mean training loss was 0.521791. Saved BF16 adapter weights occupy 28,796,120
bytes, or 28.8 MB. Peak training allocation measured by PyTorch was
8,847,247,360 bytes (8.24 GiB), excluding the separate serving process. The
longest actual input was 253 tokens; the existing trainer's 2,048-token ceiling
in its metadata does not describe actual input length.

The GPU was shared with the existing live runtime. Its health checks continued
to succeed, and a synthetic live prediction completed during training. These
checks establish coexistence for this run, not a production capacity guarantee.
Reported evaluation latency measures synchronized forward calls after warmup;
it excludes tokenization, model loading, network transport, and API scheduling.
Optional `causal_conv1d` and `flash-linear-attention` kernels were not installed;
the runtime reported using reference PyTorch implementations.

`make check` passed locally (75 Python tests, one skipped), and all five GitHub
checks passed, including database and SDK compatibility checks. Focused tests
cover future-field exclusion, deterministic sampling, duplicate identities,
historical-rate fitting, scoring, calibration, the memory floor, and child
cleanup. Independent calculations matched six reported metrics to scikit-learn
within `1e-12`. Prepared data hashes, the training export, and historical rates
were also independently checked against the frozen inputs.
After download, all aggregate metrics and confidence intervals were recomputed
from the recorded predictions and matched within `1e-12`. Frozen adapter, source
code, protocol, and dataset hashes matched the run's recorded evidence. The
supervised process exited successfully after 31 minutes 35 seconds and released
its GPU resources; the live runtime retained its original release identity.
