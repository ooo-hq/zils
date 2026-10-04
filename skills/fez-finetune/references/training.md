# Pinned training workflow

Use this reference when preparing or executing a fine-tuning run. The tools are
from Jared Palmer's Apache-2.0-licensed Kev project. They remain upstream rather
than being copied and renamed here. Read the checked-out files, not mutable
`main`, for flags, input schemas, and result fields.

## Bootstrap

Requires Git, Python 3.13, and `uv`. The cloud path also requires the user's Modal
account. CPU preparation can run without a GPU or Modal account.

Choose a fresh run directory under the user's project. The commands below use
`fez-v1`; replace it once if it already exists. Run from the user's project root,
not the installed skill directory:

```bash
export FEZ_RUN_DIR="$PWD/.private/fez-finetune/fez-v1"
export FEZ_TOOLS_REF=eb45fd2381396eb7edc3964b753ebc1b0ab1da2b
test ! -e "$FEZ_RUN_DIR" || exit 1
umask 077
mkdir -p "$FEZ_RUN_DIR/kev"
printf '*\n' > "$FEZ_RUN_DIR/.gitignore"
git -C "$FEZ_RUN_DIR/kev" init
git -C "$FEZ_RUN_DIR/kev" remote add origin https://github.com/jaredpalmer/kev.git
git -C "$FEZ_RUN_DIR/kev" fetch --depth 1 origin "$FEZ_TOOLS_REF"
git -C "$FEZ_RUN_DIR/kev" checkout --detach "$FEZ_TOOLS_REF"
git -C "$FEZ_RUN_DIR/kev" rev-parse HEAD
export FEZ_TOOLS_DIR="$FEZ_RUN_DIR/kev/skills/kev-finetune"
cd "$FEZ_TOOLS_DIR"
```

Check every command succeeds and HEAD equals `FEZ_TOOLS_REF` before continuing.
The installed Zils skill is self-contained apart from this explicit network
dependency; it needs no existing Zils checkout or separately installed Kev skill.

Read `SKILL.md` in `FEZ_TOOLS_DIR` for the tool sequence, then the relevant
`references/data-format.md` and `references/data-generation.md`. Apply Zils's
scope, evidence, and subnet requirements from the calling skill. Don't execute
the upstream deployment or teardown phases just because they are listed.

`scripts/kev_modal.py` pins its own training runtime through `KEV_REF`. Preserve
that compatible default and record both the tools revision and the report's
`kev_ref`; they are distinct. Resolve the chosen Hub checkpoint to an immutable
commit and use `repository@commit` as `FEZ_INIT_FROM`. This retains the released
decision head and adapters; a bare Qwen base does not.

For a supplied Zils checkpoint, use its pinned Hub repository or the local
hardware branch. A path on the user's computer is not visible inside Modal.
To continue a run already on the shared Modal volume, verify and use its full
`/runs/<run-name>/checkpoint` path for training: this pinned trainer does not
resolve a short run name the way its evaluation command does.

Set `KEV_APP_NAME` to a workload-specific name such as `fez-support`, retaining
it for later commands. This isolates the Modal app, but the upstream volumes
`kev-finetune-runs` and `kev-hf-cache` remain shared. Use workload-prefixed run
names and check for existing runs; never use account-wide or volume-wide cleanup
for one experiment.
Reuse existing secrets by name without printing their values. Record app and
resource names in `fez-run.json` for recovery and cleanup.

## Data preparation

Use the upstream standard-library scripts rather than creating a new converter
or splitter. All paths below are relative to `FEZ_TOOLS_DIR`:

```bash
python3 scripts/extract_workload.py /absolute/path/to/user-project --out workload.json
python3 scripts/plan_size.py workload.json --baseline-acc 0.75 --min-gain 0.05
python3 scripts/split_data.py data/labelled.jsonl --out data/fez-v1
```

Replace the project path and planning assumptions with observed inputs. Review
and complete the extracted workload first. Create `data/labelled.jsonl` from
the user's labels using `convert_data.py`, or follow the generation reference.
For a supplied model/data source, use it rather than rediscovering one.

The splitter produces `train.jsonl`, `calibration.jsonl`, `development.jsonl`,
and `summary.json`. It may drop bad records while returning success: inspect
the summary and console diagnostics. Reserve the final test set **before** this
split, outside `data/fez-v1`, with disjoint source groups. These records must
not enter sizing probes, temperature fitting, or iterative error inspection.

Token limits are checked by `validate` and again in training. CPU-only Modal
validation still uploads data and can consume cloud resources; use it only
within the agreed cloud scope.

## Cloud training

Read `scripts/kev_modal.py` and its `train --help` before launching. Use `uvx modal`
where `modal` isn't installed. Authenticate through the normal Modal flow if
needed. The default H100 supports all three sizes; change it only after checking
the requested model's **training** memory requirements. Serving memory isn't a
training budget. Verify current provider pricing before setting a cost bound;
the script's embedded GPU prices are estimates, not a billing cap.

Once the user has authorized the data destination and cloud budget, for example:

```bash
export KEV_APP_NAME=fez-support
uvx modal run scripts/kev_modal.py::validate --data data/fez-v1 --init-from "$FEZ_INIT_FROM"
uvx modal run scripts/kev_modal.py::train --data data/fez-v1 --name fez-v1 --init-from "$FEZ_INIT_FROM" --gpu H100 --timeout 3600
```

`3600` is an example one-hour job timeout, not permission to spend for an hour.
Select the actual bound from dataset size and the user's budget, including
baseline/regression evaluation. Record any separately billed setup and storage.
Keep baseline comparison and public-data replay enabled. The command trains,
fits temperatures on calibration, evaluates development, and saves reports.

Inspect `runs/fez-v1/result.json`, `config.json`, and `train.log`. The result
includes candidate and baseline raw/calibrated metrics, paired comparisons, and
a regression read. Report exactly what exists; use “not measured” for missing
fields. `config.json` records the resolved parent and base revision.

After selecting a candidate, use `evaluate --help` to score the untouched final
test set for candidate and parent. This entry point accepts a JSONL file or a
split directory. Pass **only the test JSONL file**: providing a directory with
`calibration.jsonl` causes it to fit a temperature again. This pinned entry point
forces **raw logits**, even for a calibrated checkpoint. Its printed
"calibrated" column is therefore still raw when no calibration file is supplied.
For final calibrated metrics, apply each model's previously frozen temperature
to its saved test `development/rows.json` using `kev.metrics.metrics(rows, T)`
in the recorded runtime. Use the candidate and baseline temperatures from the
selected training report, and the same clean-row population as that report.
Do not fit a new temperature on test rows. Alternatively, evaluate each frozen,
authenticated endpoint with `--remote`, which retains served probabilities.

## User-owned hardware

When the user chooses local hardware, read the pinned repository's `kev/train.py`
and `kev/benchmark.py` help and use an isolated environment in that checkout.
The Modal commands do not run on a local GPU. Follow the warm-start recipe with
`kev.train --init_from`; verify the matching base/revision, dtype, checkpointing,
and a one-batch memory check before the full run. The existing Zils miner recipe
and validator are fixed to 0.8B and must not be edited to bypass that contract.

For calibration, the pinned `kev.calibrate` command writes a **report**, not a
modified checkpoint. First copy each checkpoint to a new calibration output
directory; `scripts/calibrate_checkpoint.py` modifies `head.pt` in place. Feed it
raw rows from the **calibration** split despite its upstream "development"
terminology. Preserve the parent and apply the same calibration split to both
sides. Use `kev.benchmark --data` for held-out evaluation and preserve its
prediction/row files for paired comparison.

## Optional serving and publishing

Read the pinned `references/deploy.md` only when the user wants an endpoint,
downloaded weights, publication, or cleanup. Its commands support pulling the
checkpoint, authenticated serving, remote evaluation, and publishing to a
user-selected Hub repository. Use a Zils candidate model card that credits Kev
and names the exact parent, data, measurements, and limitations.

Check the upload file list: the upstream publisher includes reports and logs,
which can contain private data or paths. Prepare a reviewed export before any
public upload. Choose visibility from the user's request; an experiment is not
automatically a public release. Keep an endpoint authenticated unless the user
explicitly requests public access. Verify sample responses and served metrics
before changing application traffic.

For 9B, this Modal serving path merges in FP32 before casting, so loading can
need substantially more memory than steady-state BF16 inference. Use its
documented H100/A100-80GB option rather than assuming a 24 GB GPU is sufficient.
Stop only the app/resources created for this workload when cleanup is requested;
retain reports and checkpoints as requested, and leave the shared model cache
and unrelated runs intact.
