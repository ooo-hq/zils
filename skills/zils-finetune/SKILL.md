---
name: zils-finetune
description: Fine-tune a Zils decision-model candidate on a user's labelled data using Kev 0.8B, 4B, or 9B. Use for domain-specific yes/no decisions, classification, routing, or scoring, including calibration and comparison against the starting checkpoint. Supports optional serving; subnet participation is a separate workflow.
---

# Fine-tune a Zils decision model

This skill covers the legacy Kev own-data workflow. For the current JevK5 hosted
base, follow the [JevK5 queue guide](../../docs/jevk5-queue.md) and
[automatic workflow](../../docs/automatic-training.md).

Produce an experimental Zils candidate, calibrated probabilities, and a measured
comparison against its starting checkpoint. Reuse the pinned Kev training tools
described in [the training workflow](references/training.md); this skill adds
Zils's experiment and reporting requirements without maintaining a second trainer.

Zils builds on [Kev](https://github.com/jaredpalmer/kev). There is currently no
published Zils checkpoint to download. By default, start from a released
`jaredpalmer/kev-0.8b`, `jaredpalmer/kev-4b`, or `jaredpalmer/kev-9b` checkpoint.
Call the output a **Zils candidate based on Kev**, not an official Zils release.
If the user supplies an existing candidate, preserve its provenance and compare
against that exact parent. An adapter is tied to its base architecture: changing
size starts a new lineage, not a continuation of the smaller adapter.

## Establish the run

Infer answers from the request and project before asking. Resolve missing
information together: the decision and exact questions/options, labelled data,
target size, available hardware, and execution location. For cloud work, resolve
the data destination, spending limit, and timeout before uploading or launching.
Existing authorization carries forward; installation alone starts no training.

- **0.8B:** inexpensive smoke runs and the current Zils subnet architecture.
- **4B:** default for own-data experiments unless the user chose another size.
- **9B:** supported by the upstream workflow; choose hardware for training and
  checkpoint loading, not just the final inference footprint. Never silently
  substitute 4B when the user selected 9B.

The maintained cloud recipe uses Modal. If the user requests their own GPU,
follow the local branch in the training reference. Keep the current project's
environment and model settings intact; use a separate run directory and Python
environment. Store data, checkpoints, raw predictions, and logs privately.

Done when the run has a name, exact parent, workload, execution location, and
resource bound. After bootstrapping the fresh private directory below, record
these choices in its `zils-run.json`.

## Prepare data and freeze the comparison

Read [the training workflow](references/training.md), bootstrap its pinned tools,
and read their data-format reference before converting inputs. Discover existing
TypeSafe/System One calls so training uses the same instructions and option names
as the application. Prefer existing labels; synthetic data needs explicit rules,
source labels, and reviewed examples. Generating examples is not evidence of gain.

Separate training, calibration, development, and a final untouched test set.
Group related tickets, documents, users, or generated scenario families before
splitting. The upstream splitter only groups normalized states; it cannot find
semantic duplicates or related cases for you. Inspect invalid-line and conflict
counts even when the tool exits successfully. Correct unexplained exclusions
before training, record deliberate exclusions, and check label coverage in each
split. Keep public JevBench questions out of training, calibration, and selection.

Use the upstream sizing tool to assess the data available. Label a small smoke
run as a smoke run; don't invent a minimum dataset size or a guaranteed gain.
Save hashes and counts for frozen splits. Fit temperatures only on calibration;
development supports iteration, and the untouched test supports the final claim.

## Train, calibrate, and compare

Use the published Kev checkpoint as `--init-from`, with an immutable Hub revision,
and retain its matching Qwen base and adapter/head dimensions. Continue an
existing Zils candidate only through a supported checkpoint path. Keep replay and
baseline evaluation enabled unless the experiment explicitly studies them.

Follow the upstream tools for training, calibration, paired comparison, and the
general-task regression check. Calibrate candidate and baseline on the same
calibration split, then compare on identical held-out examples at the same
precision and runtime. Read rejection/truncation counts and failure logs; failed
or dropped cases must remain visible in coverage. A failed run is not a score.

Use fresh run names for iterations. If a cloud command loses contact, inspect
the existing job before retrying so a second paid job isn't started accidentally.
Stop at the agreed run/time/spending bound; more tuning requires remaining scope.

## Deliver the evidence

Preserve the upstream report and write a concise `zils-report.md` beside it:

- Parent and candidate identities, base size, code revisions, data provenance,
  split hashes/counts, seed, precision, hardware, temperature, and run duration.
- Baseline and candidate accuracy, Brier loss, ECE, and high-confidence errors;
  include paired uncertainty and the regression check where measured.
- Any missing metrics, rejected inputs, data limitations, and the exact commands
  needed to reproduce the run. Measure latency separately with matched settings
  if the user needs it; don't infer speed from parameter count.
- A decision: improvement supported, inconclusive, or regression. Preserve
  unfavorable results. Thresholds selected on development need a frozen test
  check; reported coverage is not a promised production error rate.

The upstream workflow's Brier aggregation differs from Zils's subnet
family-macro Brier skill. Label the metric actually computed; don't rename one
as the other. A domain fine-tune does not establish general superiority to Kev
or Jev, and public JevBench results are not an official leaderboard rank.

Finish with the checkpoint/report locations and what the evidence supports.
Serving, publishing weights, switching application traffic, and deleting runs
are conditional on the user's requested scope. Read the deployment branch of
the training reference for those actions; preserve necessary artifacts first.

## Subnet boundary

This skill creates own-data candidates; it does not register miners, spend
tokens on-chain, or change validator rules. The current
[Zils validator](https://github.com/ooo-hq/zils/blob/main/zils/kev_runner.py) expects
the pinned 0.8B base, rank-16 LoRA, a 256-dimensional head, and FP32 weights.
4B/9B candidates require a separate protocol/runtime update before admission.
Even a 0.8B candidate must pass that validator's checks; this skill's report is
not a subnet acceptance or promotion receipt.
