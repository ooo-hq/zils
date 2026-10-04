# JevK5 training queue

The hosted queue supports pinned JevK5 4B weights as its starting model. New
datasets are bound to a model identity when the processor validates them. The
model identity and base revision are included in the hashed job manifest, worker
assignment, customer job response and accepted release. Existing manifests that
predate model identities retain the original Kev 0.8B contract.

This changes the training workflow. It does not add a hosted prediction API or
automatically approve customer-data exports or worker assignments. The local
and testnet fleet remains on its existing Kev contract.

## Install and create the reference

Use Linux or WSL 2, Python 3.13, a BF16-capable NVIDIA GPU, its CUDA driver and
the repository's pinned model/signing dependencies. The implementation was
checked on an RTX 4090. Allow at least 25 GB free disk for models and packages;
GPU requirements depend on input length. CPU and MPS execution are not supported
by this JevK5 queue runtime. The coordinator API itself needs no GPU.

From a fresh clone:

```sh
git clone https://github.com/ooo-hq/zils.git
cd zils
uv venv --python 3.13 .venv-kev
uv pip install --python .venv-kev/bin/python --torch-backend=cu128 \
  -r requirements/model.txt -r requirements/rehearsal.txt
.venv-kev/bin/python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()"
.venv-kev/bin/python -m fez.jevk5 reference --out models/jevk5-reference
```

The reference command downloads and verifies the pinned model, tokenizer and
configuration. It creates a small `model.json` reference rather than duplicating
the full base weights. The model is `alibiserikbay/JevK5`, revision
`c4f7fdb3aeab5582336406e78d3bef11bf98833d`; the upstream runtime is pinned at
`f26426d16f59e8bbe1470e5b162cc89329e29b29` in `requirements/model.txt`.

For an already downloaded model, set `FEZ_JEVK5_BASE_DIR` to its directory and
use `--no-download`. This override must pass the same weight and tokenizer
checksums; it cannot select another model. Model workers run offline.

## Configure the services

Complete the Supabase resources, customer Auth, HTTPS edge and protected server
environment setup in [Supabase training](supabase-training.md). Set
`FEZ_TRAINING_MODEL=jevk5-4b-v0.3` in the API and processor environments. Start
the API with the existing `python -m fez.coordinator serve` command and the
processor with:

```sh
.venv-kev/bin/python -m fez.coordinator process \
  --state .private/queue-processor --reference models/jevk5-reference --device cuda
```

Use a persistent service supervisor for deployment. The processor refuses to
start if its reference does not match the configured active model. The read-only
`GET /v1/config` response identifies that active model; it is configuration, not
a worker-health or inference-availability signal.

On an approved worker, create the protected hotkey/wallet configuration described
in the queue guide, download the same reference, and run:

```sh
.venv-kev/bin/python -m miner.queue \
  --config .private/queue-miner.json --state .private/queue-miner \
  --reference models/jevk5-reference --device cuda
```

The worker receives only approved training exports and signed storage URLs. It
must not receive the processor's Supabase credential or calibration/test data.
Run worker and processor under separate operating-system identities if they share
a machine, with separate private state directories and environment files.

When multiple identities share one GPU, configure the same `FEZ_COMPUTE_LOCK`
path for them. Pre-create it in an administrator-owned directory with group
read/write access for the service identities; do not make the parent directory
writable by workers. Without this override the existing per-user/device lock is
used. Other applications need their own resource coordination.

## Data and training contract

This version supports choice, true/false and ordinal-score decisions with at
most **16 outcomes** and **2,048 prompt tokens per example**. Validation checks
those limits before worker approval. Inputs are never silently truncated. The
website's decision text is supplied as JevK5's criterion when the question has
no explicit `instructions`; the original evidence remains present.

Training uses one epoch, attention-only rank-16 LoRA, alpha 32, dropout 0.05,
learning rate 0.00002, batch size one and accumulation of four examples. Partial
final batches use their actual size. The loss is cross entropy over the allowed
answer-letter logits, not generated text. Base weights and the language-model
head remain frozen. The candidate's raw temperature is 1.0; the validator fits
the baseline and candidate separately using only calibration labels.

JevK5 candidates contain `adapter_config.json`, `adapter_model.safetensors`, and
`model.json`; accepted downloads add `release.json`. Training keeps FP32 LoRA
parameters and saves them in BF16, producing an approximately 29 MB adapter
that fits a 50 MB storage upload limit. The loader also accepts earlier FP32
adapters. The validator instantiates
its own fixed LoRA architecture and checks the exact tensor names, shapes, dtypes
and finiteness before loading. It never executes miner-provided model code or
loads a miner-chosen base. Artifact hashes include the model metadata and weights.

Calibration, positive uniform skill, the customer's accuracy target and strict
Brier improvement remain required for an accepted trained model. A strong base
may leave no qualifying fine-tuned candidate. `no_qualifying_model` remains a
valid completion; the queue does not relabel the unmodified base as a trained
customer model.

## Existing jobs and rollback

Keep existing job rows, manifests, references and release files unchanged. The
API determines old download filenames from each job's model, not today's active
setting. To evaluate older pending jobs with the new processor, also provide
`--additional-reference models/reference` and retain that Kev base cache. Workers
must use a reference matching their assigned job; an incompatible worker refuses
the assignment instead of training another model.

Before changing the active model, finish or inventory in-flight work and prepare
the appropriate workers and references. Change the API environment and primary
processor reference together. Rollback uses the same procedure with
`FEZ_TRAINING_MODEL=kev-0.8b-v1` and the Kev reference, retaining JevK5 as an
additional reference if any JevK5 jobs still need evaluation. Do not change a
prepared job's identity or rewrite its historical result.

## Verification

`make check` covers both model formats through signed upload, training fixtures,
calibration, acceptance and version-correct downloads, plus the original local
and testnet contracts. Fixture tests establish workflow behavior, not model
quality. Verify actual GPU training, artifact reload and held-out evaluation on
authorized data before operational cutover. Neither a successful workflow nor
a synthetic accuracy score establishes customer-task reliability.
