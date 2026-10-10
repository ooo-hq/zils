# JevK5 training queue

The hosted queue supports pinned JevK5 4B weights as its starting model. New
datasets are bound to a model identity when the processor validates them. The
model identity and base revision are included in the hashed job manifest, miner
assignment, customer job response and accepted release. Existing manifests that
predate model identities retain the original Kev 0.8B contract.

This changes the training workflow. It does not add a hosted prediction API or
automatically approve customer-data exports or miner assignments. The local
and testnet fleet remains on its existing Kev contract.

## Install and create the reference

Use Linux or WSL 2, Python 3.13, a BF16-capable NVIDIA GPU, its CUDA driver and
the repository's pinned model/signing dependencies. The implementation was
checked on an RTX 4090. Allow at least 25 GB free disk for models and packages;
GPU requirements depend on input length. Apple silicon miners can use
[the MPS training setup](mac-miners.md); it produces the same adapter format for
independent CUDA evaluation. CPU training remains unsupported. The coordinator
API itself needs no GPU.

From a fresh clone:

```sh
git clone https://github.com/ooo-hq/zils.git
cd zils
uv venv --python 3.13 .venv-kev
uv pip install --python .venv-kev/bin/python --torch-backend=cu128 \
  -r requirements/model.txt -r requirements/rehearsal.txt
.venv-kev/bin/python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()"
.venv-kev/bin/python -m zils.jevk5 reference --out models/jevk5-reference
```

The reference command downloads and verifies the pinned model, tokenizer and
configuration. It creates a small `model.json` reference rather than duplicating
the full base weights. The model is `alibiserikbay/JevK5`, revision
`c4f7fdb3aeab5582336406e78d3bef11bf98833d`; the upstream runtime is pinned at
`f26426d16f59e8bbe1470e5b162cc89329e29b29` in `requirements/model.txt`.

For an already downloaded model, set `ZILS_JEVK5_BASE_DIR` to its directory and
use `--no-download`. This override must pass the same weight and tokenizer
checksums; it cannot select another model. Model processes run offline.

## Configure the services

Follow [queued miner setup](queue-miners.md) to connect an approved miner.
The miner receives training data and signed upload URLs; never install a
Supabase service key on a miner. Use separate operating-system identities and
state directories when miners share hardware with a trusted evaluator.

When identities share one GPU, set the same `ZILS_COMPUTE_LOCK` path in an
administrator-owned directory with group read/write access. Miners must not be
able to replace its parent directory. Without this setting, locking is per user
and device.

Hosted API and processor deployment lives in
[zils-platform](https://github.com/ooo-hq/zils-platform/blob/main/docs/jevk5-queue.md#configure-the-services).
Bittensor validation is a separate [testnet workflow](validators.md).

## Data and training contract

This version supports choice, true/false and ordinal-score decisions with at
most **16 outcomes** and **2,048 prompt tokens per example**. Validation checks
those limits before miner approval. Inputs are never silently truncated. The
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

Retain the reference and base cache matching every in-flight job. Miners reject
an assignment whose frozen model identity differs from their reference. Never
rewrite a prepared job or historical result to match a new model setting.
Hosted processor transitions are documented in the platform repository.

## Verification

`make check` covers model identities, signed submissions, training fixtures,
calibration, acceptance, and the original local/testnet contracts. Hosted
upload-to-result integration is tested in zils-platform. Fixture results do not
establish model quality or GPU capacity; qualify actual hardware before joining.
