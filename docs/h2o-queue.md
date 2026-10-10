# Pinned H2O text adapters

New text training jobs default to `h2o-lightning-4b-v1.2.3`:

| Component | Pin |
| --- | --- |
| Base | `h2oai/h2o-lightning-4b` |
| Revision | `acaf0d4ea251e54de928c75ef4352670d33192d3` (v1.2.3) |
| Architecture | `Qwen3_5ForConditionalGeneration`, text path only |
| GPU runtime | Python 3.12, PyTorch 2.10.0 CUDA 13.0, Transformers 5.17.0, PEFT 0.21.0 |
| Kernels | FLA / fla-core 0.5.2, causal-conv1d 1.7.0, SDPA |
| Precision | BF16 backbone and serialized adapter, FP32 answer projection and live LoRA parameters |

The [miner setup](queue-miners.md) creates the isolated runtime. The coordinator
and queue client retain Python 3.13. Set `ZILS_H2O_RUNTIME_PYTHON` for both miners
and validators; neither silently substitutes the older JevK5 runtime.
Use an NVIDIA driver compatible with CUDA 13.0. The runtime rejects missing fast
kernels and different pinned package versions before loading a model.

`zils/h2o-pins.json` records file checksums, including the upstream prompt code
and serving configuration. The runtime loads only this verified code from the
operator's base cache. Miner artifacts cannot supply executable code or choose
the base model. To reuse a verified download, set `ZILS_H2O_BASE_DIR` to its
directory. Every required file is still checked.

## Training and scoring

The supported text contract is choice, true/false, or score with up to 255
outcomes and 2,048 full prompt tokens. Oversized inputs and images are rejected;
examples and options are never silently truncated. Image jobs retain ImaJev.

Prompts use H2O's native non-thinking chat template and `Answer:` prefix. Native
temperatures are 0.75 for choice, 0.8 for true/false, and 0.65 for score; the
native true/false probability floor is 0.801. These settings ignore ambient
`SHIM_*` overrides. The validator fits a further scalar temperature to that
native distribution using only the job's held-out calibration split. Its stored
`temperature=1.0` means no additional calibration, not replacement of native
temperatures. True/false answers retain H2O's true-first internal option order.

Training reuses the queue's one-epoch AdamW recipe: learning rate 0.00002, batch
one, accumulation four, rank-16 LoRA, alpha 32, dropout 0.05. Only the 152 language
attention/gated-delta projections are adapted. Vision weights, embeddings and
the output head stay frozen. Cross entropy uses the native type temperature.
Partial final accumulation windows retain their actual size.

Candidates contain `adapter_config.json`, `adapter_model.safetensors`, and
`model.json`. Metadata binds the entire H2O profile; loaders construct a fixed
architecture and verify tensor names, shapes, dtypes and finiteness. A qualifying
adapter must still improve held-out Brier loss and meet the customer's accuracy
target. Choosing H2O does not guarantee that fine-tuning improves every task.

An RTX 4090 check on 2026-10-10 trained six synthetic examples covering all three
question types at 2,040–2,048 prompt tokens. Two optimizer steps took 27.56 seconds
and peaked at 9.53 GiB of allocated GPU tensor memory. All 304 language-only LoRA
tensors were finite, and the saved adapter reloaded and scored all three types.
This establishes training and reload compatibility on that hardware; task quality
and each worker's capacity still require independent evaluation and qualification.

## Existing jobs

JevK5 and Kev identities, references and artifact formats remain supported.
Keep their references installed while their jobs finish. A worker may supply
multiple `--reference` arguments; it claims only the profiles separately approved
by the operator. Never rewrite a JevK5 job or adapter as H2O.

H2O requires fresh worker qualification and runtime hashes. Existing JevK5
capacity/quality reports do not transfer. Hosted migration, release-serving
isolation and rollback are described in
[the platform rollout guide](https://github.com/ooo-hq/zils-platform/blob/main/docs/h2o-rollout.md).
