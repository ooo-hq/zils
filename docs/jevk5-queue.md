# JevK5 training queue (retired)

JevK5 4B (`jevk5-4b-v0.3`) was the hosted queue's first pinned text profile. It was
retired on 17 October 2026 after [d3-mini](../README.md) became the only served text
base. This page is a historical record; nothing in the repository creates, trains,
transfers, serves or evaluates JevK5 checkpoints any more.

## What remains readable

- `zils.models.metadata()` still parses a frozen JevK5 `model.json` (identity,
  kind and temperature) so historical reports and release manifests stay
  legible. Every other operation rejects the identity as unsupported.
- Job manifests, releases and evidence that record the profile keep their bytes.
  The pinned identities were model `alibiserikbay/JevK5` at revision
  `c4f7fdb3aeab5582336406e78d3bef11bf98833d` and upstream runtime
  `f26426d16f59e8bbe1470e5b162cc89329e29b29`; the shared release was
  `zils-jevk5-v0.3-r1`.
- `zils/jevk5.py` keeps its name because job and trainer hashes reference it. It
  now holds only the attention-only LoRA trainer shared by the
  [H2O](h2o-queue.md) and d3 profiles.

## What was removed

The JevK5 profile and reference command (`python -m zils.jevk5 reference`), its
CUDA/MPS decision model, the Apple silicon miner guide and its MPS training test,
the upstream `jevk5` runtime requirement, the release-manifest helper and its
download script, and the `ZILS_JEVK5_BASE_DIR` override.

## Historical contract

JevK5 jobs accepted choice, true/false and ordinal-score decisions with at most
16 outcomes and 2,048 prompt tokens per example. Candidates contained
`adapter_config.json`, `adapter_model.safetensors` and `model.json`; accepted
downloads added `release.json`. Training used one epoch, attention-only rank-16
LoRA, alpha 32, dropout 0.05, learning rate 0.00002, batch size one and
accumulation of four examples. The same recipe continues for the current profiles
through the shared trainer. For current miner setup read
[queued miners](queue-miners.md); the current base is d3-mini
(`d3-mini-4.5b-v1`, pins in `zils/d3-pins.json`); the retired
[H2O model contract](h2o-queue.md) covers frozen H2O jobs only.
