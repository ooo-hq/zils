# Bounding-box feature competition

The box track is a **synthetic-only pilot** on subnet `<NETUID>`.
Miners train a bounding-box adapter for an exact parent model, publish an
immutable bundle, and commit its location with a registered hotkey. Validators
evaluate the adapter on private exams; a confirmed challenger becomes a paid
feature king only after a reviewed configuration update.

This guide describes the feature-track implementation and planned pilot
allocation. It does not establish that a particular validator has deployed or
enabled it. Before training for rewards, check that validator's signed round
result (`GET /results/<round>`) for its `feature_tracks` configuration, exact
parent, allocation, and decisions, then check its chain receipt. An accepted
commitment alone does not establish evaluation or payment, and this pilot does
not establish real-image box quality.

## Tracks and rewards

The pilot allocation is 50% base models, 20% box features, and 30% customer
adapters. Mini/lite receive 80%/20% within both the base and box budgets:

| Competition | Share of the full weight vector |
| --- | ---: |
| Mini base | 40% |
| Lite base | 10% |
| `box/d3-mini` | 16% |
| `box/d3-lite` | 4% |
| Customer adapters | 30% |

Each base or feature track has its own king history. Among its five most recent
distinct king hotkeys, those still registered split that track's allocation
equally. A sixth-oldest king does not move back into the paid history when a
newer king deregisters. A track with no eligible earner burns its share at the
subnet owner's UID; its allocation does not enlarge another track. Customer
adapter rewards continue to use the validator's configured customer evidence.

Box parents for the pilot are:

| Track | Exact parent profile | Pinned parent revision |
| --- | --- | --- |
| `box/d3-mini` | `d3-mini-4.5b-v1` | `b54402a963373cc44713f18c9596c580e31cc68b` |
| `box/d3-lite` | `d3-lite-0.8b-v1` | `03560c26f5199c3c04129a2ccb38960c913635e0` |

The validator's frozen configuration selects the accepted parent. An
adapter trained on one parent does not move to another by changing its metadata.
Train separately for each parent and repeat qualification when the accepted parent changes.

## Install from a fresh clone

Use Linux x86_64, Git, `uv`, and a BF16-capable NVIDIA GPU with a driver that
supports the pinned CUDA 13.0 training runtime. CPU and Apple MPS training are
unsupported by this recipe. Reserve model, package and checkpoint disk space
and qualify GPU capacity for the selected profile before a long run. All
commands below run from the repository root.

```bash
git clone https://github.com/ooo-hq/zils.git
cd zils
uv venv --python 3.13 .venv-kev
uv pip install --python .venv-kev/bin/python --torch-backend=cu128 \
  -r requirements/model.txt -r requirements/rehearsal.txt \
  -r requirements/testnet.txt
uv venv --python 3.12 .venv-d3
uv pip install --python .venv-d3/bin/python --torch-backend=cu130 \
  -r requirements/d3.txt
export ZILS_D3_RUNTIME_PYTHON="$PWD/.venv-d3/bin/python"
.venv-kev/bin/python -m zils.feature_miner --help
.venv-d3/bin/python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()'
```

Use a release containing the feature-track commands. If the help command is
unavailable in your checkout, update to the validator-supported release before
continuing. The Python 3.13 environment handles the wallet and commitment;
the separate Python 3.12 environment trains the model.

Register your hotkey using the [registration guide](bittensor-registration.md)
and read the [base-tier terms](base-tier-terms.md). Provision only the hotkey on the
mining host; keep the coldkey and recovery phrases on your administration
machine. No validator address or inbound miner port is required for box
commitments.

## Generate public training data

This example creates 100 synthetic images with present and absent targets.
Seed 42 is a public training seed. These files are practice data; validators
must use separate private seeds, cases, image groups and image bytes for exams.

```bash
.venv-d3/bin/python -m scripts.box_dataset \
  --out box-public --count 100 --seed 42
```

The generator writes images, `cases.jsonl`, `expanded.jsonl`, and their hash
manifest. Every image gets five questions: four edges with 101 coordinate bins
and target presence. Convert the expanded questions to the native trainer's
one-decision-per-row format, keeping `train.jsonl` beside the image directory:

```bash
.venv-d3/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("box-public")
count = 0
with (root / "train.jsonl").open("x") as output:
    for line in (root / "expanded.jsonl").read_text().splitlines():
        row = json.loads(line)
        image = "images/" + row["image"]
        assert (root / image).is_file(), image
        example = {
            "state": row["state"],
            "questions": {"decision": {**row["question"], "label": row["label"]}},
            "images": [image],
        }
        output.write(json.dumps(example, allow_nan=False) + "\n")
        count += 1
assert count == 500, count
print(f"Wrote {count} training decisions")
PY
```

Use new output directories for another run. The generator and trainer preserve
existing output instead of overwriting it. Keep your own development evaluation
separate from training; this small example supplies a working adapter-training
path, not evidence that the resulting adapter will pass the competition.

## Train on the exact parent

The complete lite example downloads and verifies its pinned parent, then trains
one epoch of the native language-only rank-16 LoRA recipe:

```bash
.venv-d3/bin/python -m zils.d3 reference \
  --model d3-lite-0.8b-v1 --out models/box-lite-reference
.venv-d3/bin/python -m zils.d3 train \
  --data box-public/train.jsonl --reference models/box-lite-reference \
  --out box-lite-adapter --seed 553 --device cuda
.venv-kev/bin/python -m zils.feature_miner pack \
  --checkpoint box-lite-adapter --out box-lite-bundle --track box/d3-lite
```

For mini, use `--model d3-mini-4.5b-v1` when creating a separate reference,
train against that reference into a new adapter directory, and pack with
`--track box/d3-mini`. The training command reads its parent from the reference;
it has no separate `--model` flag.

Submit the adapter directly. Merging with `zils.d3 candidate` produces a base
candidate for the separate [base competition](base-candidates.md). Keep the
native trainer's checkpoint temperature at `1.0`: feature intake rejects
miner-supplied temperature calibration, changed LoRA configuration, and a
different parent. The runtime checks exact tensor keys, shapes and values when
loading the adapter.

## Publish and commit

Log in to Hugging Face with a token that can write to your chosen public model
repository. Use the interactive login; keep tokens out of shell commands,
checkpoint files and the repository:

```bash
.venv-kev/bin/hf auth login
```

Replace `OWNER/BOX_REPO`, `YOUR_WALLET` and `YOUR_HOTKEY` below with your public
model repository and registered wallet names. Change the wallet directory if
your hotkey lives elsewhere. This first command is a dry run:

```bash
.venv-kev/bin/python -m zils.feature_miner commit \
  --candidate box-lite-bundle --repo OWNER/BOX_REPO \
  --wallet-name YOUR_WALLET --wallet-hotkey YOUR_HOTKEY \
  --wallet-path "$HOME/.bittensor/wallets" --network finney --netuid <NETUID>
```

Review the parent, track, hash, repository and current commitment. **There is one
active chain commitment per hotkey across base and feature competitions.** A new
box commitment replaces that hotkey's previous base or feature entry. Use
separately registered hotkeys for simultaneous entries.

Add `--yes` to the same command to authorize uploading and writing the chain
commitment. It uploads only the adapter bundle, verifies the complete remote
bundle at the returned immutable commit, writes with the hotkey, and reads the
chain value back. The final JSON receipt includes both hashes, the Hugging Face
revision, track, parent, hotkey, UID and commitment block. Save that receipt.
For an existing upload, add `--revision FULL_40_HEX_COMMIT_SHA`; branches and tags
are rejected. An unchanged existing commitment retains its original block.

The public bundle contains exactly the packaged `adapter_config.json`,
`adapter_model.safetensors`, `model.json`, and `feature.json`. The manifest pins
`zils-feature-adapter/v1`, the track, exact registered model specification,
`box-edges-101-presence/v1`, and the ordinary checkpoint hash. Its commitment is:

```text
zilsbox1:<bundle sha256>:<40-hex revision>:<owner/repo>
```

The bundle hash covers the manifest and all checkpoint artifact bytes; the
ordinary checkpoint hash remains the evaluator's model identity. Training logs
and other unrelated files are excluded from packaging. Symlinks are rejected;
adapter artifacts are limited to 512 MiB, and each metadata file to 16 KiB.
Validators download only those pinned files and execute the pinned runtime,
never miner-supplied source. Legacy `zils1` base commitments remain separate.

A bundle hash belongs to the hotkey with the earliest commitment block. Later
copies are rejected; if several hotkeys first commit it in the same block, none
owns that hash. This is byte-identical copy detection, not proof of training
provenance or detection of all reserialized copies.

## Evaluation and crowning

The fixed decoder takes each edge's highest-probability bin, orders opposite
edges, and returns a box when target-presence probability is at least 0.5.
Presence confidence estimates whether the target exists; it is not predicted
IoU. Every case receives all five questions, including absent cases.

The default pilot gates require all of the following on each draw:

- Equal-family mean IoU on present targets of at least **0.40**.
- Mean IoU gain over the incumbent of at least **0.02**, with a paired,
  family-stratified image-group bootstrap 95% lower bound **above zero**
  (2,000 resamples).
- No family's present-target mean IoU falling more than **0.02**.
- False-positive rate on absent targets of at most **0.10**.
- Binary presence Brier score no more than **0.005 worse** than the incumbent.

The same immutable candidate must qualify on two disjoint private draws at
least 12 hours apart. The validator audits case, group and image separation and
freezes its configuration and decisions. Public results contain aggregate
metrics and decisions; exam labels, image IDs, raw predictions and private paths
remain private. The frozen policy in the signed result is authoritative if an
operator chooses different thresholds.

Passing both draws produces a challenger for review. A reviewed configuration
update makes its bundle the feature incumbent and adds its hotkey to that
track's king history; exam eligibility alone pays nothing. Feature crowning
does not modify a base model or automatically graduate a feature into one.
Base graduation requires a separate reviewed release, regression evaluation,
and real-image qualification. Synthetic shape results alone do not establish
general-purpose localization, robustness, or customer-serving availability.

Operators can prepare a reviewed crown after both qualifying draws:

```bash
.venv-kev/bin/python -m scripts.crown_feature \
  --config validator.json --track box/d3-mini \
  --decision feature-tracks/box/d3-mini/decision.json \
  --candidate winning-bundle --snapshot current-chain-snapshot.json \
  --out proposed-validator.json
```

This command audits both private draws, verifies the registered claimant and exact
parent, recomputes both comparisons from the private prediction caches, and
checks the 12-hour interval using actual evaluation timestamps. It creates a new
configuration with only the feature incumbent, king history and reference path
changed. Review and install the proposal at the original configuration location
so existing relative paths retain their meaning. It performs no installation,
chain write, base release or serving change.
