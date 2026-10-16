# d3-mini Base Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `vllm-sr/d3-mini` a pinned Zils model profile that customers train text and image adapters on, that the platform serves, that a task can move to from H2O, and that the base tier measures against, so the H2O and Imajev profiles can be retired later.

**Architecture:** A new profile row in the subnet's model table plus a pins file, and a runtime module `zils/d3.py` ported from the research engine with the same `DecisionModel` contract as `zils/h2o.py`, so the existing trainer, runner, validator, releases and adapter server work through `models.text_runtime(model)`. The roughly 25 places that branch on `== models.H2O` become small table lookups only where d3 needs them. Images and video travel through the same text runtime, so the platform keeps its asset store and drops nothing yet. Cross-profile upgrades score the frozen old release with its own runtime and the new candidate with the new one, on the same test split.

**Tech Stack:** Python 3.13 (`.venv-kev`) for subnet and platform code; Python 3.12 pinned runtime (`.venv-d3`: torch 2.10.0+cu130, torchvision 0.25.0+cu130, transformers 5.17.0, peft 0.21.0, flash-linear-attention 0.5.2, causal-conv1d 1.7.0, pillow 11.3.0, opencv-python-headless 5.0.0.93); the vendor runtime `d3_runtime.py` shipped inside the model directory; systemd; PostgreSQL migrations; `unittest`, `ruff`.

**Spec:** `skills/.private/option5-spec-2026-10-16.md` (workspace, private) for the base tier, in particular §5.3 confirmation draw, §5.4 promotion into a Zils-hosted repo, §8 weights hosting, §10 two channels and the adapter regression suite; `output/d3-evidence-2026-10-16/report.md` for the evidence; the "Base direction, 16 October 2026" paragraph of `skills/.private/project-context.md` for the decision. In one paragraph: d3-mini becomes Zils's base before mainnet; existing H2O and Imajev releases keep serving until retired; a customer's task may stay on its base, be retrained on d3, or be upgraded to d3 by the existing upgrade route; the round-one exam is re-frozen with stock d3 as incumbent; miners' hardware pins change only by the new runtime environment.

## How this plan relates to Option 5

The B300 run on 16 October scored stock d3 against the stock-H2O round-one incumbent with the subnet's own `base_round.decide()`, the Option 5 round-one decision rule. d3 passed it: eligible, pending winner. So d3 is the first base to clear the Option 5 gate. It is **not** a round-one candidate under the spec's §2–§3 artifact contract, which fixes H2O's architecture, tokenizer, prompt, answer-token readout, 2,048 cap and runtime pins; d3 has its own prompt, code readout, runtime and 4,096 cap. The spec itself foresaw this: §10 makes a channel a movable alias over immutable profiles, and a new profile is exactly what d3 is. This plan therefore:

- lands d3 as a new immutable profile and moves the `latest` channel to it (Phase 1–2), which is the mechanism a future miner-won base will use too;
- generalises the candidate tooling (`candidate_files`, `promote_base.py`, the regression suite) so that from round two miners compete on d3 under the same contract shape (Task 6);
- applies the spec's own promotion conditions to d3 before `stable` can follow: a §5.3 confirmation draw on a disjoint sealed set at least 12 hours later, and the §10 adapter regression suite run across profiles (Task 12). One draw is not a promotion.
- hosts the pinned weights in a Zils-owned Hugging Face repo as §5.4 and §8 require for any promoted base (Task 1). The org name is the one §8 open item still undecided; it must be fixed before Task 1 starts.

Out of this plan: §10 channel aliases as a registry mechanism (the platform implements channels as configuration today, PR #3), the "follow stable" automatic upgrade mode, and a public round.

## Global Constraints

- Nothing in `zils/` or `miner/` may import `zils_platform`, `zils_sdk`, `stripe` or `boto3` (`tests/test_repository_boundary.py`).
- Python target 3.13, ruff line length 100, rules `E4,E7,E9,F,I`, isort `combine-as-imports`; `make check` must pass in the subnet and in the platform.
- **The H2O and Imajev spec dicts in `zils/models.py` do not change by a single byte.** Jobs and releases store those dicts verbatim and `validate_spec` compares them for equality; a change would orphan every existing release ("Adapter belongs to a different base model").
- d3 pins are authoritative: the 25 pinned files and their hashes come from `skills/.private/d3-evidence-20261016/download-verification.json` (vendor revision `a475367de0d8133a3c1a2a0dc12303d89cfa2ab4`); package versions above. The profile's `base` is the Zils-owned Hugging Face mirror of those files and `base_revision` is the mirror's commit; the pins file records the vendor repo and revision for provenance. Every profile's `base_revision` must be unique (`zils/core.py:210-214` selects the profile by revision).
- **Decision required before Task 1:** the Zils Hugging Face organisation name (Option 5 spec §8, still open). Write it into the spec dict below before executing.
- The d3 serving and training cap is 4,096 input tokens; the runtime refuses longer inputs and never truncates.
- No labels are staged on remote hosts; training rows carry labels only inside `questions.decision.label` as today.
- Secrets never enter code or docs: tokens come from `*_TOKEN` env names, hosts from unit files.

## Review Focus

1. An H2O or Imajev release served, published or upgraded after this plan lands behaves exactly as before (Task 1 test: `profile_identity(H2O)` and `spec(H2O)` unchanged; Task 9 test: an H2O release still activates).
2. A d3 adapter checkpoint uploaded to an H2O job, or vice versa, is refused at every gate: `metadata` profile check, `core.evaluate` "checkpoint belongs to a different model", adapter server "Adapter belongs to a different base model" (Task 1 and Task 9 tests).
3. A training row with an image reaching the H2O or JevK5 runtime is refused, and a d3 row with five images or a non-canonical asset is refused before any GPU work (Task 10 tests).
4. A cross-profile upgrade whose previous release cannot be scored (old runtime missing on the processor) fails the job with a clear error instead of accepting the candidate against no baseline (Task 5 test).
5. `ZILS_D3_RUNTIME_PYTHON` unset on a node that claims the d3 profile: the claim is refused by `miner_presence` identity, and `core.evaluate` raises the `settings.required` error naming the variable rather than running the wrong interpreter (Task 3 test).

---

## Phase 1 — d3 as a pinned text profile (subnet, `ooo-hq/zils`)

### Task 1: Profile entry, pins and table-driven lookups

**Files:**
- Create: `zils/d3-pins.json`
- Create: `requirements/d3.txt`
- Modify: `zils/models.py:9-14` (constants), `:90-110` (`profile_identity`), `:130-136` (`candidate_files`), `:148-176` (`metadata`), `:203-212` (`write_metadata`), `:249-254` (`text_runtime`)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `skills/.private/d3-evidence-20261016/download-verification.json` (workspace, outside the repo) for file hashes; `requirements/h2o.txt` as the template.
- Produces: `models.D3 = "d3-mini-4.5b-v1"`; `models.TEXT_ADAPTER_MODELS = (JEVK5, H2O, D3)`; `models.TEXT_RUNTIME_MODULES = {H2O: ".h2o", D3: ".d3"}` (default `.jevk5`); `models.RUNTIME_PYTHON_ENV = {H2O: "ZILS_H2O_RUNTIME_PYTHON", D3: "ZILS_D3_RUNTIME_PYTHON"}`; `models.PINS_FILES = {H2O: "h2o-pins.json", D3: "d3-pins.json"}`; `models.runtime_python_env(model) -> str | None`; `models.pins(model) -> dict`; `models.D3_CONTRACT_FILES`.

The d3 spec, copied exactly:

```python
D3 = "d3-mini-4.5b-v1"
SPECS[D3] = {
    "id": D3,
    "name": "d3-mini 4.5B",
    "base": "zils-ai/d3-mini-4.5b-v1",          # the Zils mirror, created in Step 3a
    "base_revision": "<mirror commit sha>",             # from Step 3a; never the vendor's a475367…
    "runtime_revision": "d3-vendor-runtime-a475367/v1",
    "prompt": "d3-native-v1",
    "max_input_tokens": 4096,
    "recipe": "d3-language-lora16-adamw/v1",
    "calibration": "native-then-heldout-temperature/v1",
    "inputs": {"images": 4, "video": True},
}
```

`inputs` is declared now, even though images arrive in Task 10, because the spec dict is the profile identity and must never change afterwards.

- [ ] **Step 1: Write the failing tests** in `tests/test_models.py`

```python
def test_d3_profile_is_pinned_and_unique(self):
    spec = models.spec(models.D3)
    self.assertEqual(spec["base_revision"], models.pins(models.D3)["revision"])  # the Zils mirror commit
    self.assertEqual(models.pins(models.D3)["vendor"], {"repo": "vllm-sr/d3-mini", "revision": "a475367de0d8133a3c1a2a0dc12303d89cfa2ab4"})
    self.assertEqual(spec["max_input_tokens"], 4096)
    self.assertEqual(spec["inputs"], {"images": 4, "video": True})
    revisions = [models.spec(m)["base_revision"] for m in models.SPECS]
    self.assertEqual(len(revisions), len(set(revisions)))
    self.assertIn(models.D3, models.TEXT_ADAPTER_MODELS)

def test_d3_identity_hashes_pins_file(self):
    pins = models.pins(models.D3)
    self.assertEqual(pins["python"], "3.12")
    self.assertEqual(pins["packages"]["torchvision"], "0.25.0+cu130")
    self.assertEqual(len(pins["files"]), 25)  # every manifest file except README.md and assets/*
    identity = models.profile_identity(models.D3)
    self.assertEqual(set(identity), {"profile_sha256", "runtime_sha256"})
    self.assertNotEqual(identity["runtime_sha256"], models.profile_identity(models.H2O)["runtime_sha256"])

def test_h2o_and_imajev_identities_are_unchanged(self):
    # Frozen values: compute once on main before this change and paste here.
    self.assertEqual(models.profile_identity(models.H2O)["profile_sha256"], "<paste>")
    self.assertEqual(models.profile_identity(models.IMAJEV)["profile_sha256"], "<paste>")

def test_d3_metadata_requires_profile(self):
    with tempfile.TemporaryDirectory() as tmp:
        models.write_metadata(tmp, model=models.D3)
        self.assertEqual(models.metadata(tmp)["profile"], models.spec(models.D3))
        (Path(tmp) / "model.json").write_text(json.dumps({"version": models.VERSION, "model": models.D3, "kind": "adapter", "temperature": 1.0}))
        with self.assertRaisesRegex(ValueError, "pinned profile"):
            models.metadata(tmp)

def test_text_runtime_and_interpreter_lookup(self):
    self.assertEqual(models.text_runtime(models.D3).__name__, "zils.d3")
    self.assertEqual(models.runtime_python_env(models.D3), "ZILS_D3_RUNTIME_PYTHON")
    self.assertEqual(models.runtime_python_env(models.H2O), "ZILS_H2O_RUNTIME_PYTHON")
    self.assertIsNone(models.runtime_python_env(models.JEVK5))
    self.assertEqual(models.candidate_files(models.D3), models.JEVK5_FILES)
    self.assertEqual(models.candidate_files(models.D3, kind="candidate")[-len(models.D3_CONTRACT_FILES):], models.D3_CONTRACT_FILES)
```

Before writing the frozen-identity test, run on unchanged `main`:

```bash
.venv-kev/bin/python -c "from zils import models; print(models.profile_identity(models.H2O)['profile_sha256'], models.profile_identity(models.IMAJEV)['profile_sha256'])"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `make test PYTHON=.venv-kev/bin/python` (or `.venv-kev/bin/python -m unittest tests.test_models -v`)
Expected: FAIL with `AttributeError: module 'zils.models' has no attribute 'D3'`; `text_runtime` test fails with `ModuleNotFoundError: zils.d3` (Task 2 supplies the module; leave that one failing until then).

- [ ] **Step 3a: Create the Zils mirror on Hugging Face** (operator, once)

Upload exactly the 25 pinned files (every manifest file except `README.md` and `assets/*`) from the verified local copy at `skills/.private/d3-comparison-20261011/model/` to a new public repo `zils-ai/d3-mini-4.5b-v1`, with a `README.md` that states the vendor repo, vendor revision and Apache-2.0 licence. Record the resulting commit sha as `base_revision`. Verify with `hf download zils-ai/d3-mini-4.5b-v1 --revision <sha>` into a scratch directory and `sha256sum` against `download-verification.json`: all 25 must match.

- [ ] **Step 3b: Generate `zils/d3-pins.json`**

One-off script, run from the workspace root, not committed (`MIRROR_SHA` is the Step 3a commit):

```python
import json
v = json.load(open("skills/.private/d3-evidence-20261016/download-verification.json"))
files = {k: h for k, h in v["files_sha256"].items() if k != "README.md" and not k.startswith("assets/")}
json.dump({"revision": MIRROR_SHA, "vendor": {"repo": "vllm-sr/d3-mini", "revision": v["revision"]},
           "files": files, "python": "3.12",
           "packages": {"torch": "2.10.0+cu130", "torchvision": "0.25.0+cu130", "transformers": "5.17.0",
                        "peft": "0.21.0", "flash-linear-attention": "0.5.2", "causal-conv1d": "1.7.0",
                        "pillow": "11.3.0", "opencv-python-headless": "5.0.0.93"},
           "cuda": "13.0", "attention": "noncausal_full_attention", "backbone_dtype": "bfloat16",
           "readout_dtype": "float32", "temperature": 1.0, "fast_path": False},
          open("zils/zils/d3-pins.json", "w"), indent=2, sort_keys=True)
```

Expect 25 files. `D3_CONTRACT_FILES` is the pinned non-weight set: `("config.json", "decision_config.json", "preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "readout.safetensors", "d3_runtime.py", "d3_format.py", "d3_kernels.py", "d3_fast.py", "d3_engine.py", "modeling_d3.py", "pipeline_d3.py", "LICENSE", "NOTICE")`.

- [ ] **Step 4: Write `requirements/d3.txt`**

Copy `requirements/h2o.txt`, add `torchvision==0.25.0+cu130` next to torch on the same index, and `pillow==11.3.0`, `opencv-python-headless==5.0.0.93`. Keep the file header comment pattern.

- [ ] **Step 5: Implement the table changes in `zils/models.py`**

- Add `D3`, the spec above, `D3_CONTRACT_FILES`, and `D3_CANDIDATE_FILES = ("model.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json", *D3_CONTRACT_FILES)`.
- `TEXT_ADAPTER_MODELS = (JEVK5, H2O, D3)`; add `TEXT_RUNTIME_MODULES`, `RUNTIME_PYTHON_ENV`, `PINS_FILES`.
- `pins(model) -> dict`: `json.loads(Path(__file__).with_name(PINS_FILES[model]).read_text())`.
- `runtime_python_env(model)`: `RUNTIME_PYTHON_ENV.get(model)`.
- `profile_identity`: replace `if model == H2O: runtime = json.loads(...)` with `if model in PINS_FILES: runtime = pins(model)`. The H2O result is byte-identical because it already hashed the whole file.
- `candidate_files`: `kind == "candidate"` allowed for `model in (H2O, D3)`; return `CANDIDATE_FILES` for H2O, `D3_CANDIDATE_FILES` for D3; adapter dict adds `D3: JEVK5_FILES`. Error text becomes "base candidates are supported for the H2O and d3 profiles only".
- `metadata`: candidates allowed for `(H2O, D3)`; the `elif value["model"] == H2O` branch becomes `elif value["model"] in PINS_FILES` with message "profile metadata differs from the pinned profile" (the test regex `pinned profile` matches old and new text).
- `write_metadata`: `if model in (*PINS_FILES, IMAJEV): value["profile"] = spec(model)`.
- `text_runtime`: `importlib.import_module(TEXT_RUNTIME_MODULES.get(model, ".jevk5"), __package__)`.

- [ ] **Step 6: Run the tests**

Run: `.venv-kev/bin/python -m unittest tests.test_models -v`
Expected: all PASS except `test_text_runtime_and_interpreter_lookup` (needs `zils/d3.py`, Task 2). `make lint` passes.

- [ ] **Step 7: Commit**

```bash
git checkout -b codex/d3-base-profile
git add zils/models.py zils/d3-pins.json requirements/d3.txt tests/test_models.py
git commit -m "Pin the d3-mini profile and make model lookups table-driven"
```

### Task 2: `zils/d3.py` runtime module

**Files:**
- Create: `zils/d3.py`
- Test: `tests/test_d3.py`
- Reference (read, do not import): `skills/.private/d3-evidence-20261016/adapter_engine.py` and `exam_eval.py` (workspace)

**Interfaces:**
- Consumes: `models.D3`, `models.pins(models.D3)`, `models.spec(models.D3)`, `miner_grading.workload_profile(lengths, model=D3)` (Task 3 raises its cap).
- Produces, matching `zils/h2o.py` name for name so `jevk5.train`, `jevk5_runner.py`, `runtime.prepare_base`, `qualify_node` and the platform adapter server work unchanged:
  - `MAX_TOKENS = 4096`, `TARGETS` (the nine module names), `PINS = models.pins(models.D3)`
  - `base_path(download=False) -> Path` honouring `ZILS_D3_CANDIDATE_BASE` (validated by `validate_candidate`) then `pinned_base_path`
  - `pinned_base_path(download=False) -> Path`: `ZILS_D3_BASE_DIR` or `snapshot_download("vllm-sr/d3-mini", revision=PINS["revision"], allow_patterns=list(PINS["files"]))`, then sha256 of every pinned file
  - `verify_runtime() -> None`: Python 3.12, package versions from `PINS["packages"]`, CUDA from `PINS["cuda"]`
  - `class Encoder(root)`: loads `d3_runtime` from `root` (`sys.path` insert of the model directory, as the research engine does), builds `D3.from_pretrained(root, device=..., batch_size=1, max_length=MAX_TOKENS, verify="full", local_files_only=True, permutation_average=False)` lazily on first use (`load(device)`), and `encode(state, question, images=(), videos=()) -> (ids: list[int], names: list[str])`; in this task images or videos raise `ValueError("d3 image inputs arrive in a later release")` (Task 10 replaces the body)
  - `validate_inputs(splits) -> dict` as in h2o.py but tokenising with the d3 encoder and `model=models.D3`
  - `validate_adapter_config(checkpoint, targets)`: r=16, alpha=32, dropout=0.05, `base_model_name_or_path == "vllm-sr/d3-mini"`, `revision == spec["base_revision"]`
  - `validate_candidate(checkpoint)`: `kind == "candidate"`, `D3_CANDIDATE_FILES` present, contract files hash-equal to the pins
  - `class DecisionModel(checkpoint, device, train=False)` with `.meta`, `.device`, `.encoder`, `.tokenizer`, `.base` (the vendor runtime's `backbone`), `.readout` (FP32 tensor, 255 rows), `.peft`, and methods `encode(state, question)`, `logits(ids, count) -> Tensor[count]` (hidden state of the last token `@ readout[:count].T`, float32, as `adapter_engine.Engine.logits`), `probabilities(ids, keys, question, temperature) -> dict`, `predict(state, question) -> dict[str, float]`; `train=True` wraps the backbone's language layers in a rank-16 LoRA with exactly 152 targets and 304 trainable tensors, or loads the checkpoint's adapter when present
  - `parity(model: DecisionModel, cases: list[dict]) -> dict`: compares `predict` with `d3_runtime.D3.probabilities` on the same token ids; raises if any max probability difference ≥ 1e-6 or any verdict differs; returns `{"cases": n, "max_difference": x}`
  - `main()`: `python -m zils.d3 reference --out DIR [--no-download]` (writes a reference checkpoint directory with `model.json` `kind="base"`), `train --data --reference --out --seed [--device cuda]` delegating to `jevk5.train(args, model_id=models.D3)`, and `parity --reference DIR --cases FILE`

Vendor runtime facts the implementer needs (from the research engine): `runtime.fast is None`, `runtime.attention_mode == "noncausal_full_attention"`, `runtime.temperature == 1.0`, `runtime.readout_dtype == "float32"`, `runtime.prompt == "d3"`, `runtime.backbone_type == "qwen3_5"`; `runtime.text(state, normalized.rendered)` renders the prompt; `normalize_question(question)` returns `.kind`, `.keys`, `.rendered`; `runtime.probabilities([ids], [count])` is the reference path. The backbone must have `use_cache=False` on both `config` and `config.text_config`. The LoRA target filter is `name.startswith("language_model.layers.") and name.rsplit(".", 1)[-1] in TARGETS`.

- [ ] **Step 1: Write the failing tests** in `tests/test_d3.py`, following `tests/test_h2o.py:40-56` and `:129-168`

```python
class D3ModuleTest(unittest.TestCase):
    def test_pins_and_constants(self):
        self.assertEqual(d3.MAX_TOKENS, 4096)
        self.assertEqual(d3.PINS["revision"], models.spec(models.D3)["base_revision"])
        self.assertEqual(len(d3.TARGETS), 9)

    def test_pinned_base_path_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ZILS_D3_BASE_DIR": tmp}):
            for name in d3.PINS["files"]:
                (Path(tmp) / name).parent.mkdir(parents=True, exist_ok=True)
                (Path(tmp) / name).write_bytes(b"x")
            with self.assertRaisesRegex(OSError, "d3 base file .* differs from its pin"):
                d3.pinned_base_path()

    def test_verify_runtime_rejects_wrong_torchvision(self):
        with patch("zils.d3.importlib.metadata.version", side_effect=lambda p: "0.23.0" if p == "torchvision" else d3.PINS["packages"][p]), \
             patch("zils.d3.sys.version_info", (3, 12, 12)):
            with self.assertRaisesRegex(OSError, "torchvision"):
                d3.verify_runtime()

    def test_validate_adapter_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "modules_to_save": None,
                    "target_modules": list(d3.TARGETS), "base_model_name_or_path": "vllm-sr/d3-mini",
                    "revision": models.spec(models.D3)["base_revision"]}
            (Path(tmp) / "adapter_config.json").write_text(json.dumps(good))
            d3.validate_adapter_config(tmp, d3.TARGETS)
            (Path(tmp) / "adapter_config.json").write_text(json.dumps({**good, "revision": "0" * 40}))
            with self.assertRaisesRegex(ValueError, "revision"):
                d3.validate_adapter_config(tmp, d3.TARGETS)

    def test_logits_and_probabilities_use_fp32_readout(self):
        model = d3.DecisionModel.__new__(d3.DecisionModel)
        model.device = "cpu"
        model.readout = torch.eye(255, dtype=torch.float32)
        model.base = FakeBackbone()  # returns last_hidden_state with a one-hot last token
        probabilities = model.probabilities([1, 2, 3], ["a", "b", "c"], {"type": "choice"}, 1.0)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)
        self.assertEqual(max(probabilities, key=probabilities.get), "a")

    def test_encode_refuses_images_until_release(self):
        encoder = d3.Encoder.__new__(d3.Encoder)
        with self.assertRaisesRegex(ValueError, "later release"):
            encoder.encode({}, {"type": "choice", "criteria": {"a": "a", "b": "b"}}, images=[b"png"])

    def test_cli_help_lists_commands(self):
        out = subprocess.run([sys.executable, "-m", "zils.d3", "--help"], capture_output=True, text=True, check=True).stdout
        for word in ("reference", "train", "parity"):
            self.assertIn(word, out)
```

Define `FakeBackbone` in the test as a `torch.nn.Module` whose `forward(input_ids, attention_mask, use_cache)` returns an object with `last_hidden_state` of shape `(1, len, 255)` bfloat16 with `[0, -1, 0] = 1`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv-kev/bin/python -m unittest tests.test_d3 -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zils.d3'`.

- [ ] **Step 3: Implement `zils/d3.py`** with the interfaces above. Port `Engine.__init__`, `logits`, `predict`, `add_adapter`, `save`, `load` from the research engine into `DecisionModel`; keep every `assert` from the engine as `raise RuntimeError(...)`/`OSError(...)` so CPU tests can patch around them. Do not import torch at module top (the CLI help test runs on CPU without CUDA).

- [ ] **Step 4: Run the tests**

Run: `.venv-kev/bin/python -m unittest tests.test_d3 tests.test_models -v`
Expected: PASS, including `test_text_runtime_and_interpreter_lookup` from Task 1.

- [ ] **Step 5: Commit**

```bash
git add zils/d3.py tests/test_d3.py
git commit -m "Add the d3-mini runtime module with the H2O DecisionModel contract"
```

### Task 3: Replace H2O branch points with profile lookups

**Files:**
- Modify: `zils/core.py:216-223` (runner dict), `:268-271` (interpreter)
- Modify: `miner/worker.py:101-142` (trainer module and interpreter)
- Modify: `zils/runtime.py:74-101` (`gpu_ready`), `:192` (capacity lock), keep `:277-284`
- Modify: `zils/releases.py:212-217` (runtime_revision and prompt from spec)
- Modify: `zils/miner_grading.py:58-69` (`trainer_identity`), `:72-76` (`workload_profile` cap)
- Modify: `zils/jevk5_runner.py:52-56` (no change expected; verify)
- Test: `tests/test_fleet_profiles.py:56` loop, `tests/test_miner_grading.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: `models.runtime_python_env`, `models.TEXT_RUNTIME_MODULES`, `models.pins`, `models.spec(model)["max_input_tokens"]`.
- Produces: `runtime.CAPACITY_ENV = {H2O: "ZILS_H2O_MIN_FREE_MIB", IMAJEV: "ZILS_IMAGE_MIN_FREE_MIB", D3: "ZILS_D3_MIN_FREE_MIB"}`; `miner_grading.TRAINER_FILES = {H2O: (...), D3: ("zils/d3.py", "zils/jevk5.py", "zils/d3-pins.json", "requirements/d3.txt")}`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_fleet_profiles.py:56`, extend the loop to `(models.JEVK5, models.H2O, models.IMAJEV, models.D3)` and the fixture interpreter env to set `ZILS_D3_RUNTIME_PYTHON` alongside `ZILS_H2O_RUNTIME_PYTHON`. In `tests/test_miner_grading.py` add:

```python
def test_workload_cap_follows_the_profile(self):
    miner_grading.workload_profile([4096], model=models.D3)
    with self.assertRaisesRegex(ValueError, "1..2048"):
        miner_grading.workload_profile([4096], model=models.H2O)
    self.assertNotEqual(miner_grading.trainer_identity(models.D3), miner_grading.trainer_identity(models.H2O))
```

In `tests/test_models.py` (or a new `tests/test_core_runtime.py`) add a test that runs `zils.core.evaluate` for a D3 checkpoint with `ZILS_D3_RUNTIME_PYTHON` unset and asserts the error message names `ZILS_D3_RUNTIME_PYTHON` (patch `settings.required` to raise with the variable name, as `settings.required` already does).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv-kev/bin/python -m unittest tests.test_fleet_profiles tests.test_miner_grading -v`
Expected: FAIL; the D3 loop iteration fails at the trainer module (`zils.jevk5` chosen) or the interpreter.

- [ ] **Step 3: Implement**

- `core.py`: runner dict gains `models.D3: "jevk5_runner.py"`; interpreter becomes `settings.required(env) if (env := models.runtime_python_env(model)) else args.runner_python`.
- `miner/worker.py`: module `"zils" + models.TEXT_RUNTIME_MODULES.get(model, ".jevk5")`; replace the `if model == models.H2O: command[0] = ...` with the same `runtime_python_env` lookup.
- `runtime.py`: `if model == models.H2O and device != "cuda"` becomes `if model in models.PINS_FILES and device != "cuda"`; the `(H2O, IMAJEV)` tuples become `(*models.PINS_FILES, models.IMAJEV)`; the env name comes from `CAPACITY_ENV[model]`.
- `releases.py`: `spec["runtime_revision"] if "runtime_revision" in spec else RUNTIME_REVISION`, same for `prompt`.
- `miner_grading.py`: `TRAINER_FILES` dict with the JevK5 tuple as default; `workload_profile` cap `models.spec(model)["max_input_tokens"]` for models that declare one, else 2048, with the message built from the cap.

- [ ] **Step 4: Run the full suite**

Run: `make check PYTHON=.venv-kev/bin/python`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add zils/core.py miner/worker.py zils/runtime.py zils/releases.py zils/miner_grading.py tests/
git commit -m "Route runner, trainer, capacity and release fields through the profile table"
```

### Task 4: Node provisioning and qualification for d3

**Files:**
- Modify: `scripts/provision_node.sh:65-69` (environments), `:80` (references), `:112-113` (qualify), `:121` (image hash)
- Modify: `scripts/qualify_node.py:82` (`qualify(..., model=models.H2O)`), `:85-89`, `:116-150`
- Modify: `deploy/units/zils-node.env.example`, `deploy/units/zils-queue-miner.service`, `deploy/units/zils-fleet-validator.service`
- Modify: `docs/fleet-nodes.md`, `docs/fleet-node-runbook.md`, `docs/queue-miners.md`, `docs/mining.md`, `README.md`
- Test: `tests/test_provision_script.py`, `tests/test_qualify_node.py`, `tests/test_node_units.py`

**Interfaces:**
- Consumes: `zils.d3` CLI (`reference`, `train`, `parity`), `models.pins(D3)`.
- Produces: `.venv-d3` at `/opt/zils/.venv-d3`, reference at `/opt/zils/models/d3-reference`, `ZILS_D3_RUNTIME_PYTHON` in the node env, `qualify_node.qualify(root, runtime_python, out, run, free_mib, model)`; the qualification report gains `parity: {"cases": n, "max_difference": x}` for d3; `image.json` records both pins hashes.

- [ ] **Step 1: Write the failing tests**: in `tests/test_provision_script.py` assert the dry-run output contains a `.venv-d3` phase using `requirements/d3.txt` after the `.venv-h2o` phase, and `-m zils.d3 reference --out models/d3-reference`; in `tests/test_qualify_node.py` add a d3 case of the injected `fake_run` that matches `zils.d3 train` and `zils.d3 parity`, and asserts the report has `model == models.D3` and `parity.cases == 9`; in `tests/test_node_units.py` assert the queue-miner unit passes `--reference /opt/zils/models/d3-reference` in addition to the H2O reference and the env example lists `ZILS_D3_RUNTIME_PYTHON`.

- [ ] **Step 2: Run to verify they fail.** Run: `.venv-kev/bin/python -m unittest tests.test_provision_script tests.test_qualify_node tests.test_node_units -v`. Expected: FAIL on the missing phase, model and env.

- [ ] **Step 3: Implement.** Provision adds the `.venv-d3` environment (Python 3.12, cu130 index, `requirements/d3.txt`), the d3 reference download (9.1 GB; keep H2O's), runs `qualify_node --model d3-mini-4.5b-v1 --runtime-python .venv-d3/bin/python` after the H2O qualification, and hashes both pins into `image.json`. `qualify_node.qualify` takes `model` and uses `models.text_runtime(model)` for the module name in commands, `models.profile_identity(model)`, `miner_grading.trainer_identity(model)`, the profile's cap for the six synthetic rows, and for d3 runs `-m zils.d3 parity --reference ... --cases` on the same nine rows the research engine used (eight dev-style controls plus the longest). Units pass two `--reference` flags (`miner/queue.py:249-256` accepts one per profile).

- [ ] **Step 4: Run the suite.** Run: `make check PYTHON=.venv-kev/bin/python`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/ deploy/ docs/ README.md tests/
git commit -m "Provision and qualify the d3 runtime beside H2O on fleet nodes"
```

### Task 5: Cross-profile upgrade in the validator and promotion check

**Files:**
- Modify: `zils/validator.py:56-74` (`evaluate` helper gains `base_revision=None`), `:106-136` (baseline block)
- Modify: `zils/version_selection.py:63-68` (`check_promotion`)
- Test: new `tests/test_cross_profile.py`; existing validator tests in `tests/test_rehearsal.py` stay green

**Interfaces:**
- Consumes: `models.checkpoint_model(comparison_dir)`, `core.evaluate` runtime selection by `--base-revision`.
- Produces: `evaluate_round` scores a `comparison` checkpoint of a different profile with that profile's own base revision and interpreter; the baseline report records `"baseline_model": <previous spec id>`; `check_promotion(current, entry, selection)` allows `incumbent["profile"] != entry["profile"]` only when `selection.get("base_change") is True`, else raises the existing message.

- [ ] **Step 1: Write the failing tests** in `tests/test_cross_profile.py` using the fixture interpreter from `tests/test_fleet_profiles.py`: (a) a job whose `manifest.model` is `spec(D3)` and whose `comparison/` checkpoint has `model == H2O` evaluates the baseline with `--base-revision` equal to H2O's revision (assert on the fixture runtime's recorded argv) and the candidates with d3's; (b) the same job with the H2O interpreter env missing fails with an error naming `ZILS_H2O_RUNTIME_PYTHON` and accepts nothing; (c) `check_promotion` with incumbent profile H2O and entry profile D3 raises "cannot move between model profiles" without `base_change`, and passes with `selection["base_change"] = True`.

- [ ] **Step 2: Run to verify they fail.** Run: `.venv-kev/bin/python -m unittest tests.test_cross_profile -v`. Expected: FAIL; today the baseline is evaluated under the candidate's revision and `core.evaluate` raises "checkpoint belongs to a different model".

- [ ] **Step 3: Implement.** In `validator.py` the `evaluate(submissions, cases, name, base_revision=None)` helper passes `base_revision or config["base_revision"]`; in the `previous` branch compute `previous_model = models.checkpoint_model(source)` and call `evaluate(..., base_revision=models.spec(previous_model)["base_revision"])`; store `previous_model` in the report under `baseline_model`. The interpreter for that call is chosen inside `core.evaluate` by Task 3's lookup. In `check_promotion`, guard the profile comparison with `and not selection.get("base_change")`.

- [ ] **Step 4: Run the suite.** Run: `make check PYTHON=.venv-kev/bin/python`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add zils/validator.py zils/version_selection.py tests/test_cross_profile.py
git commit -m "Score cross-profile upgrades against the frozen release on its own base"
```

### Task 6: Regression suite and base-candidate tooling take a `--model`

**Files:**
- Modify: `zils/regression_suite.py:55-157` (`--model`, env names, trainer module, stub metadata)
- Modify: `scripts/promote_base.py:17-87` (`--model`, `validate_candidate` via `models.text_runtime(model)`)
- Modify: `docs/base-candidates.md`
- Test: `tests/test_regression_suite.py`, `tests/test_base_candidate.py`

**Interfaces:**
- Consumes: `zils.d3.validate_candidate`, `models.candidate_files(D3, kind="candidate")`, `models.runtime_python_env`.
- Produces: `regression_suite.main(["--model", D3, ...])` with `ZILS_D3_CANDIDATE_BASE` for the candidate run (same-profile mode, for future d3 rounds); a cross-profile mode `--incumbent-model H2O --incumbent REF --candidate-model D3 --candidate REF` that trains every suite job on both pinned bases with each profile's own trainer and interpreter and applies the same `verdict(results, tolerance)`; `promote_base.py --model d3-mini-4.5b-v1 --candidate DIR --round N` prints a proposed `SPECS` entry `{**spec(D3), "id": f"d3-mini-4.5b-zils-r{round}", "base": repo, "parent": {...}}`.

- [ ] **Step 1: Write the failing tests**: parametrise the existing regression-suite test over `(models.H2O, models.D3)` asserting the trainer command is `-m zils.d3 train` and the candidate env sets `ZILS_D3_CANDIDATE_BASE`; add a cross-profile test with the fixture interpreter asserting the incumbent side runs `-m zils.h2o train` under `ZILS_H2O_RUNTIME_PYTHON` and the candidate side `-m zils.d3 train` under `ZILS_D3_RUNTIME_PYTHON`, on the same job files, and that the verdict compares per-job Brier with the 0.005 tolerance; in `tests/test_base_candidate.py` assert `promote_base` with `--model d3-mini-4.5b-v1` validates through `zils.d3.validate_candidate` (patched) and prints an id starting `d3-mini-4.5b-zils-r`.

- [ ] **Step 2: Run to verify they fail.** Expected: `argparse` error "unrecognized arguments: --model".

- [ ] **Step 3: Implement.** Replace each `models.H2O`, `zils.h2o`, `ZILS_H2O_*` in the two files with lookups on the chosen model: module name from `TEXT_RUNTIME_MODULES`, candidate env `f"ZILS_{'H2O' if model == H2O else 'D3'}_CANDIDATE_BASE"` expressed as a dict `CANDIDATE_BASE_ENV = {H2O: ..., D3: ...}` in `models.py`, runtime python from `runtime_python_env`. Default `--model` stays H2O until Task 12 flips the documented incumbent.

- [ ] **Step 4: Run the suite.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add zils/regression_suite.py scripts/promote_base.py zils/models.py docs/base-candidates.md tests/
git commit -m "Let the regression suite and base promotion target any pinned text profile"
```

**Phase 1 exit:** open PR `codex/d3-base-profile` against `ooo-hq/zils` main. Nothing deployed; no node has `.venv-d3` yet.

---

## Phase 2 — Platform (`ooo-hq/zils-platform`, after bumping `subnet-ref.txt` to the Phase 1 merge)

### Task 7: Migration — d3 profile row and SQL allow-lists

**Files:**
- Create: `supabase/migrations/202610170001_d3_training_profile.sql`
- Create: `tests/d3_database.py` (pattern: `tests/h2o_database.py`)
- Modify: `scripts/check_queue_db.py` (register the new assertions, as H2O at `:294-322`)

**Interfaces:**
- Consumes: `models.spec(D3)` and `models.profile_identity(D3)` for the inserted row.
- Produces: a `zils_model_profiles` row `d3-mini-4.5b-v1`; `zils_claim_profile_processing`, `fez_renew_training`, `fez_submit_training`, `fez_approve_training_job`, `zils_claim_profile_training` and the graded-scheduling trigger accept the d3 id; `zils_worker_profiles` check constraint `zils_d3_capacity_floor` requires `min_free_mib >= 12288` for d3 rows.

- [ ] **Step 1: Write the failing database test**: `tests/d3_database.py` inserts a d3 job through `zils_create_profile_job`, claims it with a qualified d3 worker, renews, submits and approves, and asserts a worker row with `min_free_mib = 8192` for d3 is rejected. Run: `make check-queue-db`. Expected: FAIL on the profile allow-list.

- [ ] **Step 2: Write the migration** following `202610100001_h2o_training_profile.sql` line for line: insert row with the exact spec JSON and identity hashes; `create or replace` each function whose literal list `('imajev-4b-v1','h2o-lightning-4b-v1.2.3')` or `('jevk5-4b-v0.3','h2o-lightning-4b-v1.2.3')` gains `'d3-mini-4.5b-v1'`; add the floor constraint. State in the header that no jobs or qualifications are rewritten.

- [ ] **Step 3: Run.** `make check-queue-db`. Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git checkout -b codex/d3-platform
git add supabase/migrations/202610170001_d3_training_profile.sql tests/d3_database.py scripts/check_queue_db.py
git commit -m "Add the d3-mini training profile to the queue database"
```

### Task 8: Coordinator, workflow and processor know the d3 profile

**Files:**
- Modify: `zils_platform/coordinator.py:115-124` (channels), `:173` (enabled set), `:386-400` (claim limits), `:722` (stale error text), `:963-968` (settings)
- Modify: `zils_platform/workflow.py:162-165` (routing), `:305-322` (section requirement)
- Modify: `zils_platform/version_selection.py:11-41` (`freeze`)
- Modify: `zils_platform/coordinator.py:806-938` (`Processor.evaluate`, comparison staging) and `:650-655` (references)
- Modify: `.env.example`, `docs/version-selection.md`
- Test: `tests/test_base_channels.py`, `tests/test_queue.py`, `tests/test_version_selection.py`, `tests/test_h2o.py`

**Interfaces:**
- Consumes: subnet Task 5 (`evaluate_round` cross-profile baseline), Task 1 (`models.D3`).
- Produces: `workflow.SECTIONS = {models.H2O: "h2o", models.IMAJEV: "image", models.D3: "d3"}` with the legacy section as default; the workflow config gains a `d3` section with the same keys as `h2o`; claim limits for d3 `min_free_mib = 12288`, `max_seconds = 3600`; `version_selection.freeze(job, parent)` accepts a different parent profile when `job["acceptance"].get("base_change") is True` and records `selection["base_change"] = True` and `selection["previous"]["profile_id"]`; `Processor` accepts a d3 `--additional-reference` and requires the previous release's runtime to be present when `base_change`.

- [ ] **Step 1: Write the failing tests**: `test_base_channels.py` adds `latest=D3` and asserts `/v1/config` publishes both; `test_queue.py` adds a d3 job routed to the `d3` workflow section and claimed with d3 limits; `test_version_selection.py` adds `freeze` with parent H2O and job D3 raising "different profile family" without `base_change` and succeeding with it; a processor test asserts that an upgrade job with `base_change` and no reference for the parent's profile fails the job with "previous release runtime unavailable" before evaluation.

- [ ] **Step 2: Run to verify they fail.** Run: `make check`. Expected: FAIL at routing (legacy section), freeze, and the missing guard.

- [ ] **Step 3: Implement** the lookups above; replace the stale string at `coordinator.py:722` with one built from the job model's name and cap.

- [ ] **Step 4: Run.** `make check`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add zils_platform/ .env.example docs/version-selection.md tests/
git commit -m "Route d3 jobs, allow base-change upgrades and publish the latest channel"
```

### Task 9: Adapter server serves any pinned text base

**Files:**
- Modify: `zils_platform/adapter_server.py:40-91` (`AdapterModel`), `:94-217` (`AdapterEngine`), `:220-259` (limits), `:344-396` (`main`)
- Modify: `zils_platform/adapter_releases.py:112-119` (`registry_entry` carries `profile` for text releases too)
- Create: `docs/d3-rollout.md` (from `docs/h2o-rollout.md`)
- Test: `tests/test_adapter_server.py`, `tests/test_adapter_releases.py`

**Interfaces:**
- Consumes: `models.text_runtime(model_id).DecisionModel`, `models.spec(model_id)["max_input_tokens"]`, `validate_adapter_config` from the runtime module.
- Produces: one adapter-server process per base, unchanged; the d3 unit `zils-d3-adapter.service` on 127.0.0.1:8934 with `ZILS_D3_ADAPTER_RUNTIME_TOKEN`, `.venv-d3` interpreter, its own `--releases` directory; `/health` and `/v1/prepare` advertise `max_pass_tokens` from the spec (4096 for d3, 2048 for H2O); the question payload is sent for every model with a `runtime_revision`; text registry entries carry `profile: spec["id"]` so `check_promotion` compares real profiles.

- [ ] **Step 1: Write the failing tests**: with the `Backend` double from `tests/test_adapter_server.py:21-49`, build an engine for `models.D3` and assert `max_pass_tokens == 4096`, that the prepared payload includes `question`, and that activating a release whose `model` is `spec(H2O)` raises "Adapter belongs to a different base model"; keep the existing H2O tests unchanged and green; in `test_adapter_releases.py` assert a text `registry_entry` has `profile == "h2o-lightning-4b-v1.2.3"` and that `merge_registry` keeps older entries without `profile`.

- [ ] **Step 2: Run to verify they fail.** Expected: FAIL on the hardcoded 2048 and the `zils.jevk5` import path.

- [ ] **Step 3: Implement**: `AdapterModel.__init__` uses `models.text_runtime(self.model_id)`; `validate_adapter_config` is called for every model whose runtime module defines it (`getattr(runtime, "validate_adapter_config", None)`); the 2048 literals at `:165`, `:173`, `:244-249` become `models.spec(model_id)["max_input_tokens"]` (2048 default for JevK5); `max_choice_options` is 255 for models with a `runtime_revision`, 16 otherwise. Write `docs/d3-rollout.md` with the unit, port, token env, release directory and rollback, mirroring the H2O document section for section.

- [ ] **Step 4: Run.** `make check`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add zils_platform/adapter_server.py zils_platform/adapter_releases.py docs/d3-rollout.md tests/
git commit -m "Serve any pinned text base from the adapter server"
```

**Phase 2 exit:** open PR `codex/d3-platform`. Migration not applied; no d3 adapter server running.

---

## Phase 3 — Images and video through the d3 runtime

### Task 10: Subnet — image and video inputs in the d3 encoder, trainer and capabilities

**Files:**
- Modify: `zils/d3.py` (`Encoder.encode`, `DecisionModel.predict`, `validate_inputs`)
- Modify: `zils/jevk5.py:235-251` (training rows may carry `images`)
- Modify: `zils/image_contract.py:9-21` (`capabilities(model)`), `zils/core.py:240-241` (image cases allowed for profiles whose spec declares `inputs`)
- Modify: `zils/jevk5_runner.py` (requests may carry `images` and `videos` as local file paths; the runner reads bytes)
- Test: `tests/test_d3.py`, `tests/test_image_contract.py` (new), `tests/test_fleet_profiles.py`

**Interfaces:**
- Consumes: the vendor runtime's `prepare(state, questions, images, videos)` which takes PIL images and video file paths (the probe used `runtime.prepare(case["state"], {"q": question}, images, videos)` and `runtime.run(prepared)`).
- Produces: `Encoder.encode(state, question, images=(), videos=()) -> (ids, names)` where `images` are canonical PNG bytes (the platform's `zils-image-rgb-png/v1` output) and `videos` are paths to MP4 files, at most `spec["inputs"]["images"]` images and one video; `DecisionModel.predict(state, question, images=(), videos=())`; `image_contract.capabilities(model) -> dict`: for IMAJEV the existing `IMAGE_CAPABILITIES`, for D3 `{"modalities": ["text", "image", "video"], "question_types": ["noul", "choice", "score"], "max_images": 4, "max_videos": 1, "max_questions": 8, "max_options": 255, "max_input_tokens": 4096, "unknown_key": None}`; training rows `{"state", "questions", "images": [asset paths]}` accepted by `jevk5.train` only when the model declares `inputs`.

- [ ] **Step 1: Write the failing tests**: `capabilities(D3)` equals the dict above and `capabilities(IMAJEV) is IMAGE_CAPABILITIES`; `Encoder.encode` with five images raises "at most 4 images"; `Encoder.encode` with a non-PNG byte string raises "canonical PNG"; `jevk5.train` on a row with `images` for `model_id=models.H2O` raises "does not accept images" before loading anything (patch `DecisionModel`); the fixture runtime in `test_fleet_profiles.py` receives a d3 request with `images` and echoes the count.

- [ ] **Step 2: Run to verify they fail.** Expected: FAIL (`capabilities` undefined; encode raises the Task 2 placeholder).

- [ ] **Step 3: Implement** using `runtime.prepare` for the multimodal prompt and `runtime.run(prepared)` only in `predict`; `logits` keeps the token-id path for text-only training (LoRA training on image rows uses the prepared inputs' embeddings exactly as the research image engine did in `skills/.private/d3-image-adapter-20261016/adapter_engine.py`, vision tower frozen).

- [ ] **Step 4: Run.** `make check PYTHON=.venv-kev/bin/python`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add zils/d3.py zils/jevk5.py zils/image_contract.py zils/core.py zils/jevk5_runner.py tests/
git commit -m "Accept images and video in the d3 runtime and declare its capabilities"
```

### Task 11: Platform — image jobs and image decisions on the d3 profile

**Files:**
- Modify: `zils_platform/image_api.py:49-57` (`training_profile.model` from the stable or latest channel when it declares `inputs`)
- Modify: `zils_platform/api.py:44-67`, `:268-274` (text entries carry capabilities from `image_contract.capabilities(profile)`; modality check uses them), `:256-317` (`Gateway.evaluate` forwards `images` and `videos` to a text runtime entry)
- Modify: `zils_platform/adapter_server.py:140-197` (`prepare` resolves asset references to canonical PNG bytes through `ImageStore.read_reference` and passes them to `encode`)
- Modify: `zils_platform/coordinator.py:184-220` (`image_intake` allowed on a d3 job; the job stays a profile job, not an image job)
- Create: `supabase/migrations/202610170002_d3_image_assets.sql` (`zils_image_create` allows jobs whose `model_profile->>'id'` declares inputs: literal list `('imajev-4b-v1','d3-mini-4.5b-v1')`)
- Test: `tests/test_api.py`, `tests/test_adapter_server.py`, `tests/image_database.py`, `tests/test_queue.py`

**Interfaces:**
- Consumes: subnet Task 10 `capabilities(model)` and `DecisionModel.predict(..., images, videos)`; existing `ImageStore`, `canonicalize`, signed read references (kept as is).
- Produces: `POST /v1/systemone` with `images` against a d3 release works; `GET /v1/models` lists d3 entries with `capabilities`; d3 training jobs with `image_intake` accept assets through the existing `/v1/image-assets` endpoints; Imajev endpoints untouched.

- [ ] **Step 1: Write the failing tests**: gateway test posting one image to a d3 text entry returns probabilities (fake runtime) while the same post to an H2O entry is refused with the existing modality error; adapter-server test that `prepare` with two asset references calls the store twice and passes two PNG byte strings to `encode`; database test that `zils_image_create` accepts a d3 job and still rejects a JevK5 job; queue test creating a d3 job with `image_intake`.

- [ ] **Step 2: Run to verify they fail.** Expected: FAIL at the IMAJEV-only checks.

- [ ] **Step 3: Implement** the lookups; keep `image_server.py` and every Imajev path unchanged (retirement is a later plan).

- [ ] **Step 4: Run.** `make check` and `make check-queue-db`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add zils_platform/ supabase/migrations/202610170002_d3_image_assets.sql tests/
git commit -m "Take image jobs and image decisions through the d3 text runtime"
```

---

## Phase 4 — Operator trial and the exam re-based on d3 (billed, one GPU)

### Task 12: One node, one adapter, one exam

**Files:**
- Create: `skills/.private/d3-trial-2026-10-<day>/` (workspace, private: receipts, qualification report, exam decision)
- Modify: `docs/base-candidates.md` (incumbent becomes stock d3 once the exam is frozen), `docs/model-research.md` (one paragraph linking the evidence report)
- Modify: `zils/regression_suite.py` default `--model` and `scripts/promote_base.py` default to `models.D3`

**Interfaces:**
- Consumes: everything above, merged and deployed to one rented node (same `provision_node.sh` flow as the fleet-nodes plan, `gpu-h100x1-base` image, B300 or H100 spot; destroy when done).
- Produces: `qualification.json` with `model == d3-mini-4.5b-v1` and `parity.max_difference < 1e-6`; one platform training job on d3 accepted and served, answered once with text and once with an image; `incumbent-decision.json` for round two with uid 0 = stock d3 on sealed-v2, scored by the deployed validator path (`zils evaluate --base-revision a475367…` on `calibration.jsonl` and `test.jsonl`, then `base_round.decide`); a cross-profile upgrade of one H2O staging task to d3 that reaches `ready` or `needs_review` with a scored baseline.

- [ ] **Step 1:** Provision and qualify the node; keep `qualification.json`.
- [ ] **Step 2:** Apply the two migrations to staging; start `zils-d3-adapter.service`; set `ZILS_TRAINING_MODEL_LATEST=d3-mini-4.5b-v1`; confirm `/v1/config` shows both channels.
- [ ] **Step 3:** Create a d3 job on `base_channel: latest` with the ABCD staging dataset; wait for acceptance; call the release through the gateway with text, then with one canonical PNG.
- [ ] **Step 4:** Create an upgrade job for an existing H2O staging task with `acceptance.base_change: true`; confirm the baseline report carries `baseline_model == h2o-lightning-4b-v1.2.3`.
- [ ] **Step 5: Confirmation draw (Option 5 §5.3).** Author a second sealed set with the round-one holdout tooling (`skills/.private/option5-holdout-r1-topup-20261016/holdout.py`, new seed, same six families, no case id or group shared with sealed-v2; `base_round.confirm` requires disjoint `dataset_sha256`). At least 12 hours after the first draw, score stock H2O (uid 0) and stock d3 (uid 1) on it through the deployed validator path (`zils evaluate` with each profile's `--base-revision`), then `base_round.decide` and `base_round.confirm(first, second)` with the 16 October decision as `first`. d3 must stay eligible.
- [ ] **Step 6: Adapter regression suite across profiles (Option 5 §10).** Run Task 6's cross-profile mode on the round-one suite jobs (`option5-regression-suite-v1`) with H2O incumbent and d3 candidate; the verdict must be `passed` (no job worse by more than 0.005 Brier, at least one better).
- [ ] **Step 7:** Only if Steps 5 and 6 pass: freeze `incumbent-decision.json` for round two with stock d3 as uid 0 (expected Brier near 0.142 from the 16 October research path) and record it in `docs/base-candidates.md`. If either fails, `latest` stays on H2O and the plan stops here with the numbers.
- [ ] **Step 8:** Destroy the node; write receipts and the time each step took against the two-week estimate into the project reference.
- [ ] **Step 9: Commit the doc changes**

```bash
git add docs/base-candidates.md docs/model-research.md zils/regression_suite.py scripts/promote_base.py
git commit -m "Make stock d3-mini the documented base-tier incumbent"
```

---

## Out of scope here, planned next (separate plans)

- **Website (`ooo-hq/zils-web`):** the closed model-id list at `lib/training.ts:29` gains `d3-mini-4.5b-v1`; the dashboard's image-versus-text branches on `imajev-4b-v1` (`components/training-dashboard.tsx:171-365`) become a check on `job.model.inputs`; a "Train on d3-mini" option on the replace flow sets `acceptance.base_change`; version detail shows each version's base; image intake widens to four images and one MP4 for d3 jobs. One plan, after Phase 2 is deployed.
- **Imajev and H2O retirement:** delete `image_server.py`, `imajev*.py`, `image_jobs.py` splits and the Imajev SQL functions once no release depends on them; H2O stays until the last H2O task is upgraded or expires.
- **Long context above 4,096 tokens:** accuracy at 8k and 16k caps on ABCD and the exam, half a day on one card, before any cap increase.

## Self-review notes

- Spec coverage: profile and pins (T1), runtime (T2), branch points (T3), nodes (T4), base change (T5, T8), regression and candidates (T6, T12), platform routing and channels (T7, T8), serving (T9), images and video (T10, T11), exam re-base (T12). Front end deferred by design.
- Type consistency: `Encoder.encode(state, question, images=(), videos=())` is the same signature in T2 and T10; `capabilities(model)` is defined in T10 and consumed in T11; `models.runtime_python_env` is defined in T1 and used in T3, T5, T6; `base_change` is the same key in the job's `acceptance`, in `selection`, and in `check_promotion`.
- Review Focus 1–5 map to tests in T1, T9, T10, T5, T3.
