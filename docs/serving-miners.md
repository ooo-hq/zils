# Serve z3 checkpoints as a miner

A miner can serve a public z3 checkpoint over HTTP so validators and gateways can
send it decision requests. Serving is text only: public z3 checkpoints, no images,
no adapters. It is separate from training and the base tier ([mining](mining.md)).

## Requirements

- Linux x86_64 with an NVIDIA GPU that supports BF16 and has enough memory for
  each checkpoint you serve.
- The pinned d3 runtime: `.venv-d3` (Python 3.12, CUDA 13.0), which the
  [installer](https://zils.ai/install.sh) builds on Linux x86_64 with an NVIDIA GPU.
  `zils-miner serve` runs inference in a child process under that interpreter, or under
  `ZILS_D3_RUNTIME_PYTHON` when it is set.
- A public IPv4 address and an open TCP port. Private, loopback and reserved
  addresses are refused.
- A hotkey registered on the subnet ([registration](bittensor-registration.md)).
  The coldkey is never read.

## Make a checkpoint directory

From the zils checkout (`~/.zils/zils` after the installer), write a stock
reference. This downloads and verifies the pinned weights and writes
`model.json`:

```bash
.venv-d3/bin/python -m zils.d3 reference --out models/d3-reference
```

Add `--model <profile>` for another z3 size (`d3-lite-0.8b-v1`, `d3-nano-2.2b-v1`,
`d3-flash-8.4b-v1`, `d3-ultra-26b-v1`). A merged base candidate written by
`zils.d3 candidate` can be served too. Adapters cannot.

Leave the base-candidate override for your model's pool (`ZILS_D3_CANDIDATE_BASE`,
`ZILS_D3_LITE_CANDIDATE_BASE`, and so on) unset. The worker refuses to start when
it is set, because the weights would no longer match the advertised hash.

## Announce, then serve

Write your endpoint on chain once. Without `--yes` this is a dry run.

```bash
zils-miner announce-serving --public-ip 203.0.113.10 --port 8091 --yes
```

Then start the server with one or more checkpoint directories, one per model:

```bash
zils-miner serve --public-ip 203.0.113.10 --port 8091 \
  --checkpoint models/d3-reference [--checkpoint DIR] [--slots 8]
```

`serve` refuses to start if the chain does not show your hotkey's axon at exactly
that address and port. Run `announce-serving` again after any change. If the
inference process dies, `serve` exits non-zero; run it under a supervisor such as
systemd with `Restart=always`.

## Serve from an existing runtime

A box that already runs the Zils adapter runtime with `--serve-base` holds the
pinned base in GPU memory and answers it as `<profile>-base`. On that box,
`zils-miner serve` can send its requests to the runtime instead of starting its own
worker, so the model is loaded once:

```bash
export ZILS_RUNTIME_TOKEN=...   # the token the runtime was started with
zils-miner serve --public-ip 203.0.113.10 --port 8091 \
  --checkpoint models/d3-reference \
  --runtime http://127.0.0.1:8935 [--runtime-token-env ZILS_RUNTIME_TOKEN]
```

- **Base only.** `--checkpoint` must be the stock reference the runtime loaded
  (`kind` `base`, `temperature` 1.0 in its `model.json`), and exactly one. Every
  request names `<profile>-base`, never an adapter the runtime also holds.
  Merged candidates cannot be served this way.
- **Identity check.** Before serving, `serve` reads the runtime's `GET /health`
  and refuses to start unless the runtime is ready, its `profile_sha256` and
  `runtime_sha256` match the checkpoint's pinned profile, and it lists
  `<profile>-base` with the matching fingerprint. The manifest's
  `weights_sha256` is still the checkpoint hash of `--checkpoint`.
- **Paid traffic first.** Requests use the runtime's bulk lane, so the
  runtime's own real-time traffic runs ahead of them. When the bulk queue is
  full the runtime answers 529 and `serve` answers 503 `busy`.
- **Adapter swaps.** The runtime has one adapter slot. A miner call activates
  the base, so the next customer-adapter request on that runtime first
  re-verifies and reloads its adapter, which adds latency to that request. Use
  `--runtime` only on a runtime whose customer adapters get little or no
  traffic. Paid base traffic causes no swap.
- **No preemption.** A miner call that is already running is not interrupted.
  One call can reserve up to 8 × 8,192 tokens, and real-time requests that
  arrive meanwhile wait behind it. Under load the bulk lane gets about one call
  in five.
- **Pauses.** While the runtime is stopped or not ready, each request gets 503
  `unavailable` and `serve` keeps running. It checks `/health` every 60 seconds
  and logs changes. If the runtime comes back with a different identity, `serve`
  exits non-zero so its supervisor restarts it.
- **Loopback only.** `--runtime` must be `http://127.0.0.1:<port>` or
  `http://localhost:<port>`; anything else is refused, so the token never leaves
  the box.
- **Token.** The runtime's bearer token is read from the variable named by
  `--runtime-token-env` (default `ZILS_RUNTIME_TOKEN`). It is never logged or
  printed.

`zils-miner doctor --model <profile> --runtime http://127.0.0.1:<port>` adds the same
identity check to its checklist.

## How callers find you

Validators and gateways read your axon from the chain, then fetch a signed
manifest from `GET /v1/serving` on that endpoint. Every signed message is
`{"payload": {...}, "signature": "<hex>"}`, signed by a hotkey over
`zils-serving/v1/<kind>`, a zero byte, and the payload as sorted, compact JSON.

- **Manifest** (`GET /v1/serving`, signed by your hotkey): `version`, `hotkey`,
  `endpoint`, `issued_at`, and `models`, a list of `model`, `weights_sha256` and
  `max_input_tokens`. Callers accept it for 120 seconds after `issued_at`.
  `weights_sha256` is the checkpoint hash of the served directory. For a stock
  reference that covers only its `model.json`, because its weights are the
  pinned release. For a candidate it covers the full weights.
- **Request** (`POST /v1/systemone`, signed by the caller's hotkey): `caller`,
  `audience` (your hotkey), `timestamp`, `nonce`, `model`, `weights_sha256`, and
  `body`, a decision request with `state`, `model` and `questions`.
- **Answer** (signed by your hotkey): `miner` (your hotkey), `request_sha256` (the
  SHA-256 of the signed request), `model`, `weights_sha256`, and `predictions`,
  with `probabilities` and `input_tokens` for each question.

## Who may call

Each request is signed by the caller's hotkey. The server accepts a hotkey that
holds a validator permit on the subnet (re-read every 10 minutes; the last set is
kept if a read fails) or one passed with `--gateway-hotkey` (repeatable SS58
address). Anyone else is refused. A request is refused when it is more than
120 seconds old or its nonce was already used.

## Limits

- One request at a time runs on the GPU; up to `--slots` connections wait. When
  full, the server answers 503 with `Retry-After: 1` and the caller retries
  elsewhere.
- One source address may hold two connections at a time. The request line and
  headers must arrive within 3 seconds, and the whole request within 15.
- At most 32 questions per request, each within the model's input limit (413
  otherwise).
- Text decisions only. Image requests and adapters are not served.
