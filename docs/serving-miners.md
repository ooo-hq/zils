# Serve z3 checkpoints as a miner

A miner can serve a public z3 checkpoint over HTTP so validators and gateways can
send it decision requests. Serving is text only: public z3 checkpoints, no images,
no adapters. It is separate from training and the base tier ([mining](mining.md)).

## Requirements

- A CUDA GPU with BF16 support, enough memory for each checkpoint you serve.
- A public IPv4 address and an open TCP port. Private, loopback and reserved
  addresses are refused.
- A hotkey registered on the subnet ([registration](bittensor-registration.md)).
  The coldkey is never read.

## Announce, then serve

Write your endpoint on chain once. Without `--yes` this is a dry run.

```bash
zils announce-serving --public-ip 203.0.113.10 --port 8091 --yes
```

Then start the server with one or more checkpoint directories:

```bash
zils serve --public-ip 203.0.113.10 --port 8091 \
  --checkpoint models/z3-mini-v1 [--checkpoint DIR] [--slots 8]
```

`serve` refuses to start if the chain does not show your hotkey's axon at exactly
that address and port. Run `announce-serving` again after any change.

## How callers find you

Validators and gateways read your axon from the chain, then fetch a signed
manifest from `GET /` on that endpoint. The manifest lists each served model, its
weights hash and its input limit, signed by your hotkey. Answers are signed too.

## Who may call

Each request is signed by the caller's hotkey. The server accepts a hotkey that
holds a validator permit on the subnet (re-read every 10 minutes; the last set is
kept if a read fails) or one passed with `--gateway-hotkey` (repeatable). Anyone
else is refused.

## Limits

- One request at a time runs on the GPU; up to `--slots` wait. When full, the
  server answers 503 and the caller retries elsewhere.
- Text decisions only. Image requests and adapters are not served.
