# Zils on Bittensor testnet

The [first closed round](testnet-round-001.md) completed on **testnet subnet 579**
on September 25, 2026. Three fresh checkpoints were trained and scored; their
timelocked weights were revealed and independently verified on-chain.

The fleet has an optional Bittensor path for a **closed testnet rehearsal**.
The subnet ID is configured explicitly; mainnet is rejected. It uses registered
wallet hotkeys for signed submissions, checks UID ↔
hotkey registrations before each round and before publication, and submits
weights through `bittensor==11.1.0`. The existing private LAN artifact transport
and validator-owned inference stay in place. This is not an open internet
competition: it has no public miner discovery or untrusted-model sandbox.

Local fleets remain offline from Bittensor. Testnet fleets also require the
validator's explicit `--publish-weights` flag before their loop sends weights.
This does not register keys, spend registration funds, change hyperparameters,
stop an existing validator, or launch background services.

## Provision the registered identities

Before generating a fleet, fund a dedicated testnet wallet, register a subnet, and
register the validator/miner hotkeys on the returned subnet ID. Use the live
chain's registration cost and activation requirements; owning a new subnet
does not itself make emissions active.

Install the additional SDK on each participant, using the existing environment:

```bash
uv pip install --python .venv-kev/bin/python -r requirements/testnet.txt
```

Use the existing `btcli` wallet files. Each machine needs only its assigned
hotkey; the owner coldkey is not needed to mine or set validator weights.
Keep wallet files outside the repository and never commit miner bundles.

Create `.private/testnet-identities.json` with this shape. Replace the placeholder
addresses, subnet ID, UIDs, wallet names and paths with the actual registrations. Wallet
paths are interpreted on the machine running that identity. Add one miner
entry for each machine:

```json
{
  "chain": {"network": "test", "netuid": 777},
  "validator": {
    "hotkey": "REGISTERED_VALIDATOR_SS58",
    "wallet": {"name": "WALLET_NAME", "hotkey": "VALIDATOR_KEY_NAME", "path": "/absolute/wallets"}
  },
  "miners": [
    {
      "uid": 1,
      "hotkey": "REGISTERED_MINER_SS58",
      "wallet": {"name": "WALLET_NAME", "hotkey": "MINER_KEY_NAME", "path": "/absolute/wallets"}
    }
  ]
}
```

`777` is an example only; use the subnet ID returned by your registration.

Create or reuse the local benchmark described in the [miner guide](mining.md).
Replace `VALIDATOR_PRIVATE_IPV4` with the validator's numeric private address.
Generate a fresh fleet. Match `--miner-ports` to the roster's order and count
(use `8901 8902 8903` for three miners). The output folder names use chain UIDs.
The command copies public wallet references into bundles, **never key files**:

```bash
.venv-kev/bin/python -m fez.fleet init --out .private/testnet-fleet \
  --host VALIDATOR_PRIVATE_IPV4 --miner-ports 8901 \
  --benchmark .private/benchmarks/local \
  --testnet-identities .private/testnet-identities.json
.venv-kev/bin/python -m fez.testnet preflight \
  --config .private/testnet-fleet/validator/config.json
```

Distribute the assigned `miner-UID.tar.gz` and provision that machine's existing
hotkey separately. [mining.md](mining.md) covers the model environment, private
networking and startup command. A changed or deregistered UID stops the chain
path; regenerate the roster with current registrations. Scores are never
silently reassigned to a replacement hotkey.

## Run one round

Start each miner with `./start-miner --rounds 1`. On the validator host, run:

```bash
.venv-kev/bin/python -m fez.fleet validator \
  --config .private/testnet-fleet/validator/config.json --rounds 1
```

This evaluates real submissions and previews the SDK transaction without
submitting. If the previous validator has recently set weights, the preview
reports the chain's cooldown instead. The private report records the registered
identities it evaluated; old local rehearsal reports cannot be published as
testnet results.

After inspection, enable publication for the registered subnet. Only one validator
process should write for its hotkey on that subnet.

Publish the completed round, substituting its directory for `ROUND`:

```bash
.venv-kev/bin/python -m fez.testnet publish \
  --config .private/testnet-fleet/validator/config.json \
  --round .private/testnet-fleet/validator/state/rounds/ROUND
```

For subsequent rounds, add `--publish-weights` to the validator startup command.
The SDK enforces chain constraints and chooses plain weights or timelocked
commit-reveal. Zils uses mechanism 0 and weights version 1. An increased required
version blocks this client rather than pretending to implement a newer rubric.

## Read the outcome

Each publication writes `chain-attempt.json` before signing and a
`chain-receipt.json` afterward. A restart returns the recorded outcome; it does
not resend. `unknown` means the connection or process stopped without a reliable
receipt. Inspect the transaction on-chain before any manual recovery; deleting
that marker and retrying can duplicate a submission.

`committed` means the timelocked transaction was included, **not** that rewards
already use the new weights. Verify after the chain's reveal period:

```bash
.venv-kev/bin/python -m fez.testnet verify \
  --config .private/testnet-fleet/validator/config.json \
  --round .private/testnet-fleet/validator/state/rounds/ROUND
```

Verification checks the current registered identities, a later `LastUpdate`,
and the expected proportions after SDK clipping/quantization. It reports the
current weight state; it cannot uniquely attribute identical weights to one
commit if another process writes for the same hotkey. `rate_limited` and
`no_weights` send nothing; previous on-chain weights remain. Zils never invents
uniform rewards when every candidate fails.

## Validation

Validation: `python -m unittest discover -v` with model, signing and testnet
dependencies installed. Chain tests fake external RPC only; wallet signatures,
SDK intent construction, report binding, durable receipts and restart handling
run against real code. The first live round also exercised transaction inclusion,
commit–reveal, and recovery after confirmation was interrupted. An SDK result
with an unknown outcome remains `unknown` even when its success flag is false;
inspect the chain before recovery or retry.

SDK reference: [Bittensor set-weights](https://www.bittensor.com/docs/tx/set-weights).
