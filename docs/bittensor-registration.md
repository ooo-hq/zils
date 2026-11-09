# Register a miner on Bittensor

This guide joins an existing subnet; a miner does not create a subnet.

Zils runs on Bittensor **mainnet subnet 127** (network `finney`). Mainnet is never
a default: every Zils chain command names the network and netuid explicitly.
Registration on 127 is the admission; no operator approval is needed. A
registered hotkey can:

- **Commit soup bodies.** See [soup pool](mining.md#soup-pool).
- **Take customer jobs.** The hosted queue admits any hotkey registered on 127;
  see [customer-job miners](queue-miners.md) and [admission](miner-admission.md).

Registration gives a hotkey a chain UID. A closed, model-pinned
[chain fleet](testnet.md) also needs the operator to add your UID and hotkey to
its roster. For validator operation, continue with [Run a validator](validators.md).

## 1. Install the pinned CLI

Use a trusted Linux or macOS workstation with Git and Python 3.13. These wallet
steps also work in WSL 2. From a fresh
clone, create a separate environment for wallet administration:

```bash
git clone https://github.com/ooo-hq/zils.git
cd zils
python3.13 -m venv .venv-btcli
.venv-btcli/bin/python -m pip install 'bittensor==11.1.0' 'typer==0.16.0'
.venv-btcli/bin/btcli --version
```

The version must be `11.1.0`, matching Zils' chain SDK. The Typer pin keeps this
CLI's help and exit handling compatible. Bittensor v11 includes `btcli`; these
commands use its [v11 CLI flags](https://www.bittensor.com/docs/cli).

Set these values in the terminal you will use for the following steps:

```bash
export ZILS_NETUID=127
export ZILS_WALLET=zils-miner
export ZILS_HOTKEY=miner-1
export ZILS_WALLET_PATH="$HOME/.bittensor/wallets"
```

127 is the Zils subnet on mainnet; for a closed fleet on another subnet or on
testnet, use the operator's network and numeric subnet ID (1–4095) instead. The wallet
and hotkey names above are examples for a new, dedicated identity.
If you already have an identity, use its names and wallet directory instead.

## 2. Create and fund the identity

For a **new** identity, create an encrypted coldkey and a separate hotkey:

```bash
.venv-btcli/bin/btcli --network finney \
  --wallet "$ZILS_WALLET" --wallet-hotkey "$ZILS_HOTKEY" \
  --wallet-path "$ZILS_WALLET_PATH" wallet create
.venv-btcli/bin/btcli --wallet-path "$ZILS_WALLET_PATH" wallet list
```

Record the recovery phrases privately when prompted. Skip creation when reusing
an existing wallet; do not overwrite its keys. Keep the coldkey on the trusted
administration workstation. Mining and weight signing use the hotkey.

Fund the **coldkey's public address** with TAO from an exchange or another wallet.
Registration burns the subnet's current price (about τ0.0074 on 3 November 2026;
read it with the commands below) plus the transaction fee.
Never send a recovery phrase or private key. Check the funded balance:

```bash
.venv-btcli/bin/btcli --network finney --wallet "$ZILS_WALLET" \
  --wallet-path "$ZILS_WALLET_PATH" wallet balance
```

## 3. Inspect and register on the subnet

Read the current subnet details, registration price, and admission settings:

```bash
.venv-btcli/bin/btcli --network finney subnets show "$ZILS_NETUID"
.venv-btcli/bin/btcli --network finney query burn --netuid "$ZILS_NETUID"
.venv-btcli/bin/btcli --network finney query subnet-hyperparameters \
  --netuid "$ZILS_NETUID"
```

Stop if the subnet is missing or registration is disabled (and, for a closed
fleet, if the operator has not confirmed admission). The price changes with chain state; check the current
collateral settings too.

Subnet 127 is full (256 UIDs). A registration replaces the registered hotkey with
the lowest pruning score that is past its immunity period (5,000 blocks, about
17 hours). Your UID can be replaced the same way once your own immunity ends and
you have earned nothing, so commit a body or start taking customer jobs before
then; an evicted hotkey must register again. Preview the coldkey-signed transaction first:

```bash
.venv-btcli/bin/btcli --network finney --wallet "$ZILS_WALLET" \
  --wallet-hotkey "$ZILS_HOTKEY" --wallet-path "$ZILS_WALLET_PATH" \
  --dry-run tx burned-register --netuid "$ZILS_NETUID"
```

The next command **submits a mainnet registration and burns TAO** after
you confirm. Review the network, subnet, hotkey, and cost in the terminal:

```bash
.venv-btcli/bin/btcli --network finney --wallet "$ZILS_WALLET" \
  --wallet-hotkey "$ZILS_HOTKEY" --wallet-path "$ZILS_WALLET_PATH" \
  tx burned-register --netuid "$ZILS_NETUID"
```

This is the v11 [Bittensor registration workflow](https://www.bittensor.com/docs/guides/mining).
A preview is not a reservation of the price. If confirmation is interrupted,
check the registration before submitting again.

## 4. Verify the chain UID

Query the registered identity and subnet membership:

```bash
.venv-btcli/bin/btcli --network finney --wallet "$ZILS_WALLET" \
  --wallet-hotkey "$ZILS_HOTKEY" --wallet-path "$ZILS_WALLET_PATH" \
  query uid --netuid "$ZILS_NETUID"
.venv-btcli/bin/btcli --network finney query metagraph --netuid "$ZILS_NETUID"
```

The UID query must return an integer, not `None`/`null`. Match that UID to the
hotkey's public SS58 address in the metagraph. Do not assume the UID is `1` or
reuse a local queue UID.

Copy only the hotkey file (`<wallet path>/<wallet>/hotkeys/<hotkey>`) to the
mining host, through a private channel. Your miner reads the hotkey from that
wallet directory; wallet names and paths in
a miner config must match the machine that runs that identity. They are not key
material. Never give anyone your recovery phrase or coldkey.

## 5. Next

- Soup pools: [install](mining.md#install), then [commit a body](mining.md#commit-a-body).
- Customer jobs: [customer-job miners](queue-miners.md), from step 2.
- A closed chain fleet: give its operator the network (`finney`), netuid, public
  hotkey address and actual UID, then follow [chain fleet](testnet.md). Registration
  alone does not establish work availability or rewards there.
