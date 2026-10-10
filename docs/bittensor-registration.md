# Register a miner on Bittensor

**Confirm the testnet subnet ID with the Zils operator before registering.**
This guide joins an existing subnet; a miner does not create a subnet.

Zils connects a **closed, model-pinned fleet** to Bittensor testnet. The operator
selects JevK5 4B text, ImaJev 4B image, or a legacy Kev 0.8B job and qualifies
the participating hardware. Registration alone does not select a model.
The [verified round on subnet 579](testnet-round-001.md) is a historical result,
not an invitation to register on that subnet today. The fleet rejects mainnet.
The current [JevK5 queue](jevk5-queue.md) uses approved hotkeys and local queue
UIDs; it does not require on-chain registration or publish chain weights.

Registration gives a hotkey a chain UID. To receive work in the closed fleet,
the operator must also add that UID and hotkey to the validator's roster.
For validator operation, continue with [Run a validator](validators.md).

## 1. Install the pinned CLI

Use a trusted Linux or macOS workstation with Git and Python 3.13. These wallet
steps also work in WSL 2; run the Bittensor fleet on Linux or macOS. From a fresh
clone, create a separate environment for wallet administration:

```bash
git clone https://github.com/ooo-hq/zils.git
cd zils
python3.13 -m venv .venv-btcli
.venv-btcli/bin/python -m pip install 'bittensor==11.1.0' 'typer==0.16.0'
.venv-btcli/bin/btcli --version
```

The version must be `11.1.0`, matching Zils' testnet SDK. The Typer pin keeps this
CLI's help and exit handling compatible. Bittensor v11 includes `btcli`; these
commands use its [v11 CLI flags](https://www.bittensor.com/docs/cli).

Set these values in the terminal you will use for the following steps:

```bash
export ZILS_NETUID=REPLACE_WITH_OPERATOR_TESTNET_NETUID
export ZILS_WALLET=zils-testnet-miner
export ZILS_HOTKEY=miner-1
export ZILS_WALLET_PATH="$HOME/.bittensor/wallets"
```

Replace `ZILS_NETUID` with the operator's numeric subnet ID (1–4095). The wallet
and hotkey names above are examples for a new, dedicated testnet identity.
If you already have an identity, use its names and wallet directory instead.

## 2. Create and fund the identity

For a **new** identity, create an encrypted coldkey and a separate hotkey:

```bash
.venv-btcli/bin/btcli --network test \
  --wallet "$ZILS_WALLET" --wallet-hotkey "$ZILS_HOTKEY" \
  --wallet-path "$ZILS_WALLET_PATH" wallet create
.venv-btcli/bin/btcli --wallet-path "$ZILS_WALLET_PATH" wallet list
```

Record the recovery phrases privately when prompted. Skip creation when reusing
an existing wallet; do not overwrite its keys. Keep the coldkey on the trusted
administration workstation. Mining and weight signing use the hotkey.

Ask the testnet operator for test TAO funding to the **coldkey's public address**.
For public testnet funding channels, use the community link on the
[official Bittensor site](https://www.bittensor.com/docs/quickstart).
Never send a recovery phrase or private key. Check the funded balance:

```bash
.venv-btcli/bin/btcli --network test --wallet "$ZILS_WALLET" \
  --wallet-path "$ZILS_WALLET_PATH" wallet balance
```

## 3. Inspect and register on the subnet

Read the current subnet details, registration price, and admission settings:

```bash
.venv-btcli/bin/btcli --network test subnets show "$ZILS_NETUID"
.venv-btcli/bin/btcli --network test query burn --netuid "$ZILS_NETUID"
.venv-btcli/bin/btcli --network test query subnet-hyperparameters \
  --netuid "$ZILS_NETUID"
```

Stop if the subnet is missing, registration is disabled, or the operator has not
confirmed admission. The price changes with chain state; check the current
collateral settings too. Preview the coldkey-signed transaction first:

```bash
.venv-btcli/bin/btcli --network test --wallet "$ZILS_WALLET" \
  --wallet-hotkey "$ZILS_HOTKEY" --wallet-path "$ZILS_WALLET_PATH" \
  --dry-run tx burned-register --netuid "$ZILS_NETUID"
```

The next command **submits a testnet registration and charges test TAO** after
you confirm. Review the network, subnet, hotkey, and cost in the terminal:

```bash
.venv-btcli/bin/btcli --network test --wallet "$ZILS_WALLET" \
  --wallet-hotkey "$ZILS_HOTKEY" --wallet-path "$ZILS_WALLET_PATH" \
  tx burned-register --netuid "$ZILS_NETUID"
```

This is the v11 [Bittensor registration workflow](https://www.bittensor.com/docs/guides/mining).
A preview is not a reservation of the price. If confirmation is interrupted,
check the registration before submitting again.

## 4. Verify the chain UID

Query the registered identity and subnet membership:

```bash
.venv-btcli/bin/btcli --network test --wallet "$ZILS_WALLET" \
  --wallet-hotkey "$ZILS_HOTKEY" --wallet-path "$ZILS_WALLET_PATH" \
  query uid --netuid "$ZILS_NETUID"
.venv-btcli/bin/btcli --network test query metagraph --netuid "$ZILS_NETUID"
```

The UID query must return an integer, not `None`/`null`. Match that UID to the
hotkey's public SS58 address in the metagraph. Do not assume the UID is `1` or
reuse a local queue UID.

Give the validator operator the network (`test`), netuid, public hotkey address,
and actual UID. Wallet names and paths in a fleet configuration must match the
machine that will run that identity; they are not key material.

## 5. Join the Zils fleet

The operator uses those public identities to generate the
[registered fleet and run preflight](testnet.md#provision-the-registered-identities).
Receive your assigned `miner-UID.tar.gz`, provision only your hotkey on the miner
host through a private channel, and follow the [miner setup](mining.md#set-up-each-machine-once).
Install `requirements/testnet.txt` in the miner's model environment as well.
Never give the validator your recovery phrase or coldkey.

From the unpacked, configured miner directory, start one round:

```bash
./start-miner --rounds 1
```

Zils uses private-network artifact exchange and an explicit roster; publishing
a generic Axon endpoint does not connect this fleet. Keep the configured miner
port reachable from the validator and the validator port reachable from miners.
If the chain replaces your UID or hotkey, stop and have the operator regenerate
the roster. Registration alone does not establish work availability or rewards.
