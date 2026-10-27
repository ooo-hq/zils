# Register a miner on Bittensor

This guide joins the existing Zils subnet; a miner does not create a subnet.

Zils runs on Bittensor mainnet (`finney`), subnet `<NETUID>`. Registration is
the admission to the base tier: register, then
[compete from outside](mining.md#compete-in-the-base-tier-from-outside) by
committing a candidate on chain. No operator approval is needed.

Customer adapter jobs from the [hosted queue](queue-miners.md) go to approved
operators; registration alone does not assign customer work.

Registration gives a hotkey a chain UID.
For validator operation, continue with [Run a validator](validators.md).

## 1. Install the pinned CLI

Use a trusted Linux or macOS workstation with Git and Python 3.13. These wallet
steps also work in WSL 2. From a fresh clone, create a separate environment for
wallet administration:

```bash
git clone https://github.com/ooo-hq/zils.git
cd zils
python3.13 -m venv .venv-btcli
.venv-btcli/bin/python -m pip install 'bittensor==11.1.0' 'typer==0.16.0'
.venv-btcli/bin/btcli --version
```

The version must be `11.1.0`, matching the Zils chain SDK. The Typer pin keeps
this CLI's help and exit handling compatible. Bittensor v11 includes `btcli`;
these commands use its [v11 CLI flags](https://www.bittensor.com/docs/cli).

Set these values in the terminal you will use for the following steps:

```bash
export ZILS_NETUID=<NETUID>
export ZILS_WALLET=zils-miner
export ZILS_HOTKEY=miner-1
export ZILS_WALLET_PATH="$HOME/.bittensor/wallets"
```

The wallet and hotkey names are examples for a new, dedicated identity. If you
already have an identity, use its names and wallet directory instead.

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
administration workstation. Mining and commitments use the hotkey only.

Fund the **coldkey's public address** with enough TAO for the registration
burn. Never send a recovery phrase or private key to anyone. Check the balance:

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

Stop if the subnet is missing or registration is disabled. The burn price
changes with demand, and a full subnet replaces its lowest-ranked UID outside
the immunity period when a new hotkey registers. Preview the coldkey-signed
transaction first:

```bash
.venv-btcli/bin/btcli --network finney --wallet "$ZILS_WALLET" \
  --wallet-hotkey "$ZILS_HOTKEY" --wallet-path "$ZILS_WALLET_PATH" \
  --dry-run tx burned-register --netuid "$ZILS_NETUID"
```

The next command **submits a mainnet registration and burns TAO** after you
confirm. Review the network, subnet, hotkey, and cost in the terminal:

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
hotkey's public SS58 address in the metagraph.

## 5. Compete

Continue with [compete from outside](mining.md#compete-in-the-base-tier-from-outside).
Zils commands take `--network finney --netuid "$ZILS_NETUID"`; mainnet is never
a default, so a command without both flags never touches mainnet. Never give
anyone your recovery phrase or coldkey.
