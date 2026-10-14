# Fleet node runbook (DigitalOcean)

Create, qualify, snapshot and destroy a node. Prices are DigitalOcean list
as of August 2026; recheck before a long-lived node. The RTX 4000 Ada size
`gpu-4000adax1-20gb` is $0.76 per GPU-hour, billed per second with a
60-second minimum.

## Create the trial node

    doctl compute ssh-key list
    doctl compute droplet create zils-node-trial --region nyc2 \
      --size gpu-4000adax1-20gb --image gpu-h100x1-base \
      --ssh-keys <SSH_KEY_ID> --tag-names zils-fleet --wait
    doctl compute droplet get zils-node-trial --format ID,PublicIPv4

If nyc2 rejects the size, try `tor1` then `atl1`.

## Provision and qualify

    ssh root@<ip> 'curl -fsSL https://raw.githubusercontent.com/ooo-hq/zils/main/scripts/provision_node.sh | ZILS_SUBNET_REF=main bash'
    ssh root@<ip> cat /var/lib/zils-node/qualification/qualification.json

The script fetches its own Python interpreters through `uv` into
`/opt/uv-python`, and the role config directory `/etc/zils-node` is group
`zils-compute`. `ZILS_NODE_DRY_RUN=1` previews the phases without changing
anything, and provisioning takes 20–40 minutes over one SSH session; if it
drops, rerun the same command, every finished phase is skipped.

Record `profile_sha256`, `runtime_sha256`, `trainer_sha256`,
`training.peak_gpu_mib` and `gpu.free_mib_at_start`; the operator registers
them in `zils_worker_profiles` for this image hash. The image hash is
`pins_sha256_16` from `/etc/zils-node/image.json`.

## Snapshot (no secrets on the node yet)

    NAME="zils-node-$(ssh root@<ip> cat /etc/zils-node/image.json | python3 -c 'import json,sys;print(json.load(sys.stdin)["pins_sha256_16"])')"
    doctl compute droplet-action snapshot <droplet-id> --snapshot-name "$NAME" --wait

## Assign a role

Create the hotkey directory and copy only the node's hotkey file into it.
Never copy `coldkey` or `coldkeypub.txt`:

    ssh root@<ip> install -d -m 0750 -g zils-compute /etc/zils-node/wallets/<wallet>/hotkeys
    scp <local wallets dir>/<wallet>/hotkeys/<hotkey> root@<ip>:/etc/zils-node/wallets/<wallet>/hotkeys/

On the node, copy `/opt/zils/deploy/role-examples/miner.json` to
`/etc/zils-node/role.json`, set `hotkey` to the hotkey's public SS58 address,
and set `wallet.name` and `wallet.hotkey` to the copied wallet and file names;
the role check refuses the file until they match. Then:

    ssh root@<ip> /opt/zils/scripts/apply_role.sh

`apply_role.sh` also re-groups everything under `/etc/zils-node/wallets` to
`zils-compute` (files 0640, directories 0750) so the service user can read it.

## Destroy

    doctl compute droplet delete zils-node-trial --force

A powered-off droplet still bills; destroy, do not power off.
