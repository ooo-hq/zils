# Fleet node runbook (DigitalOcean)

Create, qualify, snapshot and destroy a node. Prices are DigitalOcean list
as of August 2026; recheck before a long-lived node.

## Create the trial node

    doctl compute ssh-key list
    doctl compute droplet create zils-node-trial --region nyc2 \
      --size gpu-4000adax1-20gb --image gpu-h100x1-base \
      --ssh-keys <SSH_KEY_ID> --tag-names zils-fleet --wait
    doctl compute droplet get zils-node-trial --format PublicIPv4

If nyc2 rejects the size, try `tor1` then `atl1`.

## Provision and qualify

    ssh root@<ip> 'curl -fsSL https://raw.githubusercontent.com/ooo-hq/zils/main/scripts/provision_node.sh | ZILS_SUBNET_REF=main bash'
    ssh root@<ip> cat /var/lib/zils-node/qualification/qualification.json

The script fetches its own Python interpreters through `uv` into
`/opt/uv-python`, and the role config directory `/etc/zils-node` is group
`zils-compute`.

Record `profile_sha256`, `runtime_sha256`, `trainer_sha256`,
`training.peak_gpu_mib` and `gpu.free_mib_at_start`; the operator registers
them in `zils_worker_profiles` for this image hash.

## Snapshot (no secrets on the node yet)

    doctl compute droplet-action snapshot <droplet-id> --snapshot-name "zils-node-$(ssh root@<ip> jq -r .pins_sha256_16 /etc/zils-node/image.json)" --wait

## Assign a role

Copy only the node's hotkey file into `/etc/zils-node/wallets/<wallet>/hotkeys/`,
write `/etc/zils-node/role.json` from `deploy/role-examples/`, then:

    ssh root@<ip> /opt/zils/scripts/apply_role.sh

`apply_role.sh` also re-groups everything under `/etc/zils-node/wallets` to
`zils-compute` (files 0640, directories 0750) so the service user can read it.

## Destroy

    doctl compute droplet delete zils-node-trial --force

A powered-off droplet still bills; destroy, do not power off.
