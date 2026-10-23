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
    ssh root@<ip> cat /var/lib/zils-node/qualification-d3/qualification.json

The script fetches its own Python interpreters through `uv` into
`/opt/uv-python`, and the role config directory `/etc/zils-node` is group
`zils-compute`. `ZILS_NODE_DRY_RUN=1` previews the phases without changing
anything, and provisioning takes 20–40 minutes over one SSH session; if it
drops, rerun the same command, every finished phase is skipped.
The d3 environment, reference and qualification steps live inside the existing
`environments`, `references` and `qualify` phases, so they run only on a fresh
provision; a node or snapshot provisioned before d3 support needs a rebuild, or
a manual run of those three steps, before `apply_role.sh` installs the
queue-miner unit that references `models/d3-reference`.

Record `profile_sha256`, `runtime_sha256`, `trainer_sha256`,
`training.peak_gpu_mib`, `gpu.free_mib_at_start` and `parity` from the d3
report; the operator registers it in `zils_worker_profiles` for this image hash.
The image hash is `pins_sha256_16` (d3) from `/etc/zils-node/image.json`.

## Snapshot (no secrets on the node yet)

    NAME="zils-node-$(ssh root@<ip> cat /etc/zils-node/image.json | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["pins_sha256_16"]+"-"+d["subnet_commit"][:7])')"
    doctl compute droplet-action snapshot <droplet-id> --snapshot-name "$NAME" --wait

The name is `zils-node-<pins_sha256_16>-<first 7 of subnet_commit>`.

## Boot a node from the snapshot

    doctl compute snapshot list --format ID,Name
    doctl compute droplet create zils-node-1 --region nyc2 --size gpu-4000adax1-20gb \
      --image <SNAPSHOT_ID> --ssh-keys <SSH_KEY_ID> --tag-names zils-fleet --wait

Provisioning is already done on such a node; only "Assign a role" remains.

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
`zils-compute` (files 0640, directories 0750) so the service user can read it,
and does the same for `/etc/zils-node/fleet` when it exists (directories 0770,
files 0640). The miner config it writes enables graded presence
(`graded_scheduling`, `graded_min_free_mib` 12288), so the node heartbeats to the
coordinator; it still receives customer jobs only after the operator binds a
resource and qualifies it (platform `docs/miner-grading.md`). Before training,
the miner checks each job's longest example against its GPU (total memory, or
`gpu_budget_mib` in the miner config when another service shares the card) and
refuses jobs that cannot fit, so the coordinator offers them to another miner. The
`fleet-validator` role is installed but has no runbook path yet; a first
multi-host d3 round is its acceptance test.

## Verify

    ssh root@<ip> systemctl status zils-queue-miner --no-pager
    ssh root@<ip> journalctl -u zils-queue-miner -n 50 --no-pager

A healthy start shows the unit active (running) and the miner polling for
assignments.

## Destroy

    doctl compute droplet delete zils-node-trial --force

A powered-off droplet still bills; destroy, do not power off.
