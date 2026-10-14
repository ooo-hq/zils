# Fleet nodes on DigitalOcean

Status: design, 14 October 2026; phase 1 (node image, miner and fleet-validator roles, provisioning, runbook) is implemented on this branch; serving roles and release delivery are not.

Zils needs more than one CUDA host. The single RTX 4090 trains, serves H2O
text adapters and serves image models, and those workloads evict each other.
Instead of squeezing them onto one card, run each on its own rented node and
make every node a Zils fleet member: a real queue miner training customer
adapters, a serving host, or a validator, with a testnet UID when a chain round
needs one. The same nodes carry the subnet from testnet to a mainnet-shaped
fleet without a second rebuild.

Two properties are required of every node:

1. **One role per node, set by configuration**, not by hand-built hosts.
2. **Rebuildable**: destroy any node and recreate an identical one from a
   script, with a DigitalOcean snapshot as the fast path.

## What exists today

- The queue miner, its H2O runtime and qualification steps are documented in
  [queue-miners.md](queue-miners.md) and [h2o-queue.md](h2o-queue.md). They
  were installed by hand on the 4090 (WSL 2) and on the 10 October rollout.
- Fleet bundles, the validator and testnet publication are documented in
  [mining.md](mining.md), [validators.md](validators.md),
  [testnet.md](testnet.md) and [bittensor-registration.md](bittensor-registration.md).
  `zils.fleet` and `zils.testnet` are pinned to JevK5, ImaJev and Kev jobs;
  tests cover only those, and no H2O chain round has been run.
- Hosted serving on the 4090 uses a socket-activated H2O adapter runtime, a
  host-local capacity controller that pauses image serving while H2O is
  resident, and SSH tunnels to the gateway. That layout is the reference
  install this spec turns into a script.
- Accepted adapters are stored in private DigitalOcean Spaces with a catalog
  in Supabase; the serving runtime on the 4090 reads them from a local
  directory the workflow fills at activation.

## Node image

One image, built by one script, carrying everything a node might need and no
secrets. Roles differ only in which services are enabled.

| Component | Content |
| --- | --- |
| Droplet | DigitalOcean GPU Droplet, RTX 4000 Ada (20 GB) to start; RTX 6000 Ada / L40S (48 GB) if a node must serve image models and H2O together. Ubuntu LTS with the NVIDIA driver compatible with CUDA 13.0. |
| Accounts | `zilsminer` (training), `zilsapi` (serving), group `zils-compute`; no interactive login for either. |
| Source | `ooo-hq/zils` at the revision in the platform's `subnet-ref.txt`. `ooo-hq/zils-platform` at its main only on operator-owned nodes taking a serving or `hosted-validator` role, installed by a separate platform step. Both read-only to service users. |
| Environments | `.venv-kev` (Python 3.13, cu128: miner client, fleet, testnet SDK 11.1.0), `.venv-h2o` (Python 3.12, cu130: pinned H2O runtime), `.venv-image` (optional, Imajev pins), `.venv-platform` (validator and serving roles). |
| References | `models/h2o-reference` verified against `zils/h2o-pins.json`; `models/imajev-reference` only on image-capable nodes. Reference caches are part of the snapshot. |
| Units | From `zils`: `zils-queue-miner`, `zils-fleet-miner`, `zils-fleet-validator`. From `zils-platform` (operator nodes only): `zils-h2o-adapter.socket` + runtime + proxy, `zils-image-runtime`, `zils-release-sync`, `zils-training-processor`, `zils-workflow`, `zils-*-tunnel`. All installed, all disabled; the role enables a subset. |
| Qualification | `scripts/qualify_node.py` output: profile identity, `trainer_sha256`, kernel check, peak GPU memory for a 2,048-token training step and an adapter reload, timing. Stored on the node and copied to the operator for `zils_worker_profiles` registration. |

Not in the image: hotkeys, coldkeys, runtime tokens, Supabase credentials,
Spaces keys, Tailscale auth keys, tunnel SSH keys. These arrive at boot.

## Roles

`/etc/zils-node/role.json` is written at boot by the operator (or cloud-init
user data) and selects the role. A node runs exactly one role.

| Role | Enabled services | Needs at boot | Chain identity |
| --- | --- | --- | --- |
| `miner` | `zils-queue-miner` (customer jobs via coordinator HTTPS, approved hotkey); optionally `zils-fleet-miner` for a chain round using an assigned bundle | hotkey, coordinator URL, bundle when in a round | Testnet UID for its hotkey when it joins a round |
| `serve-h2o` | `zils-h2o-adapter` socket/runtime/proxy, resident with no idle shutdown; `zils-release-sync` (new, see below); `zils-h2o-tunnel` to the gateway | runtime token, Spaces read credential, tunnel key | None |
| `serve-image` | `zils-image-runtime`, `zils-image-tunnel` | as above | None |
| `fleet-validator` | `zils-fleet-validator` (`zils.fleet validator`) and the `zils.testnet` publisher for chain rounds. Pure `zils`; no platform code or credential. | validator hotkey, roster, frozen job and reference | Validator UID with permit |
| `hosted-validator` | `fleet-validator` plus the platform add-on: `zils-training-processor` (customer-job evaluation) and `zils-workflow` (assignment, activation). Only on nodes the operator owns. | the above plus the Supabase service credential | as above |

Repository boundary: `zils` is what any miner or validator operator runs, so
`provision_node.sh` must produce a qualified `miner` or `fleet-validator`
node with nothing from `zils-platform`. Serving roles and `hosted-validator`
are a second, platform-owned install step on nodes the operator controls;
they import platform code and carry hosted credentials. This matches the
subnet's boundary test, which forbids miners and validators from importing
platform modules.

The 4090 keeps the `training-host` layout it has today (miner + image serving +
capacity controller) until a `serve-image` node exists; it is the only host
where two serving runtimes share a card.

Role switching is editing `role.json` and re-running `scripts/apply_role.sh`, never an in-place edit
of units. A node that changes role loses its state directory on purpose.

## Boot-time injection

Secrets and identity are delivered once per boot through cloud-init user data
or an operator script over Tailscale, written under `/etc/zils-node/` (env
files mode 0600, wallet files as below), and never copied into snapshots:

- Wallet hotkey files under `/etc/zils-node/wallets/<wallet>/hotkeys/` for this
  node only, group `zils-compute`, files 0640 and directories 0750 so the
  service user can read them (`apply_role.sh` enforces this); coldkeys stay on
  the administration workstation, as the registration guide requires.
- `runtime.env` (`ZILS_RUNTIME_TOKEN`), `spaces.env` (read-only key for the
  release bucket), `supabase.env` (validator only), `tunnel/` SSH key.
- `role.json`.

A node refuses to start a role whose required files are missing; it does not
fall back to a weaker role.

## Release delivery to serving nodes (platform change)

Today `zils_platform.workflow` activates an accepted adapter by writing it to
a local release directory and health-checking the runtime on the same host.
A `serve-h2o` node is on another machine, so:

1. Activation publishes the verified release directory to the private Spaces
   bucket under `releases/h2o/<release_id>/` and records it in the catalog
   (both already exist for training artifacts).
2. `zils-release-sync` on the serving node polls the catalog, downloads new
   releases, verifies the manifest hashes with the existing release verifier,
   and drops them into the local release directory the runtime already watches
   (hot reload exists today).
3. The workflow's activation health check targets the serving node's runtime
   URL through the tunnel; `activation_failed` retries until the node lists
   the release, as it does today for the local runtime.

This is a `zils-platform` change (workflow publish step plus a small sync
service); the subnet repository contributes nothing new for it. Registry
entries route H2O releases to the serving node's tunnel port.

## Networking

- Tailscale on every node (already used for the 4090 and `ken-server`);
  fleet ports 8900 and 8901–8903 are allowed only from fleet peers' Tailscale
  addresses, matching the private-network assumption in `mining.md`.
- Gateway tunnels keep the existing `ssh -N` unit pattern; one unit per
  serving node. The gateway's nginx connection cap is raised when a serving
  node with the bounded request queue goes live.
- DigitalOcean firewall: inbound SSH from Tailscale only; no public ports.

## Testnet identities

Each node that participates in a chain round has its own hotkey registered
on the operator-confirmed testnet subnet (the registration guide, run from the
administration workstation; the node receives only the hotkey). The roster
file lists chain UIDs; a changed registration regenerates the roster.

Gate: the first H2O fleet round is a verification, not a rollout. `zils.fleet
init` currently takes a JevK5 or ImaJev reference and the publisher's tests
cover those profiles only. Expect an `H2O fleet support` change in this
repository (job model `h2o-lightning-4b-v1.2.3`, bundle recipe, publisher test)
before the round, and treat its first successful round as the acceptance test.

## Provisioning and rebuild

`scripts/provision_node.sh` is idempotent and runs on a bare droplet or an
already-provisioned one (see the [runbook](fleet-node-runbook.md)):

1. OS packages, driver check (`nvidia-smi` reports a CUDA 13-compatible driver).
2. Accounts, groups, directories, permissions.
3. Source checkouts at the pinned revisions; `uv` environments from the
   `requirements/*.txt` pins; no network after this step except references.
4. References downloaded and verified against `h2o-pins.json`.
5. Units installed, all disabled.
6. `scripts/qualify_node.py` run; its report is the proof the image is good.
7. Writes `/etc/zils-node/image.json`: subnet revision, platform revision,
   pins hash, qualification report hash, build time.

Rebuild paths:

- **Script**: bare droplet → provisioned node in 20–40 minutes. Authoritative.
- **Snapshot**: after step 7 on a node with no secrets injected, take a
  DigitalOcean snapshot named from `pins_sha256_16` and the short `subnet_commit` in `image.json`. New nodes boot from it
  in minutes and run only boot-time injection plus role start. A snapshot is
  retired whenever `subnet-ref.txt` or the H2O pins change; the script makes
  the next one.

Nothing in the state directories is needed to rebuild: customer job state is
reclaimed from the coordinator, releases are re-pulled from Spaces, fleet
rounds are re-run.

## Costs and sizing

| Node | Card | $/hour | Pattern |
| --- | --- | --- | --- |
| `serve-h2o` | RTX 4000 Ada | 0.76 | Always on, about $555/month; H2O resident used 9.4 GB on the 4090 |
| `miner` | RTX 4000 Ada | 0.76 | Per round or per job; a four-hour round for two miners is about $6 |
| `validator` | RTX 4000 Ada | 0.76 | Always on once chain rounds are continuous; can share the serving node at first |
| image-capable node | RTX 6000 Ada / L40S | 1.57 | Only if image serving leaves the 4090 |

Billing is per second with a 60-second minimum; a powered-off droplet still
bills, so destroy idle miners. Prices are DigitalOcean's list as of August
2026 and must be rechecked. The 20 GB card clears the 12,288 MiB training
floor on paper; training throughput and kernel qualification on Ada are
unverified until the first node runs `qualify_node.py`.

## Rollout

1. **Trial node**: provision one RTX 4000 Ada with the script, run
   qualification, snapshot. Decision point: Ada passes or the plan moves to
   the 48 GB cards.
2. **Serving node**: boot `serve-h2o` from the snapshot, land the platform
   release-delivery change, point the gateway's H2O entry at it. The 4090
   stops serving H2O; its capacity controller keeps image serving resident.
   This replaces the "Option B" co-residency work.
3. **Miner nodes**: two `miner` nodes approved as queue workers; they take
   real customer jobs. Run the first multi-host H2O chain round with them
   after the H2O fleet support change; the 4090's processor stays validator.
4. **Validator node**: a `fleet-validator` node from `zils`, with the
   platform `hosted-validator` add-on moving the processor and workflow off
   the 4090, which then becomes an ordinary miner or is retired.
5. Mainnet gates are unchanged by this document: open-miner data policy,
   emissions design, and the code's mainnet rejection.

## Out of scope

Open, permissionless miners; emissions and payouts; customer-data policy for
untrusted operators; moving the gateway off its current host.

## Open questions

1. Does the H2O kernel qualification pass on Ada-generation cards? (Trial node.)
2. Should release delivery pull from Spaces (proposed) or should the workflow
   push over the tunnel? Pull is simpler to rebuild; push needs no new
   credential on the node.
3. One validator identity or one per vertical? `mining.md` requires separate
   fleets and ports per model profile; a single node can host both validators.
4. Do rented miners count as "approved workers" automatically on rebuild, or
   does each rebuild re-register its qualification hashes? The hashes are
   image properties, so registration should be per image hash, not per node.
