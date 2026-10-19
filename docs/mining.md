# Run a text or image fleet miner

Local and Bittensor testnet fleets support **d3-mini 4.5B** text jobs
(`d3-mini-4.5b-v1`, the base since 17 October 2026). H2O Lightning 4B and the
ImaJev image profile are retired; existing bundles for them keep working, and new
bundles should pin d3. Existing Kev 0.8B bundles remain supported; JevK5 4B bundles
are retired. For workers that claim different customer jobs,
use the [queue miner guide](https://github.com/ooo-hq/zils/blob/main/docs/queue-miners.md).
Queue and fleet miners use the same trainers but different credentials and startup commands.

After the one-time setup below, run this in your assigned miner folder:

```bash
./start-miner
```

The miner trains one candidate per round, submits its signed checkpoint, receives
the validator's result, and waits for the next round. Stop with Ctrl+C, or use
`./start-miner --rounds 1` for a single result. The first start downloads the
pinned text base model. Image miners require the separately installed, verified
image runtime described below. Later starts reuse the cache.

## Prepare bundles on the validator

For a text fleet, create the d3 reference with the
[queue miner setup](https://github.com/ooo-hq/zils/blob/main/docs/queue-miners.md#2-install-and-verify-the-reference)
Prepare authorized, independent training/calibration/test JSONL files using the
[customer data format](https://github.com/ooo-hq/zils/blob/main/docs/customer-jobs.md#prepare-data).
The following freezes those files and an example acceptance policy; choose your
policy before evaluating candidates:

```bash
.venv-kev/bin/python -m zils.jobs --job-id text-v1 \
  --model d3-mini-4.5b-v1 --train .private/train.jsonl \
  --calibration .private/calibration.jsonl --test .private/test.jsonl \
  --min-accuracy 0.80 --min-brier-improvement 0.01 \
  --allow-training-data-export --out .private/jobs/text-v1
.venv-kev/bin/python -m zils.fleet init --out .private/fleet-local \
  --benchmark .private/jobs/text-v1 --checkpoint models/d3-reference \
  --host VALIDATOR_PRIVATE_IPV4
```

Replace `VALIDATOR_PRIVATE_IPV4` with the validator's numeric private address
(RFC 1918 or a Tailscale `100.64.0.0/10` address; public addresses are refused).
Use new output paths or reuse an existing audited benchmark. The default ports
are 8900 for the validator and 8901–8903 for three miners. Assign one
`miner-N.tar.gz` to each machine.

Only training data goes into a miner bundle. The validator keeps calibration
cases, test cases, and full reports. Local bundles contain disposable signing
keys and must stay private. [Testnet bundles](testnet.md) contain public wallet
references instead; provision each hotkey separately.

### Image fleets

Fleet image bundles use the retired ImaJev profile only; `zils.fleet init`
takes `--images` for ImaJev jobs and no other profile. New d3 image jobs run in
the [hosted queue](queue-miners.md), where a d3 miner downloads a job's frozen
training photos itself. Use this section only to rerun an existing ImaJev job.

Complete the pinned runtime and maximum-context hardware qualification in
[image training](https://github.com/ooo-hq/zils-platform/blob/main/docs/image-training.md#worker-and-service-setup)
for every participating miner and the validator. Use a fresh published starting
checkpoint, not an earlier customer adapter. The operator supplies an audited
image-job directory built by `zils.image_jobs.build`, its canonical image cache
(`<sha256>.png`), and measured admission limits. Those are private inputs; fleet
initialization does not download customer photos or resolve hosted predecessor jobs.

```bash
.venv-kev/bin/python -m zils.fleet init --out .private/fleet-image \
  --benchmark .private/jobs/image-v1 --checkpoint models/imajev-starting-checkpoint \
  --images .private/image-cache --min-free-mib QUALIFIED_FREE_MIB \
  --max-seconds QUALIFIED_TRAINING_SECONDS --host VALIDATOR_PRIVATE_IPV4
```

Replace both qualification placeholders with the operator's measurements. Memory
admission must cover peak reserved GPU memory plus 512 MiB and be at least
12,288 MiB; the training deadline must be 1–3,600 seconds. Use limits sufficient
for every host in this fleet. Bundles receive only training photos; calibration
and test photos stay with the validator. Export verifies image hashes and sizes.

Each model uses its own fleet directory and ports. Never combine text and image
scores into one reward vector. Only one validator may publish for a hotkey/subnet;
do not run competing vertical publishers that overwrite each other's weights.

For a legacy Kev rehearsal, keep `zils.benchmark build` and `models/reference`.
Do not relabel a Kev manifest as a different model.

## Set up each machine once

1. Unpack its assigned archive and enter the `miner-N` folder.
2. Install Git and Python 3.13, then install the pinned dependencies:

   ```bash
   python3.13 -m venv .venv-kev
   .venv-kev/bin/python -m pip install -r requirements/model.txt -r requirements/rehearsal.txt
   ```

   With `uv`, use `uv venv --python 3.13 .venv-kev` and
   `uv pip install --python .venv-kev/bin/python -r requirements/model.txt -r requirements/rehearsal.txt`.

3. Allow the assigned miner TCP port from the validator's private address.
   Allow validator TCP 8900 from the miner machines.
4. Run `./start-miner --rounds 1`. It can start before the validator and will retry.

d3 needs a qualified CUDA host and the isolated runtime below on every miner
and the validator. CPU and Apple MPS fallback apply only to Kev. Use `--device cuda`
explicitly. To reuse an environment or cache, set `ZILS_PYTHON`
to its absolute Python path and `HF_HOME` to the cache directory.

For d3 bundles, install the isolated runtime on each host and export it before
every start (the validator needs the same variable):

```bash
uv venv --python 3.12 .venv-d3
uv pip install --python .venv-d3/bin/python --torch-backend=cu130 -r requirements/d3.txt
export ZILS_D3_RUNTIME_PYTHON="$PWD/.venv-d3/bin/python"
./start-miner --device cuda --rounds 1
```

For legacy H2O bundles, do the same with `.venv-h2o`, `requirements/h2o.txt`
and `ZILS_H2O_RUNTIME_PYTHON`.

d3 training waits until at least 12,288 MiB of GPU memory is free
(`ZILS_D3_MIN_FREE_MIB` raises that floor). Neither side silently substitutes
another runtime. The pins are `zils/d3-pins.json`; the retired H2O contract is in
[h2o-queue.md](h2o-queue.md).

For legacy ImaJev image bundles, also install and verify the isolated runtime on each host:

```bash
uv venv --python 3.13 .venv-image
uv pip install --python .venv-image/bin/python --torch-backend=cu128 -r requirements/imajev.txt
.venv-image/bin/python -m scripts.download_imajev --out models/imajev-reference
export ZILS_IMAGE_RUNTIME_PYTHON="$PWD/.venv-image/bin/python"
export ZILS_IMAGE_REFERENCE="$PWD/models/imajev-reference"
./start-miner --device cuda --rounds 1
```

The validator needs these same image runtime variables. Set the shared
`ZILS_COMPUTE_LOCK` and the qualified `ZILS_IMAGE_GPU_MEMORY_FRACTION` when sharing
a GPU, as described in the image training guide. Low capacity makes a miner wait;
it never permits a smaller unverified memory allowance.

### Windows / NVIDIA GPU

Run the miner inside WSL 2, using the supported NVIDIA driver installed on
Windows. Do not install a Linux NVIDIA display driver inside WSL. Check CUDA
from the miner environment:

```bash
.venv-kev/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA is not ready"; print(torch.cuda.get_device_name(0))'
```

The validator must reach the miner directly at its private IPv4 address.
Default WSL NAT does not support this callback layout without additional
network setup. Use [WSL mirrored networking](https://learn.microsoft.com/en-us/windows/wsl/networking#mirrored-mode-networking)
and limit the Windows/Hyper-V firewall rule to the assigned port and validator.
See [NVIDIA's WSL setup](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
Native Windows service execution and NAT traversal are not implemented.

## Start the validator

From the repository root:

```bash
.venv-kev/bin/python -m zils.fleet validator \
  --config .private/fleet-local/validator/config.json --rounds 1
```

Remove `--rounds 1` for continuous rounds. The validator evaluates after every
miner submits or the collection deadline expires. It verifies signatures and
hashes, calibrates each candidate, and scores the separate evaluation cases.
Local mode never writes weights to Bittensor. Testnet mode needs registered
wallets and explicit publication; follow the [testnet guide](testnet.md).

## Two tracks on the testnet: the base tier pays first

On a testnet fleet whose validator config carries a `base_tier` track, miners
earn on two tracks, paid from one weight vector on the configured subnet:

- **Base tier (primary, continuous).** Submit a full-weight candidate for the
  pinned base, `d3-mini-4.5b-v1`, under the
  [base candidate contract](base-candidates.md). The validator's base-tier
  scorer calibrates it on the sealed calibration split and scores the sealed
  test split; you are eligible when your skill is positive, your accuracy is
  not below the incumbent's, your Brier improvement is at least 0.01 and the
  paired bootstrap 95% lower bound is above zero. King of the hill (owner
  decisions, 19 October 2026, following Teutonic): only kings are paid.
  Winning the exam makes you a challenger and pays nothing. You are crowned
  only after you also pass the confirmation draw and the adapter regression
  suite and the owner's promotion PR makes your candidate the incumbent and
  your hotkey the newest entry in the track's king history. The base share of
  every publication is split equally over the last five distinct kings still
  registered, so a dethroned king keeps earning until five newer kings have
  reigned; deregister and your slice goes to the others. The vendor base has no
  owner, so until the first promotion the base share is burned; a resubmitted
  stock base or a copy of the king is never eligible.
- **Adapter rounds (alongside).** The frozen customer job rounds above continue
  unchanged; each round's calibrated test-split skill is the adapter weight.
  A validator may instead take the adapter track from real customer work in the
  [hosted queue](queue-miners.md) (`adapter_evidence`, off by default; see
  [validators §7](validators.md#7-pay-the-adapter-track-from-hosted-queue-work)).
  Then you earn for customer jobs the queue routed to your hotkey and delivered
  to the customer, weighted by training tokens × held-out skill × your
  reliability over the window. Qualification runs and operator-owned jobs earn
  nothing, and your queue hotkey must be registered on the subnet.

Split: the base tier takes `base_tier.share` of the vector (currently **0.7**;
the validator records the share in every chain attempt and receipt), adapters
the rest. Each track is normalized over its own earners first, so a miner with
no eligible base candidate still earns its adapter share and a miner that trains
no customer adapter still earns its base share. A track nobody earned on is
not handed to the other track: its share goes to `base_tier.burn_uid` (default
0), as Teutonic burns to the subnet owner's uid. For example, with no king the
vector is 0.7 to the burn uid and 0.3 to adapter earners in their own
proportions. If nobody earned on either track, the whole vector goes to the
burn uid, so the validator keeps setting weights. Subtensor burns incentive
paid to the subnet owner's hotkey and keeps the owner's own self-weight, so
the validator refuses to publish a burn slice unless `burn_uid` is the subnet
owner hotkey's uid at that block.

Anti-gaming: one adapter submission per hotkey per round and one base candidate
commitment per hotkey on chain; a claim cannot change within a round; a candidate
hash another hotkey committed at an earlier block is refused; every
sealed split is audited against its manifest before scoring, and payment goes
only to registered roster UIDs. You receive aggregate metrics, never per-case
outcomes; a round's cases are published after it closes and are not reused.

What a base-tier miner needs: the 20–24 GB cards in today's fleet cannot
full-fine-tune the 4.5B backbone, but a d3 LoRA trains in about 10.5 GB of
GPU memory. Train a LoRA on d3 and merge it with `zils.d3 candidate`, which
writes the two bf16 shards in the parent's layout plus the untouched contract
files and validates the result (see
[the reference recipe](base-candidates.md#reference-recipe-train-a-lora-merge-it));
the merge needs about 10 GB of RAM and 10 GB of disk per candidate, and the
upload is at most 9.5 GiB. Customer jobs take priority on a shared card; base
training fills the idle time. What you train on decides everything: the 18
October rehearsal candidate, trained only on synthetic rule cases, tied stock d3
on the sealed exam.

## Compete in the base tier from outside

You do not need a bundle, an operator or an invitation to enter the base tier:
registration on subnet 579 is the admission. About an afternoon end to end, most
of it the training run.

The Zils base-tier validator on testnet subnet 579 (18 October 2026). You do not
need its address or hotkey to compete: candidates are chain commitments that
every validator reads itself. The address only serves its signed round results
(`GET /results/<round>`, step 6):

| | |
| --- | --- |
| URL | `http://165.245.195.21:8900` |
| Validator hotkey | `5HTAVUuHxV81VbYoX7DYFzebdyA1SRk3zazdZiiBN7zCcpxy` (uid 0 on subnet 579) |

Plain HTTP; every reply is signed by that hotkey. Rounds run hourly. Both values
are for the testnet only: the host is temporary, and on mainnet the validator gets
a new hotkey and a new URL.
Use the values this table shows for the network you mine on.

1. **Register a hotkey on testnet subnet 579** following the
   [registration guide](bittensor-registration.md); keep the hotkey file on the
   mining host and the coldkey elsewhere.
2. **Read the [miner terms](mining-terms.md).** Running a miner against a Zils
   validator means you accept them.
3. **Train and merge a candidate** with the
   [reference recipe](base-candidates.md#reference-recipe-train-a-lora-merge-it):
   `zils.d3 reference`, `zils.d3 train` on data of your choosing, `zils.d3 candidate`
   to merge the adapter into the parent's shards. The result is a directory that
   passes the candidate contract; the exam is realistic decisions in families you
   have not seen, so what you train on decides everything.
4. **Upload and commit.** Log in to Hugging Face once (`hf auth login`, a write
   token), then let `zils.miner commit-candidate` upload the candidate files to
   your public model repo and commit `zils1:<hash>:<revision>:<repo>` on chain
   with your hotkey (details in [base candidates: commit](base-candidates.md#commit)):

   ```bash
   .venv-kev/bin/python -m zils.miner commit-candidate \
     --wallet-name WALLET_NAME --wallet-hotkey MINER_KEY_NAME --wallet-path /absolute/wallets \
     --candidate candidate --repo YOUR_HF_USER/d3-candidate
   ```

   Without `--yes` it is a dry run that prints the commitment; with `--yes` it
   uploads, checks the public repo at the new commit, writes the commitment and
   reads it back. No validator URL, no open port, nothing left running: every
   validator reads the commitments at the start of its round and pulls the
   revision from Hugging Face, once per distinct hash.
5. **Let the doctor check.** It runs the checks in order (requirements, hotkey
   file, registration, candidate, Hugging Face repo at the revision, commitment on
   chain) and stops at the first failing one with the exact command that fixes it:

   ```bash
   .venv-kev/bin/python -m zils.miner doctor \
     --wallet-name WALLET_NAME --wallet-hotkey MINER_KEY_NAME --wallet-path /absolute/wallets \
     --candidate candidate --repo YOUR_HF_USER/d3-candidate --revision COMMIT_SHA
   ```

   You can run it before step 1 (it tells you to create the wallet), before
   step 3 (it prints the merge recipe), before step 4 (it prints the commit
   command) and again after each fix.
6. **Read the result.** Your payout is on chain: the validators' weights and your
   incentive (`btcli subnets metagraph --netuid 579 --network test`). The
   aggregate row per round (Brier and accuracy against the incumbent, eligibility,
   the pending and confirmed winner) is in each validator's signed round result on
   its own API (for the Zils validator, the address in the table above), not on
   chain. Per-case results are never returned. An ineligible or
   rejected candidate earns nothing; fix it, merge again and commit the new hash.
   Leave an unchanged commitment alone: it stays in every round at no cost, and
   committing it again moves your block later, which can hand the hash to a copier
   who committed it in between.

## Results and restarts

Miner logs, frozen candidates, and results are in `state/jobs/`. Full validator
reports are in `validator/state/rounds/`. Restarting a miner reuses completed
candidates; an interrupted training epoch starts again. The validator reloads
accepted submissions after restart. Service locks prevent duplicate launches,
and a per-device compute lock serializes work on shared hardware.

Keep started bundles at their original location: saved checkpoint paths are
absolute. Archive completed rounds between runs if disk space fills; training
and evaluation stop when less than 2 GiB is free. Regenerate bundles to upgrade
code; existing bundles are standalone snapshots.

This is a closed development fleet with a fixed one-epoch recipe. Every round
starts from the same reference with different seeds; winners are not promoted
automatically. The reused synthetic benchmark measures development progress.
Signed HTTP authenticates messages but does not encrypt them; use a trusted
private LAN. Evaluation runs each submission in a network-less namespace where
the host allows it (`sandbox` in every report says which mode ran); that bounds
network and lifetime, not every hostile-checkpoint behaviour.
