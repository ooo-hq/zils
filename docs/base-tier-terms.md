# Zils base-tier terms

Version DRAFT. These terms apply to anyone who registers a hotkey on Bittensor
mainnet subnet `<NETUID>` and commits a base or feature candidate there ("you").
They are made with Zils, the owner of that subnet ("Zils"). Customer adapter
jobs are not covered here; they go only to approved operators under separate
written terms.

## 1. What you do

- Train a candidate however you like, within the
  [candidate contract](base-candidates.md) for your [pool](model-pools.md).
- Upload it to your own public Hugging Face repository at an immutable revision
  and commit `zils1:<hash>:<revision>:<repo>` with your registered hotkey.
- Keep that revision public until validators have scored and confirmed it.
  Validators keep their own copy of each candidate they fetch; deleting the repo
  does not withdraw a submission.

## 2. Submission rules

- **One active candidate per hotkey.** A new commitment replaces your previous
  one. Use separately registered hotkeys for simultaneous entries.
- **Cooldown per pool.** A pool may set a minimum spacing between one hotkey's
  distinct submissions, measured in chain blocks. A candidate committed inside
  it is refused and recorded; commit again after the cooldown has passed.
- **First committer owns a checkpoint.** A hash another hotkey committed at an
  earlier block is refused for you.
- **No tampering.** Do not alter model IDs, hashes, metadata or contract files to
  pass a check that would otherwise fail, and do not try to obtain, reconstruct
  or probe sealed exam cases outside the published evaluation flow.

## 3. How payment works

- Validators run the [decision-quality election](base-election.md). A candidate
  that clears the bar, passes confirmation and the adapter regression suite, and
  is promoted at a release window makes its hotkey a **king**.
- Only kings are paid. A pool's share of the base budget is split equally over
  its last five distinct kings still registered. Winning an exam without being
  crowned pays nothing. A share with no king is burned at the subnet owner's UID.
- Shares between the base, feature and adapter tracks are validator
  configuration and can change.
- **No earnings are promised.** Payment is whatever the chain's consensus and
  emissions deliver to your UID. Your costs (hardware, power, registration burn,
  hosting) are yours. Nothing here is investment advice or an offer of
  securities.

## 4. What Zils may do

- **Promote and publish.** A promoted candidate's weights are republished under
  the Zils organization with a notice naming the pool, the round and your hotkey,
  and may be served to Zils customers. Its licence follows the parent model's
  licence.
- **Publish aggregate results.** Your hotkey, UID, submission hashes and
  aggregate scores may appear on zils.ai and in validator results. Per-case
  results are never returned.
- **Refuse or disqualify** a candidate or hotkey that breaks these terms, in
  which case validators stop paying it.
- **Change these terms.** A new version replaces this one; continuing to commit
  after it is published is acceptance.

## 5. Liability, briefly

You are responsible for your host, your keys and what you upload. Zils provides
the software, validators and evaluation as they are, without warranty of uptime,
scoring outcome or payment, and is not liable for your costs or losses from
mining.

## 6. Acceptance

Registering a hotkey on subnet `<NETUID>` and committing a candidate with it is
acceptance. Each hotkey accepts separately.
