# Subnet milestones

The local training loop and guarded Bittensor testnet integration are implemented.
The [first registered training-to-chain round](testnet-round-001.md) recorded a
completed commit–reveal cycle. That evidence does not establish current uptime,
open participation, mainnet support, or production model quality.

Status of the work for an open competition, as of 17 October 2026:

1. Admission rules: chain-derived admission is implemented and off by default.
2. Isolated evaluation: the sandboxed evaluation subprocess is merged.
3. Refreshed held-out benchmarks: the round-one sealed holdout is authored
   (1,458 cases in six families). The first confirmation draw, run on
   17 October 2026 on a second, disjoint sealed set (291 calibration and 1,156
   test cases), did not confirm a winner; see [d3-mini](#d3-mini).
4. Promotion and rollback: two channels and the adapter regression suite are
   decided and implemented. The suite has run once, on 17 October 2026, and
   passed on its single reference job (the rule asks for three or four). The
   channel is platform configuration today; since 17 October 2026 `stable` and
   `latest` both name d3-mini by owner decision. A registry alias and automatic
   follow-stable upgrades remain.
5. Public health and chain reporting: remaining.

## Base tier

Miners can submit a full-weight replacement for the pinned base of a text
profile. [Base candidates](base-candidates.md) describes the submission
contract, scoring, the confirmation draw, the regression suite and the `latest`
and `stable` channels. Candidate loading, round decisions, sandboxed evaluation,
the regression suite, chain-derived admission and the dry-run promotion script
are merged. No public round has been run.

## d3-mini

The profile `d3-mini-4.5b-v1` (vendor model `vllm-sr/d3-mini`, mirrored as
`zils-ai/d3-mini-4.5b-v1`) was merged on 16–17 October 2026: the profile and
pins, the runtime module, node provisioning and qualification, images and video
through the runtime, platform routing, a `latest` channel that can point at d3
when configured, and cross-profile upgrades with `base_change`
(see [version selection](version-selection.md#changing-the-base-profile)).

d3 was not promoted by the exam; it is the default for new jobs since
17 October 2026 by the owner's decision (below). On the night of
16–17 October 2026, on one rented B300 over 6.75 hours, the base tier's rules
were applied and stock d3-mini is **not promotable today**. The first draw
cleared (Brier 0.142 against 0.184 for H2O), but the confirmation draw, on a
second disjoint teacher-authored sealed set, did not confirm a winner: d3 0.1790
and H2O 0.1794, an improvement of 0.0004 with a paired-bootstrap 95% interval of
[−0.022, +0.023], where the rule needs at least 0.01 with a positive lower
bound. The adapter regression suite passed, on one reference job where the rule
asks for three or four.

Decision, 17 October 2026: the owner adopted d3-mini as the base anyway, on
capability grounds (adapters trained on d3 beat adapters trained on H2O on the
reference job and on ABCD; d3 takes images and video), recorded as an override of
the promotion rule rather than a win under it. The same afternoon the hosted
platform set `stable` and `latest` to d3-mini, qualified a d3 miner and serving
host, moved the owner's H2O task to d3 through a `base_change` upgrade and
switched H2O serving off; Imajev retired on 17 October once image training moved to
d3; JevK5 retires once nothing served depends on it.
Stock d3-mini is the base-tier incumbent for round two. The recorded evidence is in
[model research](model-research.md#d3-mini-evidence-16-october-2026).

Customer-job data contracts, calibration, and acceptance are shared with the
hosted platform. Hosted account management, APIs, billing, storage, and serving
belong to [zils-platform](https://github.com/ooo-hq/zils-platform); SDKs and agent
interfaces belong to [zils-sdk](https://github.com/ooo-hq/zils-sdk).

The [public model page](https://zils.ai/model) presents recorded research and
subnet evidence. Live views require an explicit aggregate feed; unavailable
information must remain labeled as unavailable.
