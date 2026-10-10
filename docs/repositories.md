# Repository boundaries

`ooo-hq/zils` is the miner and validator repository. Hosted product development
and client tooling belong in separate repositories.

| Repository | Code and supporting material |
| --- | --- |
| [zils](https://github.com/ooo-hq/zils) | `miner/`, shared `zils/` model/evaluation/protocol code, reference downloads, operator docs and tests |
| [zils-platform](https://github.com/ooo-hq/zils-platform) | `zils_platform/` APIs, queue orchestration, billing, storage, serving; migrations and service integration tests |
| [zils-sdk](https://github.com/ooo-hq/zils-sdk) | Python and TypeScript packages, CLI, MCP, client documentation and contract tests |
| [zils-web](https://github.com/ooo-hq/zils-web) | Website and customer-facing application |

Dependencies flow from the platform into the subnet's shared code. Miners and
validators must not import platform, SDK, Stripe, or storage-provider modules.
The subnet boundary test enforces this. SDK applications communicate over HTTP
and need no subnet or platform checkout.

Hosted entry points moved from `python -m zils.<service>` to
`python -m zils_platform.<service>`. Miner commands, Bittensor commands, HTTP
routes, model identities, signature domains, and database names are unchanged.
The hosted queue processor moved with the platform; independent model scoring
and acceptance remain here. Existing services need a planned source/install and
supervisor-command update; merging this split does not update a deployment.

The platform migration guide records installation and rollback. It retains
historical architecture notes and the old standalone benchmark preview in an
archive. The live website remains in zils-web. Public experiment methodology
and aggregate results remain here.

The Python metadata permits the platform to install this checkout as an editable
source dependency (`python -m pip install -e /path/to/zils`). Operate fleets from
a full source checkout: generated miner bundles include its requirements and
guides. No subnet package publication or standalone wheel distribution is part
of this split.
