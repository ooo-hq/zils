# Repository boundaries

`ooo-hq/zils` is the miner and validator repository. Hosted product development
and client tooling belong in separate repositories.

| Repository | Code and supporting material |
| --- | --- |
| [zils](https://github.com/ooo-hq/zils) | `miner/` and the `zils-miner` CLI, shared `zils/` model/evaluation/protocol code, validators, reference downloads, operator docs and tests |
| [zils-platform](https://github.com/ooo-hq/zils-platform) | `zils_platform/` APIs, queue orchestration, billing, storage, serving; migrations and service integration tests |
| [zils-sdk](https://github.com/ooo-hq/zils-sdk) | Python and TypeScript packages, CLI, MCP, client documentation and contract tests |
| [zils-web](https://github.com/ooo-hq/zils-web) | Website and customer-facing application |
| [zils-docs](https://github.com/ooo-hq/zils-docs) | Public documentation website (`docs.zils.ai`) |

Dependencies flow from the platform into the subnet's shared code. Miners and
validators must not import platform, SDK, Stripe, or storage-provider modules.
The subnet boundary test enforces this. SDK applications communicate over HTTP
and need no subnet or platform checkout.

Hosted services run as `python -m zils_platform.<service>`. The hosted queue
processor lives in the platform; independent model scoring and acceptance stay
here, and the platform calls them. Public experiment methodology and aggregate
results also stay here.

The Python metadata lets the platform install this checkout as an editable
source dependency (`python -m pip install -e /path/to/zils`). Miners install
from a source checkout at the stable release ([releases](releases.md)); there is
no subnet package on PyPI.
