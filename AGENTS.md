# Project standards

Zils is a professional developer tool. Write code and documentation for external
engineers, operators, and investors who have no access to the maintainer's setup.

- Public setup instructions must work from a fresh clone. Create required inputs
  explicitly and identify platform requirements and placeholders.
- Keep personal machine names, private addresses, workstation paths, operational
  diaries, and agent session plans in ignored `.private/` storage.
- Publish experiment methodology, aggregate results, relevant hardware, and
  limitations. Separate demonstrated behavior from planned capabilities;
  preserve unfavorable results and distinguish experiments from releases.
- Keep the implementation small and inspect existing code before adding a new
  dependency or abstraction. Verify commands, links, and claims before publishing.

# Repository boundary

This repository is for miners, validators, and their shared model, evaluation,
and protocol contracts. Hosted APIs, billing, database/storage adapters and
orchestration live in `zils-platform`. Client SDKs, CLI and MCP live in `zils-sdk`;
the website lives in `zils-web`. Do not add product services or their dependencies
here. Keep signed protocol versions and model identities stable during moves.

# Architecture references

- Read `docs/repositories.md` for ownership and dependency direction.
- Read `docs/evaluation.md` and `docs/customer-jobs.md` for data and acceptance.
- Read `docs/validators.md`, `docs/mining.md`, and `docs/queue-miners.md` for operators.
- Read `docs/roadmap.md` for limits. Mainnet and public discovery are not implemented.
