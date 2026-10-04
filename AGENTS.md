# Project standards

Fez is a professional developer tool. Write code and documentation for external
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

# Architecture references

- Read `docs/customer-jobs.md` for customer data, acceptance criteria, and export rules.
- Read `docs/supabase-training.md` for the hosted queue, trust boundaries, and setup.
- Read `docs/roadmap.md` for limitations and planned capabilities. Hosted customer
  inference is planned; the training queue does not deploy prediction endpoints.
