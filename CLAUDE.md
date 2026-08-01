# Repository Agent Compatibility

This file is a compact compatibility and skill-routing entrypoint for
Claude-compatible tooling and gstack. It is **not** a parallel policy source.
For Codex, the repository entrypoint is [AGENTS.md](./AGENTS.md); for every
agent, [rules.md](./rules.md) is authoritative.

## Required read order

1. [AGENTS.md](./AGENTS.md)
2. [rules.md](./rules.md)
3. The closest applicable nested instruction file and touched implementation

Do not create a worktree. Work in the checkout, protect `main` and `develop`,
keep `main`, `develop`, and `benchmarking/lab`, and use only the supported
Windows environments named in `rules.md` for repository workflows.

## Skill routing

Use an installed, namespaced gstack skill when its task matches. In particular:

- Architecture, data flow, edge cases, and test plans: `gstack-plan-eng-review`
- CEO or scope decisions: `gstack-plan-ceo-review`
- Bugs and unknown failures: `gstack-investigate`
- Code or PR review: `gstack-review`
- Web QA: `gstack-qa` or `gstack-qa-only`
- Shipping and PR preparation: `gstack-ship`
- Merge, release, and post-merge verification: `gstack-land-and-deploy`

Follow the repository browser safety rules for browser work. Do not substitute
an uninstalled skill or an unrelated browser path.

## Artifact and instruction contract

Public operational documentation belongs in `docs/`; raw validation evidence,
including images and models, belongs only in the ignored
`banchmark_result_log/` archive. Treat uncertain material as private. Follow
the manifest, metadata-preserving move, and atomic-unit safeguards in
`rules.md`; never discard large media during consolidation.

Debug and benchmark scripts that create default private output must use the
managed artifact harness with an explicit category. Its manifest behavior is a
fast CI-tested contract; an explicit user-supplied output directory remains a
deliberate local override.

When changing repository workflow, artifact handling, validation, release, or
agent policy, update `AGENTS.md`, `CLAUDE.md`, and `rules.md` together in one
commit and one PR; local pre-commit and PR CI enforce the synchronization. Also
update the hook, validator, CI workflow, or ruleset that enforces the policy
when one exists. Do not record an AI identity in Git attribution.
