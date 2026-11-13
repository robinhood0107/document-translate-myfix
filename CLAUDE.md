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

## Product-code subagent routing

When the active Codex coordinator uses a Sol-family model, Sol remains the task
owner and continues investigation, progress coordination, independent
verification, full audit, user communication, and final reporting. Delegate
product source code, implementation tests, and execution runs to a
`gpt-6-luna` subagent at `max` or `ultra` reasoning effort. Sol independently
audits the full resulting code, diff, and run evidence, then owns the final
report.

Long execution runs stay token-efficient without weakening evidence: batch
jobs into one detached chain script that runs, post-audits, proves GPU
release, and records an attempt ledger per job, stopping on the first failure;
wait for its single completion instead of polling or tailing logs; read
ledger/summary JSON rather than raw logs; pass a model-free dry audit of pins,
readiness, inputs, and audit wiring before any paid or model run; and open a
fresh implementation subagent per phase with a compact brief. Training and
follow-up absence is observed twice, at least 300 seconds apart, once per chain
start; each job still re-checks ownership, containers, and GPU state without
that wait.

Setup and launch are separate Windows entrypoints. `setup*.bat` (core tier,
including PaddleOCR VL Spotting geometry support) and `setup_full*.bat` (adds
MangaLMM plus the full-only MIT 48px AR/CTC OCR assets) run the shared PowerShell bootstrap, which owns the isolated venvs,
complete default application-model seal, and managed model volumes and never
launches the application. `run_comic.bat` and `run_comic_cuda13.bat`
remain dedicated CUDA12/CUDA13 launch entrypoints and must never create a venv,
install packages, download models, pull images, create volumes, or invoke a
`prepare_*` script. The application never repairs installation state; missing
core seals fail before page work with `setup*.bat`; MangaLMM and selected MIT
48px assets require `setup_full*.bat`. MIT 48px assets stay out of the
core/default OCR profile, and the app does not download or substitute them.
Treat Python 3.12 x64, WSL2,
Docker Desktop, and the NVIDIA driver as prerequisites. Keep CI fast and source-contract-only; do not add a clean
Windows package/model bootstrap job, but enforce release dependency closure.

For Gemma sampler work, judge only the normalized translated sentence. Preserve
meaning and naturalness; accept meaning-preserving wording or onomatopoeia
variation; strip known channel/thought framing; and retain envelope shape only
as private diagnostics. The complete invariant and reuse rule are in
`rules.md`; do not replay completed raw responses merely to apply it.

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

PR metadata must include issue references, a measurement denominator, command
results, failed/unjudged/unmeasured counts, evidence state, impact, rollback,
and unrun checks. The contract is enforced by the required
`validate-pr-metadata` check on open, reopen, synchronize, edit, and
ready-for-review events. See
`docs/repo/workflow-reporting-contract-ko.md` for the sanitized report fields,
private-run lifecycle, and limits of current app reports.
