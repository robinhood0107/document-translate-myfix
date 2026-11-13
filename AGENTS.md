# Agent Instructions

## Read order and authority

1. Read this file for the repository entrypoint.
2. Read [rules.md](./rules.md) before **any** repository-scoped work.
3. Read the closest applicable nested instructions and the touched code before editing.

`rules.md` is the canonical repository policy. If instructions conflict, it wins.
`CLAUDE.md` is a compatibility and skill-routing surface, not a second policy
source; Codex relies on this `AGENTS.md` entrypoint.

## Non-negotiable operating rules

- Treat `main` and `develop` as protected. Preserve `main`, `develop`, and
  `benchmarking/lab`; never delete them as cleanup branches.
- Work in this checkout only. Do not create Git worktrees for this repository.
- Use `.venv-win` and `.venv-win-cuda13` for supported local workflows; do not
  rely on `.venv` as the repository workflow environment.
- Keep setup and launch as separate Windows entrypoints. `setup.bat` /
  `setup_cuda13.bat` provision the core tier (HunyuanOCR, PaddleOCR VL crop,
  PaddleOCR VL Spotting geometry, Gemma) and `setup_full.bat` /
  `setup_full_cuda13.bat` add MangaLMM and the full-only MIT 48px AR/CTC OCR
  assets; all four are backed by the shared
  Windows PowerShell bootstrap, which owns venv reconciliation, the complete
  default application-model seal, and managed model volumes and never launches
  the application. `run_comic.bat` and `run_comic_cuda13.bat` stay separate
  CUDA12/CUDA13 launch entrypoints and must never create a venv, install a
  package, download a model, pull an image, create a volume, or run a
  `prepare_*` script. The application is read-only with respect to installation;
  missing core state requires setup, while MangaLMM or selected MIT 48px state
  requires setup_full before page work starts. MIT 48px assets stay out of the
  core/default OCR profile, and the app never downloads or substitutes them.
  Python 3.12 x64 (the pinned `mahotas` wheel has
  no Windows build above cp312), WSL2, Docker Desktop, and an NVIDIA driver are
  prerequisites.
- Windows launcher CI stays source-contract-only and must not perform a fresh
  package/model bootstrap. Release tests must still prove that every imported
  bootstrap module and linked setup document is included in the source ZIP.
- A repository change is complete only after its intended validation, human-only
  attribution, commit, push, and correctly targeted PR are complete, unless the
  user explicitly requests local-only work.
- Never add AI systems to author, committer, co-author, sign-off, or any other
  contributor trailer.
- Update Qt `.ts` and compiled `.qm` assets with every user-visible UI-text
  change.
- For Gemma sampler benchmarks, rank only the normalized translated sentence:
  preserve meaning and naturalness, accept meaning-preserving wording or
  onomatopoeia variation, strip known channel/thought framing first, and keep
  transport shape as private diagnostics. See the canonical quality rule in
  `rules.md`; do not replay completed raw responses merely to apply it.

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

## Public documentation and private validation artifacts

- Put public, sanitized operational and audit documentation in `docs/`.
- Put raw OCR, translation, render, benchmark, source-derived, model, cache,
  and hardware evidence in the ignored local archive
  `banchmark_result_log/`; never stage or force-add it.
- When consolidating external validation material, retain every artifact,
  including images and models, in the private archive with a metadata-preserving
  move. Move reparse points, live databases, and large media/model artifacts
  only as a verified atomic unit; do not delete, traverse, split, or silently
  rewrite them.
- If an artifact might expose a source title, local path, raw response, image,
  credential, or user data, treat it as private by default.
- New debug or benchmark scripts that create a default output directory must use
  the managed private-artifact harness with an explicit category; its manifest
  contract is covered by the repository's fast unit test.
- PR body evidence fields are checked by `scripts/validate_pr_report_contract.py` in the
  required `validate-pr-metadata` workflow on open, reopen, synchronize, edit, and
  ready-for-review events. CI reporting details and its app-report coverage limit are documented in
  `docs/repo/workflow-reporting-contract-ko.md`.

## Instruction-harness synchronization

`AGENTS.md`, `CLAUDE.md`, and `rules.md` are changed together in one commit and
one PR whenever repository workflow, artifact handling, validation, branch,
release, or agent policy changes. The local pre-commit hook and PR CI enforce
this synchronization. Update the enforcing hook, validator, CI workflow, or
ruleset in the same PR when the policy is mechanically enforceable.

Keep benchmark ranking, presets, report generation, and raw results outside
product business logic. `benchmarking/lab` holds benchmark-only harness assets;
the product branches receive only validated runtime behavior and generic
telemetry/stat surfaces.
