# Agent Instructions

Before making any repository-scoped code change, read [rules.md](./rules.md).

Mandatory requirements:

- Follow the branch, commit, push, PR, i18n, and CI rules in [rules.md](./rules.md).
- Treat `main` and `develop` as protected branches.
- Do not consider a feature complete until it is committed and pushed, unless the user explicitly requests local-only work.
- When adding or changing user-visible UI text, update the Qt translation files and compiled `.qm` assets.
- Keep benchmark policy, preset selection, ranking, and report generation outside core business code. Only generic stage hooks and telemetry/stat surfaces may remain in the pipeline/runtime layers.
- Treat `benchmarking/lab` as the dedicated long-lived benchmark branch. Benchmark-specific runners, presets, reports, and chart assets belong there, not on `main` or `develop`.
- Benchmark-validated product runtime/default promotions are allowed on `develop` as long as benchmark-only assets stay out of the PR.
- Prefer benchmark families that run the real offscreen app pipeline, keep raw results under `./banchmark_result_log/<family>/`, and ship paired pipeline/suite BAT launchers for CUDA12 and CUDA13 when feasible.
- If a benchmark family only needs an early pipeline stage for its official score, prefer a generic stage ceiling and document `execution_scope` separately from `official_score_scope`.

If these instructions conflict with any other repo-local guidance, [rules.md](./rules.md) wins.
