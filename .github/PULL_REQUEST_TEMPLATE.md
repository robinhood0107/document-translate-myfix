## Summary

<Describe one reviewable logical change.>

## Issue references

<Use `Closes #N` when this PR completes an issue; use `Refs #N` when work remains.>

## Measurement denominator

<Name the confirmed required units and denominator, or explain why a denominator does not apply.>

## Validation commands and results

<List each command and its pass/fail/not-run result.>

## Failed / unjudged / unmeasured

- failed: <count>
- unjudged: <count>
- unmeasured: <count>

## Evidence status

status: <complete|partial|unavailable> — <explain missing or partial evidence.>

## Impact

<Describe affected code, users, and data.>

## Rollback

<Describe the revert or recovery action, or explain why rollback does not apply.>

## Not run

<List checks and experiments not run, or state that none were skipped.>

## Checks

- [ ] I read `rules.md` before repository-scoped changes.
- [ ] I updated `AGENTS.md`, `CLAUDE.md`, and `rules.md` together for policy changes.
- [ ] I worked on a valid non-protected branch and pushed it before requesting review.
- [ ] I selected the intended base branch (`develop`, `main`, or `benchmarking/lab`).
- [ ] I considered documentation and changelog impact.
- [ ] I updated Qt translations and compiled `.qm` assets for user-visible UI text changes.
- [ ] I excluded private source names, local paths, raw outputs, and credentials.
- [ ] For release or main-promotion work, I built and verified the launcher-source ZIP, SHA-256 manifest, and both `COMIC_VERIFY_ONLY=1` launcher contracts.
- [ ] Required local validation passed, or the unrun reason is recorded above.
