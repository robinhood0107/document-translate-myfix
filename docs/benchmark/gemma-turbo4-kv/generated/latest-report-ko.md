# Gemma Turbo4 KV-V latest report

Status: REJECT (fixed-seed output gate)

- Median / confidence interval: not available. Fork F16 differs from shipping F16, so Turbo4 ABBA was not started.
- Fixed-seed replay: fork F16 differs from shipping F16 in 26 of 73 responses; Turbo4 differs from both controls in 72 of 73 responses. Fork F16 and Turbo4 were each stable across their two completed structural replays.
- One completed structural pass only, not promotion timing: shipping F16 46.927s, fork F16 45.515s, Turbo4 223.554s.
- Peak VRAM: 11,925 MiB. Windows available RAM minimum: 3.77 GiB. WSL swap growth: 2.44 MiB. Shared GPU growth: 231.1 MiB. These are observations, not independent rejection gates.
- OOM: 0. Orphan container: 0. GPU release was confirmed after every arm.
- R3 estimate: 13,677 MiB / 12,282 MiB = 111.36%, above the 90% preflight threshold. Active Paddle+Gemma residency was not run.
- Follow-up: read-only mounted GGUF file SHA-256 was recalculated and matched the pinned IQ4_NL identity.
