# Gemma Turbo4 KV-V results history

| Date | Stage | Decision | Notes |
|---|---|---|---|
| 2026-08-02 | fixed-seed structural (shipping F16 control) | INCOMPLETE | The former host-memory policy auto-cancelled before a completed request. This is superseded by the completed run below. |
| 2026-08-02 | fixed-seed structural (shipping F16, fork F16, Turbo4) | REJECT for v1 raw-reproducibility/speed protocol; semantic quality unreviewed | All 73 fixed-seed requests completed and all GPU releases were confirmed. Fork F16 differed from shipping F16 in 26 responses; Turbo4 differed from both controls in 72. These are raw reproducibility results, not an established semantic-quality failure. Turbo4 ABBA was not started. Peak VRAM 11,925 MiB; host-memory and swap values were telemetry only. R3 estimate was 111.36%, so active dual residency was not run. |
| 2026-08-02 | mounted GGUF identity follow-up | PASS | read-only model volume에서 실제 file SHA-256을 다시 계산해 pinned IQ4_NL SHA와 일치함을 확인했다. |

Only PASS/REJECT, timing confidence interval, peak VRAM, RAM/swap observations, and calculated R3 estimate are eligible for a user-facing checkpoint. Raw translation text, prompt, response, image and local corpus identity are never recorded here.
