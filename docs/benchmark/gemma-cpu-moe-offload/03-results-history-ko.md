# Gemma CPU-MoE / Host-KV Lab Results History

| 날짜 | 후보 | 물리 동시 상주 | raw 재현성 | 의미 품질 | 속도 승격 |
|---|---|---|---|---|---|
| 2026-08-02 | `--no-kv-offload` only | FAIL | 미실행 | 미검수 | 미진입 |
| 2026-08-02 | `--no-kv-offload --n-cpu-moe 10` | FAIL | 미실행 | 미검수 | 미진입 |
| 2026-08-02 | `--no-kv-offload --n-cpu-moe 11` | PASS — sentinel 11,890MiB, OOM/orphan 0 | FAIL — response ledger 18/73 non-exact | PASS — source-first context review에서 candidate-only 의미 회귀 0 | 미진입 — v1 raw-exact gate 실패; 단일 73-request 관측은 느림 |

`N=11`의 제품 승격 REJECT는 raw 재현성 및 속도 진입 실패의 판정이다. 18개 raw
차이 자체를 의미 품질 FAIL로 해석하지 않는다.
