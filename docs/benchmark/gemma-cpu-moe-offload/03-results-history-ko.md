# Gemma CPU-MoE / Host-KV Lab Results History

| 날짜 | 후보 | 판정 | 근거 |
|---|---|---|---|
| 2026-08-02 | `--no-kv-offload` only | REJECT | Gemma와 Paddle peak 합산이 물리 VRAM을 넘었다. |
| 2026-08-02 | `--no-kv-offload --n-cpu-moe 10` | REJECT | 합산 추정 12,370MiB로 12,282MiB를 88MiB 초과했다. |
| 2026-08-02 | `--no-kv-offload --n-cpu-moe 11` | REJECT for promotion | actual dual load는 가능했으나 fixed-seed completion 18/73이 달랐고 73-request replay도 느렸다. |

이 표의 REJECT는 제품 기본값 또는 속도 후보로의 승격 판정이다. 동시 load 가능
여부와 output/throughput 승격은 별개의 판정이다.
