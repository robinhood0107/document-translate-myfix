# Gemma CPU-MoE / Host-KV Latest Report

Status: **REJECT for product promotion**

- 최소 물리 fit 후보는 `--no-kv-offload --n-cpu-moe 11`이었다.
- 후보의 Gemma incremental peak는 9,635MiB였고, 직접 측정 Paddle peak를 합산한
  추정치는 12,042MiB였다.
- actual Gemma + Paddle residency sentinel peak는 11,890MiB였다. OOM과 orphan은
  없었고 WSL swap 증가는 0이었다.
- 하지만 73 fixed-seed completion 중 18개가 shipping F16과 달랐다. 모든 완료는
  `stop`이었지만 response ledger는 exact하지 않았다.
- 73-request wall time은 shipping F16 46.465초, 후보 67.612초였다.

따라서 이 구성은 12GiB에서 동시 model load가 가능한지 보여 주는 lab evidence일
뿐이며, 제품 설정 또는 속도 후보로 채택하지 않는다.
