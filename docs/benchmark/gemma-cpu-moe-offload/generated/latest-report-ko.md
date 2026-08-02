# Gemma CPU-MoE / Host-KV Latest Report

## N11 상태 요약

- 물리 동시 상주: **PASS**. 최소 physical-fit 후보는
  `--no-kv-offload --n-cpu-moe 11`이었고 actual residency sentinel peak는
  11,890MiB였다. OOM·orphan은 0, WSL swap 증가는 0이었다.
- raw 응답 재현성: **FAIL**. 73 fixed-seed completion 중 18개가 shipping F16과
  달라 response ledger가 exact하지 않았다. 모든 완료는 `stop`이었다.
- 의미 품질: **PASS**. 원문·인접 대사·페이지 맥락의 source-first review에서
  candidate-only 민감 대사 검열·삭제, 행위/상황 반전, 관계 붕괴는 확인되지 않았다.
  독립 OCR 조각 하나와 덜 직접적인 표현 하나는 별도 입력/문체 메모로 남겼다.
- 속도 승격: **미진입**. v1 raw-exact gate 때문에 AB/BA를 실행하지 않았다.
  단일 73-request 관측은 shipping F16 46.465초, 후보 67.612초였다.

## 제품 판정

**REJECT for product promotion** — raw 재현성 및 속도 진입 조건을 통과하지 않았기
때문이다. 이는 18개가 의미 회귀로 입증됐다는 뜻이 아니다. 이 구성은 12GiB에서
동시 model load가 가능한지 보여 주는 lab evidence로 유지한다.
