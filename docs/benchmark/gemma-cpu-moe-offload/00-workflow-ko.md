# Gemma CPU-MoE / Host-KV Lab Workflow

이 문서는 12GiB GPU에서 Gemma와 PaddleOCR-VL의 동시 model residency가
물리적으로 가능한지 확인하는 benchmark 전용 절차다. 제품의 기본 Gemma
설정, Compose, UI와 runtime contract는 바꾸지 않는다.

## 고정 조건

- shipping llama.cpp b10133과 기존 IQ4_NL GGUF만 사용한다.
- context 4096, parallel 1, threads 10, batch 2048, ubatch 512, GPU layers 23,
  F16 K/V는 고정한다.
- 후보가 바꿀 수 있는 항목은 `--no-kv-offload`와 `--n-cpu-moe N`뿐이다.
- TurboQuant, QAT, MTP/draft, n-gram, 새 GGUF, SSD paging은 이 lab에서 금지한다.

## 독립 판정 축

이 lab은 물리 동시 상주, raw 응답 재현성, 의미 품질, 속도 승격을 분리한다.
response ledger non-exact는 raw 재현성 FAIL일 뿐 의미 회귀 판정이 아니다.
번역 품질은 [공통 번역 후보 품질 판정 규칙](../translation-quality-evaluation-rule-ko.md)에 따라 별도 source-first 검수한다.

## 순서

1. shipping image, model identity, 도움말 옵션을 확인한다.
2. fixed-seed 요청 하나로 no-KV와 `n_cpu_moe`를 작은 값부터 screen한다.
3. Gemma incremental peak와 직접 측정 Paddle peak의 합이 물리 VRAM 이하인
   최초 후보만 73-request structural gate로 보낸다.
4. request ledger와 response ledger가 exact이면 raw 재현성 PASS로 기록하고 속도
   benchmark 후보로 보낸다. non-exact이면 raw 재현성 FAIL과 별도 의미 품질 결과를
   기록하되, v1 protocol의 속도 benchmark에는 보내지 않는다.
5. 사용자가 동시 load 가능 여부만 별도로 요청한 경우에는, 품질·속도 승격과 분리된
   residency sentinel가 같은 `N`을 fixed request로 다시 physical-fit 확인한 뒤
   두 model의 health/load와 반환을 한 번 확인할 수 있다.

OOM, image/model identity 불일치, container orphan, GPU 반환 미확인은 항상
중단 조건이다. Windows RAM, WSL swap, shared GPU 수치는 관측값으로 기록한다.
