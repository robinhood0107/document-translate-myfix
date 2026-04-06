# PaddleOCR-VL 1.5 벤치 워크플로

## 목적

- 공식 기본 경로는 `execution_scope=detect-ocr-only`
- 공식 점수와 품질 게이트는 `official_score_scope=detect+ocr-only`
- baseline은 `/Sample` 30장 `cold 1 + warm 3`
- 후보 비교는 `screen subset -> full confirm`

## baseline confirm

- preset: `paddleocr-vl15-baseline`
- latest `develop` 초기 상태를 기준으로 고정
- full corpus `cold 1 + warm 3`
- baseline 결과로 warm-stable profile 생성
- 안정 페이지로 screen subset 자동 선택

## suite phase

1. baseline confirm
2. `parallel_workers` / `--use_hpip`
3. `max_concurrency`
4. `gpu_memory_utilization`
5. `max_num_seqs`
6. `max_num_batched_tokens`
7. conditional layout GPU
8. optional code candidate

각 phase는 아래 프로토콜을 따릅니다.

1. screen subset `cold 1 + warm 1`
2. 상위 1개, 점수 차이 1% 이내면 상위 2개까지 confirm
3. full confirm `cold 1 + warm 3`
4. confirm winner만 다음 phase에 진입

## 승격 조건

- detection pass
- OCR pass
- `page_failed = 0`
- `ocr_cache_hit = 0`
- full confirm warm detect+ocr median 5% 이상 개선
- tie-breaker 1: warm OCR p95
- tie-breaker 2: cold detect+ocr total

## 산출물

- suite raw metrics
- page-level gold/candidate snapshot
- compare result JSON/MD
- cold/warm aggregate summary
- latest report + history snapshot
