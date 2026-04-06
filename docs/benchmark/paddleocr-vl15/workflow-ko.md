# PaddleOCR-VL 1.5 벤치 워크플로

## 목적

- 실제 offscreen 앱 파이프라인을 그대로 실행
- 공식 점수와 품질 게이트는 detect+ocr만 사용
- `/Sample` 30장, `cold 1 + warm 3` 프로토콜 고정

## baseline

- preset: `paddleocr-vl15-baseline`
- latest `develop` 초기 상태를 기준으로 고정
- baseline cold 결과에서 gold JSON 생성

## suite phase

1. baseline
2. `parallel_workers` / `--use_hpip`
3. `max_concurrency`
4. `gpu_memory_utilization`
5. `max_num_seqs`
6. `max_num_batched_tokens`
7. conditional layout GPU
8. 필요 시 code candidate

각 phase는 현재 best 기준으로만 다음 phase에 진입합니다.

## 승격 조건

- detection pass
- OCR pass
- `page_failed = 0`
- `ocr_cache_hit = 0`
- warm detect+ocr median 5% 이상 개선

## 산출물

- suite raw metrics
- page-level gold/candidate snapshot
- compare result JSON/MD
- cold/warm aggregate summary
- latest report + history snapshot
