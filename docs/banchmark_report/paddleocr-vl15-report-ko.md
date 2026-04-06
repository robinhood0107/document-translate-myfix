# PaddleOCR-VL-1.5 Runtime Benchmark 상태 보고서

이 문서는 `PaddleOCR-VL 1.5` 공식 `detect+ocr-only` suite의 최신 상태를 기록합니다.

## 보고서 메타데이터

- 생성 시각: `2026-04-07 08:45:52 KST`
- 상태: `closed-failed`
- 벤치마킹 이름: `PaddleOCR-VL-1.5 Runtime Benchmark`
- 벤치마킹 종류: `managed family suite`
- 벤치마킹 범위: `official default suite uses detect+ocr-only offscreen execution with warm-stable quality gate`
- `execution_scope`: `detect-ocr-only`
- `official_score_scope`: `detect+ocr-only`
- runtime_services: `ocr-only`
- results root: `./banchmark_result_log/paddleocr_vl15`
- active run: `./banchmark_result_log/paddleocr_vl15/20260407_052946_paddleocr-vl15-runtime_suite`

## 라운드 결론

- 최종 winner: `없음`
- develop 승격 가능: `False`
- 판정: `승격 실패`
- 종료 사유: `완료된 confirm 후보가 모두 baseline warm median 대비 5% 개선 조건을 만족하지 못함`
- 비고: `phase 3a confirm 도중 라운드를 중단하고 no-promotion으로 종료 처리함`

## Baseline Confirm

- warm median: `468.096s`
- warm spread: `1.24%`
- stable_page_count: `30`
- excluded_unstable_pages: `0`
- `page_failed_count = 0`
- `ocr_cache_hit_count = 0`

## 완료된 Confirm 결과

| phase | preset | official warm median detect+ocr sec | 결과 |
| --- | --- | --- | --- |
| phase-1-workers-and-hpip | `paddleocr-vl15-baseline-hpip-workers1` | `478.261` | `실패` |
| phase-2-max-concurrency | `paddleocr-vl15-baseline-conc32` | `475.448` | `실패` |

## 중단 시점 진행 상태

- phase 3a `gpu_memory_utilization` screen best:
  - `vram080`: `168.122s`
  - `vram076`: `172.411s`
  - `vram072`: `167.643s`
- phase 3a confirm 진행값:
  - `paddleocr-vl15-baseline-vram072`
  - warm1: `464.431s`
  - warm2: `484.472s`
  - warm3: `미완료`
- 이 시점에서 라운드를 더 진행하지 않고 `승격 없음`으로 종료 처리함

## 판정 메모

- screen 단계에서는 baseline보다 약간 빠른 후보가 반복해서 나왔음
- 그러나 full confirm으로 올리면 baseline `468.096s`를 5% 이상 개선한 후보가 나오지 않았음
- 따라서 이번 라운드에서는 제품 runtime/config 승격을 하지 않음

## 산출물

- active run root: `./banchmark_result_log/paddleocr_vl15/20260407_052946_paddleocr-vl15-runtime_suite`
- latest 자산은 이 종료 상태를 반영한 문서 기준으로 해석해야 함
- 이전 smoke 기반 차트/CSV는 참고용이며, 이번 실패 종료 라운드의 최종 승격 판단 근거는 아님
