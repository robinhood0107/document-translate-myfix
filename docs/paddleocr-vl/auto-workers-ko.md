# PaddleOCR VL Auto Workers Runtime Notes

## 목적

- `PaddleOCR VL`의 기존 block-crop 경로를 유지한 채, benchmark 전용 hidden flag로 worker scheduling 후보를 비교할 수 있게 한다.
- 제품 기본값은 `fixed_area_desc_w8`이며, single-tenant benchmark에서 승인된 winner를 기본 동작으로 사용한다.

## Hidden Flags

- 환경변수: `CT_PADDLEOCR_VL_SCHEDULER_MODE`
- benchmark preset 주입값: `ocr_generic.paddleocr_vl_scheduler_mode`
- 허용 모드:
  - `fixed`
  - `fixed_area_desc`
  - `auto_v1`

환경변수가 있으면 환경변수가 우선한다. 잘못된 값은 무시되고 기본값 `fixed_area_desc`로 돌아간다.

## Runtime Contract

- 제품 기본 `scheduler_mode`는 `fixed_area_desc`다.
- 제품 기본 `parallel_workers`는 `8`이다.
- `parallel_workers`는 기본적으로 고정 worker 수다.
- `fixed_area_desc`와 `auto_v1`에서만 `crop_area_px desc` 정렬이 적용된다.
- `auto_v1`는 local server(`localhost`, `127.0.0.1`, `::1`)일 때만 cached GPU headroom을 반영한다.
- remote server는 crop 통계만 사용한 fallback 규칙으로 worker 수를 정한다.
- `fixed`, `auto_v1`는 override/diagnostic/benchmark 용도로 계속 유지된다.

## Telemetry Surface

- `PaddleOCRVLEngine.last_page_profile`
  - `scheduler_mode`
  - `requested_cap`
  - `chosen_workers`
  - `block_count`
  - `p50_area_ratio`
  - `p90_area_ratio`
  - `large_crop_ratio`
  - `request_records`
- batch benchmark event의 `ocr_end` / `page_failed(failed_stage=ocr)`에 `ocr_page_profile`이 같이 실린다.

## 비고

- 이 문서는 제품 runtime contract만 설명한다.
- benchmark preset, gold, runner, 결과 표, 포트폴리오형 narrative 문서는 `benchmarking/lab`에서 관리한다.
- 이번 기본값 승격 근거는 `20260415_031602_paddleocr-vl-parallel-smoke` single-tenant 결과와 사용자 OCR diff 승인이다.
