# Workflow Split Runtime Gate And Threshold Proposal

아이디어 착안자: 사용자

## 목적

이 문서는 현재까지의 Requirement 1 실측 결과를 정리하고, flow 비교의 최종 판정과 Requirement 2 hybrid selector 트랙의 종료 결론을 남긴다. 아래 threshold 내용은 historical proposal이며, 2026-04-18 기준으로는 실제 제품 승격 기준으로 채택하지 않는다.

## 1. 총 실측 기준에서 가장 빠른 구성

현재 공식 Japanese 13장 suite 기준 총 소요 시간은 아래와 같다.

| scenario | 의미 | total_elapsed_sec |
| --- | --- | ---: |
| `baseline_legacy` | 기존 페이지 단위 파이프라인 | `995.846` |
| `candidate_stage_batched_single_ocr` | Japanese `Optimal` | `714.725` |
| `candidate_stage_batched_dual_resident` | Japanese `Optimal+ analysis mode` | `1664.021` |

### 현재 결론

- Docker 기동과 health wait를 포함한 총 실측 기준으로는 `candidate_stage_batched_single_ocr`가 가장 빠르다.
- `candidate_stage_batched_single_ocr`는 `baseline_legacy`보다 약 `28.2%` 빠르다.
- `candidate_stage_batched_dual_resident`는 현재 형태로는 가장 느리다.
- `candidate_stage_batched_single_ocr`와 `baseline_legacy`는 공식 quality summary 기준으로 `detect_box_total=212`, `ocr_non_empty_total=212`, `page_failed_count=0`를 동일하게 만족한다.

이 값은 다음을 뜻한다.

1. `stage_batched_pipeline` 자체는 일본어 기준에서 시간 이득 가능성이 있다.
2. 하지만 `Optimal+ analysis mode`처럼 `PaddleOCR VL + MangaLMM`를 모든 페이지에 다 돌리는 방식은 steady-state 기본값으로는 비효율적이다.
3. 따라서 `Optimal+ Japanese`는 지금 단계에서는 “정식 기본 OCR 전략”이 아니라 “selector 기준을 만들기 위한 분석 모드”로 봐야 한다.
4. 2026-04-18 기준 최종 승격 후보는 `stage_batched_pipeline + Japanese Optimal(PaddleOCR VL 중심)`이다.

## 2. Docker 기동/대기 시간을 포함해서 봤을 때의 해석

stage-batched 시나리오의 분해표는 아래와 같다.

| scenario | compose_up_sec | health_wait_sec | pure_processing_sec |
| --- | ---: | ---: | ---: |
| `candidate_stage_batched_single_ocr` | `2.283` | `119.021` | `707.423` |
| `candidate_stage_batched_dual_resident` | `5.860` | `231.691` | `1656.400` |

정리하면:

- `single_ocr`는 OCR 1계열만 올리므로 OCR stage 준비 시간이 상대적으로 짧다.
- `dual_resident`는 `PaddleOCR VL + MangaLMM`를 모두 health ready까지 기다려야 하므로 startup 비용이 커진다.
- 게다가 현재 `analysis mode`는 모든 페이지에서 sidecar까지 다 돌리므로 pure processing 시간도 매우 커진다.

즉, 지금 benchmark에서 속도 1등은 확실히 `Japanese Optimal(single)`이다.

## 3. 그럼 `Optimal+ Japanese`는 왜 필요한가

이 모드는 속도 1등을 위한 모드가 아니라, 아래 문제를 해결하기 위한 준비 단계다.

- `MangaLMM`이 잘 되는 페이지는 MangaLMM을 쓰고 싶다.
- 하지만 `p_016.jpg` 같은 hard page는 품질을 지키기 위해 `PaddleOCR VL` fallback이 필요하다.
- 그래서 “어느 정도 mismatch면 Paddle로 넘길지” 기준선이 필요하다.

즉, 현재 dual-resident run의 역할은 다음과 같다.

1. `MangaLMM`을 전 페이지에 sidecar로 돌려 본다.
2. detector box 수와 `bbox_2d` 성공 수 차이를 모은다.
3. 사용자 O/X 검수로 허용 가능한 mismatch 구간을 잠근다.
4. 그 뒤에야 selector-enabled final run을 다시 측정한다.

## 4. 현재 데이터로 본 provisional threshold 제안

이번 문서에서는 `bbox_2d`를 기준으로 아래 지표를 제안한다.

- `bbox_mismatch_ratio = (detect_box_count - bbox_2d_success_block_count) / detect_box_count`

추가로 함께 본다.

- `miss_count = detect_box_count - bbox_2d_success_block_count`

### 제안 규칙

#### A. MangaLMM 통과 후보

- `bbox_mismatch_ratio <= 0.15`

#### B. 수동 검토 구간

- `0.15 < bbox_mismatch_ratio <= 0.25`

#### C. PaddleOCR VL fallback 후보

- `bbox_mismatch_ratio > 0.25`

추가 safeguard:

- `detect_box_count >= 20` and `miss_count >= 5`
  - ratio가 경계선이어도 fallback 쪽으로 본다.
- hard page (`p_016.jpg`)는 기본 fallback 후보로 본다.

## 5. 페이지별 provisional band

| page | detect | bbox_2d_success | miss_count | bbox_mismatch_ratio | suggested_band |
| --- | ---: | ---: | ---: | ---: | --- |
| `094.png` | 20 | 15 | 5 | `0.250` | `fallback_candidate_large_gap` |
| `097.png` | 13 | 12 | 1 | `0.077` | `keep_candidate` |
| `101.png` | 6 | 4 | 2 | `0.333` | `fallback_candidate` |
| `i_099.jpg` | 17 | 14 | 3 | `0.176` | `review_band` |
| `i_100.jpg` | 16 | 14 | 2 | `0.125` | `keep_candidate` |
| `i_102.jpg` | 19 | 15 | 4 | `0.211` | `review_band` |
| `i_105.jpg` | 10 | 6 | 4 | `0.400` | `fallback_candidate` |
| `p_016.jpg` | 30 | 5 | 25 | `0.833` | `fallback_candidate_hard_page` |
| `p_017.jpg` | 15 | 15 | 0 | `0.000` | `keep_candidate` |
| `p_018.jpg` | 18 | 15 | 3 | `0.167` | `review_band` |
| `p_019.jpg` | 9 | 9 | 0 | `0.000` | `keep_candidate` |
| `p_020.jpg` | 30 | 26 | 4 | `0.133` | `keep_candidate` |
| `p_021.jpg` | 9 | 8 | 1 | `0.111` | `keep_candidate` |

## 6. 현재 시점의 운영 권장안

### 지금 당장 속도가 가장 중요한 경우

- `stage_batched_pipeline + Japanese Optimal`

이 구성이 가장 빠르다.

### 품질 기준선을 만들면서 나중에 MangaLMM을 최대한 살리고 싶은 경우

- `stage_batched_pipeline + Optimal+ Japanese analysis mode`
- 다만 지금은 분석/검수용으로만 사용
- 사용자 O/X가 쌓이기 전에는 steady-state 기본 모드로 승격하지 않음

### 현재 권장되는 fallback 방향

- selector 전:
  - downstream 기준은 `PaddleOCR VL`
- selector 후:
  - `MangaLMM` 먼저 평가
  - `bbox_mismatch_ratio`가 threshold를 넘으면 `PaddleOCR VL` fallback

## 7. 다음 단계

1. Requirement 1은 `flow gain confirmed`로 잠근다.
2. `feature/workflow-split-runtime`에서 `stage_batched_pipeline`를 제품 승격 대상으로 준비한다.
3. hybrid selector 대신, CTD 마스킹 경로를 실제 배치/benchmark 경로에 연결하고 residue cleanup smoke를 통과시킨 뒤 제품 승격으로 넘어간다.

## 8. Requirement 2 종료 결론

2026-04-18 기준으로 Requirement 2의 `MangaLMM` hybrid selector benchmark는 `failed_closed`로 종료한다.

근거는 아래와 같다.

1. 공식 suite에서 `candidate_stage_batched_dual_resident`는 `1664.021s`로, `candidate_stage_batched_single_ocr`(`714.725s`)보다 현저히 느리다.
2. review pack 기준으로 `text_bubble` 누락, 인접 bubble 병합, block 분할 불안정성이 반복 관찰되었다.
3. 따라서 현재 상태의 `MangaLMM first + selective Paddle fallback`은 benchmark 기준으로 설명 가능하고 재현 가능한 운영안까지 올라오지 못했다.

결론적으로 이 문서의 threshold 제안은 historical note로만 남기고, 현 시점 제품 승격 기준은 다음으로 단순화한다.

- `legacy_page_pipeline`
- `stage_batched_pipeline + Japanese Optimal(PaddleOCR VL 중심)`

## 9. CTD 마스킹 연결 상태

2026-04-18 기준 benchmark 경로에서 CTD 마스킹 연결과 residue cleanup smoke가 통과했다.

- smoke inputs:
  - `094.png`
  - `p_016.jpg`
- representative metadata:
  - `banchmark_result_log/inpaint_debug/20260418_150438_sample-debug-export/japan/debug_metadata/094_debug.json`
  - `banchmark_result_log/inpaint_debug/20260418_150441_sample-debug-export/japan/debug_metadata/p_016_debug.json`

두 output 모두 아래를 만족한다.

- `mask_refiner == "ctd"`
- `keep_existing_lines == true`
- `protect_mask_applied == true`
- `cleanup_applied == true`

따라서 Requirement 1 gate 이후 남아 있던 마지막 benchmark blocker는 “CTD 마스크 경로가 실제로 연결되었는가”였고, 이 항목은 현재 `passed_smoke_on_benchmarking_lab` 상태로 본다.
