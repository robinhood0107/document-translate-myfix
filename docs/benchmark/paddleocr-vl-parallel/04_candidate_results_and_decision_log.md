# 04 Candidate Results And Decision Log

## 목적

이 문서는 candidate matrix 결과, 탈락 사유, 최종 winner를 기록하는 결정 로그다.

이 문서의 역할은 단순 결과 표를 복사해 두는 것이 아니다. 어떤 후보가 왜 살아남았고 왜 탈락했는지를, 이후 제품 승격 판단과 회고 문서에서 다시 사용할 수 있도록 narrative 형태로 남기는 것이 목적이다.

## candidate matrix

- `fixed_w1..w8`
- `fixed_area_desc_w1..w8`
- `auto_v1_cap4`
- `auto_v1_cap6`
- `auto_v1_cap8`

## 기록 규칙

- 각 후보는 `warmup 1 + measured 3`
- speed rank는 `ocr_total_sec median`
- tie-break는 `ocr_page_p95_sec`, 그 다음 `elapsed_sec`
- baseline은 항상 `fixed_w8`
- 품질 게이트는 자동 탈락 규칙이 아니라 `검수 우선순위 보조 지표`다.
- 사용자 검수용 diff pack은 `speed rank 기준 top-2 non-baseline candidates`만 생성한다.

즉, 이 문서의 순위는 “최종 자동 승자”를 선언하는 문서가 아니라, “단독 상주 속도 랭킹과 사용자 검수 대상 후보”를 정리하는 결정 로그다.

## 문서화 원칙

- 각 후보별로 통과/실패를 기록한다.
- 실패 후보는 반드시 `품질 게이트 실패 사유` 또는 `runtime failure 사유`를 남긴다.
- latest report와 conclusion card를 이 문서에서 링크한다.
- `quality_gate_winner`와 `review_top1/review_top2`를 분리해 기록한다.

추가로 아래 정보를 남긴다.

- baseline 대비 어떤 점이 빨라졌는지
- baseline 대비 어떤 점이 나빠졌는지
- `fixed_area_desc`가 단순 정렬만으로 어떤 효과를 냈는지
- `auto_v1`가 실제 local headroom에서 worker를 얼마나 낮췄는지
- 향후 2차에서 검토할 가치가 있는 가설
- `PaddleOCR VL 단독 상주 상한선 benchmark`라는 실행 계약

## 현재 기록 상태

실측 결과는 suite가 완료될 때마다 아래 자산으로 동기화한다.

- `docs/banchmark_report/paddleocr-vl-parallel-report-ko.md`
- `docs/assets/benchmarking/paddleocr-vl-parallel/latest/latest_summary.md`
- `docs/assets/benchmarking/paddleocr-vl-parallel/latest/conclusion_card.md`
- `docs/assets/benchmarking/paddleocr-vl-parallel/latest/`

이 문서는 보고서의 요약판이 아니라, “어떤 결정이 어떤 근거로 내려졌는가”를 추적하는 의사결정 문서다. 따라서 결과가 누적될수록 탈락 사유와 해석을 중심으로 채워 넣는다.

이번 family는 `runtime_services=ocr-only`, `stage_ceiling=ocr`로 고정되며, Gemma와 MangaLMM이 올라오지 않는 `PaddleOCR VL 단독 상주` 상태를 기준으로 한다.

## 최신 single-tenant smoke 결과

사용자 승인으로 잠긴 latest suite는 `20260415_031602_paddleocr-vl-parallel-smoke`다.

- runtime_contract: `paddleocr-vl-single-tenant-ocr-only`
- runtime_services: `ocr-only`
- stage_ceiling: `ocr`
- runtime container names: `['paddleocr-server', 'paddleocr-vllm']`
- `gemma-local-server booted = False`

속도 랭킹은 아래와 같다.

- `fixed_area_desc_w8`: `ocr_total_sec_median=292.526`
- `fixed_w8`: `ocr_total_sec_median=293.048`
- `auto_v1_cap4`: `ocr_total_sec_median=295.765`

baseline은 계속 `fixed_w8`다. 이번 최종 프로토콜에서는 숫자 게이트를 자동 탈락 기준으로 쓰지 않고, `speed rank top-2 non-baseline`을 OCR diff 검수 후보로 고정한다.

- `quality_gate_winner = fixed_area_desc_w8`
- `review_top1 = fixed_area_desc_w8`
- `review_top2 = auto_v1_cap4`
- `final_promotion_status = approved_fixed_area_desc_w8`

이번 단독 상주 smoke에서는 상위 3개 후보 모두 aggregate 품질 지표가 동일했다.

- `mean_CER = 0.0`
- `mean_exact_match = 1.0`
- `empty_block = 0`
- `page_failed_count = 0`

추가로 review pack 기준 `fixed_area_desc_w8`, `auto_v1_cap4` 모두 baseline `fixed_w8` 대비 changed block이 없었다. 따라서 이번 라운드의 최종 develop promotion winner는 `fixed_area_desc_w8`로 확정되었다.

## 기록 예시 원칙

- `fixed_w8`이 baseline으로 채택된 이유
- `fixed_area_desc_w8`가 tail latency를 줄였는지 여부
- `auto_v1_cap4/6/8` 중 품질 게이트를 통과한 후보가 있는지 여부
- winner가 hidden flag 상태로만 promotion 되는 이유

## 다음 단계

suite가 완료되면 이 문서에는 다음을 추가한다.

- speed rank top-2 요약
- quality gate winner 요약
- 탈락 후보 대표 사례
- 품질 게이트 실패 패턴
- 사용자 검수용 diff pack 링크
- develop 승격 보류/승인 상태

현재 결정 이후의 다음 단계는 다음과 같다.

- `fixed_area_desc_w8`를 develop 기본값으로 승격
- `fixed`, `auto_v1`는 hidden override/diagnostic mode로 유지
- 단독 상주 계약과 승격 근거는 benchmarking/lab 문서와 latest/history 자산에 계속 보존
- 이후 2차에서만 `weighted concurrency`, `22장 full corpus`, `MangaLMM 동시 상주`를 다시 검토

## 저자 및 기여

- 핵심 문제 해결 방향은 사용자가 착안했다.
- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
