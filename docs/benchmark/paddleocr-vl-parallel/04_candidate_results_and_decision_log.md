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
- 품질 게이트를 통과하지 못한 후보는 speed rank 대상에서 제외

즉, 이 문서의 순위는 “무조건 가장 빠른 후보”가 아니라 “품질을 유지한 채 가장 빠른 후보”를 가리킨다.

## 문서화 원칙

- 각 후보별로 통과/실패를 기록한다.
- 실패 후보는 반드시 `품질 게이트 실패 사유` 또는 `runtime failure 사유`를 남긴다.
- latest report와 conclusion card를 이 문서에서 링크한다.

추가로 아래 정보를 남긴다.

- baseline 대비 어떤 점이 빨라졌는지
- baseline 대비 어떤 점이 나빠졌는지
- `fixed_area_desc`가 단순 정렬만으로 어떤 효과를 냈는지
- `auto_v1`가 실제 local headroom에서 worker를 얼마나 낮췄는지
- 향후 2차에서 검토할 가치가 있는 가설

## 현재 기록 상태

실측 결과는 suite가 완료될 때마다 아래 자산으로 동기화한다.

- `docs/banchmark_report/paddleocr-vl-parallel-report-ko.md`
- `docs/assets/benchmarking/paddleocr-vl-parallel/latest/latest_summary.md`
- `docs/assets/benchmarking/paddleocr-vl-parallel/latest/conclusion_card.md`
- `docs/assets/benchmarking/paddleocr-vl-parallel/latest/`

이 문서는 보고서의 요약판이 아니라, “어떤 결정이 어떤 근거로 내려졌는가”를 추적하는 의사결정 문서다. 따라서 결과가 누적될수록 탈락 사유와 해석을 중심으로 채워 넣는다.

## 최신 smoke 결과

현재 latest smoke suite 기준 결과는 아래와 같다.

- winner: `fixed_w8`
- speed only 관점의 빠른 후보:
  - `fixed_area_desc_w8`: `ocr_total_sec_median=290.731`
  - `auto_v1_cap4`: `ocr_total_sec_median=292.363`
- baseline:
  - `fixed_w8`: `ocr_total_sec_median=305.231`

속도만 보면 두 후보 모두 baseline보다 빨랐다. 하지만 둘 다 품질 게이트를 근소하게 넘겨 탈락했다.

- `fixed_area_desc_w8`
  - `mean_CER = 0.0011`
  - `mean_exact_match = 0.9953`
  - failures:
    - `mean_CER > baseline + 0.001`
    - `exact_match < baseline - 0.002`
- `auto_v1_cap4`
  - `mean_CER = 0.0011`
  - `mean_exact_match = 0.9953`
  - failures:
    - `mean_CER > baseline + 0.001`
    - `exact_match < baseline - 0.002`

이 smoke 결과만 놓고 보면, `큰 crop 먼저 보내기`와 `local headroom 기반 auto worker` 모두 latency 개선 가능성은 확인했지만, 아직 품질 보존 조건을 통과하지는 못했다.

## 기록 예시 원칙

- `fixed_w8`이 baseline으로 채택된 이유
- `fixed_area_desc_w8`가 tail latency를 줄였는지 여부
- `auto_v1_cap4/6/8` 중 품질 게이트를 통과한 후보가 있는지 여부
- winner가 hidden flag 상태로만 promotion 되는 이유

## 다음 단계

winner가 정해지면 이 문서에는 다음을 추가한다.

- winner 요약
- 탈락 후보 대표 사례
- 품질 게이트 실패 패턴
- 22장 full corpus 승격 검증 계획
- 필요 시 weighted concurrency 후보 설계 메모

현재 smoke 이후의 즉시 다음 단계는 다음과 같다.

- `fixed_area_desc`와 `auto_v1`의 품질 열화 원인이 공통인지 확인
- request ordering 변경이 exact match에 미치는 영향을 block 단위로 점검
- full candidate matrix 실행 전 smoke 로그를 기반으로 gate 근처 후보를 세밀하게 해석
- hidden flag 제품 승격은 유지하되, default-on 논의는 보류

## 저자 및 기여

- 핵심 문제 해결 방향은 사용자가 착안했다.
- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
