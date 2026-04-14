# 02 Measurement Methodology And Quality Gate

## 배경과 문제 상황

translation 단계는 이번 의사결정에 필요하지 않다. 이번 비교의 목적은 “OCR 품질이 baseline과 같을 때 어떤 스케줄러가 가장 빠른가”를 판단하는 것이다. 따라서 translation, inpaint, render가 개입하면 판단이 오염된다.

이 문서의 목적은 speed와 quality를 동시에 보는 측정 방법을 고정하고, 이후의 winner 판단이 임의로 흔들리지 않게 만드는 것이다.

## 사용자가 제안한 핵심 발상

사용자는 `정답지를 먼저 만들고 그 뒤에 여러 번 돌려서 분배를 판단해야 한다`는 방향을 분명히 제시했다. 즉, 휴리스틱부터 넣는 것이 아니라, baseline을 고정하고 반복 실험으로 승자를 고르는 순서를 먼저 정립하자는 제안이었다.

## 측정 설계와 기준

- execution scope: `detect-ocr-only`
- baseline gold: `fixed_w8` measured run seed
- telemetry:
  - `summary.json`
  - `metrics.jsonl`
  - `page_snapshots.json`
  - `gpu_samples.jsonl`
  - `page_profiles.jsonl`
  - `request_events.jsonl`

여기서 중요한 점은 GPU를 `요청마다 직접 질의`하지 않는다는 것이다. benchmark branch에서는 별도의 background sampler가 `200ms` 간격으로 `gpu_samples.jsonl`을 남기고, request timestamp와 nearest join 방식으로 분석한다. 이 방식은 local runtime에 측정 오버헤드를 과도하게 주지 않으면서도, request 직전/직후의 headroom 변화를 추적할 수 있다.

또한 detector는 baseline seed run의 geometry를 고정하고, 이후 후보 비교는 OCR만 바꾼다. 이렇게 해야 candidate별 품질 차이가 detector 흔들림 때문인지, OCR scheduling 때문인지 구분할 수 있다.

## 품질 게이트

- `page_failed_count` 증가 없음
- `empty_block_count` 증가 없음
- aggregate `mean CER <= baseline + 0.001`
- aggregate `exact match rate >= baseline - 0.002`

이 게이트는 “무조건 가장 빠른 후보”를 뽑기 위한 것이 아니라, `shipping parity`를 벗어난 후보를 초기에 제거하기 위한 hard gate다. 속도 개선이 있더라도 품질이 baseline보다 눈에 띄게 나빠지면 제품 승격 대상으로 삼지 않는다.

## 구현 방식과 설계 선택

- detector는 baseline seed run의 geometry를 gold에 고정한다.
- OCR 비교는 block geometry match 후 문자 단위 CER로 계산한다.
- baseline candidate는 항상 `fixed_w8`로 고정한다.
- smoke에서는 `fixed_w8`, `fixed_area_desc_w8`, `auto_v1_cap4`만 빠르게 확인한다.
- full candidate matrix에서는 `fixed_w1..w8`, `fixed_area_desc_w1..w8`, `auto_v1_cap4/6/8`을 비교한다.

이 baseline 선택은 현재 shipping에 가장 가까운 설정을 기준점으로 쓰기 위함이다. baseline이 흔들리면 모든 품질 비교가 의미를 잃기 때문에, 가장 먼저 baseline run에서 detector manifest와 gold seed를 생성한다.

## 결과와 효과

이 방법은 absolute OCR correctness가 아니라 `shipping parity under faster scheduling`을 검증하는 데 최적화되어 있다. 운영 환경에서 실제로 필요한 것은 “더 똑똑한 OCR”이 아니라 “지금과 같은 품질을 더 빠르게 내는 OCR”이기 때문이다.

또한 request-level telemetry, GPU sampling, page-level quality gate가 분리되어 있기 때문에, 이후 winner를 설명하거나 회고 문서를 작성할 때도 어느 수준에서 병목이 발생했는지 구조적으로 서술할 수 있다.

## 남은 한계와 다음 단계

baseline seed는 사람이 교정한 gold가 아니므로, 향후 실제 정답 gold와는 분리해 관리해야 한다.

또한 1차의 목적은 subset winner 선별이다. subset winner가 정해져도, 22장 full corpus와 더 긴 soak run에서 같은 결론이 유지되는지 추가 검증이 필요하다.

## 저자 및 기여

- 핵심 문제 해결 방향은 사용자가 착안했다.
- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
