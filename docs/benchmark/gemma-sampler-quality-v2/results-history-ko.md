# Gemma sampler quality v2 결과 이력

기존 v1 sampler 결과와 품질 REJECT 기록은 수정하거나 재해석하지 않는다.

## 1. Reference freeze

사용자 승인으로 478 case reference를 freeze했다.

- case 수 `478`, occurrence `758`, split `tuning 382 / holdout 96`
- flagged 항목은 사용자 지시로 전부 canonical 확정, 미해결 `0`
- 필수 24개 사용자 표본 검수 승인 완료
- reference SHA-256 `4159e15ae1c8d22ab0dfe954713efae931dd313ed2cd2e4554e6955a43e576bc`

## 2. 단일 캠페인 실행

CUDA13 launcher 1회 실행으로 r6 재사용 검증 → joint → min-p → Router unload까지 진행하고
`WAITING_FOR_FINAL_JUDGMENT`에서 정상 종료했다.

| 구간 | arm | 응답 |
|---|---:|---:|
| r6 재사용 (온도 10개 × `p.95/k64/m0`) | 10 | 9,560 |
| Joint (온도 10개 × top-p `.85/.90/.95/1` × top-k `0/32/64`) | 110 | 105,160 |
| Min-p (온도 10개 × `p1/k0` × min-p `0/.05/.10`) | 20 | 19,120 |
| 합계 | **140** | **133,840** |

- arm당 `956`응답, `(arm, seed)`당 `478`, 논리 slot 중복 `0`
- campaign plan SHA-256 `08a14fe020bf938a28f782f6c7a757e1947dc63cdb10ea0c6d7c8329e6293516`
- runtime fingerprint `c32a36042100fb06daa7e777bb02d8ad1fa1b79b680989f7092c2e382d750e1a` 단일
- case별 request identity 단일, seed `2`개 고정

## 3. 판정

478 case 전체를 arm·sampler 숨긴 blind cluster로 판정했다. holdout 개봉 단계는 제거하고
`tuning/holdout`은 provenance로만 보존했다.

- 고유 번역 cluster `2,341` (exact canonical 자동 일치 `82` 포함)
- 미판정·review-required `0`
- 판정 표준 정렬을 위한 감사 가능 amendment `4`회 / `113` cluster
- 최종 분포: PASS `113,512` / MINOR `7,205` / MAJOR `8,514` / CATASTROPHIC `4,609`

transport 진단값은 순위에 반영하지 않았다.

- `finish_reason` 전부 `stop`, choice 전부 `1`, `reasoning_content` `0`
- JSON/schema/order/count 오류 `0`
- 번역 JSON 앞의 channel 프레임 `133,840 / 133,840` — 번역문은 정상 추출되며 140 arm에 동일하게 발생

## 4. 필수 gate

고정 인물명·나이 case는 별도 필수 gate로 두었다.

- 두 seed 모두 통과한 arm `0 / 140`
- 한쪽 seed만 통과한 arm `23 / 140`
- 나머지 응답은 이름 마스킹 또는 빈 응답

계획서 기준 자동 승격 조건은 미충족이며, 제품 승격은 사용자 명시 승인으로만 진행했다.

## 5. 순위와 선택

계획서의 lexicographic 순위(`catastrophic → major → minor → 오류 case → 자연스러움 → latency/token`)에서
1위와 하위 후보의 차이는 짝지은 부호 검정으로 유의하지 않았다. 오히려 집계 1위가 5위보다 유의하게
나빴다(`p=0.043`).

사용자는 `필수 gate 유지 → 검열·민감표현 삭제 최소`를 우선 기준으로 지정했다. 이 기준의 선택은
아래와 같다.

| 항목 | 선택 tuple | 기존 기본값 |
|---|---|---|
| tuple | `T0.5 / p1.00 / k32 / m0` | `T0.7 / p0.95 / k64 / m0` |
| 필수 gate | `1/2` | `0/2` |
| 검열·삭제·마스킹·동의 변경 | `13` | `20` |
| 민감표현 약화 | `3` | `3` |
| catastrophic / major / minor | `28 / 60 / 56` | `41 / 57 / 48` |
| 자연스러움 평균 | `4.41` | `4.39` |
| latency 평균 | `604 ms` | `910 ms` |

짝지은 비교(956 case×seed):

- 민감표현 손실 기준 개선 `7` / 악화 `0` / `p=0.0156` — 유의하게 개선
- 전체 품질 등급 기준 개선 `36` / 악화 `32` / `p=0.72` — 동등

전체 `140` arm의 catastrophic 분포는 `26`~`41`, 평균 `32.9`, 표준편차 `3.55`이며 arm별 seed 간
catastrophic 차이는 평균 `1.92`, 최대 `9`다. 따라서 총합 순위의 소폭 차이는 유의한 품질 우위로
해석하지 않는다.

## 6. sampler로 해결되지 않는 잔여 결함

선택 tuple과 기존 기본값이 함께 실패하는 `107`건이 남는다. 거부 표현 삭제, 성적 행위 일반화,
여성 지정 누락, 의미 반전, 고유명 변경, 원문 토큰 노출이 주 유형이며 온도·top-p·top-k·min-p
조정으로는 제거되지 않았다. 후속 과제는 prompt·시스템 지시 층에서 다룬다.

## 7. 제품 반영

사용자 승인 후 제품 기본값을 선택 tuple로 승격했다. QSettings migration version `2`가 네 값을
조건 없이 한 번 덮어쓰고 marker 기록 이후 사용자 수정은 보존한다. raw corpus·요청·응답·판정·검토
HTML은 private archive에만 남기며 이 문서에는 집계와 hash만 기록한다.
