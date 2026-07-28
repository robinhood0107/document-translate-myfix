# Gemma 최종 A/B blind 품질 검수 latest

## 현재 상태

- protocol: v4 report-only
- source contract revalidation: PASS
- imported candidate families: 2
- imported rounds per family: 2
- verified pages: 22/22
- verified rows: 292/292
- Q8 candidate inclusion: 0
- Docker/model request count: 0
- 사용자 요청 Codex blind 전수 검수: 292/292
- 의미 회귀 판정: 27행 / 54출력
- A: `current-contextual-single`, 14행 / 18출력
- B: `grouped-f16`, 21행 / 36출력
- grouped round 1: 19출력
- grouped round 2: 17출력
- grouped 두 라운드 공통 회귀: 15행
- unblind: 완료
- 상태: `quality_rejected`
- 상대 품질 우위: `current-contextual-single`
- grouped 제품 기본값 구현: 금지
- 전체 파이프라인 비교: 실행하지 않음

## 최종 판정

protocol v4의 292행 불변 열, 순서, 네 판정값, 회귀 유형과 메모 검증을
통과한 뒤에만 unblind했다. 후보와 실제 제품 기본 프로필은 다음처럼
서로 다르다.

- A: `IQ4_XS + contextual-single + chunk 6 + ngram draft8 + F16`
- B: `IQ4_XS + contextual-grouped + chunk 7 + no-spec + F16`
- 실제 제품 기본값:
  `IQ4_NL + contextual-single + chunk 6 + no-spec + F16`

grouped F16은 `だけ` 누락, 인물명과 침대의 잘못된 결합, 절정 행위 누락,
긍정 대답의 신음 변환, 심리적 감정의 쾌감 변환, 화자·행위 방향 반전
등 21행 36출력에서 의미 회귀가 확인됐다. 15행은 두 라운드에서 모두
반복됐다.

계획의 절대 게이트인 candidate-only 의미 회귀 0을 B 조합이 통과하지
못했으므로
`feature/gemma-iq4xs-grouped-default`와 22페이지 전체 파이프라인 비교는
진행하지 않는다. 두 후보 중 상대적으로 나은 번역은
`current-contextual-single`이지만, 이 후보도 14행 18출력의 회귀가 있어
새로운 품질 우승 설정이나 현재 제품 프로필의 우승으로 간주하지 않는다.
두 후보는 모델·요청 방식·chunk·speculative decoding 설정이 함께
달랐으므로, 이 결과만으로 grouping 하나가 모든 회귀의 원인이라고
단정하지 않는다. 승격 대상이었던 B 조합 전체를 품질 탈락으로 확정하고
제품 기본 프로필은 변경하지 않는다. 22페이지 전체 파이프라인 비교는
미완료가 아니라 품질 게이트에 따른 정상 실행 취소다.
