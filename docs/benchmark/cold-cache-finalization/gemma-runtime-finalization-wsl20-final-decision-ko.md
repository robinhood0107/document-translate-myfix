# Gemma WSL 20GB 런타임 최종 판정

## 결론

WSL 20GB 환경에서 남아 있던 Gemma 모델·런타임 축을 모두 다시 실행했지만,
속도와 의미 품질을 동시에 통과한 후보는 없었습니다. 제품 프로필은 다음 값을
유지합니다.

```text
model: HauhauCS IQ4_NL
request mode: contextual-single
chunk size: 6
speculative decoding: none
KV: F16
batch / ubatch: 2048 / 512
GPU layers: 23
prompt cache RAM: 0 MiB
```

최소 속도 향상률은 사용하지 않았습니다. 후보의 paired end-to-end와 request-only
시간이 모두 기준선보다 빠르고 단측 95% bootstrap 하한이 0%보다 커야 속도 후보로
인정했습니다. 그 뒤 원문 기준 candidate-only 의미 회귀를 검사했습니다.

## 실행 계약

- 입력: 일본어·중국어·영어 각 18블록, 총 54블록
- 입력 요약 SHA-256:
  `5797ab2c130107a5b3b7c531f3f0a2d35175582f2db6595dc29a85e3779c13ac`
- 각 후보 3회, 순서 교차
- `contextual-single`, no-spec, F16, completion 512 고정
- 번역 result cache·Exact TM·project checkpoint 비활성화
- 실행 사이 관리형 컨테이너 `stop`, `down` 미사용
- 각 실행 시작 대비 WSL swap 증가 0 요구

## 최종 결과

| 축 | 후보 | paired 결과 | 의미 품질 | 판정 |
|---|---|---|---|---|
| 모델 | IQ4_XS | request 평균 +1.527%, 하한 -2.026% | 2행·5출력 candidate-only 회귀 | IQ4_NL 유지 |
| batch | 4096 | E2E +3.512%, 하한 +2.397%; request +2.589%, 하한 +0.869% | 관계·행동 손실과 비정상 혼합 문자열 | 탈락 |
| NGL | 30 | request 평균 -58.469% | 관계·고유명사 회귀 관찰 | 탈락 |
| NGL | 31 | request 평균 -54.194% | 관계·고유명사 회귀 관찰 | 탈락 |
| ubatch | 256 | request 평균 -4.858% | 속도 선별에서 종료 | 탈락 |
| ubatch | 384 | request 평균 -5.009% | 속도 선별에서 종료 | 탈락 |
| ubatch | 768 | request 평균 -1.240%, 라운드 승패 반전 | 속도 선별에서 종료 | 탈락 |
| prompt cache | RAM 256 MiB | E2E +1.915%, 하한 +0.362%; request +2.105%, 하한 +1.227% | 중국어 짧은 원문을 신음으로 바꾼 candidate-only 회귀 1건 | 탈락 |
| chunk | 9 | E2E +0.029%, 하한 -2.968%; request -0.981% | 속도 선별에서 종료 | 탈락 |
| chunk | 12 | E2E -0.816%; request -2.106% | 고유명사 표기 회귀 관찰 | 탈락 |

`IQ4_XS`는 파일이 약 4.58% 작지만 속도 우위가 입증되지 않았고, `needle`을
`syringe`로 구체화하거나 접촉 대상·친족 관계를 바꾸는 회귀가 반복됐습니다.
따라서 파일 크기만으로 승격하지 않습니다.

`batch=4096`과 `cache-ram=256`은 작은 이득도 실제로 찾아내는 새 속도 규칙을
통과했습니다. 그러나 전자는 친족·행동 관계를 손실하거나 비정상 혼합 문자열을
만들었고, 후자는 모든 기준선 라운드가 `나, 나`로 옮긴 짧은 중국어를 한 후보
라운드에서 신음으로 바꿨습니다. 속도가 빨라도 의미 회귀가 한 건이면 탈락한다는
계약에 따라 제품에 적용하지 않습니다.

## 292행 검수와 제품 변경

짧은 54블록 품질 게이트를 통과한 누적 후보가 없으므로 292행 blind 검수는 실행하지
않았습니다. 이는 미완료가 아니라 사전 품질 게이트에 따른 정상 종료입니다. 제품
기본값과 one-time migration은 변경하지 않습니다.

원시 번역, 라운드별 시간, private review와 모델 경로는 Git 밖 validation log에만
보존합니다.

