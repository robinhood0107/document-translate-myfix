# Gemma sampler quality v2 latest report

이 tracked 파일에는 원문, 번역문, raw 요청·응답, private archive 경로를 기록하지 않는다.

## 실행 identity

| 항목 | 값 |
|---|---|
| reference SHA-256 | `4159e15ae1c8d22ab0dfe954713efae931dd313ed2cd2e4554e6955a43e576bc` |
| campaign plan SHA-256 | `08a14fe020bf938a28f782f6c7a757e1947dc63cdb10ea0c6d7c8329e6293516` |
| runtime fingerprint | `c32a36042100fb06daa7e777bb02d8ad1fa1b79b680989f7092c2e382d750e1a` |
| 상태 | `WAITING_FOR_USER_APPROVAL` → 사용자 승인 완료 |
| case / seed / arm | `478` / `2` / `140` |
| 증거 응답 | `133,840` (신규 `124,280` + r6 재사용 `9,560`) |
| 판정 cluster | `2,341`, 미판정 `0` |

## 전체 판정 분포

| 등급 | 응답 | 비율 |
|---|---:|---:|
| PASS | 113,512 | 84.8 % |
| MINOR | 7,205 | 5.4 % |
| MAJOR | 8,514 | 6.4 % |
| CATASTROPHIC | 4,609 | 3.4 % |

## 필수 gate

| 결과 | arm |
|---|---:|
| 두 seed 통과 | 0 |
| 한 seed 통과 | 23 |
| 미통과 | 117 |

## 선택 tuple 대비 기존 기본값

| 지표 | `T0.5 / p1.00 / k32 / m0` | `T0.7 / p0.95 / k64 / m0` |
|---|---:|---:|
| 필수 gate | 1 / 2 | 0 / 2 |
| 검열·삭제·마스킹·동의 변경 | 13 | 20 |
| 민감표현 약화 | 3 | 3 |
| CATASTROPHIC | 28 | 41 |
| MAJOR | 60 | 57 |
| MINOR | 56 | 48 |
| 오류 발생 고유 case | 89 | 84 |
| 자연스러움 평균 | 4.41 | 4.39 |
| latency 평균 (ms) | 604 | 910 |
| completion token 평균 | 19.03 | 20.10 |

짝지은 부호 검정(956 case×seed):

| 기준 | 개선 | 악화 | p |
|---|---:|---:|---:|
| 민감표현 손실 | 7 | 0 | 0.0156 |
| 전체 품질 등급 | 36 | 32 | 0.72 |

## arm 분산

| 지표 | 값 |
|---|---|
| CATASTROPHIC 범위 | 26 ~ 41 |
| CATASTROPHIC 평균 / 표준편차 | 32.9 / 3.55 |
| seed 간 CATASTROPHIC 차이 평균 / 최대 | 1.92 / 9 |

## Transport 진단값 (순위 미반영)

| 항목 | 값 |
|---|---|
| `finish_reason` | 전부 `stop` |
| choice 수 | 전부 `1` |
| `reasoning_content` | `0` |
| JSON/schema/order/count 오류 | `0` |
| 번역 JSON 앞 channel 프레임 | `133,840 / 133,840` |

## 잔여 결함

선택 tuple과 기존 기본값이 함께 실패하는 항목 `107`건은 sampler 조정으로 제거되지 않았다.
후속 과제는 prompt·시스템 지시 층에서 다룬다.
