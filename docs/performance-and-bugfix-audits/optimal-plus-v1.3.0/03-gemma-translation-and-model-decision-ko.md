# Gemma 번역 및 모델 판정

v1.3.0의 번역 기본 프로필은 다음과 같다. 이 선택은 빠른 단일 실험이 아니라
구조·의미·runtime 안정성을 함께 본 누적 판정이다.

| 항목 | 확정값 |
|---|---|
| 모델 | IQ4_NL |
| 요청 방식 | contextual-single |
| 문맥 묶음 | chunk 6 |
| speculative decoding | 사용 안 함 |
| KV | F16 |
| context | 4096 |
| batch / ubatch | 2048 / 512 |
| GPU layers | 23 |
| completion limit | 512 |
| server prompt cache RAM | 0 MiB |

prompt, sampler, JSON Schema, sanitizer, 원자성 안전장치는 이 모델 선정 과정에서
변경하지 않았다. 긴 입력을 마침표로 임의 분할하는 새 경로도 만들지 않았다.
context capacity 문제는 기존 contextual-single strict retry, block 격리 fallback,
필요 시 chunk bisection, 원문 보존 규칙으로 처리한다.

## 품질 판정 방식

모델 후보는 먼저 민감 15 block과 일본어·중국어·영어 각 18 block의 54 block
검증을 통과해야 한다. 구조 검증은 키·순서·타입, 누락·빈 값·잘림·중첩/중복/후행
JSON, fallback과 `finish_reason=length`를 확인한다. 의미 검수는 원문을 직접 읽어
다음을 후보 전용 회귀로 분류한다.

- 화자, 관계·친족, 부정·가능·의무
- 행동, 주체·대상, 숫자·시간·수량, 고유명사
- 명시적 성적·폭력 의미의 누락·순화, 번역 거부
- 두 라운드 사이 심각한 의미 반전

자연스러운 말투·어순·문장부호 차이는 의미가 같으면 회귀로 세지 않는다. 최종
후보는 292행 blind 검수까지 통과해야 한다.

## 후보별 결과

| 후보 또는 축 | 결과 | 최종 판정 |
|---|---|---|
| contextual-grouped, group 7, no-spec, F16 | 번역 전용은 약 40.057% 빨랐지만 292행 blind에서 21행·36출력 의미 회귀 | 퇴역 |
| Q8 KV | F16보다 4.730% 느리고 fallback·invalid 발생 | 탈락 |
| IQ4_XS | 초기 명목 0.9354% 빠름. WSL 20GB 재측정 request 평균 1.527%이나 신뢰구간 하한 −2.026%, candidate-only 의미 회귀 2행·5출력 | IQ4_NL 유지 |
| QAT 모델군 | 크기·속도·의미 품질에서 현행보다 우위 없음 | 탈락 |
| MTP / draft 모델군 | 일부는 느리거나 GPU draft 계약 오류, 총시간과 품질 우위 없음 | 탈락 |
| ngram speculative decoding | 반복 검증에서 총시간 우위 없음 | 탈락 |
| batch 1024 및 ubatch 후보 | 초기 구조 통과 수치는 있었으나 최종 누적 의미/속도 게이트 미통과 | 2048/512 유지 |
| chunk 9 / chunk 12 | 관측 0.731% / 1.357% 이득이나 최종 의미 검수 우승 후보 없음 | chunk 6 유지 |
| NGL 23 이상 | VRAM·swap·총시간을 포함한 우위 없음 | NGL 23 유지 |
| server prompt cache 256 MiB | E2E 1.915%, request 2.105% 단축이나 candidate-only 의미 회귀 1건 | 0 MiB 유지 |
| `np=2` | 실질적인 총시간 이득 없음 | 미사용 |

grouped 경로의 퇴역은 JSON decoder, request context, contextual-single retry,
HTTP 오류 분류, 논리/HTTP telemetry, 번역 cache를 되돌린다는 뜻이 아니다.
이 공통 안전장치는 유지하고 grouped dispatch, split, partial grouped fallback,
strict grouped retry만 제품 경로에서 제거했다.

## GGUF page cache와 모델 상주

정상 `stop` 뒤 즉시 재시작할 때 healthy 시간이 66.811초에서 16.865초로
줄어 74.757% 단축됐다. 이것은 GGUF를 VRAM에 영구 상주시키지 않고, named
volume의 파일 읽기가 OS page cache에서 재사용된 결과다.

따라서 제품은 다음을 유지한다.

- 모델과 manifest는 versioned named volume에 read-only 보관한다.
- OCR·인페인트·Gemma를 동시에 GPU에 상주시켜 page cache 이득을 얻으려 하지
  않는다.
- 정상 종료는 `stop`을 사용하고, 명시적 초기화/삭제 이외에는 `down`을 쓰지
  않는다.
- mlock, 강제 RAM pin, drop cache, 자동 WSL 종료를 사용하지 않는다.

OCR 후 저우선위 explicit read-ahead도 시험했지만, 신뢰구간 하한 −0.201%와
swap 증가가 확인되어 채택하지 않았다. 자연 OS page cache는 보존하되, 적극적인
prefetch는 제품 경로에 넣지 않는다.

## 번역 cache와의 관계

OS page cache는 model file load를 줄일 뿐 HTTP 요청과 생성 decode를 생략하지
않는다. 반대로 SQLite result cache와 승인형 Exact TM은 전체 hit에서 Gemma
시작과 HTTP 요청을 0으로 만들 수 있다. cache identity, 부분 hit 문맥 유지,
사용자 사전 적용 순서는 [번역 메모리 가이드](../../gemma/translation-memory-ko.md)에
따른다.

제품 runtime 준비·volume fingerprint의 운영 계약은
[Gemma 로컬 서버 설정 가이드](../../gemma/local-server-ko.md), 모든 후보의
한눈에 보는 판정은 [완전 실험 등록부](01-complete-experiment-register-ko.md)를
참고한다.
