# Translation Memory Fast Path 결과 이력

## 2026-07-28 — 첫 54블록 실측

조건:

- 제품 커밋: `323c806`
- lab 통합 커밋: `18051c1`
- 3개 언어 × 18블록
- grouped 크기 7, completion cap 512
- IQ4_XS, no-spec, F16 KV, context 4096, threads 10, GPU layers 23
- pinned llama.cpp b10133 이미지

자동 구조 게이트 결과:

- cold와 warm 빈 캐시 모두 54/54, severe telemetry 0
- 중지 result-cache hit 54/54, runtime ensure 0, HTTP 0, 컨테이너 미시작
- mixed hit 27 + miss 27, 54/54 복원, HTTP 9, severe telemetry 0
- 승인 TM 54/54, runtime ensure 0, HTTP 0, cold 결과와 54/54 byte-for-byte 동일
- sampler 변경 21/21 stale reject
- 손상 DB는 삭제·수정 없이 `DatabaseError` fail-open

시간:

| 상태 | 시간 | 비교 |
|---|---:|---:|
| 중지 + 빈 캐시 | 60.301초 | startup 포함 기준 |
| warm + 빈 캐시 | 34.624초 | 번역 기준 |
| 중지 + result-cache 전체 hit | 1.978초 | cold 대비 96.720% 단축 |
| warm + result-cache 전체 hit | 2.019초 | warm 대비 94.170% 단축 |
| 중지 + 승인 TM 전체 hit | 1.967초 | cold 대비 96.738% 단축 |
| requested_blocks 27개 no-hit | 22.177초 | HTTP 9 |
| mixed 27 hit + 27 miss | 22.101초 | HTTP 9 |

prefix 행렬:

| cache-ram | cache_prompt | median | prompt eval median | fastest 대비 |
|---:|---:|---:|---:|---:|
| 0 MiB | on | 1.948초 | 39.923ms | 기준 |
| 0 MiB | off | 2.079초 | 364.274ms | 6.737% 느림 |
| 256 MiB | on | 2.010초 | 51.741ms | 3.192% 느림 |

판정:

- `cache-ram 0 + cache_prompt on` 유지
- 256 MiB는 가장 빠른 후보보다 3% 이내 조건을 0.192%p 초과하므로 승격하지 않음
- 제품의 field 생략 요청에서도 pinned llama.cpp가 cached prompt token을 보고했으므로 현재 효과적인 on 동작과 일치함
- mixed 출력은 누락·빈 값·구조 오류 없이 통과했고, 현재 full grouped 결과와 비교해 요청 방식 때문에 생긴 명백한 의미 회귀는 발견하지 못함
- 확률적 표현 차이와 기존 원문/OCR 애매함은 남아 있으므로 최종 대규모 blind 품질 승인을 대신하지 않음
