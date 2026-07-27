# Translation Memory Fast Path 최신 요약

측정일: 2026-07-28
범위: 54블록 다국어 실제 Gemma 제품 경로

## 결론

- Persistent result cache와 승인 Exact TM 모두 중지 상태에서 Gemma를 시작하지 않고 54/54 결과를 복원했습니다.
- result-cache 전체 hit는 cold 대비 96.720%, warm 대비 94.170% 단축됐습니다.
- mixed 27 hit/27 miss 요청은 전체 문맥을 유지한 `requested_blocks` 9회로 완료됐고 안전 telemetry는 0이었습니다.
- `cache-ram 0 + cache_prompt on`이 세 prefix 후보 중 가장 빨랐습니다. 256 MiB는 3.192% 느려 승격하지 않습니다.
- SQLite 손상은 DB를 삭제하거나 덮어쓰지 않고 해당 실행에서만 cache를 끄는 fail-open으로 확인됐습니다.
- 강화된 10개 자동 게이트를 기존 공개 summary에 다시 적용한 결과 모두 통과했습니다.

## 남은 게이트

이 결과는 구조와 속도 게이트입니다. 최종 대규모 번역 후보의 화자·관계·부정·행동·대상·숫자·고유명사 품질은 사용자 blind 검수 전까지 미승인 상태입니다. 따라서 전체 페이지 pipeline 실행과 최종 default 승격은 아직 수행하지 않습니다.
