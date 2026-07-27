# Translation Memory Fast Path 벤치마크 흐름

## 목적

이 벤치마크는 Gemma 번역 결과 캐시와 승인형 Exact Translation Memory가 실제 제품 엔진에서 다음 불변조건을 지키는지 확인합니다.

- 전체 캐시 hit이면 중지된 Gemma 컨테이너를 시작하지 않는다.
- 부분 hit에서도 전체 정렬 문맥은 유지하고 누락된 `requested_blocks`만 생성한다.
- 승인되지 않은 후보는 Gemma를 우회하지 않는다.
- prompt, sampler, 모델 SHA-256, runtime fingerprint가 달라지면 stale 결과를 재사용하지 않는다.
- SQLite 잠금·손상은 해당 실행의 캐시만 끄고 번역을 계속하는 fail-open으로 처리한다.
- 정상 종료는 Docker `stop`만 사용하고, 실행 전 컨테이너 모델을 중지 상태로 복원한다.

## 입력 계약

입력은 짧은 다국어 제품 경로 실험에서 생성한 JSON 요약입니다. 일본어·중국어·영어가 각각 18블록이어야 하며 언어 경계를 넘지 않습니다.

러너는 입력 파일 SHA-256, Git branch/commit, 모델과 runtime identity만 공개 요약에 기록합니다. 원문·번역·로컬 경로는 Git에 들어가지 않으며, 상세 비교 자료는 사용자가 지정한 무시된 validation 디렉터리에만 저장합니다.

## 실행 순서

1. 현재 Gemma 컨테이너 모델과 상태를 기록한 뒤 `stop`합니다.
2. 중지+빈 캐시 54블록을 실행하여 startup 포함 cold 기준선을 만듭니다.
3. 컨테이너가 중지된 상태와 warm 상태에서 전체 result-cache hit를 각각 측정합니다.
4. warm+빈 캐시 54블록으로 startup을 제외한 기준선을 만듭니다.
5. 27개 `requested_blocks` no-hit와 27 hit/27 miss mixed 요청을 비교합니다.
6. sampler stale, cache-off, 손상 DB fail-open을 확인합니다.
7. cold 실행에서 수집된 후보를 명시적으로 승인한 뒤 result cache를 비우고 Exact TM 54/54 hit를 확인합니다.
8. `cache-ram 0 + cache_prompt on`, `cache-ram 0 + cache_prompt off`, `cache-ram 256 + cache_prompt on`을 비교합니다.
9. 컨테이너를 실행 전 모델 계약의 중지 상태로 복원합니다.

## 승격 게이트

- result cache와 승인 TM 모두 54/54 비어 있지 않은 결과
- 중지 상태 all-hit에서 runtime ensure 0회, HTTP 0회, 컨테이너 미시작
- cold 대비 중지·warm result-cache hit와 승인 TM 결과 모두 54/54 byte-for-byte 동일
- mixed 요청에서 hit 27, miss 27, 전체 54개 순서 복원
- 누락·빈 값·잘림·parser·split·partial fallback·심각 반복 0
- sampler 변경 21/21 stale reject
- 손상 DB 보존 및 정상 번역 경로로 fail-open
- prefix 후보는 가장 빠른 성공값 대비 3% 이내인 후보 중 가장 낮은 `cache-ram` 선택

자동 검사는 명시적 의미 품질을 판정하지 않습니다. 화자·관계·부정·행동·대상·숫자·고유명사 검수와 최종 대규모 품질 승인은 별도 사용자 게이트로 남깁니다.
