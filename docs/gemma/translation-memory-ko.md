# Gemma 결과 캐시와 정확 일치 번역 메모리

## 두 기능의 차이

Comic Translate는 Gemma 반복 번역을 줄이기 위해 서로 다른 두 저장소 계약을 사용합니다.

### 영구 블록 결과 캐시

같은 번역 요청의 sanitized raw 결과를 재사용합니다. 단순히 원문만 비교하지 않고 다음과 같은 출력 영향값을 identity에 포함합니다.

- 원래 순서의 전체 그룹 문맥과 target index/key
- source/target language와 extra context
- grouped 크기와 요청 모드
- prompt/profile/schema와 모든 sampler 값
- 모델 파일명·SHA-256과 runtime image/config fingerprint
- sanitizer, 반복 guard, 정규화, TM revision
- 현재 번역 결과 사전 규칙의 hash

하나라도 달라지면 기존 결과를 사용하지 않고 miss 또는 stale reject로 처리합니다. 사용자 번역 결과 사전 치환 전의 sanitized 번역을 저장하고, 현재 사전 규칙은 hit와 miss 모두에서 정확히 한 번만 적용합니다.

### 정확 일치 번역 메모리

사용자가 승인한 원문→번역 쌍을 다른 문맥에서도 재사용할 수 있는 별도 저장소입니다.

- Unicode NFC
- CRLF/CR을 LF로 통일
- 문자열 바깥 공백 제거

위 세 가지 외에는 정규화하지 않습니다. 대소문자, 내부 공백, 문장부호가 다르면 다른 원문입니다. embedding, fuzzy match, semantic cache는 사용하지 않습니다.

번역 결과는 처음에는 미승인 후보로만 수집됩니다. 후보는 자동으로 Gemma를 우회하지 않습니다. 같은 원문과 언어에 서로 다른 승인 번역이 여러 개면 ambiguous miss로 처리하여 Gemma가 번역합니다.

## 실행 흐름

1. OCR 블록과 현재 설정으로 cache identity를 만듭니다.
2. result cache를 먼저 확인합니다.
3. miss인 블록만 승인 Exact TM에서 찾습니다.
4. 전체 hit이면 Gemma runtime을 시작하지 않습니다.
5. 부분 hit이면 전체 원문 문맥은 그대로 유지하고 `requested_blocks`에 miss key만 넣습니다.
6. 번역 성공 뒤 결과를 한 transaction으로 기록하고 LRU/use count를 batch 갱신합니다.
7. 현재 번역 결과 사전을 최종 결과에 한 번 적용합니다.

한 블록의 최종 실패가 발생하면 원본 블록 전체를 부분 상태로 커밋하지 않는 기존 Gemma 원자성은 유지됩니다.

## 설정과 보존

`설정 → 사용자 사전 → 정확 일치 번역 메모리`에서 다음 항목을 관리합니다.

- 영구 블록 결과 캐시 사용 여부
- Exact TM과 미승인 후보 수집 사용 여부
- result-cache 보존 한도: 기본 50,000
- 미승인 후보 보존 한도: 기본 5,000
- 후보 승인·승인 해제·삭제
- result cache 비우기
- Exact TM JSON 가져오기·내보내기

한도를 넘은 result-cache 항목과 미승인 후보는 오래 사용하지 않은 순서로 정리됩니다. 승인 항목은 이 보존 정리로 자동 삭제하지 않습니다. Exact TM 변경은 revision을 올려 이전 result-cache identity가 그대로 재사용되지 않게 합니다.

## 개인정보와 신뢰 경계

SQLite DB에는 원문과 번역문이 평문으로 저장되므로 민감한 로컬 사용자 데이터입니다. raw 이미지는 저장하지 않습니다. DB는 앱 user-data 디렉터리에만 생성되며 Git이나 프로젝트 파일에 포함되지 않습니다.

가져오기 파일의 `approved: true` 항목은 Gemma를 즉시 우회할 수 있습니다. 앱은 가져오기 전에 확인을 요구하지만, 사용자가 파일 내용을 신뢰할 수 있는 경우에만 승인해야 합니다.

result cache 비우기는 Exact TM 후보와 승인 항목을 유지합니다. Exact TM 항목 삭제는 선택 항목에만 적용됩니다. 자동 복구라는 이름으로 DB 전체를 삭제하거나 새 파일로 덮어쓰지 않습니다.

## 장애 동작

SQLite lock, 손상, schema 불일치를 감지하면 다음처럼 fail-open합니다.

- 해당 앱 실행에서 cache/TM을 비활성화
- 사용자에게 경고
- Gemma 정상 번역 경로 계속 실행
- 기존 DB를 삭제하거나 다시 작성하지 않음

관리 화면의 승인·삭제·clear·import·export는 실패를 성공처럼 표시하지 않고 오류를 반환합니다. 손상 DB를 export해 빈 정상 파일처럼 덮어쓰지도 않습니다.

## server-side prompt cache와의 차이

llama.cpp의 `cache_prompt`와 `cache-ram`은 prompt prefill 계산을 재사용할 뿐 번역 결과 자체를 저장하지 않습니다. SQLite result cache와 Exact TM은 HTTP 요청과 출력 decoding까지 생략할 수 있는 제품 레벨 fast path입니다.

현재 제품 runtime은 `cache-ram=0`을 유지합니다. 54블록 fast-path 검증에서 전체 result-cache hit와 승인 TM hit는 각각 HTTP 0회로 완료됐지만, 이 결과는 최종 대규모 번역 품질 승인을 대신하지 않습니다.
