# PaddleOCR-VL exact 영구 결과 캐시 명세

이 문서는 관리형 PaddleOCR-VL 폴더 처리의 영구 결과 캐시 계약을 고정한다. benchmark 후보 순위와 원시 결과는 저장소 밖에서 관리하며, 이 문서에는 제품 불변 조건만 둔다.

## 목적

- 동일한 crop과 동일한 관리형 런타임 계약을 다시 처리할 때 OCR 추론을 생략한다.
- 폴더 전체가 cache hit이면 PaddleOCR-VL 런타임 시작과 OCR HTTP 요청을 모두 생략한다.
- 사전 변경은 raw OCR 추론 캐시를 버리지 않고 현재 실행 결과에 정확히 한 번 반영한다.

## 저장 계약

- 사용자 데이터 경로의 전용 SQLite WAL 데이터베이스를 사용한다.
- 기본 보존 한도는 50,000개 crop 결과이며 `last_used_at` 기준 LRU로 정리한다.
- crop 이미지와 원본 이미지는 저장하지 않는다.
- 사전 적용 전 raw OCR 결과와 상태, 빈 결과 사유, 원문·정규화 텍스트, crop 진단만 저장한다.
- JSONL 내보내기와 명시적 전체 지우기를 제공한다. 자동 손상 복구를 이유로 기존 데이터베이스를 삭제하거나 덮어쓰지 않는다.

## exact identity

캐시 키에는 다음 입력을 canonical JSON으로 직렬화한 SHA-256을 사용한다.

- 연속 메모리 기준 raw crop pixel SHA-256과 실제 전송 JPEG SHA-256
- crop shape·dtype, block 좌표·text class, bubble 유무·좌표
- text-free guard 입력과 판정 근거
- source language, max token, prettify, visualize
- crop·encoder·parser·sanitizer·guard schema version
- Paddle 모델명, 공식 image digest와 실제 image ID
- compose, command, vLLM config, pipeline config SHA-256과 종합 runtime fingerprint

사용자 OCR 결과 사전은 raw 추론 identity에 포함하지 않는다. hit와 miss 모두 raw 결과를 복원한 뒤 현재 사전을 정확히 한 번 적용한다.

## 런타임 계약

- 공식 PaddleOCR-VL 이미지는 digest로 고정한다.
- 평상시 이미지를 반복 pull하지 않는다.
- 중지된 컨테이너의 image ID, command/config fingerprint 또는 label이 다르면 `docker compose up -d --force-recreate`로 재생성한다.
- 모든 fingerprint가 정확히 같을 때만 기존 컨테이너를 `docker start`로 재사용한다.
- 사용자 지정 또는 관리되지 않는 endpoint에서는 영구 캐시를 비활성화하고 기존 OCR 경로를 그대로 실행한다.
- 정상 종료는 `stop`을 사용한다. 이 기능은 `down`이나 광범위 컨테이너 삭제를 호출하지 않는다.

## 폴더 처리 순서

1. detection이 끝난 모든 페이지의 Paddle crop을 준비한다.
2. exact identity를 계산하고 SQLite를 한 번에 조회한다.
3. hit는 raw 결과를 복원하고, miss만 runtime 작업 목록에 넣는다.
4. runtime miss가 하나라도 있을 때만 관리형 Paddle runtime을 준비한다.
5. OCR 완료 뒤 raw 결과를 저장하고 현재 사전을 한 번 적용한다.
6. 기존 품질 검사가 low-quality를 판정하면 실제 no-cache 재시도를 수행하고 최종 raw 결과만 저장한다.

PaddleOCR-VL 자동 폴더 처리에서는 기존 sampled-image 또는 fuzzy-coordinate 메모리 캐시를 사용하지 않는다.

## 실패 정책

- DB lock, 손상, schema mismatch, invalid cached JSON은 해당 실행의 영구 캐시만 비활성화한다.
- 손상 DB를 삭제하거나 덮어쓰지 않고 OCR은 실제 런타임 경로로 계속한다.
- cache plan 준비 중 일반 오류는 해당 페이지만 실패시키고 나머지 페이지는 계속 처리한다.
- 사용자 취소는 즉시 상위 취소로 전파한다.
- clear 또는 export 실패는 사용자에게 알리고 기존 DB와 기존 export 파일을 보존한다.

## 검증 게이트

- cold miss와 warm hit의 raw OCR·진단이 완전히 같아야 한다.
- all-hit에서 logical OCR request, HTTP attempt, request bytes가 모두 0이어야 한다.
- 사전 규칙은 hit와 miss에서 각각 정확히 한 번만 적용돼야 한다.
- pixel, JPEG, 언어, guard, schema, 모델, 이미지, command/config fingerprint가 하나라도 달라지면 miss여야 한다.
- schema mismatch와 손상 DB는 원본 파일을 변경하지 않고 fail-open해야 한다.
- `.venv-win` 단위 테스트, `.venv-win-cuda13` Python/headless/CUDA runtime 검사, 번역 자산 검사, Windows launcher 계약 검사를 통과해야 한다.
