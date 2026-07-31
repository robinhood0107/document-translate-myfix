# 아키텍처와 경계

## 제품 코드와 벤치마크 코드

제품 OCR은 전략별 모듈로 분리되어 있습니다.

```text
modules/ocr/
  paddle_crop/
  paddle_spotting/
  mangalmm_full_page/
  common/
```

이 디렉터리에는 prompt, parser, retry, 이미지 크기 정책, runtime 경계가 서로 섞이지
않습니다. 공통 결과 계약과 좌표 유틸리티만 `common`에 있습니다.

`scripts/benchmark_ocr_three_way_human_truth.py`는 `benchmarking/lab` 전용
report-only 계층입니다. 후보 순위, 사람 정답, blind key, raw 결과는 제품 코드에
들어가지 않습니다.

## 표준 결과

세 route는 가져오기 단계에서 다음 두 목록으로 표준화됩니다.

- `raw_regions`: 모델이 실제로 낸 text와 geometry
- `canonical_units`: detector block 또는 안전한 full-page extra 단위

Paddle crop은 detector block과 1:1입니다. Paddle Spotting은 여러 line region을
읽기 순서대로 하나의 detector unit에 보존합니다. MangaLMM은 raw full-page region을
geometry로 연결하되, 여러 detector block과 비슷하게 겹치는 region은 ambiguous로
남깁니다.

이 가져오기 결과는 제품 reconciliation을 대신하지 않습니다. 사람 검수에서 현재
실패를 재현하고 다음 제품 PR의 global reconciliation 전후를 같은 계약으로 비교하기
위한 중립 표면입니다.

## 무결성 사슬

```text
source image SHA
  + detector snapshot SHA
  → corpus manifest canonical SHA
  → truth page and crop SHA
  → truth lock SHA
  + route source binding SHA
  + raw route result SHA
  + llama.cpp runtime contract SHA
  → normalized run SHA
  → blind payload and key SHA
  → completed review
```

모든 실제 경로는 Git 저장소 밖이어야 합니다. JSON duplicate key, path traversal,
source/result hash 변경, vLLM runtime contract, route 누락, truth 변경, 불완전 review는
즉시 거부합니다. truth lock은 초기 detector block의 ID·bbox·순서·class·direction을
전부 요구하므로 페이지나 detector 행을 조용히 삭제할 수 없습니다. 기존 full-page
결과처럼 raw 응답에 원본 SHA가 없을 수 있으므로 `source-bindings.json`으로 원본
SHA·크기·manifest와 route별 primary result 파일 SHA를 먼저 묶은 뒤에만 가져옵니다.
blind payload도 canonical self-hash로 잠그므로 생성 뒤 row 수, evidence hash 또는
unblind 계약을 바꾸면 finalization이 실패합니다.

검수표에는 route 고유 내부 상태명이 아니라 `matched`, `compound`, `ambiguous`,
`missing_or_partial`, `unmatched_extra`의 공통 geometry 상태만 공개합니다. CSV 값은
스프레드시트 수식으로 실행되지 않도록 가역 escape하며, 통계 계산 때 원문으로 복원합니다.

후보가 `assets`에 mask, cleaned crop, render, diff 같은 파일을 선언하면 해당 파일도
원시 결과 SHA 사슬에 포함됩니다. 블라인드 패키지에는 route 이름 대신 A/B/C 경로로
행별 고유 경로에 복사됩니다. 복사본 전체의 상대 경로와 SHA-256도 봉인하므로 행 간
덮어쓰기, 검수 뒤 변경, 누락 또는 임의 파일 추가를 finalization에서 거부합니다.
자산은 점수 자체가 아니라 파괴적 편집을 사람이 확인하는 증거입니다.

## 정확도 정의

- `normalized exact`: NFC 후 공백·줄바꿈·문장부호를 제거한 문자열 완전 일치
- `normalized character accuracy`: 같은 정규화 후 Levenshtein 문자 정확도
- `semantic correct`: 사람이 원본과 후보를 직접 읽고 `yes`로 판정한 경우만 인정
- `meaning text recall`: `translate_inpaint` 정답 region 중 semantic correct 비율
- `candidate extra false positive`: full-page-only 후보를 사람이 오처리로 판정한 수
- `destructive edit`: OCR 검출 자체가 아니라 실제 번역·인페인트가 그림을 훼손한 수
- `page complete`: 의미 텍스트가 전부 맞고 merge/split·파괴 편집·extra 오처리 판정이
  모두 확정된 페이지. `uncertain`은 검수 완료 값으로 보존되지만 완전 성공으로 세지 않음
- `full-page-only meaning recall`: detector가 놓쳐 사람이 추가한 의미 region의 recall

자동 문자열 유사도를 semantic truth로 승격하는 코드는 두지 않습니다.
