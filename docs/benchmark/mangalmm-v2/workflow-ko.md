# MangaLMM v2 검증 workflow

## 브랜치와 데이터 경계

- benchmark runner·manifest 계약·보고서는 `benchmarking/lab`에만 둔다.
- 제품 runtime·parser·UI는 별도 `develop` PR로 승격한다.
- 원본 이미지, annotation, crop, raw response, overlay, 번역 결과는 Git
  밖에 둔다.
- manifest와 결과 출력 경로가 Git working tree 안이면 runner가 거부한다.
- 제품 코드와 Git 문서에는 실제 페이지명·좌표·원문 token을 넣지 않는다.

## 외부 manifest 계약

`benchmarks/mangalmm_v2/evaluation-manifest.example.json`은 형식만 보여준다.
실제 manifest는 외부 검증 폴더에 만든다.

각 case는 다음을 고정한다.

- neutral `case_id`
- `development`, `holdout`, `negative_control`, `final` 중 하나의 split
- source image와 SHA-256
- annotation JSON과 SHA-256
- holdout이면 후보 실행 전 `frozen_before_candidate_run=true`

annotation의 각 region은 다음을 포함한다.

- bbox 또는 polygon
- 원문
- semantic role
- processing action
- bubble 유형
- 인간 번역 대상 여부

지원 role:

- `dialogue_bubble`, `dialogue_free`, `narration`, `ui_or_sign`
- `sfx`, `decorative`, `ambiguous`

지원 action:

- `translate_inpaint`, `preserve`, `review`

불변식:

- `sfx`, `decorative`는 `preserve`
- `ambiguous`는 `review`
- 인간 번역 대상이 아닌 영역은 `translate_inpaint` 금지
- 인간 번역 대상인 의미 영역은 `preserve` 금지

## 실행

과거 이력 감사:

```powershell
.venv-win\Scripts\python.exe scripts\benchmark_mangalmm_v2.py audit-history `
  --output C:\external-validation\mangalmm-v2\history-audit.json
```

외부 평가 계약 검증:

```powershell
.venv-win\Scripts\python.exe scripts\benchmark_mangalmm_v2.py validate-manifest `
  --manifest C:\external-validation\mangalmm-v2\evaluation-manifest.json `
  --output C:\external-validation\mangalmm-v2\manifest-contract.json
```

`--skip-file-hashes`도 외부 annotation 파일과 schema는 검사하지만 SHA 일치
검사만 생략한다. 정식 benchmark 실행 자격을 주지 않는다.

## v2 실행 순서

1. 사람이 원본을 보고 development/holdout annotation을 먼저 고정한다.
2. history audit와 manifest SHA 검증을 통과시킨다.
3. Manga named volume·full-page runtime·strict parser를 제품 코드에 구현한다.
4. development에서 `1728/8192`와 `2048/12288`를 비교한다.
5. 손상·잘림·반복·coverage gap에서만 상위 profile을 한 번 재시도한다.
6. semantic role routing과 COO shadow 신호를 개발 세트에서 고정한다.
7. 잠금 holdout을 정확히 한 번 실행한다.
8. 공통 최종 세트와 영어 회귀를 실행한다.
9. Paddle과 Manga를 같은 후속 Stage-Batched 파이프라인의 별도 사용자
   선택지로 노출한다.

## 실패 조건

- crop·tile·Paddle fallback 요청
- dense profile의 해상도나 completion 용량 감소
- parser 손상·잘림·반복 또는 좌표 clipping 오류
- N:1·1:N 영역의 자동 병합
- 의미 텍스트 누락 또는 SFX·작은 UI의 잘못된 삭제
- mask 밖 픽셀 변경
- holdout 결과를 본 뒤 같은 holdout에 맞춰 rule을 수정
