# Workflow Split Runtime Review Pack Scoring Guide

아이디어 착안자: 사용자

## 목적

이 문서는 Japanese `Optimal+ analysis mode` 실측에서 생성된 Review Pack을 어떻게 읽고, 어떤 기준으로 `O / X`를 매겨 selector threshold를 잠글지 안내하는 문서다.

이번 단계의 채점 목적은 다음 하나로 고정한다.

- 향후 `stage_batched_pipeline + Optimal+ Japanese`에서, 각 페이지를 `MangaLMM 그대로 사용`할지 아니면 `PaddleOCR VL로 fallback`할지를 결정하기 위한 기준점을 만든다.

## 열어야 할 파일

### 1. 핵심 요약표

- `banchmark_result_log/workflow-split-runtime/20260415_091848_candidate_stage_batched_dual_resident/sidecar_review_pack.md`

이 파일은 GitHub에서 바로 읽기 좋은 표 형식이다.

### 2. raw 데이터

- `banchmark_result_log/workflow-split-runtime/20260415_091848_candidate_stage_batched_dual_resident/sidecar_review_pack.json`

이 파일은 수치를 더 자세히 확인할 때 쓴다.

### 3. 채점용 시트

- `banchmark_result_log/workflow-split-runtime/20260415_091848_candidate_stage_batched_dual_resident/review_decision_sheet-ko.md`

이 파일에 각 페이지별 `O / X`를 적으면 된다.

### 4. 참고 출력 이미지

- `banchmark_result_log/workflow-split-runtime/20260415_091848_candidate_stage_batched_dual_resident/094/*_translated.jpg`

주의:

- 이 translated image들은 이번 analysis mode에서 `PaddleOCR VL` downstream 기준으로 만들어진 출력이다.
- 즉 이번 Review Pack은 `MangaLMM 결과 자체의 최종 렌더 미리보기`가 아니라, `MangaLMM을 통과시켜도 되는 페이지인지`를 판단하기 위한 metric-based review 단계다.

## 이번 채점에서 `O / X`의 의미

- `O`
  - 이 페이지는 `MangaLMM` 결과를 허용 가능하다고 보고, 향후 selector에서 `PaddleOCR VL` fallback 없이 통과시켜도 된다고 판단한다.
- `X`
  - 이 페이지는 `MangaLMM` 결과만으로는 품질이 부족하다고 보고, 향후 selector에서 반드시 `PaddleOCR VL`로 fallback해야 한다고 판단한다.

중요한 점:

- 현재 `Optimal+ Japanese` 실측은 `PaddleOCR VL`을 downstream 기준으로 사용했고, `MangaLMM`은 sidecar 비교 데이터만 수집했다.
- 따라서 이번 Review Pack은 “최종 렌더 결과물 채점”이 아니라 “MangaLMM을 통과시켜도 되는 페이지인지”를 판정하는 단계다.

## 이 단계에서 봐야 할 핵심 수치

각 페이지마다 아래 수치를 본다.

- `detect_box_count`
  - detector가 찾은 박스 수
- `sidecar_non_empty`
  - `MangaLMM`이 실제 텍스트를 채운 박스 수
- `bbox_2d_success_block_count`
  - `MangaLMM`이 `bbox_2d`를 정상적으로 매핑한 블록 수

이번 threshold 제안에서 핵심 기준은 이 값이다.

- `bbox_mismatch_ratio = (detect_box_count - bbox_2d_success_block_count) / detect_box_count`

해석:

- 비율이 낮을수록 `MangaLMM`이 detector 박스와 더 잘 맞았다는 뜻이다.
- 비율이 높을수록 `MangaLMM`이 많이 놓쳤거나 `bbox_2d` 매핑이 잘 안 됐다는 뜻이다.

## 채점 방법

### 기본 규칙

1. `bbox_mismatch_ratio`가 매우 낮으면 `O`
2. `bbox_mismatch_ratio`가 높거나, hard page 느낌이 강하면 `X`
3. 애매하면 `비고`에 이유를 남기고 우선 `X` 쪽으로 보수적으로 판단

### 이번 benchmark에서 권장하는 provisional 해석

- `bbox_mismatch_ratio <= 0.15`
  - 기본적으로 `O` 후보
- `0.15 < bbox_mismatch_ratio <= 0.25`
  - 수동 검토 구간
- `bbox_mismatch_ratio > 0.25`
  - 기본적으로 `X` 후보

추가 보수 규칙:

- `detect_box_count >= 20` 이면서 `miss_count >= 5`면 `X` 쪽으로 본다.
- `p_016.jpg` 같이 hard page로 드러난 사례는 기본적으로 `X`로 본다.

## 사용자가 실제로 해주면 되는 일

1. `review_decision_sheet-ko.md`를 연다.
2. 각 페이지의 `suggested_band`를 참고한다.
3. 최종 판단을 `reviewer_decision` 열에 `O` 또는 `X`로 적는다.
4. 애매했던 이유가 있으면 `reviewer_note`에 짧게 적는다.

사용자가 GitHub에서 직접 수정하지 않아도 된다.

- 페이지 이름별로 `O / X`만 따로 보내줘도 된다.
- 예시:
  - `097.png O`
  - `i_102.jpg X`
  - `p_018.jpg O`

## 이 채점이 끝나면 다음에 하는 일

1. `O / X` 결과를 기준으로 threshold를 잠근다.
2. page-level selector rule을 작성한다.
3. 그 rule을 적용한 `Optimal+ Japanese` 최종 rerun을 다시 측정한다.
4. 그 뒤에야 `develop` 승격 후보로 올린다.
