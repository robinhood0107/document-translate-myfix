# 문제 해결 명세서 04 - Review Candidate Selection

핵심 문제 해결 방향은 사용자가 착안했다.

## 가설

속도 랭킹 상위 2개 비-baseline 후보를 사용자 OCR diff 검수 대상으로 고정하고, quality gate winner는 보조 해석 지표로만 유지한다.

## 실험 조건

- rank_metric=ocr_total_sec median

## 측정값

- quality_gate_winner=fixed_area_desc_w8 review_top1=fixed_area_desc_w8 review_top2=auto_v1_cap4

## 해석

속도 랭킹 상위 2개 비-baseline 후보를 사용자 OCR diff 검수 대상으로 고정하고, quality gate winner는 보조 해석 지표로만 유지한다.

## 다음 행동

- user OCR diff review

## 저자 및 기여

- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
