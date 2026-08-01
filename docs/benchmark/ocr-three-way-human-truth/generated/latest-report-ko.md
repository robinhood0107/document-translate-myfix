# 세 OCR 사람 정답 벤치마크 latest

## 완료 상태

- protocol: `ocr-three-way-human-truth-v1`
- 관리형 backend: llama.cpp only
- 경로: Paddle crop / Paddle full-page Spotting / MangaLMM full-page
- source-first truth lock: 완료
- Japan 22페이지 원본·확대 crop 직접 판독: 완료
- A/B/C blind 검수·완료 전 unblind 차단: 완료
- 구조 실패·추가 영역·merge/split·파괴 편집 집계: 완료
- 제품 기본값 변경: 없음

## 사람 기준 결과

| 경로 | 구조 성공 | 페이지 완전 | 글자 정확 | 문자 정확도 | 의미 recall | 추가 영역 | 오탐 | merge/split | 파괴 편집 | 요청 시간 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paddle crop | 22/22 | 9.09% | 59.31% | 76.10% | 177/187, 94.65% | 0 | 0 | 2 | 0 | 196.589초 |
| Paddle full-page Spotting | 22/22 | 4.55% | 44.16% | 62.59% | 156/187, 83.42% | 36 | 0 | 10 | 0 | 41.868초* |
| MangaLMM full-page | 22/22 | 0.00% | 39.12% | 57.69% | 151/187, 80.75% | 265 | 6 | 30 | 0 | 별도 live CUDA 804.550초 |

\* Paddle Spotting 시간은 당시 direct 요청 합계이며 startup과 전체 pipeline을 포함하지
않아 Paddle crop의 제품 전체시간과 직접 비교할 수 없습니다.

## 최종 판정

Paddle crop이 의미 텍스트 recall, 원문 문자 정확도, merge/split 안정성에서 모두
우세했습니다. Paddle Spotting과 MangaLMM은 parser와 CUDA 실행 자체는 안정화됐지만
line/block 단위 불일치와 추가 영역 routing 손실을 현재 기준선 수준까지 줄이지
못했습니다.

따라서 기본·추천 경로는 계속 Paddle crop입니다. Spotting과 MangaLMM은 결과 재현과
향후 연구를 위한 Experimental 경로로만 유지하며 자동 fallback이나 동시 상주를
사용하지 않습니다. COO는 세 경로 어디에서도 안전한 SFX 분리점을 만들지 못해 제품에
적용하지 않습니다.

원시 이미지, truth, route 결과, blind key와 완료 검수표는 Git 밖 validation log에만
보존합니다.
