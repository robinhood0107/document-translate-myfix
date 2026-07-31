# 세 OCR 사람 정답 벤치마크 latest

## 현재 상태

- protocol: `ocr-three-way-human-truth-v1`
- 관리형 backend: llama.cpp only
- 경로: Paddle crop / Paddle full-page Spotting / MangaLMM full-page
- source-first truth 도구: 구현 완료
- 후보 결과 import와 source binding: 구현 완료
- A/B/C blind review와 완료 전 unblind 차단: 구현 완료
- 실제 source-first truth lock: 대기
- 전체 사람 정답 검수: 대기
- 제품 기본값 변경: 금지

## 현재 판정

기존 자동 일치율은 사람 기준 OCR 정확도가 아니다. 원본과 detector crop만 본 상태에서
정답을 먼저 잠그고, exact source/result binding을 통과한 세 경로를 같은 A/B/C 표에서
전수 검수한 뒤에만 정확도와 추천 경로를 판정한다. 현재 제품 기본값은 변경하지 않는다.

원시 이미지, 원문 정답, route 결과, blind key와 완료 CSV는 모두 Git 밖 검증 폴더에
보존한다. 이 파일은 재현 가능한 도구와 현재 gate 상태만 기록한다.
