# Gemma 최종 A/B blind 품질 검수 결과 이력

## 2026-07-28

- protocol v4 report-only importer 구현
- source protocol v3 canonical fingerprint 재검증 통과
- input manifest, OCR snapshot, IQ4_XS model, runtime, 번역 동작 계약 통과
- source 일곱 run의 고정 SHA-256·상대 경로·실행 순서 재검증 통과
- baseline 1·2라운드와 grouped F16 1·2라운드 clean gate 통과
- Q8은 3회차 partial fallback·invalid value 실패 증거만 확인하고 후보에서 제외
- 네 clean 결과의 22페이지·292블록 키·원문·페이지·블록 순서 일치
- 후보명·속도·mapping을 숨긴 292행 A/B HTML·CSV 생성
- 초기 빈 검수표의 1,168개 미검수 판정 거부 및 unblind 미생성 확인
- 실제 브라우저에서 292행·1,168개 판정 입력, 진행 카운터, CSV export,
  console error 0 확인
- Windows CUDA12·CUDA13 환경의 단위 테스트, headless smoke, 번역 asset,
  launcher/runtime contract 검사 통과
- CUDA13 환경에서 실제 source import와 report 생성 통과
- 사용자 품질 검수: 대기
- 제품 기본값 승격: 시작하지 않음
- 전체 파이프라인 비교: 시작하지 않음
