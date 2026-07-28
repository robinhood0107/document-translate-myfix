# Gemma 최종 번역 전용 비교 결과 이력

## 2026-07-28

- 상태: runner 구현 및 pre-landing review 수정 검증
- 고정 비교: current contextual-single, grouped F16, grouped Q8
- 고정 구조: 22페이지, 292블록, grouped size 7, completion token 512
- 실측 결과: 실행 전
- 사용자 품질 판정: 대기
- 22페이지 전체 파이프라인: 실행 금지 상태
- review 보강: 알려진 corpus digest, 번역 동작·resume contract, loopback
  container, pinned container ID, result hash/path, 정상 request 수, Latin-square
  순서, process-attributed VRAM

실측 뒤에는 raw 번역을 복사하지 않고 median, 편차, 구조 gate, VRAM 차이,
사용자 판정만 이 문서에 추가한다.
