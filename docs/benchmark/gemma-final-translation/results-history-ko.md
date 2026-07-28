# Gemma 최종 번역 전용 비교 결과 이력

## 2026-07-28

- 상태: 조건부 3 round에서 Q8 hard gate 실패로 중단
- 고정 비교: current contextual-single, grouped F16, grouped Q8
- 고정 구조: 22페이지, 292블록, grouped size 7, completion token 512
- 기본 2 round 실측:
  - current contextual-single median 173.665초
  - grouped F16 median 104.100초, 기준선 대비 40.057% 단축
  - grouped Q8 median 109.025초, F16보다 4.730% 느림
- 조건부 3 round: F16/Q8 차이가 5% 미만이어서 실행
- 중단 사유: Q8이 partial response 1건을 contextual-single 1회로
  fallback하여 logical/HTTP 요청 계약과 clean-run gate를 위반
- 출력 보존: 22페이지, 292블록, 순서, 빈 값, 구조 출력은 정상
- 사용자 품질 판정: blind 산출물을 만들지 않고 중단
- 22페이지 전체 파이프라인: 실행 금지 상태
- review 보강: 알려진 corpus digest, 번역 동작·resume contract, loopback
  container, pinned container ID, result hash/path, 정상 request 수, Latin-square
  순서, process-attributed VRAM

실측 뒤에는 raw 번역을 복사하지 않고 median, 편차, 구조 gate, VRAM 차이,
사용자 판정만 이 문서에 추가한다.

### 후속 protocol v4

Q8을 후보에서 제외한 뒤 저장된 baseline/F16 clean 결과만 가져오는
report-only A/B 검수 도구를 별도 family로 추가했다. v3 source suite는
수정하지 않으며, 새 모델 요청 없이 두 후보의 두 라운드를 모두 사용자
검수 대상으로 제공한다.

- [workflow](../gemma-final-ab/workflow-ko.md)
- [latest report](../gemma-final-ab/generated/latest-report-ko.md)
