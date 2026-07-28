# Gemma 최종 번역 전용 비교 latest

- 상태: `STOPPED_Q8_HARD_GATE`
- 측정 범위: translation-only
- 입력 계약: 22페이지 / 292블록
- 후보: 3
- 기본 round: 2
- round 순서: Latin-square
- 조건부 round: 후보 편차 5% 초과 또는 F16/Q8 차이 5% 미만
- protocol v3 Docker preflight: 통과
- lab PR CI: 통과
- 기본 2 round: 세 후보 모두 통과
- 조건부 3 round: Q8 첫 실행에서 중단
- 기준선 median: 173.665초
- grouped F16 median: 104.100초, 기준선 대비 40.057% 단축
- grouped Q8 통과 round median: 109.025초, F16보다 4.730% 느림
- 중단 사유: Q8 partial fallback 1블록, invalid value 1건
- 구조 품질 gate: Q8 실패
- 성능 gate: F16 통과, Q8 탈락
- 사용자 blind review: 미생성
- full pipeline: 사용자 승인 전 금지

## 후속 검수

Q8을 탈락으로 고정하고 baseline/F16의 저장된 clean 2라운드만 가져오는
protocol v4 report-only A/B 검수가
[별도 family](../../gemma-final-ab/generated/latest-report-ko.md)에서
완료됐다. 292행 blind 전수 검수에서 grouped F16은 21행 36출력의 의미
회귀가 확인되어 `quality_rejected`로 종료됐다. 상대적으로는
current contextual-single이 나았으나 이 source v3 결과와 Q8 중단 판정은
변경하지 않는다.
