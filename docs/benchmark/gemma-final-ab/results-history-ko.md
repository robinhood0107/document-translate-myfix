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
- 사용자 요청에 따른 Codex blind 전수 검수 292/292 완료
- protocol v4 review validator와 명시적 확인문 통과 뒤 unblind
- A=`current-contextual-single`: 14행, 18출력 회귀
- B=`grouped-f16`: 21행, 36출력 회귀
- A 전체 프로필:
  `IQ4_XS + contextual-single + chunk 6 + ngram draft8 + F16`
- B 전체 프로필:
  `IQ4_XS + contextual-grouped + chunk 7 + no-spec + F16`
- 실제 제품 기본 프로필:
  `IQ4_NL + contextual-single + chunk 6 + no-spec + F16`
- grouped 회귀 출력: round 1은 19개, round 2는 17개
- grouped 회귀 행 중 15개는 두 라운드에서 반복
- 상대 품질 우위: `current-contextual-single`
- 최종 상태: `quality_rejected`
- B 조합 전체의 제품 기본값 승격: 금지
- 이 결과만으로 grouping 하나를 모든 회귀의 원인으로 단정하지 않음
- 실제 제품 기본 프로필: 변경하지 않음
- 전체 파이프라인 비교: 품질 gate 실패에 따른 정상 실행 취소
- grouped 제품 코드 재현 기준: PR #141 `develop` merge
  `fbc131c73eb260abc9be6aec1334dba6a7da738c`
- 최종 검수 증거 기준: PR #148 `benchmarking/lab` merge
  `034a6e85172d438e9e1fe5d29560a105493b6f6b`
- 제품 퇴역 기준: PR #149 `develop` merge
  `24a7fb8ae194e5d1510ee6e0a288ec636cdba2b9`
