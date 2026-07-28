# cold/cache 최종화 결과 이력

## protocol v1 도구 추가

- frozen 제품 기준선과 후보 matrix를 추가했습니다.
- 실제 offscreen stage-batched pipeline을 최대 6페이지로 실행합니다.
- cold 후보는 3회 순서 교차, cache는 disabled/empty miss overhead와
  all-hit를 분리 측정합니다.
- project checkpoint는 기존 output, missing output materialization,
  한 페이지 OCR downstream 무효화를 검증합니다.
- stopped Docker container의 불필요한 300초 health wait를 제거했습니다.
- wrapper 사전 부팅을 제거하고 staged runtime을 제품 manager에 주입해
  cold startup과 all-hit runtime 0을 실제 제품 경로에서 측정합니다.
- 첫 실제 스모크에서 staged Compose의 암시적 project 이름이 달라 기존
  stopped 컨테이너와 이름 충돌하는 문제를 재현했습니다. 제품과 같은
  Compose project 이름을 명시해 원인을 수정했습니다.
- 같은 stopped container 조건의 후속 1페이지·20블록 스모크는 페이지
  실패 0, Paddle runtime 시작 1회, OCR HTTP 20회로 완료됐습니다.
  runtime 시작은 82.276초, 전체 pipeline wall time은 105.318초였으며
  종료 뒤에도 제품 Compose project identity가 유지됐습니다.
- 첫 project checkpoint cold control에서는 staged Gemma command에
  `-ctk/-ctv`와 같은 의미의 긴 옵션이 중복돼 제품 runtime의 명령 완전
  일치 계약이 실패했습니다. staging이 기존 `-ctk/-ctv` 값을 직접
  교체하도록 수정하고 중복 옵션 금지 회귀 검사를 추가했습니다.
- staged Gemma command에 no-spec/F16 계약을 직접 고정했습니다.
- full baseline의 실제 stage share를 검증해 stage 5% 또는 예상 전체 1%
  게이트를 판정할 수 있게 했습니다.
- 의미 검수 대상은 모든 라운드를 외부 private review JSON으로 묶습니다.
- raw 결과는 Git 밖에만 기록합니다.
- provisional global OCR cache 1페이지 검증은 miss overhead 0.036%,
  all-hit 99.413% 단축, runtime 시작 0, HTTP 0, raw OCR 완전 동일로
  통과했습니다.
- provisional project checkpoint 1페이지 검증은 all-stage hit,
  runtime 시작 0, Gemma HTTP 0, render-only 복원, 페이지 단위 재계산,
  all-hit 91.406% 단축을 확인했습니다. 다만 checkpoint에서 복원한
  과거 OCR profile을 현재 telemetry가 다시 합산해 Paddle HTTP 30회로
  보이던 오류와 최초 runtime 준비 편차 때문에 정식 판정은 보류했습니다.
- 과거 OCR telemetry 재합산은 제품 PR #156에서 수정했습니다.
- cache runner는 cache-disabled/enabled-empty를 각각 한 번 비채점
  안정화한 뒤 격리된 측정 3회를 실행하도록 보강했습니다. 안정화 결과도
  보존하고 실패 시 즉시 중단하지만 miss overhead와 variance에는 넣지
  않습니다.
- 새 project 검증에서 cold median miss overhead -4.375%, all-hit
  92.230% 단축, 모든 stage hit, runtime/HTTP 0, render-only 복원,
  한 페이지 OCR downstream 재계산이 통과했습니다.
- cold wall variance는 disabled 8.676%, enabled-empty 15.284%였습니다.
  outlier는 각각 inpainter 시간 증가와 Paddle/Gemma runtime wait 증가로,
  cache I/O와 무관했습니다. cache 보고서에는 이를 비차단 diagnostic으로
  유지하고 승인된 median miss overhead 3% gate와 분리합니다.
- 첫 partial 재계산은 프로젝트 복원 UI callback이 창 종료 뒤 실행되는
  benchmark race로 실패했습니다. 프로젝트 UI 작업을 pipeline 시작 전에
  drain하도록 수정한 뒤 동일 partial smoke와 정식 재계산이 통과했습니다.
- commit `47b6480`의 clean checkout에서 protocol v1
  (`7dfc2baa57ebdafc0e9ca2944e20397ccc9cd008d92f366f2a046dbffd3af04f`)
  최종 cache suite를 다시 실행했습니다.
- global OCR exact cache는 disabled cold 중앙값 20.101초,
  enabled-empty 중앙값 19.999초로 miss overhead -0.510%였고,
  all-hit는 0.120초로 99.401% 단축됐습니다. raw OCR은 완전히 같았고
  Paddle runtime 시작, 논리 요청, HTTP 요청은 모두 0이었습니다.
- project checkpoint는 disabled cold 중앙값 129.793초,
  enabled-empty 중앙값 129.657초로 miss overhead -0.105%였고,
  all-hit는 9.551초로 92.633% 단축됐습니다. 모든 project stage hit,
  detector/Paddle/Gemma/inpainter inference 0, Paddle/Gemma runtime 시작
  및 HTTP 0, missing-output render-only 복원, 단일 페이지 OCR downstream
  재계산, exact output과 비영향 페이지 보존을 모두 통과했습니다.
- 최종 cold variance는 global OCR disabled/enabled 0.885%/0.494%,
  project disabled/enabled 1.587%/2.971%로 diagnostic 기준 5% 안에
  들어왔습니다.

이 문서는 도구 구현 이력과 검증된 cache gate 결과입니다. raw 입력,
OCR/번역 결과, DB와 sidecar는 Git 밖에만 보존합니다. cold 후보의 제품
기본값은 별도 family 선별과 품질 게이트를 통과하기 전에는 변경하지
않습니다.
