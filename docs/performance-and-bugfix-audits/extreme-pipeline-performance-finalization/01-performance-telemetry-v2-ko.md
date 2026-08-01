# Performance telemetry v2 구현 기록

## 목적

후속 scheduler, runtime arbiter와 ETA가 같은 측정 계약을 사용하도록 제품에
후보 순위와 무관한 범용 계측 기반을 추가한다. 이 변경은 모델·prompt·출력과
파이프라인 순서를 바꾸지 않는다.

## 추가된 계약

- 전체 stage window와 page별 기존 stage 합계를 분리한다.
- image decode, decoded hash, detector inference, OCR queue/request/parse,
  inpaint mask/model load/forward/cleanup, translation plan/prefill/decode,
  render window를 구분한다.
- runtime의 process/model load, health, ready, release, sleeping/stopped 상태
  전이를 기록한다.
- work-DAG node에 실제 시작 offset, elapsed와 dependency를 기록한다.
- page, pixel, block, mask, ROI, source-character 수만 workload feature로
  기록한다.
- GPU memory/utilization과 WSL swap의 시작·끝·peak·delta를 기록한다.
- `CT_PERFORMANCE_NVTX=1`인 lab 실행에서만 NVTX range를 방출한다.

문자열 feature는 안전한 identifier만 허용한다. 파일 경로, raw text, prompt,
이미지와 응답 내용은 performance snapshot에 들어가지 않는다. 계측은 lock-safe,
fail-open이며 예외를 삼키지 않는다.

## 검증 계약

- schema version과 이전 stage/runtime/cache 집계를 보존하는 단위 테스트
- private 문자열 차단, work graph, runtime transition, 실패 예외 전달 테스트
- `.venv-win`과 `.venv-win-cuda13` 전체 단위 테스트
- Python 정적 검사, headless smoke, Qt translation asset 검사
- cache OFF 실제 S1 full-auto에서 15/15 OCR·번역과 terminal snapshot 확인

원시 timing, local path와 runtime log는 public 문서에 포함하지 않고 private
validation archive에서만 보존한다.

## 첫 실측 결과

- `.venv-win`: 888 tests 통과, 7 skipped
- `.venv-win-cuda13`: 888 tests 통과, 7 skipped
- cache OFF S1 full-auto: OCR 15/15, 번역 15/15, terminal status `completed`
- schema v2 workload: 1 page, 15 OCR blocks, 15 translation blocks,
  163 source characters, 600,841 mask pixels, 14 inpaint model calls
- critical-path work graph: 7 nodes

이 S1은 OS page-cache 상태가 통제된 후보 AB가 아니므로 성능 승격 수치로 쓰지
않는다. 실행 시작 대비 WSL swap이 5.711 MiB 증가했으므로, 이후 scheduler
후보에서는 사용자 계획의 swap 비증가 게이트를 그대로 적용한다.
