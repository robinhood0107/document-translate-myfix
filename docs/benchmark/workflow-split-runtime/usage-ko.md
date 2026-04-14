# Workflow Split Runtime Usage

## 현재 상태

이 family는 아직 실행 scaffold 단계다. runner, preset, BAT launcher, report generator는 이후 Requirement 1 구현 단계에서 추가된다.

## 준비 조건

1. corpus: `Sample/japan_vllm_parallel_subset`
2. 로컬 환경:
   - `.venv-win`
   - `.venv-win-cuda13`
3. Docker health-first reuse 정책 사용
4. Gemma / OCR runtime health endpoint 확인 가능

## 실행 계약

- 레거시 page pipeline과 stage-batched pipeline을 같은 corpus 13장에 대해 비교한다.
- Docker compose up 시간, health wait 시간, reuse hit, timeout/retry, VRAM snapshot을 반드시 저장한다.
- 결과는 `banchmark_result_log/workflow_split_runtime/` 아래에 저장한다.

## 문서 계약

suite가 생기면 아래 문서를 함께 갱신한다.

- `workflow-ko.md`
- `architecture-ko.md`
- `results-history-ko.md`
- `docs/banchmark_report/workflow-split-runtime-report-ko.md`
- history 아래 `problem_solving_specs/*.md`

## 예정 BAT 쌍

- `scripts/workflow_split_runtime_benchmark_pipeline.bat`
- `scripts/workflow_split_runtime_benchmark_pipeline_cuda13.bat`
- `scripts/workflow_split_runtime_benchmark_suite.bat`
- `scripts/workflow_split_runtime_benchmark_suite_cuda13.bat`

현재는 이름만 잠근 상태이며, 실제 스크립트는 후속 단계에서 추가한다.
