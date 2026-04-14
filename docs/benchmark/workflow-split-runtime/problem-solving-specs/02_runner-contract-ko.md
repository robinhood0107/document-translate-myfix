# 문제 해결 명세서 02 - Runner Contract

핵심 문제 해결 방향은 사용자가 착안했다.

## 문제

Requirement 1을 실제로 돌리기 전에, 어떤 시나리오를 어떤 파일 구조로 남길지 계약이 잠겨 있지 않으면 나중에 결과 비교표와 포트폴리오 문서가 흐트러질 수 있었다.

## 사용자 착안 요약

사용자는 `13장` 기준으로 기존 페이지 단위 워크플로우와 단계형 후보 워크플로우를 같은 프레임에서 비교하고, Docker 기동/healthcheck/VRAM/timeout까지 전부 근거 파일로 남겨야 한다고 요구했다.

## 이번 단계에서 잠근 실행 계약

1. family 이름은 `workflow-split-runtime`으로 유지한다.
2. 실제 코퍼스 루트는 `Sample/japan`으로 잠근다.
3. 공식 13장 파일 목록을 고정한다.
4. 공식 smoke는 `094.png` + `p_016.jpg` 2장으로 잠근다.
5. run 산출물은 아래 7종을 기본 계약으로 둔다.
   - `benchmark_request.json`
   - `events.jsonl`
   - `timing_summary.json`
   - `quality_summary.json`
   - `vram_snapshots.jsonl`
   - `docker_timeline.json`
   - `report.md`
6. Windows launcher는 아래 두 이름으로 고정한다.
   - `scripts/run_workflow_split_runtime_cuda12.bat`
   - `scripts/run_workflow_split_runtime_cuda13.bat`

## 현재 구현 내용

- baseline scenario는 실제 `benchmark_pipeline.py`를 감싼 family runner에서 실행된다.
- stage-batched candidate 두 개는 아직 experimental runner가 없으므로, 같은 파일 구조를 가진 blocked 계약 run으로 먼저 기록한다.
- family runner는 마지막 실행 결과를 `banchmark_result_log/workflow-split-runtime/last_workflow_split_runtime_suite.json`에 잠근다.
- report generator는 최신 suite record를 읽어 `results-history-ko.md`와 generated report를 갱신한다.

## 왜 이렇게 설계했는가

- 지금 당장 baseline smoke를 시작할 수 있다.
- stage-batched 실험 코드를 나중에 붙여도 산출물 구조를 다시 바꿀 필요가 없다.
- “지금 실행 가능한 것”과 “아직 막혀 있는 것”을 같은 문서 틀에서 정직하게 보여줄 수 있다.

## 기대 효과

- Requirement 1 실측 패키지가 코드/문서/BAT 기준으로 먼저 잠긴다.
- baseline evidence를 쌓으면서도, 나중에 candidate runner를 붙일 자리가 명확해진다.
- 포트폴리오 문서가 실측 전 단계부터 자연스럽게 이어진다.

## 남은 리스크

- stage-batched candidate는 아직 실제 파이프라인 실행이 불가능하다.
- 따라서 Requirement 1 성공 게이트는 아직 판정할 수 없다.

## 다음 액션

1. `run_workflow_split_runtime_cuda13.bat smoke`로 baseline smoke를 돌린다.
2. baseline 13장 measured run을 누적한다.
3. benchmarking/lab에 experimental stage-batched runner를 추가한다.
4. 같은 산출물 구조로 candidate 두 시나리오를 실제 측정한다.

## 저자 및 기여

- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
