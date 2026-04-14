# Workflow Split Runtime Usage

## 현재 상태

이 family는 “실측 준비 완료” 단계까지 고정되었다.

- runner: 구현 완료
- preset 3종: 구현 완료
- CUDA12/CUDA13 BAT: 구현 완료
- generated report updater: 구현 완료
- `baseline_legacy`: 실행 가능
- `candidate_stage_batched_single_ocr`: blocked 계약 run
- `candidate_stage_batched_dual_resident`: blocked 계약 run

## 준비 조건

1. corpus: `Sample/japan`
2. 공식 13장:
   - `094.png`
   - `097.png`
   - `101.png`
   - `i_099.jpg`
   - `i_100.jpg`
   - `i_102.jpg`
   - `i_105.jpg`
   - `p_016.jpg`
   - `p_017.jpg`
   - `p_018.jpg`
   - `p_019.jpg`
   - `p_020.jpg`
   - `p_021.jpg`
3. 로컬 환경:
   - `.venv-win`
   - `.venv-win-cuda13`
4. Docker health-first reuse 정책 사용
5. Gemma / OCR runtime health endpoint 확인 가능

## 실행 계약

- 레거시 page pipeline과 stage-batched 후보 워크플로우를 같은 family에서 추적한다.
- Docker compose up 시간, health wait 시간, reuse hit, timeout/retry, VRAM snapshot을 반드시 저장한다.
- 결과는 `banchmark_result_log/workflow-split-runtime/` 아래에 저장한다.
- 각 run은 아래 파일을 만든다.
  - `benchmark_request.json`
  - `events.jsonl`
  - `timing_summary.json`
  - `quality_summary.json`
  - `vram_snapshots.jsonl`
  - `docker_timeline.json`
  - `report.md`

## Windows 실행

### CUDA12

```bat
scripts\run_workflow_split_runtime_cuda12.bat
scripts\run_workflow_split_runtime_cuda12.bat smoke
scripts\run_workflow_split_runtime_cuda12.bat summary
```

### CUDA13

```bat
scripts\run_workflow_split_runtime_cuda13.bat
scripts\run_workflow_split_runtime_cuda13.bat smoke
scripts\run_workflow_split_runtime_cuda13.bat summary
```

## Python 직접 실행

### smoke

```bash
python scripts/workflow_split_runtime_benchmark.py run --scenario all --smoke
```

### full

```bash
python scripts/workflow_split_runtime_benchmark.py run --scenario all
```

### report refresh

```bash
python scripts/workflow_split_runtime_benchmark.py summary
```

## 문서 계약

suite가 갱신되면 아래 문서를 함께 갱신한다.

- `workflow-ko.md`
- `architecture-ko.md`
- `results-history-ko.md`
- `docs/banchmark_report/workflow-split-runtime-report-ko.md`
- `problem-solving-specs/*.md`

## 현재 해석

이 family는 이제 baseline smoke를 시작할 준비가 되었고, stage-batched experimental runner가 붙는 즉시 같은 산출물 규약으로 candidate 두 시나리오를 실제 측정할 수 있다.
