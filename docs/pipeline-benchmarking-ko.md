# 자동번역 벤치마킹 가이드

기준 날짜: `2026-04-05`

이 문서는 `Gemma + PaddleOCR VL` 조합을 반복적으로 벤치마킹하는 방법과 기록 규칙을 정리합니다.

## 1. 코퍼스 규칙

기본 코퍼스는 저장소 루트의 로컬 `/Sample` 폴더입니다.

- representative corpus: `30장`
- smoke corpus: representative의 앞 `5장`

이 폴더는 로컬 검증 전용입니다.

- Git에 올리지 않음
- `.gitignore`에 `/Sample/` 추가

## 1-1. 실행 환경 전제

오프스크린 benchmark 스크립트는 실제 앱 파이프라인을 import해서 돌립니다.

따라서 아래와 같은 full runtime 의존성이 준비된 Python 환경에서 실행해야 합니다.

- `PySide6`
- `opencv-python` 또는 이에 준하는 `cv2`
- 현재 앱이 사용하는 OCR / ONNX / 기타 런타임 의존성

Windows에서는 시스템 Python이 아니라 아래 전용 환경을 사용합니다.

- `.venv-win`
- `.venv-win-cuda13`

## 2. 산출물 위치

벤치 결과는 저장소가 아니라 Windows 사용자 문서 폴더 아래의 `Comic Translate` 폴더에 저장합니다.

기본 경로:

```text
%USERPROFILE%\Documents\Comic Translate\<timestamp>_<label>\
```

선택적으로 `CT_BENCH_OUTPUT_ROOT` 환경변수로 출력 루트를 덮어쓸 수 있습니다.

각 run 디렉터리에는 아래 파일이 생성됩니다.

- `benchmark_request.json`
- `preset_resolved.json`
- `runtime_snapshot.json`
- `docker_snapshot.json`
- `metrics.jsonl`
- `summary.json`
- `summary.md`

## 3. 계측 항목

`metrics.jsonl`에는 아래 정보가 들어갑니다.

- `tag`
- `pipeline_mode`
- `run_type`
- `image_path`, `image_name`
- `rss_mb`
- app cache 상태
- GPU snapshot
  - `memory_total_mb`
  - `memory_used_mb`
  - `memory_free_mb`
  - `gpu_util_percent`
  - `memory_util_percent`

단계 태그는 아래를 사용합니다.

- `page_start`
- `detect_start`, `detect_end`
- `ocr_start`, `ocr_end`
- `inpaint_start`, `inpaint_end`
- `translate_start`, `translate_end`
- `render_start`, `render_end`
- `page_done`
- `page_failed`

추가 run-level 태그도 남습니다.

- `batch_run_start`, `batch_run_done`, `batch_run_cancelled`
- `webtoon_run_start`, `webtoon_run_done`, `webtoon_run_cancelled`
- `benchmark_run_start`, `benchmark_run_finished`

## 4. preset 적용 방식

preset은 [benchmarks/presets](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/benchmarks/presets)에 JSON으로 관리합니다.

현재 기본 preset:

- `repo-default`
- `live-ops-baseline`
- `gpu-shift-ocr-front-cpu`
- `gemma-heavy-offload`

runtime staging 스크립트:

```bash
.venv/bin/python scripts/apply_benchmark_preset.py --preset live-ops-baseline --runtime-dir /tmp/ct_runtime
```

## 5. attach-running vs managed

### attach-running

현재 이미 떠 있는 Docker runtime에 붙어서 계측만 수행합니다.

```bash
.venv/bin/python scripts/benchmark_pipeline.py --preset live-ops-baseline --mode batch --repeat 1 --runtime-mode attach-running
```

### managed

preset에서 staged runtime 파일을 만들고 Docker 서비스를 recreate한 뒤 벤치를 수행합니다.

```bash
.venv/bin/python scripts/benchmark_pipeline.py --preset gpu-shift-ocr-front-cpu --mode batch --repeat 1 --runtime-mode managed
```

Windows에서는 아래 배치 파일로 더 간단하게 실행할 수 있습니다.

```bat
scripts\benchmark_suite.bat
scripts\benchmark_suite_cuda13.bat
scripts\benchmark_pipeline.bat
scripts\benchmark_pipeline_cuda13.bat
scripts\benchmark_pipeline.bat run gpu-shift-ocr-front-cpu batch managed 1
scripts\benchmark_pipeline.bat summary
```

- `benchmark_suite.bat`
  사용자용 원클릭 풀스위트, `.venv-win`
- `benchmark_suite_cuda13.bat`
  사용자용 원클릭 풀스위트, `.venv-win-cuda13`
- `benchmark_pipeline.bat`
  고급/수동 실행용, `.venv-win`
- `benchmark_pipeline_cuda13.bat`
  고급/수동 실행용, `.venv-win-cuda13`
- 원클릭 스위트는 `managed` 측정 뒤 원래 Docker 런타임 상태로 자동 복원

## 6. 권장 실행 순서

1. `repo-default`
2. `live-ops-baseline`
3. `gpu-shift-ocr-front-cpu`
4. `gemma-heavy-offload`
5. 필요 시 OCR vLLM backend 조정 preset 추가

## 7. 집계

여러 run 디렉터리를 모아서 표로 보고 싶으면 아래를 사용합니다.

```bash
.venv/bin/python scripts/summarize_benchmarks.py --input "%USERPROFILE%\\Documents\\Comic Translate"
```

## 8. 합격 기준

- 오류 `0회`
- `free VRAM floor >= 1.5 GiB`
- 대표 코퍼스 기준 median total page time 개선
- one-page auto latency 악화 없음
- OCR retry 증가 없음
- truncated / empty translation 없음

## 9. 실사용 문서

실제로 어떻게 측정하고 결과를 다음 preset 선택에 활용할지는 아래 문서를 기준으로 봅니다.

- [pipeline-benchmark-usage-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/pipeline-benchmark-usage-ko.md)
