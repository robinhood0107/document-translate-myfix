# 자동번역 벤치마킹 가이드

기준 날짜: `2026-04-05`

이 문서는 `Gemma + PaddleOCR VL` 조합을 translation-only 기준으로 반복 측정하는 방법과 기록 규칙을 정리합니다.

## 1. 코퍼스 규칙

기본 코퍼스는 저장소 루트의 로컬 `/Sample` 폴더입니다.

- representative corpus: `30장`
- audit subset: representative의 앞 `5장`

이 폴더는 로컬 검증 전용입니다.

- Git에 올리지 않음
- `.gitignore`에 `/Sample/` 추가
- 벤치 스크립트는 생성 산출물 하위 폴더를 코퍼스에서 자동 제외함

## 1-1. 실행 환경 전제

오프스크린 benchmark 스크립트는 실제 앱 파이프라인을 import해서 돌립니다.

필수 런타임:

- `PySide6`
- `cv2`
- 현재 앱이 사용하는 OCR / ONNX / 기타 런타임 의존성

Windows에서는 시스템 Python이 아니라 아래 전용 환경만 사용합니다.

- `.venv-win`
- `.venv-win-cuda13`

런처 매핑:

- `benchmark_pipeline.bat`, `benchmark_suite.bat` -> `.venv-win`
- `benchmark_pipeline_cuda13.bat`, `benchmark_suite_cuda13.bat` -> `.venv-win-cuda13`

## 1-2. 벤치 전용 폰트 폴더

렌더링 벤치에서는 target language 기준으로 `benchmarks-fonts/<언어>/`를 먼저 찾습니다.

- 한국어: `benchmarks-fonts/Korean/`
- 일본어: `benchmarks-fonts/Japanese/`
- 영어: `benchmarks-fonts/English/`

규칙:

- 지원 확장자: `.ttf`, `.ttc`, `.otf`, `.woff`, `.woff2`
- 파일이 여러 개면 사전순 첫 번째 파일 사용
- 폴더가 비어 있으면 현재 앱 폰트 설정 유지

fallback:

- `Simplified Chinese` -> `Chinese`
- `Traditional Chinese` -> `Chinese`
- `Brazilian Portuguese` -> `Portuguese`

## 2. 산출물 위치

벤치 결과는 저장소가 아니라 Windows 사용자 문서 폴더 아래의 `Comic Translate` 폴더에 저장합니다.

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
- translated text export가 켜진 경우 `translated_texts/`

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

단계 태그:

- `page_start`
- `detect_start`, `detect_end`
- `ocr_start`, `ocr_end`
- `inpaint_start`, `inpaint_end`
- `translate_start`, `translate_end`
- `render_start`, `render_end`
- `page_done`
- `page_failed`

summary에서 핵심으로 보는 품질/속도 지표:

- `page_failed_count`
- `gemma_json_retry_count`
- `gemma_chunk_retry_events`
- `gemma_truncated_count`
- `gemma_empty_content_count`
- `ocr_empty_rate`
- `ocr_low_quality_rate`
- `ocr_median_sec`
- `translate_median_sec`
- `inpaint_median_sec`
- `elapsed_sec`

## 4. preset 적용 방식

preset은 [benchmarks/presets](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/benchmarks/presets)에 JSON으로 관리합니다.

현재 active preset:

- `translation-baseline`
- `translation-ngl20`
- `translation-ngl21`
- `translation-ngl22`
- `translation-ngl23`
- `translation-ngl24`
- `translation-ngl24-ctx3072`
- `translation-t04`
- `translation-t05`
- `translation-t06`
- `translation-t07`

runtime staging 예시:

```bash
.venv/bin/python scripts/apply_benchmark_preset.py --preset translation-baseline --runtime-dir /tmp/ct_runtime
```

## 5. attach-running vs managed

### attach-running

현재 이미 떠 있는 Docker runtime에 붙어서 계측만 수행합니다.

```bash
.venv/bin/python scripts/benchmark_pipeline.py --preset translation-baseline --mode batch --repeat 1 --runtime-mode attach-running
```

### managed

preset에서 staged runtime 파일을 만들고 Docker 서비스를 recreate한 뒤 벤치를 수행합니다.

```bash
.venv/bin/python scripts/benchmark_pipeline.py --preset translation-ngl23 --mode batch --repeat 1 --runtime-mode managed
```

Windows 런처:

```bat
scripts\benchmark_suite.bat
scripts\benchmark_suite_cuda13.bat
scripts\benchmark_pipeline.bat
scripts\benchmark_pipeline_cuda13.bat
scripts\benchmark_pipeline_cuda13.bat run translation-baseline batch attach-running 1
scripts\benchmark_pipeline_cuda13.bat run translation-ngl23 batch managed 1
scripts\benchmark_pipeline_cuda13.bat summary
```

- `benchmark_suite*.bat`
  원클릭 풀스위트
- `benchmark_pipeline*.bat`
  고급/수동 실행용
- 원클릭 스위트는 `managed` 측정 뒤 원래 Docker 런타임 상태로 자동 복원

## 6. 권장 실행 순서

### 1차 기준선 재확정

1. `translation-baseline` one-page `attach-running`
2. `translation-baseline` batch `attach-running`

### 2차 `n_gpu_layers` 스크리닝

1. `translation-ngl20`
2. `translation-ngl21`
3. `translation-ngl22`
4. `translation-ngl23`
5. `translation-ngl24`

위 5개는 먼저 one-page `managed`로 스크리닝하고, 통과 상위 3개만 batch `managed`로 올립니다.

### 3차 temperature 스크리닝

best `n_gpu_layers`를 고정한 뒤:

1. `translation-t04`
2. `translation-t05`
3. `translation-t06`
4. `translation-t07`

역시 one-page `managed`를 먼저 돌리고, 상위 2개만 batch `managed`로 올립니다.

### 4차 rescue

`n_gpu_layers=24`가 품질 또는 안정성 기준을 못 넘으면 `translation-ngl24-ctx3072`를 한 번만 추가로 측정합니다.

## 7. 집계

여러 run 디렉터리를 모아서 표로 보고 싶으면 아래를 사용합니다.

```bash
.venv/bin/python scripts/summarize_benchmarks.py --input "%USERPROFILE%\\Documents\\Comic Translate"
```

또는 Windows:

```bat
scripts\benchmark_pipeline_cuda13.bat summary
```

## 8. 합격 기준

hard reject:

- `page_failed_count > 0`
- `gemma_truncated_count > 0`
- `gemma_empty_content_count > 0`
- `ocr_empty_rate` 증가
- `ocr_low_quality_rate` 증가
- `gemma_json_retry_count` 증가

survivor ranking:

1. `batch elapsed_sec` 최소
2. `translate_median_sec` 최소
3. `gemma_json_retry_count` 최소
4. `one-page elapsed_sec` 최소

## 9. 현재 translation-baseline

현재 브랜치의 translation-only 기준선은 아래와 같습니다.

- `paddleocr-server` front service = `cpu`
- `paddleocr-vllm` = `gpu`
- Gemma sampler = `temperature=0.6`, `top_k=64`, `top_p=0.95`, `min_p=0.0`
- Gemma = `n_gpu_layers=23`, `threads=12`, `ctx=4096`
- OCR client = `parallel_workers=8`, `max_new_tokens=1024`

## 10. 실사용 문서

- [pipeline-benchmark-usage-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/pipeline-benchmark-usage-ko.md)
- [pipeline-benchmark-results-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/pipeline-benchmark-results-ko.md)
