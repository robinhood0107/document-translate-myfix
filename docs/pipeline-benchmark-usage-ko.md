# 자동번역 벤치 측정 / 활용 방법

기준 날짜: `2026-04-05`

이 문서는 현재 프로젝트에서 벤치를 실제로 어떻게 돌리고, 나온 결과를 다음 튜닝에 어떻게 활용하는지 설명합니다.

## 1. 먼저 준비할 것

### 로컬 코퍼스

- 저장소 루트의 `/Sample` 폴더를 사용합니다.
- 기준은 `30장`입니다.
- smoke 용도는 앞 `5장`, representative 용도는 전체 `30장`입니다.
- `/Sample`은 `.gitignore`에 포함되어 Git에 올라가지 않습니다.

### Python 실행 환경

오프스크린 benchmark는 실제 앱 파이프라인을 import해서 돌립니다.

따라서 아래 의존성이 있는 환경이 필요합니다.

- `PySide6`
- `cv2`
- 앱 실행에 필요한 OCR / ONNX / 기타 런타임 의존성

Windows에서는 시스템 Python이 아니라 아래 전용 환경을 사용합니다.

- `.venv-win`
- `.venv-win-cuda13`

배치 파일 이름으로 어떤 환경을 쓸지 구분합니다.

- `benchmark_pipeline.bat`, `benchmark_suite.bat` -> `.venv-win` 전용
- `benchmark_pipeline_cuda13.bat`, `benchmark_suite_cuda13.bat` -> `.venv-win-cuda13` 전용

### 벤치 전용 언어별 폰트

렌더링 벤치에서는 target language 기준으로 `benchmarks-fonts/<언어>/` 폴더를 먼저 찾습니다.

- 예: 한국어는 `benchmarks-fonts/Korean/`
- 지원 확장자: `.ttf`, `.ttc`, `.otf`, `.woff`, `.woff2`
- 폴더 안에 폰트가 여러 개 있으면 사전순 첫 번째 파일을 사용합니다.
- 폴더가 비어 있으면 현재 앱 폰트 설정을 그대로 씁니다.

추가 fallback:

- `Simplified Chinese` -> `Chinese`
- `Traditional Chinese` -> `Chinese`
- `Brazilian Portuguese` -> `Portuguese`

## 2. 가장 쉬운 실행 방법

Windows에서는 배치 파일 이름으로 환경을 고릅니다.

- 표준 환경: [benchmark_suite.bat](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/benchmark_suite.bat)
- CUDA13 환경: [benchmark_suite_cuda13.bat](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/benchmark_suite_cuda13.bat)

```bat
scripts\benchmark_suite.bat
scripts\benchmark_suite_cuda13.bat
```

이 명령은 아래 3개를 자동으로 순서대로 수행합니다.

1. `translation-baseline` `one-page` `attach-running`
2. `translation-baseline` `batch` `attach-running`
3. `translation-ngl23` `batch` `managed`

실행이 끝나면 자동으로 아래를 수행합니다.

- 현재 스위트만 묶은 `suite_report.md` 생성
- 결과 폴더 열기
- `suite_report.md` 열기
- 콘솔에 PASS / WARN / FAIL 요약과 추천 결과 출력
- `managed` 단계 뒤에는 Gemma / OCR Docker 런타임을 시작 시점 상태로 복원

## 3. 결과는 어디에 쌓이나

기본 저장 위치는 아래입니다.

```text
%USERPROFILE%\Documents\Comic Translate
```

필요하면 `CT_BENCH_OUTPUT_ROOT` 환경변수로 다른 출력 루트를 강제로 지정할 수 있습니다.

예시:

```text
C:\Users\<사용자이름>\Documents\Comic Translate\20260405_223000_suite
```

이 폴더 아래에 각 단계별 하위 폴더와 스위트 리포트가 함께 생깁니다.

- `01_translation_baseline_one_page`
- `02_translation_baseline_batch`
- `03_translation_ngl23_managed`
- `suite_report.md`
- `suite_report.json`
- `suite_console_summary.txt`

## 4. 고급 / 수동 실행

세부 실행을 직접 제어하고 싶을 때는 환경에 맞는 고급 런처를 사용합니다.

- 표준 환경: [benchmark_pipeline.bat](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/benchmark_pipeline.bat)
- CUDA13 환경: [benchmark_pipeline_cuda13.bat](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/scripts/benchmark_pipeline_cuda13.bat)

### 4-1. 현재 떠 있는 서버에 붙어서 batch 측정

```bat
scripts\benchmark_pipeline.bat run translation-baseline batch attach-running 1
scripts\benchmark_pipeline_cuda13.bat run translation-baseline batch attach-running 1
```

### 4-2. one-page auto 성격으로 짧게 측정

```bat
scripts\benchmark_pipeline.bat run translation-baseline one-page attach-running 1
scripts\benchmark_pipeline_cuda13.bat run translation-baseline one-page attach-running 1
```

### 4-3. preset 기준으로 Docker runtime을 다시 띄워서 측정

```bat
scripts\benchmark_pipeline_cuda13.bat run translation-ngl20 one-page managed 1
scripts\benchmark_pipeline_cuda13.bat run translation-ngl21 one-page managed 1
scripts\benchmark_pipeline_cuda13.bat run translation-ngl22 one-page managed 1
scripts\benchmark_pipeline_cuda13.bat run translation-ngl23 one-page managed 1
scripts\benchmark_pipeline_cuda13.bat run translation-ngl24 one-page managed 1
scripts\benchmark_pipeline_cuda13.bat run translation-t06 one-page managed 1
scripts\benchmark_pipeline_cuda13.bat run translation-t06 batch managed 1
```

### 4-4. 누적 결과 요약표 만들기

```bat
scripts\benchmark_pipeline.bat summary
scripts\benchmark_pipeline_cuda13.bat summary
```

### 4-5. translated text export audit 비교

batch 후보를 baseline과 비교할 때는 아래 스크립트를 사용합니다.

```bat
.venv-win-cuda13\Scripts\python.exe scripts\compare_translation_exports.py ^
  --baseline-run-dir "C:\Users\pjjpj\Documents\Comic Translate\<baseline-run>" ^
  --candidate-run-dir "C:\Users\pjjpj\Documents\Comic Translate\<candidate-run>" ^
  --sample-dir "C:\Users\pjjpj\Desktop\openai_manga_translater\comic-translate\Sample" ^
  --sample-count 5 ^
  --output "C:\Users\pjjpj\Documents\Comic Translate\<candidate-run>\translation_audit.json"
```

이 스크립트는 아래를 검사합니다.

- 첫 5장 translated export 존재 여부
- block key mismatch
- 과도한 장문 생성 또는 반복 생성 의심 블록

### 4-6. 결과 폴더 열기

```bat
scripts\benchmark_pipeline.bat open
scripts\benchmark_pipeline_cuda13.bat open
```

## 5. 어떤 파일을 보면 되나

### 가장 먼저 볼 파일

- `suite_report.md`
- `suite_report.json`

여기서 바로 확인할 핵심 값:

- `elapsed_sec`
- `page_done_count`
- `page_failed_count`
- `gpu_peak_used_mb`
- `gpu_floor_free_mb`
- `ocr_median_sec`
- `translate_median_sec`
- `inpaint_median_sec`

### 더 자세히 볼 파일

- 각 단계 폴더의 `metrics.jsonl`
- 각 단계 폴더의 `summary.json`
- 각 단계 폴더의 `command_stdout.txt`, `command_stderr.txt`

이 파일은 단계별 이벤트 로그입니다. 아래 태그가 중요합니다.

- `detect_start`, `detect_end`
- `ocr_start`, `ocr_end`
- `inpaint_start`, `inpaint_end`
- `translate_start`, `translate_end`
- `render_start`, `render_end`
- `page_done`
- `page_failed`

## 6. 결과를 어떻게 해석하나

### 좋은 조합

- `page_failed_count = 0`
- `gemma_truncated_count = 0`
- `gemma_empty_content_count = 0`
- `elapsed_sec` 감소
- `translate_median_sec` 감소
- `gemma_json_retry_count` 증가 없음
- `ocr_empty_rate` 증가 없음
- `ocr_low_quality_rate` 증가 없음

### 버려야 할 조합

- OOM, CUDA 오류, HTTP 오류
- `page_failed_count > 0`
- 잘린 응답 / 빈 응답 발생
- OCR 품질 지표 악화

## 7. 실제 활용 예시

### 예시 A: `translation-ngl24`가 one-page는 나쁘지 않지만 `translation-ngl23`보다 느리고 representative 초반 retry 패턴도 같은 경우

다음 액션:

1. `translation-ngl24`는 탈락
2. `translation-ngl24-ctx3072`를 rescue candidate로 한 번만 재측정
3. 그래도 개선이 없으면 `23` baseline 유지

### 예시 B: `translation-ngl23`가 batch elapsed와 `translate_median_sec`를 모두 줄이는 경우

다음 액션:

1. `n_gpu_layers=23`을 임시 승자로 고정
2. 같은 조건에서 `translation-t04/t05/t06/t07` one-page 스크리닝
3. 상위 2개만 representative batch까지 올림

### 예시 C: `translation-t06`가 retry를 유지하면서 `truncated=0`으로 줄이고 속도도 더 빨라지는 경우

다음 액션:

1. 이를 새 `translation-baseline` 후보로 기록
2. 첫 5장 translated text export를 baseline과 비교
3. 누락, key mismatch, 과도한 장문 생성이 없으면 채택

## 8. 다음에 결과를 문서화하는 위치

실제 승자 조합은 아래 문서에 누적합니다.

- [pipeline-benchmark-results-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/pipeline-benchmark-results-ko.md)
- [gemma-translation-optimization-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/gemma-translation-optimization-ko.md)

전략은 아래 문서를 기준으로 유지합니다.

- [pipeline-resource-strategy-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/pipeline-resource-strategy-ko.md)
- [pipeline-benchmarking-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/pipeline-benchmarking-ko.md)
- [pipeline-benchmark-checklist-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/pipeline-benchmark-checklist-ko.md)
