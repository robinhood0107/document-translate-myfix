# 자동번역 벤치 실행 워크플로우

이 문서는 현재 저장소에서 benchmark를 실제로 어떻게 돌리는지 정리한 실행 중심 문서입니다.

## 결과 저장 위치

모든 benchmark raw 결과는 repo 내부 아래 경로에 쌓습니다.

```text
./banchmark_result_log
```

이 경로는 benchmark 스크립트의 기본 출력 루트이며, 현재 문서와 자동 보고서도 이 위치를 전제로 작성됩니다.

## 실행 환경

Windows에서는 시스템 Python을 쓰지 않고 아래 환경만 사용합니다.

- `scripts\benchmark_pipeline.bat`, `scripts\benchmark_suite.bat` -> `.venv-win`
- `scripts\benchmark_pipeline_cuda13.bat`, `scripts\benchmark_suite_cuda13.bat` -> `.venv-win-cuda13`

기본 코퍼스는 repo 루트의 `/Sample` 폴더이며, representative corpus는 `30장`, audit subset은 앞 `5장`입니다.

## 가장 쉬운 실행

CUDA13 기준 원클릭 실행:

```bat
scripts\benchmark_suite_cuda13.bat
```

이 런처는 아래를 순서대로 수행합니다.

1. `translation-baseline` one-page `attach-running`
2. `translation-baseline` batch `attach-running`
3. `translation-ngl23` batch `managed`
4. `docs/banchmark_report/report-ko.md` 자동 생성

## 수동 실행

대표 batch:

```bat
scripts\benchmark_pipeline_cuda13.bat run translation-baseline batch attach-running 1
```

one-page:

```bat
scripts\benchmark_pipeline_cuda13.bat run translation-baseline one-page attach-running 1
```

managed candidate:

```bat
scripts\benchmark_pipeline_cuda13.bat run translation-ngl23 batch managed 1
scripts\benchmark_pipeline_cuda13.bat run translation-t06 batch managed 1
```

요약 + 문서 재생성:

```bat
scripts\benchmark_pipeline_cuda13.bat summary
```

직접 보고서만 다시 생성:

```bat
.venv-win-cuda13\Scripts\python.exe scripts\generate_benchmark_report.py
```

## 생성되는 파일

각 run 디렉터리에는 보통 아래가 생깁니다.

- `benchmark_request.json`
- `preset_resolved.json`
- `runtime_snapshot.json`
- `docker_snapshot.json`
- `metrics.jsonl`
- `summary.json`
- `summary.md`
- 필요 시 `translation_audit.json`, `translation_audit.md`

자동 문서 생성 결과는 아래에 기록됩니다.

- [report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/banchmark_report/report-ko.md)
- [latest](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/latest)

## 관련 문서

- [usage-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/usage-ko.md)
- [checklist-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/checklist-ko.md)
- [architecture-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/architecture-ko.md)
- [optimization-journey-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/optimization-journey-ko.md)
