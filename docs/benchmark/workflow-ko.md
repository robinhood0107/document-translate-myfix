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

`b8665 + Gemma 4 parser` 실험 라운드는 아래처럼 별도 profile로 실행합니다.

```bat
scripts\benchmark_suite_cuda13.bat --suite-profile b8665-gemma4
```

이 profile은 `attach-running`이 아니라 `managed-only`로 동작합니다. 다만 현재 managed 정책은 무조건 recreate가 아니라 `health-first reuse`입니다. 즉, 이미 떠 있는 Docker 서비스가 healthy면 그대로 재사용하고, health-check가 실패한 서비스만 재기동합니다.

`b8665-gemma4` profile의 큰 순서는 아래와 같습니다.

1. `Gemma 4 verification` managed startup + raw API smoke
2. `translation-old-image-baseline` one-page / batch control
3. `b8665-object-control` one-page / batch control
4. `b8665-schema-control` one-page / batch control
5. format winner 기준 `chunk_size` sweep
6. format + chunk winner 기준 `temperature` coarse/fine sweep
7. format + chunk + temperature winner 기준 `n_gpu_layers` sweep
8. 필요 시 `ctx=3072` rescue, low-think fallback
9. `docs/banchmark_report/report-ko.md` 자동 재생성

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

`managed`는 이제 healthy한 Docker 서비스를 우선 재사용합니다. 살아 있는 서비스가 정상일 때는 재기동하지 않고, 실패한 서비스만 복구합니다.

요약 + 문서 재생성:

```bat
scripts\benchmark_pipeline_cuda13.bat summary
```

직접 보고서만 다시 생성:

```bat
.venv-win-cuda13\Scripts\python.exe scripts\generate_benchmark_report.py
.venv-win-cuda13\Scripts\python.exe scripts\generate_benchmark_report.py --manifest .\banchmark_result_log\<b8665-suite-run>\report_manifest_b8665.yaml
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

- [report-ko.md](../banchmark_report/report-ko.md)
- `docs/assets/benchmarking/latest`
- `docs/banchmark_report/history/<snapshot-id>/report-ko.md`
- `docs/assets/benchmarking/history/<snapshot-id>/`

`b8665` managed run에는 추가로 아래 파일이 같이 남습니다.

- `docker_logs/gemma-local-server.log`
- `docker_logs/paddleocr-server.log`
- `docker_logs/paddleocr-vllm.log`
- `_server_verification/verification.json`
- `_server_verification/request_object.json`
- `_server_verification/response_object.json`
- `_server_verification/request_schema.json`
- `_server_verification/response_schema.json`
- `_server_verification/gemma_log_tail.txt`

이 로그는 `b8665` build/version 확인, Gemma 4 template/parser 관련 startup 메시지 확인, structured output 이상 징후 확인에 사용합니다.

## 관련 문서

- [usage-ko.md](./usage-ko.md)
- [checklist-ko.md](./checklist-ko.md)
- [architecture-ko.md](./architecture-ko.md)
- [optimization-journey-ko.md](./optimization-journey-ko.md)
