# cold/cache 최종화 벤치 사용법

## 지원 환경

- CUDA12: `.venv-win`
- CUDA13: `.venv-win-cuda13`

아래 예시는 CUDA13입니다. 결과 경로는 반드시 존재하지 않는 새
디렉터리이며 Git 저장소 밖이어야 합니다.

## protocol 확인

```bat
scripts\cold_cache_finalization_suite_cuda13.bat describe ^
  --output-dir C:\comic-translate-validation\cold-cache\protocol-v2
```

## pipeline family

```bat
scripts\cold_cache_finalization_suite_cuda13.bat run-pipeline ^
  --family paddle-workers ^
  --full-reference-suite C:\comic-translate-validation\cold-cache\http\suite_state.json ^
  --input-dir C:\ExampleWorkspace\example-input ^
  --sample-count 6 ^
  --source-lang Japanese ^
  --output-dir C:\comic-translate-validation\cold-cache\paddle-workers
```

generated runtime axis는 `--axis`를 추가합니다.

```bat
scripts\cold_cache_finalization_suite_cuda13.bat run-pipeline ^
  --family paddle-runtime ^
  --axis max_num_seqs ^
  --input-dir C:\samples ^
  --output-dir C:\comic-translate-validation\cold-cache\max-num-seqs
```

## 번역 family

```bat
scripts\cold_cache_finalization_suite_cuda13.bat run-translation ^
  --family gemma-model ^
  --source-summary C:\comic-translate-validation\source-summary.json ^
  --output-dir C:\comic-translate-validation\cold-cache\gemma-model
```

prefill 병목용 제한 matrix는 batch 축부터 실행합니다.

```bat
scripts\cold_cache_finalization_suite_cuda13.bat run-translation ^
  --family gemma-batch ^
  --axis batch_size ^
  --source-summary C:\comic-translate-validation\source-summary.json ^
  --output-dir C:\comic-translate-validation\cold-cache\gemma-batch
```

batch 후보가 게이트를 통과했을 때만 우승값을 고정해 ubatch 축을
실행합니다.

```bat
scripts\cold_cache_finalization_suite_cuda13.bat run-translation ^
  --family gemma-batch ^
  --axis ubatch_size ^
  --base-batch-size 1024 ^
  --source-summary C:\comic-translate-validation\source-summary.json ^
  --output-dir C:\comic-translate-validation\cold-cache\gemma-ubatch
```

## persistent cache

```bat
scripts\cold_cache_finalization_suite_cuda13.bat run-cache ^
  --scenario global-ocr ^
  --input-dir C:\samples ^
  --sample-count 6 ^
  --source-lang Japanese ^
  --output-dir C:\comic-translate-validation\cold-cache\global-ocr

scripts\cold_cache_finalization_suite_cuda13.bat run-cache ^
  --scenario project ^
  --input-dir C:\samples ^
  --sample-count 6 ^
  --source-lang Japanese ^
  --output-dir C:\comic-translate-validation\cold-cache\project
```

project checkpoint는 임의의 miss-overhead 또는 all-hit 개선율 문턱을
사용하지 않는다. `캐시 첫 실행 + all-hit`와
`캐시 첫 실행 + 한 페이지 수정`을 각각 `캐시 없이 같은 작업 두 번`과
비교하며, 두 누적시간이 모두 실제로 짧아야 통과한다. 부분 수정은
최소 2페이지에서만 검증한다.

## 산출물

- `protocol_state.json`: commit과 protocol/baseline SHA
- `suite_state.json`: 중단 복구용 실행 상태와 private run 참조
- `summary.json`: 자동 게이트 결과
- `report.md`: 사람이 읽는 요약
- `private/`: 이미지, OCR/번역 snapshot, DB, project sidecar, 로그
- `private/pipeline-review.json`: HTTP 계열 page/block 의미 비교표
- `private/translation-review.json`: 54블록 후보·라운드 비교표

도구는 기존 output 디렉터리를 재사용하지 않습니다. 실패한 실행도
`suite_state.json`과 private 로그를 보존하며 자동 반복하지 않습니다.
`run-pipeline`, `run-translation`, `run-cache`는 재현 가능한 증거를 위해
clean Git checkout에서만 실행됩니다. 미커밋 개발 스모크는 하위
`benchmark_pipeline.py`를 직접 사용하고 공식 suite 결과로 승격하지
않습니다.
