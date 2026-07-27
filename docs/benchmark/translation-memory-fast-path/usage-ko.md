# Translation Memory Fast Path 벤치마크 사용법

## 사전 조건

- prepared Gemma external volume과 pinned llama.cpp 이미지가 준비되어 있어야 합니다.
- 입력 JSON은 일본어·중국어·영어 각각 18블록의 기존 다국어 제품 경로 요약이어야 합니다.
- 출력 디렉터리는 Git 밖이거나 `git check-ignore`로 확인되는 validation 디렉터리여야 합니다.
- 컨테이너를 삭제하거나 `docker compose down`하지 않습니다.

## Windows 실행

기본 지원 환경:

```bat
scripts\translation_memory_fast_path_benchmark_suite.bat ^
  --source-summary <multilingual-summary.json> ^
  --output-dir <validation-log-root>\translation-memory-fast-path\run-01
```

CUDA13 지원 환경:

```bat
scripts\translation_memory_fast_path_benchmark_suite_cuda13.bat ^
  --source-summary <multilingual-summary.json> ^
  --output-dir <validation-log-root>\translation-memory-fast-path\run-01
```

Gemma 추론은 두 경우 모두 pinned Docker runtime에서 수행됩니다. 두 launcher의 차이는 앱 측 Python 환경 검증입니다.

## 주요 옵션

- `--model`: prepared volume에 등록된 모델 파일명
- `--group-size`: 제품 grouped 크기, 기본 7
- `--max-completion-tokens`: 이 벤치에서는 512로 고정
- `--prefix-repeat-count`: prefix 후보 반복 수, 기본 3
- `--skip-prefix-matrix`: cache/TM 계약만 재검증할 때 사용

## 결과 파일

- `summary.json`: 원문·번역·로컬 경로를 제외한 공개 가능한 계측
- `private-comparisons.json`: 의미 검수용 원문과 후보 번역
- `*.sqlite3`: 각 cache/TM 상태의 로컬 DB
- `runner.stdout.txt`, `runner.stderr.txt`: 호출자가 선택적으로 남기는 실행 로그

`private-comparisons.json`, SQLite DB, 실행 로그는 raw validation 자료이므로 Git에 추가하지 않습니다.

## 실패와 복구

러너는 실패해도 `finally`에서 Gemma를 `stop`하고 실행 전 컨테이너 모델의 prepared contract로 재생성합니다. 모델 파일이나 volume을 삭제하지 않습니다. `summary.json`은 각 주요 시나리오 뒤 원자적으로 checkpoint되므로 어느 단계에서 멈췄는지 확인할 수 있습니다.

전체 페이지 pipeline launcher는 제공하지 않습니다. 해당 실행은 번역 품질 사용자 승인 뒤 한 번만 수행하는 별도 게이트이기 때문입니다.
