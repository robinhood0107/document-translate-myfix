# Benchmark 브랜치 / 머지 정책

이 문서는 benchmark 자산을 제품 브랜치와 분리해 유지하는 기준을 정리합니다.

## 목적

- `main`에는 배포에 필요한 제품 코드만 남깁니다.
- `develop`에는 통합 대상 제품 코드만 남깁니다.
- benchmark 실험, preset, 보고서, 차트, 결과 로그는 `benchmarking/lab` 브랜치에서만 유지합니다.

핵심 원칙은 다음과 같습니다.

- core는 번역/렌더/OCR 같은 비즈니스 동작만 책임집니다.
- benchmark 레이어는 측정, 비교, 판단, 보고서 생성을 책임집니다.
- core와 benchmark 사이의 연결점은 얇은 계측 훅과 통계 surface만 허용합니다.

## 브랜치 역할

### `main`

- 배포 브랜치
- benchmark 파일 금지
- benchmark 결과 문서 금지

### `develop`

- 제품 통합 브랜치
- benchmark 파일 금지
- benchmark 결과 문서 금지
- 아래 공용 계측만 허용
  - stage event emission
  - retry / truncated / empty / quality 통계 surface
  - generic memlog / gpu snapshot helper

### `benchmarking/lab`

- benchmark 전용 장기 브랜치
- 실험 preset, runner, suite, compare 스크립트, generated report, 차트, 실험 문서를 유지
- `develop`의 최신 제품 코드를 주기적으로 받아서 실험

## `develop`로 가져와도 되는 파일

benchmark 실험 결과를 바탕으로 아래 종류의 변경만 `develop` 후보로 가져옵니다.

- 제품 기능을 실제로 바꾸는 runtime 설정 변경
  - 예: Gemma sampler 기본값, `n_gpu_layers`, OCR front device 기본값
  - 예: Docker image/pull policy, `response_format_mode`, `response_schema_mode`, `chunk_size`, prompt profile
- benchmark를 위해서가 아니라 제품 관측 안정성을 위해 필요한 얇은 계측 훅
  - 예: `emit_memlog(tag, **extra)` 같은 generic hook
  - 예: translator engine의 retry/truncated 통계 surface
- 제품 사용 문서에서 실제 현재 동작을 설명하는 문구
  - 예: Gemma local server 설정값

위와 같은 변경은 benchmark 결과를 근거로 했더라도 benchmark 자산이 아니라 제품 동작 변경으로 취급합니다. 따라서 `benchmarks/`, generated report, 차트, preset을 포함하지 않는 한 `develop` 대상 PR로 수용할 수 있습니다.

## `develop`로 가져오면 안 되는 파일

아래는 benchmark 전용 자산이므로 `develop`와 `main`에 넣지 않습니다.

- `benchmarks/`
- `benchmarks-fonts/`
- `scripts/benchmark_*`
- `scripts/generate_benchmark_report.py`
- `scripts/summarize_benchmarks.py`
- `scripts/compare_translation_exports.py`
- `scripts/apply_benchmark_preset.py`
- `docs/benchmark/`
- `docs/banchmark_report/`
- `docs/assets/benchmarking/`
- benchmark 결과 수치, 승자 preset 설명, 차트 링크가 들어간 README 변경

## benchmark 결과를 `develop`로 반영하는 절차

1. `benchmarking/lab`에서 실험을 완료합니다.
2. 결과를 읽고 제품에 실제로 필요한 변경만 추립니다.
3. 새 `codex/*` 작업 브랜치를 `develop`에서 분기합니다.
4. benchmark 전용 파일은 제외하고 아래만 수동으로 옮깁니다.
   - 제품 runtime/config 변경
   - 공용 계측 훅
   - 제품 사용자 문서
5. `develop` 대상 검증을 수행합니다.
6. benchmark 결과 자체는 merge하지 않고, 제품에 필요한 결정만 merge합니다.

즉, `benchmarking/lab`에서 `develop`로 전체 브랜치를 merge하는 것이 아니라, 검증된 제품 변경만 별도 브랜치에서 다시 정리해 반영하는 것이 원칙입니다.

## 유지보수 관점 기준

이 구조는 순수 DDD 강제라기보다, 이 저장소에 맞는 실용적 경계 설정입니다.

- 단일 책임 분리
  - core: 제품 동작
  - benchmark: 실험/비교/보고서
- 결합도 감소
  - core가 benchmark runner를 import하지 않음
- 리팩토링 안전성
  - 이벤트 스키마와 통계 surface만 안정적으로 유지하면 benchmark 레이어는 덜 깨짐
- 운영 단순화
  - `main`과 `develop`는 항상 배포/통합 관점으로 깨끗하게 유지 가능

## 문서 운영 기준

- 제품 브랜치의 README는 제품 사용 문서만 유지합니다.
- benchmark 설명과 결과 해석은 `benchmarking/lab` 쪽 문서에서만 유지합니다.
- 같은 목적의 benchmark 문서는 가능한 한 통합합니다.
  - 사용법/운영 절차는 하나의 문서
  - 아키텍처/경계 원칙은 하나의 문서
  - 생성 결과는 generated report 하나로 유지
