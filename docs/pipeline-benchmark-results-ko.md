# 자동번역 벤치 결과 이력

이 문서는 translation-only benchmark 결과를 커밋 기준으로 누적 기록하는 곳입니다.

## 기록 규칙

- 이전 preset 체계 결과는 active 기준선으로 취급하지 않습니다.
- 새 baseline을 승격할 때만 이 문서를 갱신합니다.
- 항상 commit SHA와 preset 이름을 함께 남깁니다.
- representative corpus 기준 결과를 우선 기록합니다.

## 현재 active preset 집합

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

## 2026-04-05 translation-only 재정리

현재 브랜치:

- `codex/feature/pipeline-gpu-benchmarking`

현재 active baseline:

- preset: `translation-baseline`
- OCR front: `cpu`
- Gemma sampler: `0.6 / 64 / 0.95 / 0.0`
- Gemma compose: `n_gpu_layers=23`, `threads=12`, `ctx=4096`

## 기준선 재확정 결과

`/Sample` 코퍼스에서 생성 산출물 하위 폴더를 자동 제외하고, 입력 이미지를 각 run 디렉터리의 `corpus/`로 staging하는 구조로 다시 수집한 결과입니다.

### 기준선 one-page

- run:
  - `C:\Users\pjjpj\Documents\Comic Translate\20260405_224354_translation-baseline_one-page_r1`

| 지표 | 값 |
| --- | --- |
| elapsed_sec | `35.106` |
| translate_median_sec | `13.011` |
| ocr_median_sec | `9.363` |
| page_failed_count | `0` |
| gemma_json_retry_count | `0` |
| gemma_truncated_count | `0` |
| ocr_empty_rate | `0.0` |
| ocr_low_quality_rate | `0.0` |

### 기준선 batch

- run:
  - `C:\Users\pjjpj\Documents\Comic Translate\20260405_231837_translation-baseline_batch_r1`

| 지표 | 값 |
| --- | --- |
| elapsed_sec | `1067.117` |
| translate_median_sec | `13.511` |
| ocr_median_sec | `16.358` |
| inpaint_median_sec | `2.081` |
| page_failed_count | `0` |
| gemma_json_retry_count | `1` |
| gemma_truncated_count | `1` |
| gemma_empty_content_count | `0` |
| ocr_empty_rate | `0.0` |
| ocr_low_quality_rate | `0.0397` |

## `n_gpu_layers` sweep

### one-page screen 결과

| preset | elapsed_sec | translate_median_sec | ocr_median_sec | retry | fail |
| --- | --- | --- | --- | --- | --- |
| `translation-ngl20` | `44.145` | `15.856` | `20.171` | `0` | `0` |
| `translation-ngl21` | `45.113` | `14.211` | `22.989` | `0` | `0` |
| `translation-ngl22` | `44.549` | `12.113` | `23.648` | `0` | `0` |
| `translation-ngl23` | `42.305` | `11.781` | `22.132` | `0` | `0` |
| `translation-ngl24` | `43.485` | `14.806` | `20.976` | `0` | `0` |

해석:

- managed one-page 기준으로는 `translation-ngl23`가 가장 빨랐습니다.
- 다만 warm `attach-running` 기준선 one-page(`35.106s`)보다 빠르지는 않았습니다.
- 따라서 batch에서는 `translation-ngl23`를 우선 검증하고, 나머지는 지배 관계가 명확해지면 pruning합니다.

### representative batch 진행 상태

#### `translation-ngl23`

- run:
  - `C:\Users\pjjpj\Documents\Comic Translate\20260405_233628_translation-ngl23_batch_r1`

| 지표 | 값 |
| --- | --- |
| elapsed_sec | `1053.787` |
| translate_median_sec | `12.999` |
| ocr_median_sec | `16.821` |
| inpaint_median_sec | `2.215` |
| page_failed_count | `0` |
| gemma_json_retry_count | `1` |
| gemma_truncated_count | `1` |
| gemma_empty_content_count | `0` |
| ocr_empty_rate | `0.0` |
| ocr_low_quality_rate | `0.0397` |

baseline 대비:

- elapsed `1067.117 -> 1053.787`
- translate median `13.511 -> 12.999`
- retry/truncated/quality 지표는 baseline과 동급

audit subset 5장 비교:

- 결과: `PASS`
- 파일:
  - `C:\Users\pjjpj\Documents\Comic Translate\20260405_233628_translation-ngl23_batch_r1\translation_audit.json`

참고:

- audit heuristic은 짧은 감탄사 반복(`응`, `아`, `으응`)을 과생성으로 오탐하지 않도록 보정했습니다.
- 현재까지는 `translation-ngl23`가 대표 batch 기준 선두 후보입니다.

#### pruning 결과

- `translation-ngl24`
  - one-page에서 `translation-ngl23`보다 느렸고, representative batch 초반 패턴도 동일한 retry 흐름이라 pruning
- `translation-ngl20`
  - one-page에서 `translation-ngl23`보다 명확히 열세라 representative batch 승격 전 pruning

## `temperature` sweep

### one-page screen 결과

temperature sweep는 `best n_gpu_layers=23` 기준으로 다시 측정했습니다.

| preset | elapsed_sec | translate_median_sec | ocr_median_sec | retry | fail |
| --- | --- | --- | --- | --- | --- |
| `translation-t04` | `46.417` | `15.402` | `21.136` | `0` | `0` |
| `translation-t05` | `49.127` | `16.508` | `24.024` | `0` | `0` |
| `translation-t06` | `41.801` | `13.144` | `19.472` | `0` | `0` |
| `translation-t07` | `42.597` | `14.678` | `19.965` | `0` | `0` |

해석:

- `n_gpu_layers=23` 축에서는 `translation-t06`이 가장 빨랐습니다.
- `translation-t07`도 나쁘지 않았지만, `t06`보다 전체 elapsed와 translate median이 모두 밀렸습니다.
- `t04`, `t05`는 one-page 단계에서 바로 탈락입니다.
#### representative batch 결과

##### `translation-t06`

- run:
  - `C:\Users\pjjpj\Documents\Comic Translate\20260406_001330_translation-t06_batch_r1`

| 지표 | 값 |
| --- | --- |
| elapsed_sec | `1048.742` |
| translate_median_sec | `12.150` |
| ocr_median_sec | `16.604` |
| inpaint_median_sec | `2.253` |
| page_failed_count | `0` |
| gemma_json_retry_count | `1` |
| gemma_truncated_count | `0` |
| gemma_empty_content_count | `0` |
| ocr_empty_rate | `0.0` |
| ocr_low_quality_rate | `0.0397` |

baseline 대비:

- elapsed `1067.117 -> 1048.742`
- translate median `13.511 -> 12.150`
- truncated `1 -> 0`
- retry / OCR 품질 지표는 동급 유지

audit subset 5장 비교:

- 결과: `PASS`
- 파일:
  - `C:\Users\pjjpj\Documents\Comic Translate\20260406_001330_translation-t06_batch_r1\translation_audit.json`

정리:

- `translation-t06`은 `translation-ngl23`보다도 빠르고, `gemma_truncated_count`가 `0`으로 내려가 더 안정적입니다.
- 따라서 `temperature=0.6 / n_gpu_layers=23` 조합을 새 active translation baseline으로 승격합니다.

## 현재 잠정 결론

- corrected `translation-baseline`은 representative batch 기준으로 재확정 완료
- one-page `n_gpu_layers` 후보 중 승자는 `translation-ngl23`
- `n_gpu_layers=23` 기준 one-page temperature 후보 중 승자는 `translation-t06`
- representative batch 최종 승자는 `translation-t06`
- 따라서 새 active translation baseline은 아래와 같습니다.
  - `temperature=0.6`
  - `top_k=64`
  - `top_p=0.95`
  - `min_p=0.0`
  - `n_gpu_layers=23`
  - `threads=12`
  - `ctx=4096`
  - `paddleocr-server=cpu`
