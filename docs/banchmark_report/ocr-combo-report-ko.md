# OCR Combo Benchmark Report

이 보고서는 `ocr-combo-runtime` 정식 실행에서 확보한 evidence를 기준으로 작성한 수동 정리본입니다.

## Metadata

- run_dir: `./banchmark_result_log/ocr_combo/20260407_161019_ocr-combo-runtime_suite`
- status: `early stop after convergence`
- execution_scope: `full-pipeline`
- official_score_scope: `full-pipeline elapsed_sec + geometry/quality/translation hard gate`
- entrypoint: `scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime`

## Fixed Gemma

- image: `local/llama.cpp:server-cuda-b8665`
- response_format_mode: `json_schema`
- chunk_size: `6`
- temperature: `0.6`
- n_gpu_layers: `23`

## Round Conclusion

- China corpus 권장 OCR: `PaddleOCR VL + Gemma`
- japan corpus 권장 OCR: `PaddleOCR VL + Gemma`
- mixed corpus 운영 권장 라우팅: `중국어/일본어 혼합 운영도 우선 PaddleOCR VL + Gemma 단일 OCR로 시작`
- develop 승격 권장: `아니오`

이번 라운드의 핵심은 OCR 엔진 선택 자체는 수렴했지만, 현재 hard gate가 reference 재실행까지 안정적으로 통과시키지 못했다는 점입니다. 그래서 엔진 추천은 가능하지만, 제품 기본값 승격까지는 보류합니다.

## Why No Promotion

- China default compare에서 `PPOCRv5 + Gemma`, `HunyuanOCR + Gemma` 모두 hard gate 실패
- japan default compare에서 `MangaOCR + Gemma`, `HunyuanOCR + Gemma` 모두 hard gate 실패
- China final confirm에서 reference 계열 재실행 3회 모두 `translation_similarity_avg < 0.98`로 실패
- japan은 외부 비교군 2개가 모두 탈락한 시점에서 승자 엔진은 이미 수렴했고, 남아 있던 `PaddleOCR VL` reference-only tuning ladder는 엔진 선택을 바꾸지 못하므로 중단

즉 이번 결과는 `OCR family winner`는 보여주지만, `current gate로 promotion 가능한 winner`는 만들지 못했습니다.

## Default Comparison

| corpus | engine | elapsed_sec | translate_median_sec | ocr_total_sec | gpu_peak_used_mb | gpu_floor_free_mb | quality_gate_pass | hard gate summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| china | PaddleOCR VL + Gemma (reference) | 216.196 | 13.066 | 89.484 | 11969 | 31 | `true` | self-compare pass |
| china | PPOCRv5 + Gemma | 88.932 | 6.264 | 13.080 | 11955 | 45 | `false` | `non_empty_retention 0.9583`, `empty_rate 0.0417`, `translation_similarity 0.6147` |
| china | HunyuanOCR + Gemma | 101.550 | 8.097 | 10.480 | 11946 | 54 | `false` | `translation_similarity 0.8597` |
| japan | PaddleOCR VL + Gemma (reference) | 785.343 | 10.305 | 453.191 | 11986 | 14 | `true` | self-compare pass |
| japan | MangaOCR + Gemma | 268.781 | 7.488 | 8.363 | 11965 | 35 | `false` | `page_failed 1`, `gemma_truncated 3`, `translation_similarity 0.6941` |
| japan | HunyuanOCR + Gemma | 293.750 | 8.084 | 36.569 | 11983 | 17 | `false` | `single_char_like_rate 0.1672`, `translation_similarity 0.6879` |

## China Final Confirm

| run | elapsed_sec | translate_median_sec | ocr_total_sec | quality_gate_pass | hard gate summary |
| --- | --- | --- | --- | --- | --- |
| run-1 | 193.320 | 10.688 | 85.963 | `false` | `translation_similarity 0.8710` |
| run-2 | 199.575 | 10.614 | 90.363 | `false` | `translation_similarity 0.8982` |
| run-3 | 199.709 | 10.875 | 90.793 | `false` | `translation_similarity 0.8806` |

- China official median elapsed: `199.575s`
- 해석: reference 재실행조차 현재 translation similarity hard gate를 안정적으로 통과하지 못하므로 `promotion_recommended=false`

## Reference Tuning Observations

| corpus | candidate | elapsed_sec | translate_median_sec | ocr_total_sec | quality_gate_pass | hard gate summary |
| --- | --- | --- | --- | --- | --- | --- |
| china | `vram080` | 194.484 | 10.536 | 88.063 | `false` | `translation_similarity 0.8908` |
| china | `vram084` | 195.378 | 10.899 | 85.718 | `false` | `translation_similarity 0.8710` |
| japan | `pw4` | 771.377 | 10.093 | 446.962 | `false` | `translation_similarity 0.8905` |
| japan | `pw8` | 873.440 | 15.715 | 459.175 | `false` | `gemma_truncated 1`, `translation_similarity 0.8974` |

속도만 보면 China에서 `PaddleOCR VL` 튜닝으로 `216.196s -> 194.484s`까지 줄일 수 있었지만, 현재 quality gate를 통과하는 candidate는 아니었습니다. japan도 `pw4`가 baseline보다 소폭 빨랐지만 gate를 못 넘었습니다.

## VRAM And Runtime Notes

- 모든 조합에서 `gpu_peak_used_mb`는 대체로 `11.9 GiB` 수준으로 비슷했습니다.
- `gpu_floor_free_mb`는 `14~54 MB` 범위까지 내려가서, 이번 비교는 전반적으로 VRAM headroom이 매우 작은 상태에서 이뤄졌습니다.
- China에서는 `PPOCRv5`, `HunyuanOCR`가 매우 빠르지만 translation fidelity가 reference 대비 크게 흔들렸습니다.
- japan에서는 `MangaOCR`, `HunyuanOCR` 모두 raw speed는 훨씬 빠르지만 reference 대비 translation similarity가 크게 떨어졌고, `MangaOCR`는 Gemma truncation/page failure까지 동반했습니다.

## Recommendation

- 현재 benchmark evidence 기준 운영 OCR 추천은 `China=PaddleOCR VL + Gemma`, `japan=PaddleOCR VL + Gemma`입니다.
- 다만 이번 결과는 `promotion-ready winner`가 아니라 `safest engine family recommendation`입니다.
- 다음 라운드에서는 아래 중 하나가 필요합니다.
  - translation similarity hard gate를 reference 재실행 안정성에 맞게 완화
  - stable page / stable block 기반 gate로 전환
  - japan corpus에서 Gemma `chunk_size`를 더 보수적으로 낮춘 별도 fairness run 수행
