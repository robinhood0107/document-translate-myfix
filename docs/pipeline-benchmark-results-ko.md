# 자동번역 벤치 결과 이력

이 문서는 실제 벤치 결과를 커밋 기준으로 누적 기록하는 곳입니다.

## 기록 규칙

- 새 baseline을 승격할 때만 이 문서를 갱신합니다.
- 항상 commit SHA와 preset 이름을 함께 남깁니다.
- representative corpus 기준 결과를 우선 기록합니다.

## 템플릿

```text
### <commit-sha> <preset-name>

- representative elapsed_sec:
- page_done_count:
- page_failed_count:
- gpu_peak_used_mb:
- gpu_floor_free_mb:
- 판단:
```

## 현재 상태

- 벤치 인프라와 preset/문서가 먼저 들어간 상태입니다.
- 다음 단계부터 `/Sample` 30장 코퍼스로 실제 수치를 누적합니다.

## 2026-04-05 현재 브랜치 실험

작업 브랜치:

- `codex/feature/pipeline-gpu-benchmarking`

기준 커밋:

- `ecd3817` `feat(benchmark): add language-specific benchmark fonts`
- `6b6b15e` `feat(benchmark): add stable gemma translation presets`

### 기준선 A: `live-ops-baseline`

기록 출처:

- `C:\Users\pjjpj\Documents\Comic Translate\20260405_180352_suite`

#### representative batch

| 항목 | 값 |
| --- | --- |
| elapsed_sec | `789.204` |
| page_done_count | `30` |
| page_failed_count | `0` |
| gpu_peak_used_mb | `11937` |
| gpu_floor_free_mb | `63` |
| ocr_median_sec | `12.368` |
| translate_median_sec | `11.278` |
| inpaint_median_sec | `2.158` |

#### one-page

| 항목 | 값 |
| --- | --- |
| elapsed_sec | `34.548` |
| page_done_count | `1` |
| page_failed_count | `0` |
| gpu_peak_used_mb | `11706` |
| gpu_floor_free_mb | `294` |

### 기준선 B: `gpu-shift-ocr-front-cpu`

이 기준선은 `paddleocr-server=cpu`, `paddleocr-vllm=gpu`를 유지한 상태입니다.

기록 출처:

- 이전 suite: `C:\Users\pjjpj\Documents\Comic Translate\20260405_180352_suite`
- 새 계측 run:
  - `C:\Users\pjjpj\Documents\Comic Translate\20260405_194023_gpu-shift-ocr-front-cpu_one-page_r1`
  - `C:\Users\pjjpj\Documents\Comic Translate\20260405_194322_gpu-shift-ocr-front-cpu_batch_r1`

#### representative batch

| 항목 | 값 |
| --- | --- |
| elapsed_sec | `1016.410` |
| page_done_count | `30` |
| page_failed_count | `0` |
| gpu_peak_used_mb | `11929` |
| gpu_floor_free_mb | `71` |
| ocr_median_sec | `15.644` |
| translate_median_sec | `12.977` |
| inpaint_median_sec | `2.121` |
| gemma_json_retry_count | `1` |
| gemma_chunk_retry_events | `1` |
| gemma_truncated_count | `0` |
| gemma_empty_content_count | `0` |
| ocr_empty_rate | `0.0` |
| ocr_low_quality_rate | `0.0313` |

#### one-page

| 항목 | 값 |
| --- | --- |
| elapsed_sec | `43.051` |
| page_done_count | `1` |
| page_failed_count | `0` |
| gpu_peak_used_mb | `11702` |
| gpu_floor_free_mb | `298` |
| ocr_median_sec | `21.353` |
| translate_median_sec | `12.811` |
| inpaint_median_sec | `2.989` |
| gemma_json_retry_count | `0` |
| gemma_chunk_retry_events | `0` |
| gemma_truncated_count | `0` |
| gemma_empty_content_count | `0` |
| ocr_empty_rate | `0.0` |
| ocr_low_quality_rate | `0.0` |

#### 해석

- 기존 suite에서는 `gpu-shift-ocr-front-cpu`가 `live-ops-baseline`보다 빨랐습니다.
- 새 계측 기준으로도 품질 지표는 양호하지만, representative batch에서는 `gemma_json_retry_count=1`이 확인됐습니다.
- 즉 다음 최적화 초점은 OCR보다 Gemma 번역 안정화입니다.

### 후보 1: `gemma-translation-stable-22`

설정:

- OCR front = `cpu`
- Gemma sampler = `1.0 / 64 / 0.95 / 0.0`
- `n_gpu_layers=22`
- `threads=12`
- `ctx=4096`

기록 출처:

- `C:\Users\pjjpj\Documents\Comic Translate\20260405_200301_gemma-translation-stable-22_one-page_r1`
- representative batch는 조기 중단

#### one-page

| 항목 | 값 |
| --- | --- |
| elapsed_sec | `42.115` |
| page_done_count | `1` |
| page_failed_count | `0` |
| gpu_peak_used_mb | `11901` |
| gpu_floor_free_mb | `99` |
| ocr_median_sec | `19.399` |
| translate_median_sec | `15.124` |
| inpaint_median_sec | `2.374` |
| gemma_json_retry_count | `0` |
| gemma_chunk_retry_events | `0` |
| gemma_truncated_count | `0` |
| gemma_empty_content_count | `0` |
| ocr_empty_rate | `0.0` |
| ocr_low_quality_rate | `0.0` |

#### representative batch 중단 판단

이 조합은 batch 초반에 `gemma_json_retry_count=2`가 확인되어 기준선 B(`1`)를 이미 초과했습니다.

- representative batch 진행 중 `page_done=4`
- `gemma_json_retry_count=2`
- `gemma_chunk_retry_events=2`

즉 `temperature=1.0 / top_k=64 / top_p=0.95 / min_p=0.0`만으로는 batch JSON 안정성이 충분히 확보되지 않았습니다.

### 후보 2: `gemma-translation-stable-22-t07`

설정:

- OCR front = `cpu`
- Gemma sampler = `0.7 / 64 / 0.95 / 0.0`
- `n_gpu_layers=22`
- `threads=12`
- `ctx=4096`

#### one-page

| 항목 | 값 |
| --- | --- |
| elapsed_sec | `41.298` |
| page_done_count | `1` |
| page_failed_count | `0` |
| gpu_peak_used_mb | `11852` |
| gpu_floor_free_mb | `148` |
| ocr_median_sec | `20.658` |
| translate_median_sec | `12.781` |
| inpaint_median_sec | `2.334` |
| gemma_json_retry_count | `0` |
| gemma_chunk_retry_events | `0` |

#### representative batch 중단 판단

이 조합도 batch 초반에 기준선 B보다 많은 retry가 확인되어 조기 중단했습니다.

- representative batch 진행 중 `page_done=7`
- `gemma_json_retry_count=2`
- `gemma_chunk_retry_events=2`

### 후보 3: `gemma-translation-stable-22-t05`

설정:

- OCR front = `cpu`
- Gemma sampler = `0.5 / 64 / 0.95 / 0.0`
- `n_gpu_layers=22`
- `threads=12`
- `ctx=4096`

기록 출처:

- `C:\Users\pjjpj\Documents\Comic Translate\20260405_201853_gemma-translation-stable-22-t05_one-page_r1`
- `C:\Users\pjjpj\Documents\Comic Translate\20260405_202151_gemma-translation-stable-22-t05_batch_r1`

#### one-page

| 항목 | 값 |
| --- | --- |
| elapsed_sec | `41.592` |
| page_done_count | `1` |
| page_failed_count | `0` |
| gpu_peak_used_mb | `11886` |
| gpu_floor_free_mb | `114` |
| ocr_median_sec | `22.142` |
| translate_median_sec | `12.063` |
| inpaint_median_sec | `2.236` |
| gemma_json_retry_count | `0` |
| gemma_chunk_retry_events | `0` |
| gemma_truncated_count | `0` |
| gemma_empty_content_count | `0` |

#### representative batch

| 항목 | 값 |
| --- | --- |
| elapsed_sec | `825.772` |
| page_done_count | `30` |
| page_failed_count | `0` |
| gpu_peak_used_mb | `11987` |
| gpu_floor_free_mb | `13` |
| ocr_median_sec | `12.264` |
| translate_median_sec | `11.269` |
| inpaint_median_sec | `1.982` |
| gemma_json_retry_count | `1` |
| gemma_chunk_retry_events | `1` |
| gemma_truncated_count | `0` |
| gemma_empty_content_count | `0` |
| ocr_empty_rate | `0.0` |
| ocr_low_quality_rate | `0.0081` |

#### 판단

- 기준선 B 대비 `gemma_json_retry_count`는 같은 수준으로 유지
- representative batch `elapsed_sec`는 `1016.410 -> 825.772`로 개선
- representative batch `translate_median_sec`는 `12.977 -> 11.269`로 개선
- representative batch `ocr_median_sec`는 `15.644 -> 12.264`로 개선
- `ocr_low_quality_rate`도 `0.0313 -> 0.0081`로 개선
- 단점은 `gpu_floor_free_mb=13`으로 VRAM 여유가 거의 없다는 점

### 현재 채택 판단

현재 representative benchmark 결과 기준 최종 채택 후보는 `gemma-translation-stable-22-t05`입니다.

선정 이유:

- 속도 개선폭이 큼
- page failure 없음
- JSON retry는 baseline과 동률
- OCR 품질 지표도 개선

보류 이유:

- VRAM headroom이 극단적으로 낮아 다른 워크로드 동시 실행에는 불리함

현재 사용자 목표가 `VRAM 여유보다 속도와 품질 유지`에 있으므로, 현 시점 추천 운영값은 이 조합입니다.
