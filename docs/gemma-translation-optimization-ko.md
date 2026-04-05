# Gemma 번역 안정화 / GPU 재배분 최적화

기준 날짜: `2026-04-05`

이 문서는 현재 브랜치에서 진행 중인 `Gemma 번역 경로 안정화 + GPU 재배분` 실험의 목적, 순서, 채택 기준을 정리합니다.

## 1. 현재 판단

현재 기준선은 `translation-baseline`입니다.

핵심 해석:

- `paddleocr-vllm`이 OCR 본 추론을 담당하므로, `paddleocr-server` front service를 `cpu`로 내려도 품질 손실 없이 전체 배치 시간이 줄어들 수 있습니다.
- 실제 다음 병목은 OCR보다 Gemma 번역 단계일 가능성이 높습니다.
- 따라서 1차는 OCR backend보다 Gemma 번역 경로의 JSON 안정성과 GPU 활용을 먼저 올립니다.

## 2. 1차 고정 조건

품질 유지가 최우선이므로 아래는 1차에서 유지합니다.

- Gemma 모델: `gemma-4-26b-a4b-it-heretic.q3_k_m.gguf`
- OCR 모델: `PaddleOCR-VL-1.5-0.9B`
- Gemma `chunk_size=4`
- Gemma `max_completion_tokens=512`
- Gemma `reasoning=off`
- Gemma `--swa-full=enabled`
- OCR client `parallel_workers=8`
- OCR client `max_new_tokens=1024`
- OCR runtime
  - `gpu_memory_utilization=0.84`
  - `max_num_seqs=32`
  - `max_num_batched_tokens=98304`

## 3. 바꾸는 항목

### OCR front

- `paddleocr-server`: `gpu -> cpu`
- `paddleocr-vllm`: 계속 `gpu`

### Gemma sampler

translation-only 기본 분포:

- `temperature=0.5`
- `top_k=64`
- `top_p=0.95`
- `min_p=0.0`

핵심은 번역용 분포를 유지한 채 `temperature`만 `0.4 ~ 0.7` 범위에서 탐색하고, representative batch에서 retry와 품질 지표가 baseline을 넘지 않는 지점을 찾는 것입니다.

### Gemma GPU 확대 순서

1. `n_gpu_layers=20`, `threads=12`, `ctx=4096`
2. `n_gpu_layers=21`
3. `n_gpu_layers=22`
4. `n_gpu_layers=23`
5. `n_gpu_layers=24`
6. `24`가 불안정하면 `ctx=3072` rescue

## 4. preset 순서

### 기준선

- `translation-baseline`

### 1차 후보

- `translation-ngl20`
- `translation-ngl21`
- `translation-ngl22`
- `translation-ngl23`
- `translation-ngl24`
- `translation-ngl24-ctx3072`

### 2차 후보

- `translation-t04`
- `translation-t05`
- `translation-t06`
- `translation-t07`

탐색 순서는 `n_gpu_layers`를 먼저 고른 뒤, 그 우승 설정에서만 `temperature`를 조정하는 방식으로 고정합니다.

## 5. 새로 기록하는 지표

이 실험에서는 아래 지표를 summary에 직접 남깁니다.

- `gemma_json_retry_count`
- `gemma_chunk_retry_events`
- `gemma_truncated_count`
- `gemma_empty_content_count`
- `ocr_empty_block_count`
- `ocr_low_quality_block_count`
- `ocr_empty_rate`
- `ocr_low_quality_rate`
- `ocr_median_sec`
- `translate_median_sec`
- `inpaint_median_sec`

## 6. 채택 기준

아래를 모두 만족해야 새 조합을 채택합니다.

- `page_failed_count = 0`
- `gemma_truncated_count = 0`
- `gemma_empty_content_count = 0`
- `gemma_json_retry_count <= translation-baseline`
- `ocr_empty_rate <= translation-baseline`
- `ocr_low_quality_rate <= translation-baseline`
- representative batch `elapsed_sec` 개선
- representative batch `translate_median_sec` 개선
- one-page `elapsed_sec` 악화 없음

## 7. 현재 결론

현재 active translation baseline은 아래 조합입니다.

- `paddleocr-server --device cpu`
- `paddleocr-vllm` GPU 유지
- Gemma `temperature=0.6`, `top_k=64`, `top_p=0.95`, `min_p=0.0`
- Gemma `n_gpu_layers=23`
- Gemma `threads=12`
- Gemma `ctx=4096`

이 조합을 fixed baseline으로 두고, `n_gpu_layers`와 `temperature`의 한계값을 찾는 방식으로만 확장합니다.

현재까지의 실측 해석:

- corrected baseline batch: `elapsed=1067.117`, `translate_median=13.511`, `truncated=1`
- `translation-ngl23` batch: `elapsed=1053.787`, `translate_median=12.999`, `truncated=1`
- `translation-t06` batch: `elapsed=1048.742`, `translate_median=12.150`, `truncated=0`
- `translation-t06`은 baseline과 동일한 retry / OCR 품질 지표를 유지하면서 속도와 안정성을 함께 개선했습니다.
- audit subset 5장도 통과했으므로, 현재 최종 승자는 `translation-t06`입니다.

즉, 현재 최적 방향은 아래와 같습니다.

- OCR front는 계속 `cpu`
- OCR backend VRAM 설정은 그대로 유지
- Gemma sampler는 `0.6 / 64 / 0.95 / 0.0`
- Gemma `n_gpu_layers`는 `23`
- 다음 개선 단계는 OCR backend VRAM 튜닝이 아니라, 이 baseline을 유지한 채 `ctx`/`chunk_size`와 장기 corpus 안정성을 보는 것입니다.

## 8. CUDA13 실행 규칙

이 최적화는 CUDA13 기준으로만 측정합니다.

- `scripts\\benchmark_pipeline_cuda13.bat`
- `scripts\\benchmark_suite_cuda13.bat`

표준 `.venv-win` 런처는 이번 실험 기준선에 포함하지 않습니다.

## 9. 결과 기록 위치

- raw 결과: `C:\\Users\\pjjpj\\Documents\\Comic Translate`
- repo 요약: [pipeline-benchmark-results-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/pipeline-benchmark-results-ko.md)
- 설정 이력: [gemma-profiles-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/gemma-profiles-ko.md)
