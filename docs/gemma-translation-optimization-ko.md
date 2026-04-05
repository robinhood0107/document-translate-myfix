# Gemma 번역 안정화 / GPU 재배분 최적화

기준 날짜: `2026-04-05`

이 문서는 현재 브랜치에서 진행 중인 `Gemma 번역 경로 안정화 + GPU 재배분` 실험의 목적, 순서, 채택 기준을 정리합니다.

## 1. 현재 판단

현재 기준선에서는 `gpu-shift-ocr-front-cpu`가 가장 실용적인 출발점입니다.

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

기존 creative 쪽 분포:

- `temperature=1.0`
- `top_k=0`
- `top_p=1.0`
- `min_p=0.05`

실측 결과 최종 채택 분포:

- `temperature=0.5`
- `top_k=64`
- `top_p=0.95`
- `min_p=0.0`

핵심은 creative 분포를 번역용 분포로 조이고, representative batch에서 retry가 baseline 수준을 넘지 않는 지점까지 온도를 낮추는 것입니다.

### Gemma GPU 확대 순서

1. `n_gpu_layers=22`, `threads=12`, `ctx=4096`
2. `24`는 실험 후보로만 유지
3. 현재 운영 기본값은 `22`

## 4. preset 순서

### 기준선

- `live-ops-baseline`
- `gpu-shift-ocr-front-cpu`

### 1차 후보

- `gemma-translation-stable-22`
- `gemma-translation-stable-24`
- `gemma-translation-stable-24-ctx3072`

### fallback

- `gemma-translation-stable-22-t07`
- `gemma-translation-stable-22-t05`

fallback은 자동 채택이 아니라, `1.0 / 64 / 0.95 / 0.0`에서도 JSON retry가 줄지 않을 때만 검토합니다. 현재 기준으로는 `t05`가 representative batch 결과상 최종 채택 후보입니다.

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
- `gemma_json_retry_count <= gpu-shift-ocr-front-cpu baseline`
- `ocr_empty_rate <= baseline`
- `ocr_low_quality_rate <= baseline`
- representative batch `elapsed_sec` 개선
- representative batch `translate_median_sec` 개선
- one-page `elapsed_sec` 악화 없음

## 7. 현재 결론

현재 representative benchmark 결과 기준 최종 채택 후보는 아래 조합입니다.

- `paddleocr-server --device cpu`
- `paddleocr-vllm` GPU 유지
- Gemma `temperature=0.5`, `top_k=64`, `top_p=0.95`, `min_p=0.0`
- Gemma `n_gpu_layers=22`
- Gemma `threads=12`
- Gemma `ctx=4096`

이 조합은 기준선 B 대비:

- `gemma_json_retry_count`는 같은 수준(`1`)
- `translate_median_sec`는 개선
- `ocr_median_sec`도 개선
- representative batch 총 시간도 개선

## 8. CUDA13 실행 규칙

이 최적화는 CUDA13 기준으로만 측정합니다.

- `scripts\\benchmark_pipeline_cuda13.bat`
- `scripts\\benchmark_suite_cuda13.bat`

표준 `.venv-win` 런처는 이번 실험 기준선에 포함하지 않습니다.

## 9. 결과 기록 위치

- raw 결과: `C:\\Users\\pjjpj\\Documents\\Comic Translate`
- repo 요약: [pipeline-benchmark-results-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/pipeline-benchmark-results-ko.md)
- 설정 이력: [gemma-profiles-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/gemma-profiles-ko.md)
