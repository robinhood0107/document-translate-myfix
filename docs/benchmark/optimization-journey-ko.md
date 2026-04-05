# 자동번역 최적화 여정

이 문서는 현재 로컬 Gemma + PaddleOCR VL 경로를 어떻게 최적화했는지 순서대로 설명합니다.

## 1. 초기 상태

처음에는 OCR front도 GPU를 쓰고, Gemma sampler도 번역보다는 creative 쪽에 가까웠습니다. 이 상태에서는:

- VRAM 상주 경쟁이 컸고
- Gemma JSON 재시도가 생겼고
- 속도/안정성 균형이 좋지 않았습니다

## 2. OCR front를 CPU로 이동

핵심 판단은 이것이었습니다.

- OCR 본 추론은 `paddleocr-vllm`이 GPU에서 수행
- `paddleocr-server` front는 CPU로 내려도 품질 손실이 거의 없음

이 조정은 전체 batch 시간을 줄이는 데 도움이 됐고, 이후 VRAM을 Gemma에 더 배분할 수 있는 기반이 됐습니다.

## 3. translation-only preset 체계 정리

creative/legacy preset을 걷어내고 translation-only 후보만 남겼습니다.

탐색 축은 두 개로 정리했습니다.

- `n_gpu_layers`
- `temperature`

고정한 값:

- `top_k=64`
- `top_p=0.95`
- `min_p=0.0`
- `chunk_size=4`
- `max_completion_tokens=512`

## 4. `n_gpu_layers` 탐색

one-page 기준으로 `20 -> 24`를 먼저 스크리닝했습니다.

이 단계에서 `23`이 가장 좋은 후보로 올라왔고, batch에서도 `translation-ngl23`가 baseline보다 나은 결과를 냈습니다.

## 5. `temperature` 탐색

`n_gpu_layers=23`을 고정한 상태에서 `0.4 ~ 0.7`를 비교했습니다.

결과:

- `0.4`, `0.5`는 느렸고
- `0.7`은 괜찮았지만 `0.6`보다 밀렸고
- `0.6`이 속도와 안정성의 균형이 가장 좋았습니다

## 6. 현재 결론

현재 채택한 운영 조합은 아래입니다.

- `temperature=0.6`
- `top_k=64`
- `top_p=0.95`
- `min_p=0.0`
- `n_gpu_layers=23`
- `threads=12`
- `ctx=4096`
- `paddleocr-server=cpu`

최신 수치와 비교 차트는 [report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/banchmark_report/report-ko.md)에서 봅니다.
