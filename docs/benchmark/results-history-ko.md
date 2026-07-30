# 자동번역 벤치 결과 이력

이 문서는 `./banchmark_result_log`에 남겨둔 주요 run을 사람 읽기용으로 정리한 레지스트리입니다.

## 2026-07-30 전체 감사 및 런타임 상주 후속 실측

- [전체 실험 감사·품질 예시·후속 속도 계획](./20260730-full-experiment-audit-and-next-speed-plan-ko.md)
- 후속 Gemma 실측 요약:
  - 오래 멈춘 cold start-to-healthy: 66.811초
  - 정상 `stop` 직후 재시작: 16.865초, 74.757% 단축
  - 현재 stage-batched 1페이지의 Gemma runtime start/wait: 18.832초
  - 모델을 VRAM에 둔 채 시작했을 때 GPU 최소 여유: 79MiB
- 판정: OS 페이지 캐시 재사용은 보존하고, full-auto GPU 상주는 기본값으로
  채택하지 않는다. 새 이미지 속도는 PaddleOCR runtime hot-window 실험을 먼저
  검토한다.

## 현재 active preset

- `translation-baseline`

현재 preset 파일 값:

- Gemma `temperature=0.6`
- `top_k=64`
- `top_p=0.95`
- `min_p=0.0`
- `n_gpu_layers=23`
- `threads=12`
- `ctx=4096`
- `paddleocr-server=cpu`

## 기준 run

| 용도 | run dir |
| --- | --- |
| baseline one-page | `./banchmark_result_log/20260405_224354_translation-baseline_one-page_r1` |
| baseline batch | `./banchmark_result_log/20260405_231837_translation-baseline_batch_r1` |
| ngl23 batch finalist | `./banchmark_result_log/20260405_233628_translation-ngl23_batch_r1` |
| t06 batch winner candidate | `./banchmark_result_log/20260406_001330_translation-t06_batch_r1` |

## sweep run

### `n_gpu_layers`

- `./banchmark_result_log/20260405_225129_translation-ngl20_one-page_r1`
- `./banchmark_result_log/20260405_225451_translation-ngl21_one-page_r1`
- `./banchmark_result_log/20260405_225737_translation-ngl22_one-page_r1`
- `./banchmark_result_log/20260405_230023_translation-ngl23_one-page_r1`
- `./banchmark_result_log/20260405_230307_translation-ngl24_one-page_r1`

### `temperature`

- `./banchmark_result_log/20260406_000110_translation-t04_one-page_r1`
- `./banchmark_result_log/20260406_000423_translation-t05_one-page_r1`
- `./banchmark_result_log/20260406_000711_translation-t06_one-page_r1`
- `./banchmark_result_log/20260406_000955_translation-t07_one-page_r1`

## 참고

최신 비교 표와 차트는 [report-ko.md](../banchmark_report/report-ko.md)에서 봅니다.
