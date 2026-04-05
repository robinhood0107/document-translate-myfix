# 자동번역 벤치 결과 읽는 법

## 먼저 볼 파일

가장 먼저 보는 파일은 아래 둘입니다.

- [report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/report-ko.md)
- `./banchmark_result_log/<run>/summary.json`

`report-ko.md`는 여러 run을 비교한 문서이고, `summary.json`은 개별 run의 raw 요약입니다.

## 핵심 지표

속도:

- `elapsed_sec`
- `translate_median_sec`
- `ocr_median_sec`
- `inpaint_median_sec`

품질/안정성:

- `page_failed_count`
- `gemma_json_retry_count`
- `gemma_truncated_count`
- `gemma_empty_content_count`
- `ocr_empty_rate`
- `ocr_low_quality_rate`

## 좋은 후보의 조건

- `page_failed_count = 0`
- `gemma_truncated_count = 0`
- `gemma_empty_content_count = 0`
- `gemma_json_retry_count` 증가 없음
- `ocr_empty_rate` 증가 없음
- `ocr_low_quality_rate` 증가 없음
- `elapsed_sec`와 `translate_median_sec`가 함께 개선

## translated export audit

대표 batch 후보는 앞 `5장` translated export를 비교합니다.

```bat
.venv-win-cuda13\Scripts\python.exe scripts\compare_translation_exports.py ^
  --baseline-run-dir ".\banchmark_result_log\<baseline-run>" ^
  --candidate-run-dir ".\banchmark_result_log\<candidate-run>" ^
  --sample-dir ".\Sample" ^
  --sample-count 5 ^
  --output ".\banchmark_result_log\<candidate-run>\translation_audit.json"
```

이 단계는 다음 문제를 잡는 용도입니다.

- translated export 누락
- block key mismatch
- 명백한 과생성 또는 반복 생성

## 차트 해석

자동 보고서는 보통 아래 차트를 만듭니다.

- batch elapsed 비교
- `n_gpu_layers` 대비 `translate_median_sec`
- `temperature` 대비 `translate_median_sec`
- retry / truncated / OCR low-quality 비교

실전에서는 `translate_median_sec`와 `gemma_truncated_count`를 같이 봐야 합니다. 속도만 빠르고 truncated가 늘어나면 채택하면 안 됩니다.

## 관련 문서

- [workflow-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/workflow-ko.md)
- [results-history-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/results-history-ko.md)
- [resource-strategy-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/resource-strategy-ko.md)
