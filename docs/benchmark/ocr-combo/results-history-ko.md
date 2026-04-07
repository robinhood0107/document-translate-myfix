# OCR Combo 결과 이력

이 문서는 `ocr-combo` family의 latest 결과와 운영 해석을 기록합니다.

## latest

- report: [ocr-combo-report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/banchmark_report/ocr-combo-report-ko.md)
- raw run: `./banchmark_result_log/ocr_combo/20260407_161019_ocr-combo-runtime_suite`
- status: `early stop after convergence / no-promotion`

## 2026-04-07 convergence run

- 실행 진입점: `scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime`
- fixed Gemma: `b8665 + json_schema + chunk_size 6 + temperature 0.6 + n_gpu_layers 23`
- China 결론:
  - 외부 비교군 `PPOCRv5`, `HunyuanOCR`는 모두 hard gate 실패
  - reference 계열 `PaddleOCR VL + Gemma`만 합격 후보로 남음
  - final confirm median elapsed는 `199.575s`
  - 하지만 reference 재실행 3회 모두 `translation_similarity_avg < 0.98`로 fail
- japan 결론:
  - 외부 비교군 `MangaOCR`, `HunyuanOCR`는 모두 hard gate 실패
  - `PaddleOCR VL + Gemma`만 합격 후보로 남음
  - reference-only tuning을 일부 진행했지만 `pw4`, `pw8`도 quality gate 실패
- 최종 정책:
  - China corpus 권장 OCR: `PaddleOCR VL + Gemma`
  - japan corpus 권장 OCR: `PaddleOCR VL + Gemma`
  - mixed corpus 운영: 우선 `PaddleOCR VL + Gemma` 단일 OCR로 시작
- 승격 판단:
  - `develop` 승격 비권장
  - 이유: current gate가 reference 재실행까지 안정적으로 통과시키지 못해 promotion-ready winner를 만들지 못함

## 해석 원칙

- 결과는 China/japan corpus를 분리해서 읽습니다.
- 이번 latest는 `OCR family recommendation`은 확정했지만, `promotion-ready runtime default`는 확정하지 못한 라운드입니다.
- benchmark 자산은 계속 `benchmarking/lab`에만 보존합니다.
