# OCR Combo Benchmark Report

이 파일은 `scripts/generate_ocr_combo_report.py`가 suite manifest를 기준으로 갱신합니다.

## Metadata

- generated_at: `2026-04-07 21:36:35 대한민국 표준시`
- status: `awaiting_gold_review`
- benchmark_name: `OCR Combo Runtime Benchmark`
- benchmark_kind: `managed family suite`
- benchmark_scope: `language-aware OCR+Gemma comparison using benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime`
- execution_scope: `full-pipeline`
- speed_score_scope: `full-pipeline elapsed_sec`
- quality_gate_scope: `OCR-only`
- gold_source: `human-reviewed`
- baseline_sha: `b040c799ec74862ab43572a4274de61b54797007`
- develop_ref_sha: `c1cd90d4da7419df213893e8ddfd1451d16ed0eb`
- entrypoint: `scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime`
- results_root: `./banchmark_result_log/ocr_combo`

## Fixed Gemma

- image: `local/llama.cpp:server-cuda-b8665`
- response_format_mode: `json_schema`
- chunk_size: `6`
- temperature: `0.6`
- n_gpu_layers: `23`

## Awaiting Gold Review

이번 latest run은 사람 검수 OCR gold가 아직 잠기지 않아 bootstrap 모드로 종료되었습니다.
다음 단계는 `benchmarks/ocr_combo/gold/<corpus>/gold.json`을 검수해 `review_status=locked`로 저장한 뒤 같은 명령을 다시 실행하는 것입니다.

## Gold Review Packets

| corpus | gold_review_status | gold_path | gold_review_packet_dir | gold_generated_from_run_dir | gold_page_count | example_page |
| --- | --- | --- | --- | --- | --- | --- |
| china | draft | ./benchmarks/ocr_combo/gold/china/gold.json | ./banchmark_result_log/ocr_combo/20260407_211341_ocr-combo-runtime_suite/gold-review/china | ./banchmark_result_log/ocr_combo/20260407_211341_ocr-combo-runtime_suite/reference-seed/china | 8 | 0006_0005 |
| japan | draft | ./benchmarks/ocr_combo/gold/japan/gold.json | ./banchmark_result_log/ocr_combo/20260407_211341_ocr-combo-runtime_suite/gold-review/japan | ./banchmark_result_log/ocr_combo/20260407_211341_ocr-combo-runtime_suite/reference-seed/japan | 22 | 094 |

검수는 [gold-review-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/ocr-combo/gold-review-ko.md)의 절차를 따릅니다.

## Artifacts

- gold review csv: `./docs/assets/benchmarking/ocr-combo/latest/gold_review_packets.csv`