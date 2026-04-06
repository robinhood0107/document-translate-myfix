# PaddleOCR-VL 1.5 벤치 아키텍처

이 family는 translation benchmark와 달리 **OCR runtime 튜닝**이 목적이지만, 실행 자체는 실제 앱 파이프라인을 그대로 탑니다.

## 실행 범위

- 이미지 로드
- detect
- OCR
- translate
- inpaint
- render
- save

즉, 실제 제품과 같은 offscreen 앱 경로를 사용합니다.

## 공식 점수 범위

합격/탈락과 winner 선정은 아래만 사용합니다.

- `detect wall time`
- `ocr wall time`
- `detect+ocr page total`
- `warm detect+ocr suite total`
- `warm p50/p95/p99 per-page OCR time`
- detection/OCR gold compare 결과

번역, 인페인트, 렌더, 저장은 실행은 하되 공식 점수에는 포함하지 않습니다.

## 품질 게이트

- `page_failed_count = 0`
- `ocr_cache_hit_count = 0`
- block count 동일
- bubble/box IoU 기반 1:1 매칭
- normalized OCR text 완전 일치
- `non_empty`, `empty`, `single_char_like` 무회귀

## 결과 루트

- raw 결과: `./banchmark_result_log/paddleocr_vl15/`
- 최신 보고서: [paddleocr-vl15-report-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/banchmark_report/paddleocr-vl15-report-ko.md)
- 최신 자산: [latest](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/assets/benchmarking/paddleocr-vl15/latest)

## 브랜치 정책

- benchmark harness, preset, report, 차트는 `benchmarking/lab`에만 둡니다.
- 제품 승격은 별도 `codex/* -> develop` PR에서 runtime/config/policy만 옮깁니다.
