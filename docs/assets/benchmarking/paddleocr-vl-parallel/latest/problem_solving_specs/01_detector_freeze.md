# 문제 해결 명세서 01 - Detector Freeze

핵심 문제 해결 방향은 사용자가 착안했다.

## 가설

detector geometry를 baseline seed로 고정하고 이후 OCR만 비교한다.

## 실험 조건

- baseline manifest=./banchmark_result_log/paddleocr_vl_parallel/20260415_020217_paddleocr-vl-parallel-smoke/detector_manifest.json

## 측정값

- pages=13 baseline=fixed_w8

## 해석

detector geometry를 baseline seed로 고정하고 이후 OCR만 비교한다.

## 다음 행동

- baseline gold seed 생성

## 저자 및 기여

- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
