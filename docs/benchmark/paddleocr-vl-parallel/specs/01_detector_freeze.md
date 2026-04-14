# 문제 해결 명세서 01 - Detector Freeze

핵심 문제 해결 방향은 사용자가 착안했다.

## 가설

baseline shipping candidate의 detect-ocr-only snapshot을 detector freeze 기준으로 사용하면 이후 OCR candidate 비교가 흔들리지 않는다.

## 실험 조건

- corpus: `Sample/japan_vllm_parallel_subset`
- baseline candidate: `fixed_w8`

## 측정값

suite 실행 후 latest spec에서 채운다.

## 해석

suite 실행 후 latest spec에서 채운다.

## 다음 행동

baseline gold seed를 생성하고 candidate sweep으로 넘어간다.
