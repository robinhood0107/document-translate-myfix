# Gemma IQ4_NL Japan Full-GPU Live Progress

## 현재 상태

- 업데이트 시각: `2026-04-12 19:59:59 KST`
- 공식 suite: `./banchmark_result_log/gemma_iq4nl_japan/20260411_171639_gemma_iq4nl_japan_fullgpu_suite`
- suite 상태: `completed`
- 현재 stage: `confirm`
- 현재 candidate: `max512`
- 마지막 heartbeat: `2026-04-12 19:58:41 대한민국 표준시`
- infra retry count: `4`

## 고정 파이프라인

- translator: `Custom Local Server(Gemma)`
- ocr: `PaddleOCR VL`
- detector: `RT-DETR-v2 ONNX + CUDAExecutionProvider`
- inpainter: `lama_large_512px`
- mask refiner: `ctd`
- use_gpu: `true`
- OCR / detector / CTD / inpainter: 모두 `cuda`
- corpus: `Sample/japan` 전체 22장

## 진행 중인 candidate

_현재 활성 attempt 정보를 찾지 못했습니다._

## stage1 완료 후보 요약

| candidate | attempt | elapsed_sec | page_failed | translate_median_sec | gpu_floor_free_mb | note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ov08-ngl23 | attempt01_t07_infra01 | 1109.261 | 0 | 20.710 | 12 | passed |
| ov08-ngl22 | attempt01_t07_infra01 | 1109.715 | 0 | 21.122 | 15 | passed |
| ov08-ngl20 | attempt01_t07_infra01 | 1135.166 | 0 | 22.796 | 12 | passed |
| ov08-ngl16 | attempt01_t07_infra01 | 1179.316 | 0 | 26.230 | 82 | passed |
| ov08-ngl18 | attempt01_t07_infra01 | 1189.276 | 0 | 26.494 | 22 | passed |
| ov08-ngl14 | attempt03_t05_infra01 | 1242.492 | 0 | 28.301 | 132 | passed |
| ov08-ngl14 | attempt02_t06_infra01 | 1251.539 | 0 | 26.305 | 120 | truncated |
| ov072-ngl23 | attempt01_t07_infra01 | 1373.196 | 0 | 30.907 | 10 | passed |
| ov068-ngl23 | attempt01_t07_infra01 | 1403.564 | 0 | 34.372 | 9 | passed |
| ov072-ngl22 | attempt01_t07_infra01 | 1404.367 | 0 | 34.442 | 15 | passed |
| ov068-ngl22 | attempt01_t07_infra01 | 1417.897 | 0 | 35.335 | 8 | passed |
| ov072-ngl20 | attempt01_t07_infra01 | 1520.610 | 0 | 39.253 | 13 | passed |
| ov068-ngl20 | attempt01_t07_infra01 | 1553.597 | 0 | 41.645 | 12 | passed |
| ov076-ngl22 | attempt01_t07_infra01 | 1634.221 | 0 | 42.254 | 31 | passed |
| ov068-ngl18 | attempt01_t07_infra01 | 1653.387 | 0 | 44.136 | 27 | passed |
| ov076-ngl23 | attempt01_t07_infra01 | 1693.197 | 0 | 42.176 | 9 | passed |
| ov076-ngl20 | attempt01_t07_infra01 | 1741.225 | 0 | 44.711 | 50 | passed |
| ov072-ngl18 | attempt01_t07_infra01 | 1798.925 | 0 | 51.488 | 36 | passed |
| ov068-ngl16 | attempt01_t07_infra01 | 1871.466 | 0 | 53.742 | 79 | passed |
| ov072-ngl16 | attempt01_t07_infra02 | 1922.773 | 0 | 58.626 | 68 | passed |
| ov072-ngl14 | attempt01_t07_infra02 | 1942.822 | 0 | 58.052 | 39 | passed |
| ov076-ngl18 | attempt01_t07_infra01 | 1968.816 | 0 | 51.157 | 31 | passed |
| ov08-ngl14 | attempt01_t07_infra01 | 2002.261 | 0 | 54.850 | 69 | truncated |
| ov076-ngl16 | attempt01_t07_infra01 | 2053.637 | 0 | 60.083 | 87 | truncated |
| ov076-ngl16 | attempt02_t06_infra01 | 2065.875 | 0 | 60.917 | 39 | passed |
| ov076-ngl14 | attempt01_t07_infra02 | 2282.597 | 0 | 72.185 | 283 | passed |
| ov068-ngl14 | attempt02_t06_infra01 | 2632.192 | 0 | 88.301 | 131 | passed |
| ov068-ngl14 | attempt01_t07_infra01 | 2773.300 | 1 | 91.195 | 84 | failed |

## 현재까지의 해석

- 현재까지 가장 빠른 통과 후보는 `ov08-ngl23 / attempt01_t07_infra01`이며 `elapsed_sec=1109.261`, `translate_median_sec=20.710` 입니다.
- 현재까지 rescue가 필요했던 후보는 `ov08-ngl14`이며, `attempt03_t05_infra01`에서 통과했습니다.

## 메모

- 이 문서는 벤치가 도는 동안의 live progress 문서입니다. 최종 채택값은 suite 완료 후 최종 보고서에 다시 정리합니다.
