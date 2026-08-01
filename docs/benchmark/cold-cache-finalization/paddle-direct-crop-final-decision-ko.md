# Paddle crop direct llama.cpp 최종 판정

## 판정

`PaddleOCR-VL detector + crop OCR`의 관리형 제품 경로는 PaddleX CPU relay를
제거하고 llama.cpp의 OpenAI 호환 endpoint를 직접 호출하는 구성을 제품 승격한다.

- 모델, detector, crop, guard, parser, max token은 바꾸지 않는다.
- 제품 JPEG crop을 PaddleX와 동일하게 PNG로 변환한다.
- 요청은 `image_url` 다음 `OCR:` 텍스트 순서로 보낸다.
- PaddleX가 반환하던 문단 줄바꿈과 텍스트 정규화를 direct adapter에서 재현한다.
- custom/unmanaged endpoint는 사용자가 명시한 외부 계약이므로 별도로 보존한다.
- COO는 이 전환에 사용하지 않는다. 기존 shadow 평가의
  `reject_no_safe_operating_point` 판정을 유지한다.

초기 JPEG·text-first direct 실험은 출력 차이가 있었으므로 폐기했다. 아래 수치는
실행 중인 PaddleX 3.6.1 predictor의 실제 llama.cpp 요청 계약을 확인한 뒤 얻은 최종
PNG·image-first 결과만 사용한다.

## CUDA 실측

환경은 RTX 4070 SUPER 12GB, `.venv-win-cuda13`, 고정 llama.cpp image digest,
named model volume이다. 각 실행 전후에는 `paddleocr-server`와
`paddleocr-llamacpp`를 정상 `stop`했고 `down`은 사용하지 않았다.

### Japan S6, 73블록 AB/BA

| 구간 | PaddleX relay 중앙값 | direct 중앙값 | 단축 |
|---|---:|---:|---:|
| runtime startup | 10.696718초 | 1.489541초 | 86.074785% |
| OCR request | 47.883237초 | 5.804315초 | 87.878190% |
| 전체 runner wall | 60.216087초 | 8.333160초 | 86.161240% |

두 라운드 모두 73/73에서 raw text, normalized text, OCR status, crop geometry,
semantic role을 포함한 block 계약이 완전히 같았다. HTTP retry는 0이었다.

### 언어 회귀와 전체 일본어 세트

| 세트 | 결과 동일성 | relay wall | direct wall | 단축 |
|---|---:|---:|---:|---:|
| 영어 10페이지, fireplace 음성 대조군 포함 | 10/10 block 완전 동일 | 22.433474초 | 3.601996초 | 83.943655% |
| 중국어 6페이지 | 41/41 block 완전 동일 | 41.921130초 | 7.360335초 | 82.442422% |
| Japan 22페이지 | normalized OCR 311/311 동일 | 비교 가능한 relay wall 없음 | 23.969736초 | 산출 안 함 |

Japan 22페이지의 사람 정답 통계도 두 경로가 완전히 같았다.

- normalized exact: 188/311
- mean character accuracy: 77.570322%
- direct-only exact regression: 0
- direct-only exact improvement: 0

이는 transport 변경이 기존 OCR 품질을 높였다는 뜻이 아니라, 기존 제품 OCR 결과를
그대로 유지하면서 CPU relay의 startup과 per-request overhead를 제거했다는 뜻이다.

## 고정 런타임 계약

- backend: llama.cpp
- image:
  `ghcr.io/ggml-org/llama.cpp@sha256:22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb`
- model SHA-256:
  `f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8`
- mmproj SHA-256:
  `204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a`
- named volume: `comic-translate-paddleocr-vl-llamacpp-models-v1`
- endpoint: `http://127.0.0.1:18000/v1/chat/completions`
- prompt: `OCR:`
- image format: PNG
- content order: image, text
- completion limit: 1024
- special tokens: off

벤치마크 summary에는 container ID, image ID, command, runtime labels, named-volume
mount를 함께 저장해 결과가 다른 런타임과 섞이지 않게 했다.

## Git 외부 근거

원시 crop, 응답, manifest, runtime snapshot과 비교 결과는 다음에 있다.

`<validation-log-root>\paddle-crop-transport\`

최종 판정에 사용한 하위 폴더는 다음과 같다.

- `20260801_s6_compare_v2_round1`, `20260801_s6_compare_v2_round2`
- `20260801_japan22_direct_png_v2`, `20260801_japan22_compare_v2`
- `20260801_english10_relay_v1`, `20260801_english10_direct_v2`,
  `20260801_english10_compare_v2`
- `20260801_china6_relay_v1`, `20260801_china6_direct_v2`,
  `20260801_china6_compare_v2`

## 제품 PR 요구사항

1. 관리형 기본 URL과 health check를 direct llama.cpp endpoint로 전환한다.
2. PNG·image-first·relay-compatible normalization을 별도 paddle crop transport
   모듈로 구현한다.
3. 기존 기본 relay URL만 one-time migration하고 custom URL은 보존한다.
4. active Compose와 runtime manager에서 `paddleocr-server`를 제거한다.
5. runtime/cache fingerprint를 새 direct 계약으로 갱신한다.
6. obsolete relay container/image는 명시적 ID manifest를 만든 뒤에만 제거한다.
7. unit, headless, CUDA12/13 launcher, 실제 offscreen S1/S6를 통과시킨다.

제품 승격 이후에도 full-page Spotting과 MangaLMM은 독립 전략으로 유지하며 이
transport 변경과 섞지 않는다.

## 후속 folder-global queue 판정

direct 제품 경로에서 `page barrier w8`과 `folder-global queue w4`를 7회 paired
CUDA로 다시 비교했다. 7/7 결과는 완전히 같았지만 후보 승패 4승 3패,
pipeline 중앙값 명목 +0.390529%, 단측 95% bootstrap 하한 -0.276427%로 실제
속도 우위를 입증하지 못했다. 따라서 direct transport는 유지하되 folder-global
queue 제품 코드는 승격하지 않는다. 상세 근거는
[folder-global queue 최종 판정](../paddleocr-vl-parallel/folder-global-queue-final-decision-ko.md)에 기록한다.
