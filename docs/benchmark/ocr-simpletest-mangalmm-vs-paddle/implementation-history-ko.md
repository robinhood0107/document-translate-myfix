# OCR Simpletest MangaLMM vs PaddleOCR VL 구현 히스토리

## 브랜치 분리 이유

- 제품 기능 구현은 `feature/mangalmm-ocr-optimal-plus`에서 진행했다.
- benchmark runner, preset, generated report, copied asset, simpletest 비교 결과는 장기 benchmark 브랜치인 `benchmarking/lab`에만 유지한다.
- 이 분리는 `rules.md`와 `docs/repo/benchmark-branch-policy-ko.md`의 benchmark 자산 분리 원칙을 따른다.

## 핵심 커밋과 상태

- 제품 feature branch
  - `cdc9254` `feat(ocr): add MangaLMM OCR mode and runtime`
  - 상태: `feature/mangalmm-ocr-optimal-plus` 브랜치에 존재
- benchmark branch 반영
  - `5c62d5f` `feat(ocr): add MangaLMM OCR mode and runtime`
  - 상태: `benchmarking/lab`에서 benchmark runner가 같은 OCR/runtime surface를 이해하도록 반영
  - 주의: benchmark branch에는 benchmark 전용 자산만 유지하고, 제품 기본값 승격은 하지 않음
- simpletest 비교 family
  - `c1b66e8` `feat(benchmark): add MangaLMM simpletest OCR family`
  - 상태: `benchmarking/lab`에만 유지
- PR 상태
  - draft PR `#55`
  - 링크: `https://github.com/robinhood0107/comic-translate-myfix/pull/55`
  - 현재는 제품 기능 검토용이며, benchmark 전용 runner/preset/report는 이 PR에 포함하지 않는다.

## 설계 결정과 근거

### 1. `MangaLMM`은 full-page OCR이 아니라 block-crop OCR

- 기존 파이프라인은 `image + blk_list`를 받아 block crop 기준으로 OCR을 수행한다.
- 이번 통합도 이 구조를 유지해, `MangaLMM`이 페이지 전체 대신 검출된 `TextBlock` crop만 처리하도록 설계했다.
- 이유:
  - 기존 detector/inpaint/render geometry를 건드리지 않는다.
  - OCR 엔진 교체 범위를 최소화한다.
  - 페이지 전체 입력 대비 image size 실패 위험을 크게 줄인다.

### 2. 응답 형식은 plain text 강제가 아니라 JSON OCR

- 실제 llama.cpp + MangaLMM 실험에서 가장 안정적인 출력은 `bbox_2d + text_content` JSON array였다.
- 그래서 OCR 엔진은 plain text 강제를 하지 않고 아래 프롬프트를 기준으로 JSON OCR을 받는다.
  - `Please perform OCR on this image and output only a JSON array of recognized text regions, where each item has "bbox_2d" and "text_content". Do not translate. Do not explain. Do not add markdown or code fences.`
- 파이프라인에서는 `text_content`를 `blk.text`에 넣고, region bbox는 메타데이터로만 저장한다.

### 3. `blk.xyxy`는 유지하고 region bbox는 메타데이터만 저장

- 메인 파이프라인의 주 geometry는 계속 `blk.xyxy`를 사용한다.
- `MangaLMM`이 반환한 region bbox는 `blk.ocr_regions`, `blk.ocr_crop_bbox`, `blk.ocr_resize_scale`에만 저장한다.
- 이유:
  - 기존 render/inpaint alignment를 보존한다.
  - OCR region 좌표 실험을 하더라도 제품 geometry 축을 흔들지 않는다.

### 4. `ctx=4096` 유지

- `MangaLMM` 서버는 `-c 4096`, `-np 1`로 고정했다.
- 판단 근거:
  - block crop + 짧은 고정 OCR 프롬프트 + `max_completion_tokens=256` 조합에서는 텍스트 컨텍스트 압박이 작다.
  - 실제 실패 원인은 컨텍스트 부족보다 이미지 크기와 crop 복잡도 쪽이 더 강했다.
  - 최신 `PaddleOCR VL + Gemma4` full-pipeline 공식 winner에서 `gpu_floor_free_mb=14`가 나왔기 때문에, 불필요한 context 확대는 VRAM 안정성에 더 불리하다.

### 5. resize는 조건부 안전장치로만 사용하고, 원본 좌표로 역매핑

- 기본 정책은 무리사이즈다.
- 단, crop이 큰 경우에만 아래 조건으로 다운스케일한다.
  - `crop area > 1_200_000`
  - 또는 `long side > 1280`
- 스케일은 단일 비율 유지로만 적용한다.
  - `scale = min(1.0, sqrt(1_200_000 / area), 1280 / long_side)`
- 응답 `bbox_2d`는 resized crop 좌표에서 original crop 좌표로 역스케일한 뒤, crop origin을 더해 원본 페이지 좌표로 복원한다.
- 이유:
  - 성공률을 높이면서도 원본 좌표 정합성을 잃지 않는다.

### 6. `Gemma + OCR` 동시 상주, health-first reuse, 재기동 최소화

- `Gemma`와 현재 OCR 엔진은 동시에 상주 가능하게 두고, stage 전환 때문에 내리지 않는다.
- OCR 엔진끼리는 한 번에 하나만 managed active 상태로 둔다.
- runtime manager는 healthy URL이 이미 살아 있으면 재기동하지 않고 그대로 재사용한다.
- 이유:
  - 자동 번역 파이프라인의 stage 전환 시 startup penalty를 줄인다.
  - 재기동 횟수를 줄여 health miss와 Docker churn을 최소화한다.

## 현재 상태

- 제품 기능은 feature branch와 draft PR에 존재한다.
- simpletest 비교 family, preset, generated report, copied asset 정책은 `benchmarking/lab`에만 존재한다.
- `Optimal+`는 새 설계 상태이며, 이번 단계에서는 기본값으로 승격하지 않는다.

## 승격 조건

- `Sample/simpletest` 3장 full-pipeline 속도 비교에서 `MangaLMM`이 `PaddleOCR VL`보다 빠르거나 최소 비슷해야 한다.
- translated image 3장에 대해 사용자가 수동 품질 검수를 수행해야 한다.
- 필요 시 후속 Gemma 소형 retune을 수행할 수 있으나, 1차 기본 추천은 현행 promoted 값 유지다.

## Gemma 유지 기준

- 현재 비교의 baseline Gemma 값은 아래와 같다.
  - `context_size=4096`
  - `threads=10`
  - `n_gpu_layers=23`
  - `chunk_size=6`
  - `temperature=0.7`
  - `max_completion_tokens=512`
- 이 값은 최신 공식 full-pipeline winner 리포트인 `docs/banchmark_report/gemma-iq4nl-japan-report-ko.md`를 기준으로 한다.
- simpletest 결과가 안정적이면 이 값을 그대로 유지한다.
- 아래 조건 중 하나라도 보이면 `MangaLMM` 기준의 후속 소형 retune을 검토한다.
  - `MangaLMM`의 전체 `warm_median_elapsed_sec`가 `PaddleOCR VL`보다 `5%` 이상 느림
  - `page_failed_count > 0`
  - restart, health miss, empty response와 같은 안정성 이슈가 실제 benchmark에서 관측됨
