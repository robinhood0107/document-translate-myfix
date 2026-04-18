# Workflow Mode And Runtime Design

## 현재 제품 구조

이번 승격에서 제품에 남는 workflow는 두 개다.

1. `stage_batched_pipeline`
2. `legacy_page_pipeline`

웹툰 경로와 benchmark runner는 이번 설계 범위에서 제외한다.

## workflow_mode 저장 모델

```yaml
tools:
  workflow_mode: stage_batched_pipeline | legacy_page_pipeline
```

이 키는 `QSettings` round-trip과 Settings > Tools UI에서 직접 사용한다.

## 라우팅 구조

### `pipeline/main_pipeline.py`

- `workflow_mode == stage_batched_pipeline`
  - `StageBatchedProcessor.batch_process(...)`
- 그 외
  - 기존 `BatchProcessor.batch_process(...)`

## stage-batched 순서

제품 승격 대상 순서는 아래로 고정한다.

```text
detect all
-> OCR all
-> inpaint all
-> translate all
-> render/export all
```

즉 benchmark의 최종 결론과 사용자의 최신 요구를 반영해, 번역은 인페인팅 뒤에 둔다.

## OCR 정책

이번 승격에서 제품 운영안은 `PaddleOCR VL 중심`이다.

- Japanese `Optimal` -> `PaddleOCR VL`
- legacy `best_local_plus` / `Optimal+` 저장값은 `Optimal`로 정규화
- stage-batched 제품 모드에서는 single-runtime route만 허용
- `MangaLMM` sidecar/selector route는 benchmark 실패로 제외

그래서 `pipeline_config.validate_workflow_mode()`는 sidecar가 필요한 조합을 차단한다.

## 마스킹 정책

핵심 설계는 “detector는 그대로 두고, 마스킹 경로만 CTD로 교체”다.

현재 제품 승격에서 기대하는 값:

- `mask_refiner == "ctd"`
- `keep_existing_lines == True`
- residue cleanup 실제 호출

그리고 legacy builder는 명시적 호환 모드일 때만 사용한다.

## CTD 연결 지점

1. `SettingsPage.get_mask_refiner_settings()`
   - persisted CTD 설정을 실제 반환
2. `modules/utils/image_utils.generate_mask()`
   - `ctd` / `legacy_bbox` 분기
3. `BatchProcessor` / `StageBatchedProcessor`
   - `generate_mask()` 뒤 residue cleanup 실제 호출

## 현재 승격 판단

이번 승격은 단순한 UI 토글 추가가 아니라, benchmark winner를 실제 제품 런타임으로 연결하는 작업이다. 따라서 최종 확인 포인트는 OCR이 아니라 다음이다.

1. `workflow_mode`가 실제 배치를 바꾸는가
2. CTD 마스크 경로가 실제로 타는가
3. cleaned output에서 신규 심각 artifact가 없는가
