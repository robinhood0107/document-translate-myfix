# Workflow Split Runtime Problem Definition And Journey

## 사용자 문제 정의

사용자는 배치 파이프라인이 페이지 단위로 `detect -> OCR -> inpaint -> translate`를 반복하면서 Docker 기동/healthcheck와 VRAM 회수가 비효율적으로 일어나는 점을 문제로 보았다. 특히 `Gemma` 번역 런타임의 기동 시간이 길기 때문에, 전체 워크플로우를 단계형으로 분리하면 시간이 줄어드는지 검증해 달라고 요청했다.

## Requirement 1 결론

benchmark canonical 문서 기준 Requirement 1은 성공했다.

- 기존 `legacy_page_pipeline`: `995.846s`
- `stage_batched_pipeline`의 승격 후보인 Japanese `Optimal(PaddleOCR VL 중심)`: `714.725s`
- 시간 이득: `281.121s`
- 개선률: 약 `28.2%`

동시에 OCR parity도 benchmark 기준으로 유지됐다.

- `detect_box_total=212`
- `ocr_non_empty_total=212`
- `page_failed_count=0`

즉 이번 승격의 핵심은 OCR 품질 재논쟁이 아니라, benchmark winner를 제품 코드로 안전하게 연결하는 일이다.

## Requirement 2 결론

`MangaLMM` hybrid selector는 benchmark 실패로 종료했다.

이유는 아래 두 가지다.

1. 속도
   - `Paddle OCR-only warm`: 약 `300.437s`
   - `MangaLMM OCR-only warm`: 약 `365.250s`
2. 품질
   - detector block과 `MangaLMM` 결과를 좌표 매칭으로 비교했을 때 `text_bubble` 누락과 merge/split 오류가 반복됐다.

따라서 제품 승격 범위에서 `MangaLMM hybrid`는 제외하고, `stage_batched + PaddleOCR VL 중심`만 남긴다.

## 마지막 blocker

benchmark 결론을 제품에 옮기기 전 마지막 blocker는 `순서`가 아니라 `마스킹 경로`였다.

문제는 세 군데에서 동시에 발생했다.

1. `SettingsPage.get_mask_refiner_settings()`가 실질적으로 `legacy_bbox`를 강제
2. `modules/utils/image_utils.py`가 들어온 설정을 다시 legacy로 덮어씀
3. `generate_mask()`가 최종적으로 legacy builder만 호출

즉 detector는 최신이었지만, 실제 배치 루트는 계속 legacy mask path를 타고 있었다.

## 현재 해결 방향

이번 제품 승격에서 해결하는 것은 아래다.

1. `CTD + keep_existing_lines=True`를 실제 배치/단계형 루트에 연결
2. residue cleanup을 placeholder가 아니라 실제로 호출
3. `workflow_mode`를 Settings > Tools에 추가
4. 제품 노출 플로우를 아래 두 개로 고정
   - `Stage-Batched Pipeline (Recommended)`
   - `Legacy Page Pipeline (Legacy)`

## 지금 남은 일

1. product branch에서 대표 smoke와 i18n 갱신 완료
2. `feature/workflow-split-runtime` commit / push
3. `develop` 승격 PR 갱신

benchmark raw docs와 canonical 하네스는 계속 `benchmarking/lab`에 두고, `develop`에는 이 요약만 남긴다.
