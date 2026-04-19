# 시리즈 프로젝트 저장/무결성 메모

## 저장 구조

- `series_manifest`
- `series_items`
- `embedded_projects`
- `series_settings`
- `series_navigation_history`
- `series_queue_runtime`

## child 동기화 규칙

- child open: embedded `.ctpr` blob -> temp working copy
- child save/autosave/back to series: temp working copy -> embedded `.ctpr` blob
- canonical data는 `.seriesctpr` 내부 blob이다.

## 무결성 목표

- `.ctpr`와 `.seriesctpr` 모두 temp file 완성 후 replace 방식으로 저장한다.
- in-place SQLite save를 기본 경로로 두지 않는다.
- series child temp는 캐시일 뿐 canonical 저장소가 아니다.

## 주의 포인트

- 기존 `.ctpr`는 page 식별이 path 기반이므로 child import 시 item UUID와 child path를 분리 보관한다.
- large blob/history 누적으로 파일이 커질 수 있으므로 series item 메타와 embedded project blob을 분리 저장한다.
- autosave/recovery는 series용 경로와 naming을 별도로 둔다.

## 현재 구현 상태

- `.ctpr` 저장은 `project_state_v2`에서 temp SQLite 작성 후 `os.replace`로 교체한다.
- `.seriesctpr` 저장은 `series_state_v1`의 snapshot writer가 temp SQLite 작성 후 `os.replace`로 교체한다.
- series child 편집은 temp child `.ctpr`를 materialize해서 열고, 저장 시에는 child blob을 다시 container에 덮어쓴다.
- canonical series 데이터는 항상 `.seriesctpr` 파일 자체이며, temp child는 캐시/편집 작업본으로만 사용한다.
