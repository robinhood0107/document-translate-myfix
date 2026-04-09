# `comic-translate 2.7.0 -> 2.7.1` 수동 백포트 검수 문서

## 요약

이 포크는 현재 로컬 적응형 `2.7.0` 기준선에서, upstream `2.7.1`의 필요한 버그 수정만 선택해서 반영합니다.

upstream `2.7.1`은 바뀐 파일 수가 많지 않지만, 이 포크는 릴리스를 통째로 merge하지 않고 현재 구조에 맞게 수동으로 적응 이식합니다.

## upstream 파일 변경 목록

upstream `v2.7.0...v2.7.1`에서 실제로 바뀐 파일은 아래 8개입니다.

- `.github/workflows/build-macos-dmg.yml`
- `app/controllers/image.py`
- `app/controllers/projects.py`
- `app/controllers/psd_importer.py`
- `app/controllers/task_runner.py`
- `app/ui/list_view_image_loader.py`
- `app/ui/main_window/builders/nav.py`
- `app/version.py`

## 현재 포크에 반영한 항목

### `app/controllers/image.py`

- PSD importer의 `prepare_psd_font_catalog()`를 import
- threaded PSD import 전에 폰트 카탈로그를 미리 준비
- 비동기 navigation callback을 `QTimer.singleShot(..., self.main, ...)`로 정리

### `app/controllers/projects.py`

- autosave worker의 error/finished callback을 main-thread receiver를 거쳐 실행

### `app/controllers/psd_importer.py`

- 공개 helper `prepare_psd_font_catalog()` 추가
- `get_image_data()`, `get_channel_by_id()`, `get_channel_by_index()` 실패 시 로깅 추가
- RGB 채널이 완전히 비는 경우 오류 로그 추가
- `_can_build_font_catalog_in_current_thread()`를 추가해 잘못된 thread에서 font catalog를 무리하게 빌드하지 않도록 수정

### `app/controllers/task_runner.py`

- queue continuation을 `QTimer.singleShot(..., self.main, ...)`로 정리
- `result`, `error`, `finished` callback이 모두 일관되게 main thread로 돌아오도록 수정

### `app/ui/list_view_image_loader.py`

- worker 출력 타입을 `QPixmap`에서 `QImage`로 전환
- `QPixmap.fromImage(...)`는 main thread에서만 수행
- 썸네일 전달 시 temporary numpy buffer lifetime에 의존하지 않도록 deep-copied `QImage` 사용
- cross-thread 직접 호출 대신 queued signal 전달로 정리
- 현재 포크의 card/avatar 구조는 유지하면서 upstream의 thread 안정화 수정만 흡수

### `app/ui/main_window/builders/nav.py`

- `Project File` 옆에 `PSD`가 오도록 유지
- 표시 문자열을 `PSD File`에서 `PSD`로 단순화

### `app/version.py`

- 앱 버전을 `2.7.1`로 올림

## 명시적으로 제외한 항목

### `.github/workflows/build-macos-dmg.yml`

이번 라운드에서는 upstream macOS DMG workflow를 가져오지 않습니다.

이유:

- 현재 포크의 패키징/릴리스 경로는 Windows 중심입니다.
- 사용자 요구사항에서 `2.7.1` 선택 이식 범위에서 macOS workflow를 명시적으로 제외했습니다.

## 적응 이식 메모

이번 백포트는 upstream 파일을 그대로 복사한 것이 아닙니다.

- 이 포크는 이미 upstream와 다른 OCR/runtime 구조를 가지고 있습니다.
- 현재 list view는 upstream와 동일한 item-decoration 구조가 아니라 card/avatar 구조를 씁니다.
- PSD import/export도 이 포크의 round-trip 경로에 맞게 이미 diverge되어 있습니다.

따라서 `2.7.1` 변경은 **파일 단순 복원**이 아니라, 필요한 동작 수정만 현재 구조에 맞게 적응 이식한 것입니다.

## 결론

이 포크의 `2.7.1` 백포트는 범위를 좁게 유지합니다. 핵심은 PSD import 안전성, UI-thread 안전성, 썸네일 로더 안정성, navigation 정리, 버전 갱신이며, 기존 로컬 OCR/runtime 구조는 유지합니다.
