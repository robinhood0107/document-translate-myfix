# 렌더링 동작 / 설계 메모

이 문서는 헤더 버튼 흐름, 렌더 설정의 현재 동작, 그리고 렌더 설정 확장 구현 결과를 한곳에 모아 정리한 문서입니다.

## 헤더 버튼과 모드 흐름

### 헤더 UI 구성

- 헤더 컨트롤은 [workspace.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/main_window/builders/workspace.py)에서 생성됩니다.
- `self.hbutton_group`에는 아래 6개 버튼이 들어갑니다.
  - `Detect`
  - `Recognize`
  - `Translate`
  - `Segment`
  - `Clean`
  - `Render`
- `Manual` / `Automatic` 라디오와 `Translate All`도 같은 헤더 레이아웃에 있습니다.

### 모드 전환

- `Manual`은 `controller.manual_mode_selected()`로 연결됩니다.
  - 수동 버튼 그룹 활성화
  - `Translate All` 비활성화
- `Automatic`은 `controller.batch_mode_selected()`로 연결됩니다.
  - 수동 버튼 그룹 비활성화
  - `Translate All` 활성화

즉, 헤더 버튼 라우팅은 원래부터 올바르게 동작했고, 이후 렌더 관련 확장에서도 이 구조는 유지했습니다.

### 수동 / 자동 진입점

- 수동 버튼은 모두 [controller.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/controller.py)의 기존 진입점을 그대로 탑니다.
- 자동 처리는 `controller.start_batch_process()`를 유일한 진입점으로 유지합니다.
- 일반 배치는 [batch_processor.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/pipeline/batch_processor.py), 웹툰 배치는 `pipeline.webtoon_batch_process()` 경로를 사용합니다.

## 렌더 설정의 현재 동작

### 기존/현재 전역 설정

아래 항목은 수동 렌더와 배치 렌더에 공통으로 적용됩니다.

- `font_family`
- `min_font_size`
- `max_font_size`
- `color`
- `upper_case`
- `outline`
- `outline_color`
- `outline_width`
- `bold`
- `italic`
- `underline`
- `line_spacing`
- `direction`

즉, 윤곽선은 원래도 전역 렌더 설정이었습니다.

### 자동 판단과 개별 편집

- 글꼴 색상은 기본적으로 감지된 색이 있으면 그것을 쓰고, 없으면 패널 색상을 사용합니다.
- 현재는 `Use Selected Color` 옵션으로 이 자동 판단을 강제 override할 수 있습니다.
- 글꼴 크기 드롭다운은 여전히 선택된 텍스트 아이템만 수정하는 `ITEM` 성격입니다.
- 실제 새 렌더에서는 최소/최대 글꼴 크기 범위를 기준으로 `pyside_word_wrap()`가 최종 크기를 맞춥니다.

### 현재 UI 그룹

- `Text Color`
- `Horizontal / Vertical`
- `Style / Outline`

상하 정렬과 윤곽선 `OFF / ON`은 렌더 패널 전용 체크 스타일을 사용합니다.

## 렌더 설정 확장 구현

### 추가된 모델/상태

- `TextRenderingSettings`
  - `force_font_color`
  - `smart_global_apply_all`
  - `vertical_alignment_id`
- `TextItemProperties`
  - `vertical_alignment`
  - `source_rect`
  - `block_anchor`
- `TextBlockItem`
  - `vertical_alignment`
  - `source_rect`
  - `block_anchor`

### 공통 정책 계층

공통 렌더 정책은 아래를 담당합니다.

- 자동 색상 판단
- 색상 강제 오버라이드
- 원본 박스를 기준으로 한 상하 정렬 계산

내부 정책은 아래처럼 유지합니다.

- `GLOBAL`
- `SMART`
- `ITEM`

다만 실제 UI에는 이 내부 용어를 노출하지 않고 설명형 문구만 보여 줍니다.

### 통합 경로

같은 렌더 정책이 아래 경로에서 공통으로 쓰이도록 정리했습니다.

- 수동 `Render`
- 수동 `Translate` 후 재래핑
- 일반 배치
- 웹툰 배치
- 저장/복원
- 내보내기 렌더
- 검색/치환 매칭

## 호환성과 설계 원칙

- 헤더 버튼 라우팅은 바꾸지 않습니다.
- 기존 프로젝트 파일은 새 필드 없이도 열려야 합니다.
- `vertical_alignment` 기본값은 `top`입니다.
- `source_rect`가 없으면 현재 위치/크기 또는 블록 지오메트리로 보정합니다.
- `block_anchor`는 수동 이동/리사이즈 후에도 원래 블록 식별을 유지하기 위해 `source_rect`와 분리합니다.

## 관련 구현 파일

### 공통 정책 / 상태

- [render_style_policy.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/modules/utils/render_style_policy.py)
- [render.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/modules/rendering/render.py)
- [text_item_properties.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/canvas/text/text_item_properties.py)
- [text_item.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/canvas/text_item.py)

### UI / 컨트롤러

- [workspace.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/main_window/builders/workspace.py)
- [window.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/main_window/window.py)
- [controller.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/controller.py)
- [text.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/controllers/text.py)

### 저장 / 복원 / 배치

- [manual_workflow.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/controllers/manual_workflow.py)
- [batch_processor.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/pipeline/batch_processor.py)
- [render.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/pipeline/webtoon_batch/render.py)
- [image_viewer.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/canvas/image_viewer.py)
- [save_renderer.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/canvas/save_renderer.py)

## 검증 메모

- 헤더 모드 전환 동작은 유지되었습니다.
- 상하 정렬은 저장/복원 이후에도 유지됩니다.
- 윤곽선은 여전히 전역 설정으로 동작합니다.
- 색상은 자동판단 또는 강제 적용 중 하나로 일관되게 동작합니다.
- 오프스크린 Qt 환경 스모크와 관련 파일 `py_compile` 검증을 수행한 이력이 있습니다.
