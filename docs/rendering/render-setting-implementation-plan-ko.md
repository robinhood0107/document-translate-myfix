# 렌더 설정 구현 계획 및 결과

## 목표

- 현재 헤더 버튼 라우팅을 유지한다.
- 스크린샷에서 본 수동/자동 모드 동작을 그대로 유지한다.
- 기존 윤곽선 의미를 바꾸지 않으면서 UI를 더 분명하게 만든다.
- 상하 정렬 `top / center / bottom`을 새 전역 렌더 설정으로 추가한다.
- 자동판단하던 색상 동작에는 필요할 때만 강제 옵션을 추가한다.

## 확장 대상

### 데이터 모델

- `TextRenderingSettings`에 아래 필드를 추가한다.
  - `force_font_color`
  - `smart_global_apply_all`
  - `vertical_alignment_id`
- `TextItemProperties`에 아래 필드를 추가한다.
  - `vertical_alignment`
  - `source_rect`
  - `block_anchor`
- `TextBlockItem`에도 아래 상태를 추가한다.
  - `vertical_alignment`
  - `source_rect`
  - `block_anchor`

### 스타일 해석 계층

공통 헬퍼 계층에서 다음을 담당하도록 구성했다.

- 기존 자동 색상 판단
- 색상 강제 오버라이드
- 원본 박스를 기준으로 한 상하 정렬 계산

내부 규칙은 다음과 같다.

- `GLOBAL`: 현재 패널 값을 항상 사용
- `SMART`: 기본은 자동판단, 강제 토글이 켜지면 패널 값 사용
- `ITEM`: 현재 선택된 아이템에만 적용

### 수동 경로 통합

- `Render`는 기존 진입점을 유지하되, 생성되는 텍스트 아이템에
  - `source_rect`
  - `vertical_alignment`
  - 공통 색상 정책
  를 적용한다.
- `Translate`도 기존 진입점을 유지하되, 번역 후 재래핑 시
  - 공통 렌더 정책 재사용
  - 상하 위치 재계산
  - 색상 강제 옵션 반영
  이 되도록 바꿨다.

### 자동 경로 통합

- 일반 `Translate All`은 `text_items_state` 생성 시 공통 렌더 정책을 사용한다.
- 웹툰 `Translate All`도 같은 정책을 사용한다.
- 최종 저장 렌더는 기존처럼 직렬화된 `text_items_state`를 읽는 구조를 유지한다.

## 호환성

- 기존 프로젝트 파일도 새 필드 없이 정상적으로 열려야 한다.
- `vertical_alignment`가 없으면 기본값은 `top`이다.
- `source_rect`가 없으면 현재 아이템 위치/크기 또는 블록 지오메트리로 보정한다.

## 검증 목표

- 수동 모드 스크린샷 동작이 그대로여야 한다.
- 자동 모드 스크린샷 동작이 그대로여야 한다.
- 윤곽선은 여전히 전역 설정으로 동작해야 한다.
- 색상은 자동판단 또는 강제 적용 중 하나로 일관되게 작동해야 한다.
- 상하 정렬은 저장/불러오기 이후에도 유지되어야 한다.

## 실제 구현 파일 맵

### 공통 정책과 상태

- [render_style_policy.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/modules/utils/render_style_policy.py)
- [render.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/modules/rendering/render.py)
- [text_item_properties.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/canvas/text/text_item_properties.py)
- [text_item.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/canvas/text_item.py)

### UI와 컨트롤러 연결

- [workspace.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/main_window/builders/workspace.py)
- [window.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/main_window/window.py)
- [controller.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/controller.py)
- [text.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/controllers/text.py)

### 수동/자동 파이프라인 통합

- [manual_workflow.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/controllers/manual_workflow.py)
- [batch_processor.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/pipeline/batch_processor.py)
- [render.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/pipeline/webtoon_batch/render.py)

### 저장/복원 및 호환성 경로

- [image_viewer.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/canvas/image_viewer.py)
- [save_renderer.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/canvas/save_renderer.py)
- [text_item_manager.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/canvas/webtoons/scene_items/text_item_manager.py)
- [projects.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/controllers/projects.py)
- [search_replace.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/controllers/search_replace.py)
- [base.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/commands/base.py)

## 이 브랜치에서 실제로 수행한 검증

실행한 항목:

- 변경된 구현 파일에 대해 `./.venv/bin/python -m py_compile ...` 실행
- 오프스크린 Qt 환경에서 `ComicTranslate()` 생성 스모크 테스트
- `vertical_alignment='center'` 상태의 `TextBlockItem` 오프스크린 스모크 테스트

확인한 결과:

- 새 렌더 패널 컨트롤이 창 생성 시 정상적으로 만들어진다.
- 기존 윤곽선 체크박스는 내부 호환용 상태값으로 유지된다.
- 화면에는 더 분명한 윤곽선 `OFF | ON` UI가 보인다.
- 상하 정렬은 이동 후 다시 정렬해도 Y 위치가 누적 드리프트하지 않는다.
- 렌더 패널은 `Text Color`, `Horizontal / Vertical`, `Style / Outline` 그룹으로 재배치할 수 있다.
- 상하 정렬과 윤곽선 `OFF / ON`은 렌더 패널 내부 전용 체크 스타일로 강조 표시할 수 있다.

## 주요 설계 선택

- 헤더 버튼 라우팅은 의도적으로 바꾸지 않았다.
- 글꼴 크기 드롭다운은 계속 `ITEM` 전용이며, 새 렌더는 최소/최대 글꼴 크기 설정을 기준으로 자동 맞춤한다.
- `block_anchor`는 수동 이동/리사이즈 후에도 블록 식별을 유지하기 위해 `source_rect`와 분리했다.
- 실제 UI에서는 내부 정책 용어를 숨기고, 설명형 문구만 보여 주도록 정리했다.
