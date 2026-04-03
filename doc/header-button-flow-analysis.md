# 헤더 버튼 흐름 분석

## 스크린샷 대응

- 첫 번째 스크린샷은 `수동(Manual)` 모드에 대응한다.
  - 상단의 `감지`, `인식`, `번역`, `분할`, `정리`, `렌더링` 버튼이 활성화된다.
  - `모두 번역(Translate All)` 버튼은 비활성화된다.
- 두 번째 스크린샷은 `자동(Automatic)` 모드에 대응한다.
  - 상단의 `감지`, `인식`, `번역`, `분할`, `정리`, `렌더링` 버튼이 비활성화된다.
  - `모두 번역(Translate All)` 버튼은 활성화된다.

## 헤더 UI 구성

- 헤더 컨트롤은 [workspace.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/main_window/builders/workspace.py)에서 생성된다.
- `self.hbutton_group`에는 아래 6개 버튼이 이 순서대로 들어간다.
  - `Detect`
  - `Recognize`
  - `Translate`
  - `Segment`
  - `Clean`
  - `Render`
- `self.manual_radio`, `self.automatic_radio`, `self.translate_button`도 같은 헤더 레이아웃에 있다.

## 모드 전환

- `Manual` 라디오는 `controller.manual_mode_selected()`에 연결된다.
  - `enable_hbutton_group()` 호출
  - `translate_button` 비활성화
  - `cancel_button` 비활성화
- `Automatic` 라디오는 `controller.batch_mode_selected()`에 연결된다.
  - `disable_hbutton_group()` 호출
  - `translate_button` 활성화
  - `cancel_button` 활성화

즉, 스크린샷에서 보이는 헤더 버튼의 동작 자체는 원래부터 올바르게 구현되어 있었고, 이번 작업에서도 그대로 유지했다.

## 버튼 연결 구조

6개의 수동 헤더 버튼은 모두 [controller.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/controller.py)에 그대로 연결되어 있다.

- `Detect` -> `controller.block_detect()` -> `ManualWorkflowController.block_detect()`
- `Recognize` -> `controller.ocr()` -> `ManualWorkflowController.ocr()`
- `Translate` -> `controller.translate_image()` -> `ManualWorkflowController.translate_image()`
- `Segment` -> `controller.load_segmentation_points()` -> `ManualWorkflowController.load_segmentation_points()`
- `Clean` -> `controller.inpaint_and_set()` -> `ManualWorkflowController.inpaint_and_set()`
- `Render` -> `TextController.render_text()`

`Translate All`은 별도의 자동 경로를 탄다.

- `Translate All` -> `controller.start_batch_process()`
- 일반 페이지 -> `pipeline.batch_process()`
- 웹툰 모드 -> `pipeline.webtoon_batch_process()`

## 수동 모드 흐름 요약

### 감지

- 현재 페이지 한 장:
  - `pipeline.detect_blocks()`
  - `pipeline.on_blk_detect_complete()`
- 여러 페이지 선택:
  - `ManualWorkflowController.block_detect()`가 페이지별로 감지를 수행하고,
  - 결과 사각형을 `image_states[file]["viewer_state"]["rectangles"]`에 저장한다.

### 인식

- 현재 페이지 한 장:
  - `pipeline.OCR_image()` 또는 `pipeline.OCR_webtoon_visible_area()`
- 여러 페이지 선택:
  - `ManualWorkflowController.ocr()`가 페이지별 OCR을 수행하고 `blk_list`를 갱신한다.

### 번역

- 현재 페이지 한 장:
  - `pipeline.translate_image()` 또는 `pipeline.translate_webtoon_visible_area()`
  - 이후 `ManualWorkflowController.update_translated_text_items()`
- 여러 페이지 선택:
  - `ManualWorkflowController.translate_image()`가 선택 페이지를 번역하고,
  - 이후 `update_translated_text_items()`가 보이는 텍스트 아이템을 다시 줄바꿈하고 스타일을 적용한다.

### 분할

- `ManualWorkflowController.load_segmentation_points()`
- 라이브 사각형과 텍스트 아이템을 지운 뒤,
- `inpaint_bboxes`를 계산하고,
- 분할 스트로크 상태를 복원한다.

### 정리

- `ManualWorkflowController.inpaint_and_set()`
- 단일 페이지는 `pipeline.inpaint()`를 사용한다.
- 여러 페이지는 `inpainting.inpaint_page_from_saved_strokes()`를 사용한다.

### 렌더링

- `TextController.render_text()`
- 단일 페이지:
  - 번역 문구를 포맷팅하고
  - `manual_wrap()`을 호출하고
  - `blk_rendered`를 발생시키고
  - `TextBlockItem`을 만든다.
- 여러 페이지:
  - `pyside_word_wrap()`으로 줄바꿈하고
  - 각 페이지의 `text_items_state`를 저장한다.

## 자동 모드 흐름 요약

### 모두 번역

- `controller.start_batch_process()`가 설정을 검증한 뒤 아래 경로로 분기한다.
  - 일반 페이지 -> `pipeline.batch_process()`
  - 웹툰 모드 -> `pipeline.webtoon_batch_process()`

### 일반 배치

- `pipeline.batch_process()`는 아래 단계를 수행한다.
  - 감지
  - OCR
  - 인페인팅
  - 번역
  - 렌더링 상태 생성
- 최종 렌더 결과는 `image_states[image_path]["viewer_state"]["text_items_state"]`에 직렬화된다.

### 웹툰 배치

- `pipeline.webtoon_batch_process()`는 가상 페이지를 순회한다.
- 렌더 결과는 동일한 `text_items_state` 개념으로 저장된다.
- 최종 저장은 `ImageSaveRenderer`를 사용한다.

## 안전한 변경 경계

현재 헤더/모드 구조를 깨뜨리지 않으려면 다음 원칙을 지켜야 했다.

- [controller.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/controller.py)의 기존 버튼 연결은 유지한다.
- `manual_mode_selected()`와 `batch_mode_selected()`의 역할은 유지한다.
- `ManualWorkflowController`의 수동 진입 함수명과 흐름은 유지한다.
- `start_batch_process()`를 자동 처리의 유일한 진입점으로 유지한다.
- 새 동작은 헤더 라우팅이 아니라 렌더 설정 해석과 텍스트 상태 생성 계층에 넣는다.

## `codex/header-render-policy` 브랜치의 구현 상태

이 브랜치는 위에서 분석한 헤더 라우팅을 그대로 유지한다.

유지된 점:

- [workspace.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/main_window/builders/workspace.py)는 같은 헤더 버튼 세트를 계속 만든다.
- [controller.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/controller.py)는 6개 수동 버튼과 `Translate All`을 여전히 같은 진입점으로 연결한다.
- `manual_mode_selected()`와 `batch_mode_selected()`는 하위 파이프라인을 바꾸지 않고, 활성/비활성 상태만 바꾼다.

헤더 라우팅을 바꾸지 않고 추가된 점:

- 오른쪽 렌더 패널에 색상 강제, 상하 정렬, 명시적 윤곽선 ON/OFF UI를 추가했다.
- 기존 전역 윤곽선 불리언은 유지하면서, 화면에는 더 분명한 `OFF | ON` 컨트롤을 보여 준다.
- 공통 렌더 정책 헬퍼를 만들어 아래 경로가 같은 규칙을 쓰게 했다.
  - `TextController.render_text()`
  - `ManualWorkflowController.update_translated_text_items()`
  - `pipeline.batch_process()`
  - `pipeline.webtoon_batch_process()`

실질적인 결과:

- 스크린샷에서 보인 모드별 헤더 동작은 그대로다.
- 새 기능은 모두 오른쪽 렌더 패널 뒤쪽 로직에 들어갔고, 헤더 버튼 구조는 건드리지 않았다.
