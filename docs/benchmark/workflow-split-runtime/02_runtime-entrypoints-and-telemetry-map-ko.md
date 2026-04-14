# Workflow Split Runtime Runtime Entry Points And Telemetry Map

## 문서 목적

- Requirement 1과 Requirement 2 구현 시 실제 제품 코드 진입점을 고정한다.
- 계측 추가 위치와 구조화 이벤트 포인트를 미리 잠근다.
- 아이디어 착안자: 사용자

## 현재 제품 진입점

### 1. 배치 전체 실행 중심

- [batch_processor.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/pipeline/batch_processor.py:730)
  - `detect_start`
  - `ocr_start`
  - `inpaint_start`
  - `translate_start`
  - `render_start`
  - `page_done`
- 현재는 페이지별로 위 순서를 모두 소화한 뒤 다음 페이지로 넘어간다.

### 2. OCR 런타임 진입점

- [modules/ocr/processor.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/modules/ocr/processor.py:36)
  - `OCRProcessor.initialize()`에서 runtime manager 호출
- [modules/ocr/local_runtime.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/modules/ocr/local_runtime.py:96)
  - `ensure_engine()`에서 compose up / health wait / reuse / shutdown 처리

### 3. Gemma 런타임 진입점

- [modules/translation/processor.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/modules/translation/processor.py:38)
  - `Translator.__init__()`에서 runtime manager 호출
- [modules/translation/local_runtime.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/modules/translation/local_runtime.py:107)
  - `ensure_server()`에서 compose up / health wait / reuse 처리

### 4. 설정 저장 진입점

- [settings_page.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/settings/settings_page.py:580)
  - `get_all_settings()`
- [settings_page.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/settings/settings_page.py:658)
  - `save_settings()`
- [settings_page.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/settings/settings_page.py:728)
  - `load_settings()`

## Requirement 1 계측 체크포인트

아래 항목은 하네스 요구를 제품 이벤트/로그로 바꾼 체크포인트다.

1. batch start
2. detect stage start/end
3. OCR compose up start/end
4. OCR health wait start/end
5. OCR runtime reuse hit
6. MangaLMM ready
7. PaddleOCR VL ready
8. OCR actual start/end
9. OCR stage end
10. Gemma compose up start/end
11. Gemma health wait start/end
12. Gemma runtime reuse hit
13. translation actual start/end
14. translation stage end
15. inpaint stage start/end
16. render/export start/end
17. page done
18. batch done
19. timeout / retry / restart

## 예상 구현 위치

### 제품 코드

- `pipeline/batch_processor.py`
  - legacy vs stage-batched workflow 분기
  - stage aggregation
  - benchmark event emission 확장
- `modules/ocr/local_runtime.py`
  - OCR runtime lifecycle 이벤트 추가
  - Requirement 2에서 dual-resident 정책 분기
- `modules/translation/local_runtime.py`
  - Gemma lifecycle 이벤트 추가
- `app/ui/settings/settings_page.py`
  - `workflow_mode` 저장/로드
- `app/ui/settings/settings_ui.py`
  - workflow mode UI

### benchmark/lab 코드

- future family runner
- timing summary builder
- docker lifecycle breakdown collector
- report generator

## 현재 판단

가장 안전한 방식은 기존 `mark_processing_stage()`와 `report_runtime_progress()`를 버리지 않고, stage-batched workflow에서도 동일 스키마를 유지하면서 runtime lifecycle 세부 이벤트만 더 얹는 것이다. 이렇게 해야 상태 패널과 batch report가 회귀 없이 동작하고, benchmark harness도 generic event surface만 읽으면 된다.
