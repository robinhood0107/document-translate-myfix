# Runtime/UI 버그 헌팅 체크리스트

이 문서는 사소한 버그와 의도되지 않은 동작을 차근차근 찾고, 재현 가능하게 기록하고, 필요한 경우에만 국소적으로 고치기 위한 감사 장부다. 항목은 삭제하지 않고 상태를 갱신한다.

## 검사 원칙

- 보수적으로 판단한다. 현재 정상 동작을 깨뜨릴 수 있는 추측성 수정은 하지 않는다.
- root cause가 확인되기 전에는 코드 수정 PR로 넘기지 않는다.
- 수정은 원인 파일과 회귀 테스트에 국소화한다. 리팩터링, 문구 변경, 출력 규칙 변경은 해당 버그 해결에 꼭 필요한 경우만 포함한다.
- 데이터 손실, 대량 skip, 저장/로드 호환성, 전체 파이프라인 구조 변경이 의심되면 수정 전에 별도 보고한다.
- 감사 중 발견한 의심 항목은 먼저 `candidate`로 기록하고, 재현/로그/불변식 위반/사용자 흐름 증거 중 하나 이상을 확보한 뒤 `confirmed`로 올린다.
- 체크리스트 항목은 삭제하지 않는다. 완료, 보류, 오탐도 근거와 함께 남긴다.
- 이미지 또는 결과물 품질과 관련된 항목은 반드시 테스트 이미지를 만들어 실제 산출물을 확인한다.
- 이미지 또는 결과물 변경 PR은 결과 이미지/아카이브/디버그 산출물 경로를 사용자에게 전달하고, 병합 전 사용자 검토를 요청한다.
- 문서/브랜치/커밋/PR/검증은 항상 `rules.md`와 로컬 hook 정책을 따른다. 예를 들어 문서 작업도 `docs/*` 브랜치가 아니라 허용 prefix인 `chore/*`를 사용한다.

## 상태 규칙

| 상태 | 의미 |
| --- | --- |
| `[ ]` | 아직 검사 전 |
| `[~]` | 조사 중 또는 candidate 상태 |
| `[x]` | 확인 및 필요한 조치 완료 |
| `blocked` | 외부 조건 또는 사용자 판단 필요 |
| `not-a-bug` | 의도된 동작 또는 수정 불필요로 판정 |

## 검증 로그 규칙

- 감사 로그 루트: `/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate_validation_logs/2026-05-29/bug-hunt/<audit-id>/`
- 각 명령은 `2>&1 | tee <logfile>`로 저장한다.
- 로그와 빌드 산출물은 커밋하지 않는다.
- 버그 수정 PR 본문에는 실행 명령, pass/fail 요약, 로그 경로를 남긴다.
- 이미지/결과물 검증 로그에는 입력 테스트 이미지, 생성 결과물, 비교 기준, 사용자가 검토해야 할 파일 경로를 같이 남긴다.

## 전수검사 프로토콜

현재 감사 대상 런타임 코드는 `app`, `pipeline`, `modules`, `imkit` 아래 Python 파일 347개다. AST 인벤토리 기준으로 `app` 149개 파일/2618개 함수, `modules` 175개 파일/1660개 함수, `pipeline` 17개 파일/193개 함수, `imkit` 6개 파일/36개 함수를 순차 검토한다.

1. 각 BH 항목을 시작할 때 관련 파일 목록을 먼저 만든다.
2. 각 파일은 AST로 class/function/method 목록을 뽑고, public entrypoint와 signal/callback/write path를 표시한다.
3. 함수 단위로 입력, 출력, 상태 변경, 예외 처리, 파일 쓰기, UI signal, cancel/retry 조건을 추적한다.
4. 단순 검색만으로 끝내지 않는다. `except Exception`, `emit`, `write_image`, `rmtree`, `mkdtemp`, `setEnabled`, `mark_project_dirty`, `image_skipped`, `page_failed` 같은 위험 패턴은 호출자와 호출 결과까지 따라간다.
5. 검토한 파일은 감사 로그에 파일명, 확인한 함수 범위, 발견 사항, 테스트 필요 여부를 남긴다.
6. vendor/adapted 코드도 runtime 호출 경로에 닿으면 검사한다. 단, 큰 외부 코드 리팩터가 필요하면 즉시 사용자에게 보고하고 별도 계획으로 분리한다.

## 이미지/결과물 검토 규칙

- 대상: inpaint, mask, render, OCR crop/overlay, export output, archive output, PSD import/export, webtoon stitched image, series output, preview/debug artifact.
- 최소 검증:
  - 작은 synthetic 이미지 1개
  - 경계/투명도/비정상 bbox 등 재현용 이미지 1개
  - 필요 시 실제 사용자 흐름에 가까운 샘플 이미지 1개
- 결과물은 repo 밖 validation log 폴더에 저장하고 커밋하지 않는다.
- PR에는 결과물 경로와 확인 포인트를 적고, "사용자 검토 필요"를 명시한다.
- 사용자가 결과물을 확인하기 전에는 시각 품질이나 출력 포맷 변경 PR을 병합하지 않는다. 단, 문서만 바꾸거나 순수 내부 상태 테스트만 바꾸는 PR은 이 규칙에서 제외한다.

## 우선 코드 흐름

| 흐름 | 따라갈 주요 entrypoint | 확인할 상태 |
| --- | --- | --- |
| 일반 batch | `ComicTranslate.batch_process`, `BatchProcessor.process` 계열 | page stage, skip/report, output token, cancel/retry |
| stage-batched | `StageBatchedProcessor.process` 계열 | stage별 실패 격리, context 상태, per-page 결과 |
| webtoon batch | `webtoon_batch.processor`, `ChunkMixin`, `RenderMixin` | chunk 실패, stitched image, page offset, render 저장 |
| 수동 workflow | `ManualWorkflowController` | detect/OCR/translate/inpaint 후 dirty/project state |
| 이미지 컨트롤러 | `ImageStateController` | page state, skip message, render dirty, sorting/source mapping |
| 프로젝트 저장/로드 | `project_state`, `project_state_v2`, `series_state_v1` | lazy blob, temp dir, patch/order, export source, recovery |
| 출력/아카이브 | `automatic_output`, `export_paths`, `file_handler` | result/log naming, archive materialization, source stem fallback |
| 캔버스/명령 | `app/ui/commands`, `canvas/save_renderer`, `text_item` | undo/redo, text box state, brush/inpaint patch, save render |
| 런타임 설정 | settings pages, local OCR/translation runtimes | setup failure message, cancel wait, model availability |

## 검사 상태표

| ID | 상태 | 영역 | 위험도 | 검사할 증상/의도치 않은 동작 | 재현/검사 방법 | 관련 경로 | 테스트 필요 | Issue/PR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BH-001 | `[ ]` | batch/stage-batched | 높음 | stage-batched와 legacy batch의 실패 기록/skip/retry 동작이 다르게 남는 경로가 더 있는지 확인 | 실패 stage별 테스트와 report entry 비교 | `pipeline/batch_processor.py`, `pipeline/stage_batched_processor.py` | 예 | - |
| BH-002 | `[ ]` | inpaint/mask boundary | 높음 | bbox/polygon/mask 좌표가 음수, width/height 경계, zero-area에서 조용히 잘못된 영역을 만지는 경로 확인 | 좌표 경계 회귀 테스트와 existing mask tests 확장 가능성 검토 | `imkit/transforms.py`, `modules/masking`, `modules/inpainting` | 예 | - |
| BH-003 | `[ ]` | report/UI skip mapping | 중간 | 실패 stage가 일반 `Page processing failed`로 묻히는 잔여 reason 확인 | `image_skipped.emit` reason 전수 검색과 UI/report mapping 대조 | `app/controllers/image.py`, `app/controllers/batch_report.py`, `pipeline` | 예 | - |
| BH-004 | `[ ]` | cancellation/retry | 중간 | 사용자가 취소한 작업이 실패/skip으로 기록되거나 retry 대상에 섞이는 경로 확인 | cancel tests, retry report payload 검사 | `pipeline`, `app/controllers/batch_report.py`, `app/controllers/image.py` | 예 | - |
| BH-005 | `[ ]` | project save/load | 높음 | `.ctpr`/`.ctprv2` 저장 후 재열기에서 page mapping, patches, output preferences, lazy blob 상태가 깨지는 경로 확인 | round-trip tests와 lazy materialization tests 검토 | `app/projects/project_state.py`, `app/projects/project_state_v2.py` | 예 | - |
| BH-006 | `[ ]` | export/output naming | 중간 | source/project stem fallback이 temp page, archive child, project reload 후 흔들리는 경로 확인 | output resolver tests와 archive/project source records 대조 | `modules/utils/export_paths.py`, `modules/utils/automatic_output.py` | 예 | - |
| BH-007 | `[ ]` | webtoon pipeline | 중간 | chunk OCR/translation/render 실패가 page/report/preview 상태에 일관되게 반영되는지 확인 | webtoon chunk failure tests와 UI reason mapping 대조 | `pipeline/webtoon_batch`, `app/ui/canvas/webtoons` | 예 | - |
| BH-008 | `[ ]` | series queue | 중간 | series queue 실패/skip/cancel/open failed item 상태가 섞이거나 복구가 꼬이는 경로 확인 | series workspace runtime tests와 queue policy inspection | `app/controllers/series.py`, `app/projects/series_state_v1.py` | 예 | - |
| BH-009 | `[ ]` | undo/rerender state | 중간 | 텍스트 박스 수정/삭제/저장/재열기 후 dirty badge, rerender target, undo stack이 어긋나는 경로 확인 | editor/rerender tests와 project reload tests | `app/ui/commands`, `app/controllers/text.py`, `tests/test_text_box_commands.py` | 예 | - |
| BH-010 | `[ ]` | external model/runtime config | 중간 | OCR/translator/inpainter 외부 런타임 설정 실패가 사용자에게 모호하게 표시되거나 batch를 잘못 중단하는 경로 확인 | config/runtime tests, startup smoke, error message mapping | `app/ui/settings`, `modules/ocr`, `modules/translation`, `pipeline` | 예 | - |
| BH-011 | `[ ]` | temp/cache cleanup | 낮음 | 내부 temp/cache 정리가 너무 이르거나 늦어서 파일 누락 또는 불필요한 누수가 생기는 경로 확인 | cleanup path inspection과 targeted temp-dir tests | `controller.py`, `modules/utils/file_handler.py`, `app/projects` | 경우에 따라 | - |
| BH-012 | `[ ]` | UI 상태/문구 | 낮음 | 버튼 enabled 상태, 안내 문구, 상태 badge가 실제 처리 상태와 어긋나는 경로 확인 | headless smoke와 controller/UI tests | `app/ui`, `app/controllers` | 경우에 따라 | - |
| BH-013 | `[ ]` | image/result artifacts | 높음 | 이미지 결과물 변경이 테스트 이미지 없이 병합되는 경로 차단 | synthetic/edge/sample 이미지 산출물 생성 후 사용자 검토 요청 | `modules/utils/automatic_output.py`, `pipeline`, `app/ui/canvas` | 예 | - |
| BH-014 | `[ ]` | PSD import/export | 중간 | PSD import/export가 temp image, font fallback, layer order, output image를 조용히 손상시키는 경로 확인 | PSD 경로 AST 추적, 가능한 경우 테스트 PSD/이미지 fixture 산출물 확인 | `app/controllers/psd_importer.py`, `app/controllers/psd_exporter.py` | 예 | - |
| BH-015 | `[ ]` | canvas save/render | 높음 | 캔버스 저장과 final render가 text item, patch, alpha, RGB 변환을 다르게 처리하는 경로 확인 | 테스트 이미지 저장 결과 비교, render/save_renderer 호출 추적 | `app/ui/canvas/save_renderer.py`, `modules/rendering/render.py` | 예 | - |
| BH-016 | `[ ]` | text item state | 중간 | text box 생성/수정/삭제가 project state, undo stack, render_dirty와 불일치하는 경로 확인 | command AST 추적, existing text box tests 확장 | `app/ui/commands`, `app/controllers/text.py`, `app/ui/canvas/text_item.py` | 예 | - |
| BH-017 | `[ ]` | startup/home/recent projects | 낮음 | 시작 화면 recent project, missing file handling, window title 상태가 꼬이는 경로 확인 | startup/recent project tests와 controller path inspection | `app/ui/startup_home.py`, `app/controllers/projects.py` | 경우에 따라 | - |
| BH-018 | `[ ]` | source txt/md exchange | 중간 | TXT/MD export/import가 페이지명, 누락 페이지, 예상 밖 페이지를 잘못 매칭하는 경로 확인 | exchange tests, page matching AST 추적 | `app/controllers`, `modules/utils` | 예 | - |
| BH-019 | `[ ]` | runtime progress panel | 낮음 | progress panel preview/status/cancel/report 버튼이 실제 batch 상태와 어긋나는 경로 확인 | headless UI tests, event payload mapping 점검 | `app/ui/pipeline_status_panel.py`, `controller.py` | 경우에 따라 | - |
| BH-020 | `[ ]` | file materialization | 높음 | archive/project/lazy blob materialization이 원본 경로와 temp 경로를 혼동하는 경로 확인 | archive tests, project round-trip, export source mapping | `modules/utils/file_handler.py`, `app/path_materialization.py`, `app/projects` | 예 | - |

## 감사 진행 방식

1. `develop`을 최신화하고 감사용 문서/조사 브랜치에서 시작한다.
2. 한 번에 하나의 BH 항목만 조사한다.
3. 비파괴 조사로 코드 경로, 기존 테스트, 관련 이슈/PR을 확인한다.
4. 증거가 부족하면 `candidate`로만 남기고 수정하지 않는다.
5. 버그가 확정되면 GitHub issue를 만들고 `fix/<short-slug>` 브랜치에서 새 회귀 테스트와 함께 고친다.
6. PR이 merge되면 이 문서의 해당 행을 `[x]`로 바꾸고 issue/PR/log 경로를 남긴다.
7. 이미지/결과물 관련 항목은 결과물을 사용자에게 검토 위임하고, 검토 요청/결과를 해당 BH 항목에 남긴다.

## 버그 수정 PR 기준

- PR 하나는 위험도별 확인된 버그 하나만 다룬다.
- PR 본문에는 root cause, 사용자 영향, 수정 범위, 테스트 명령, 로그 경로, `Closes #<issue>`를 포함한다.
- 필수 검증:
  - branch-specific pytest
  - `.venv-win/Scripts/python.exe scripts/compile_translations.py --check`
  - `.venv-win/Scripts/python.exe scripts/validate_changed_python.py --all`
  - 필요한 경우 `.venv-win/Scripts/python.exe scripts/headless_smoke.py`
- UI 문구 변경 시 `resources/translations/ct_*.ts`와 `resources/translations/compiled/*.qm`을 갱신한다.
- Windows launcher/runtime 변경 시 `scripts/verify_windows_launchers.py`와 필요한 CUDA13 검증을 추가한다.
- 이미지/결과물 변경 시 테스트 이미지와 결과물을 validation log에 저장하고, PR merge 전 사용자 검토를 요청한다.

## 큰 문제 보고 기준

아래 조건 중 하나라도 해당하면 바로 사용자에게 보고하고 별도 계획을 만든다.

- 저장/로드 호환성을 깨뜨릴 가능성이 있는 변경
- 대량 page skip, 데이터 손실, 원본/결과 파일 오염 가능성
- batch, webtoon, series, editor를 동시에 바꾸는 구조 문제
- public output naming, project schema, archive materialization 규칙 변경
- 테스트로 충분히 격리하기 어려운 외부 런타임/모델 의존 문제
