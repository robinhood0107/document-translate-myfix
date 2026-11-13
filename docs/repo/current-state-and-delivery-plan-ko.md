# N0 구현 현황과 후속 전달 계획

이 문서는 저장소의 workflow reporting contract와 후속 작업의 책임 경계를 정리한다. 현재 게시·검증·병합 상태는 [전체 추적 #315](https://github.com/robinhood0107/document-translate-myfix/issues/315)와 [N0 추적 #316](https://github.com/robinhood0107/document-translate-myfix/issues/316)의 커밋·PR·CI 근거를 따른다. 코드 작성, 검사 통과, 게시, 병합을 별도 상태로 기록한다.

## 프로그램 책임 경계

| 영역 | 진입점·소유 경로 | 책임 |
|---|---|---|
| 앱 진입과 UI 조정 | `comic.py`, `controller.py`, `app/controllers/` | 준비 상태 확인, 사용자 작업과 실행 연결 |
| 페이지 처리 | `pipeline/` | 단일·stage·웹툰 처리, 취소, 출력 완결성 |
| 인식과 번역 | `modules/ocr/`, `modules/translation/` | 선택한 엔진의 입출력 계약, 캐시와 runtime 연결 |
| 이미지 편집 | `modules/masking/`, `modules/inpainting/`, `modules/rendering/` | 보호 영역, 편집 mask, 배경 복원과 텍스트 배치 |
| 실행 자원 | `modules/utils/local_llama_router/`, `modules/utils/gpu_handoff.py` | 모델 소유권, 단계 간 자원 교대와 반환 |
| 사용자 작업 보존 | `app/projects/`, `app/ui/commands/` | 프로젝트·checkpoint, 편집과 undo 상태 |
| 설치와 실행 | Windows setup/run 진입점, `scripts/bootstrap_windows.ps1` | 설치 소유권과 실행 전 준비 상태의 분리 |
| 수치와 보고 | telemetry, `modules/utils/run_report.py`, validation harness | 실제 수집값과 미수집 상태, 실행·검증 근거 |

이 표는 책임 경계이며 각 경로의 미커밋 변경이나 품질 검증이 끝났다는 뜻이 아니다. 누적 제품 변경은 #317–#323의 재현·검증·PR 단위로 나눠 추적한다.

## 현재 구현 범위

- managed validation run은 시작·종료 event journal과 JSON·Markdown 보고서를 생성한다. 새 실행은 manifest schema 2를 사용하고, `running`은 완료 검증에서 거부한다. 기존 schema 1 완료 manifest는 그대로 읽으며 raw 결과나 완료 manifest를 다시 쓰지 않는다.
- CI는 통합 브랜치 push와 허용된 base branch PR을 검사하고, stale run을 취소한다. 기존 required job ID는 유지하고 PR report guard를 추가한다.
- CI report guard는 모든 선행 job 결과를 받아 정제 JSON·Markdown, job summary, sanitized artifact를 생성하고 검증한다. source로 확인되지 않은 CI duration은 `null`과 `unavailable`로 기록한다.
- PR 본문 계약 검사는 issue link, 분모, 검증 명령, 실패·미판정·미측정, 근거 상태, 영향, 되돌리기, 미실행 내용을 검사한다. issue form은 bug, task, experiment를 분리한다.
- Ruff는 고정된 품질 의존성으로 E9/F 규칙을 변경된 Python 소스에 적용한다. vendored·generated 파일은 대상에서 제외하고 저장소 전체에 대한 legacy baseline을 만들지 않는다.
- PR comment publisher는 기본 브랜치에 도달한 뒤 실행되는 `workflow_run` 경로로 구성했다. 정제 artifact를 다시 검증하고 안정 marker를 upsert하며, PR 코드는 publisher 권한으로 실행하지 않는다. 그 전까지는 CI summary와 sanitized artifact가 보고 근거다.

## 미완료 범위

- `pipeline/stage_batched_processor.py`와 `modules/utils/run_report.py`의 앱 내부 보고는 성공 렌더 경로 중심이다. 취소·예외 종료와 보고서 쓰기 실패를 앱 처리 상태와 연결하는 작업은 N0에 포함하지 않았다.
- 앱 보고서의 품질 분모, 원본 대응, logical request와 retry·관리 요청 분리, CPU/RSS/GPU 계측, cleanup 관측은 소스가 현재 보유한 값부터 별도로 연결해야 한다. 보고 근거 없이 완료를 표시하지 않는다.
- CI가 전체 취소 또는 runner 강제 종료를 만나면 해당 `report-guard` required check는 통과하지 않는다. GitHub의 native workflow/job 상태가 취소를 기록한다. 모든 실행 환경에서 별도 완료 문서를 복구해 쓴다고 보장하지 않는다.
- PR comment 게시가 실패하면 별도 publisher workflow 상태가 실패한다. CI의 summary와 sanitized artifact는 생성·검증 상태를 나타내며 comment 게시 성공으로 오인하지 않는다.

## 후속 전달 순서

1. N0 코드와 정책 파일을 독립 PR로 검토하고 report guard 및 issue 본문 검사를 required path에서 확인한다.
2. 병합 뒤 실제 CI 추가 시간을 측정하고, main/develop ruleset의 기존 context와 새 report guard context를 대조한다.
3. 앱 terminal report 작업은 기존 pipeline lifecycle과 source-of-truth 경계를 확인한 별도 이슈에서 진행한다. 출력 생성, 취소, 오류, cleanup 상태를 실제 page 결과와 분리해 검증한다.

검증 문서에는 성공뿐 아니라 실패, 취소, 미판정, 미측정, 미실행을 남긴다. 새 UI나 PSD 처리 기능은 이 변경에 추가하지 않았다.
