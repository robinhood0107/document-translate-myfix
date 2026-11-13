# 저장소 작업과 검증 보고 계약

이 문서는 issue, PR, CI, 로컬 검증 실행에서 관측 가능한 근거를 기록하는 기준이다. GitHub의 issue·PR timeline과 CI job 결과를 원본 이벤트로 사용한다. 모든 수동 명령, GUI 동작, 앱 내부 작업을 자동 수집한다고 가정하지 않는다.

## Issue와 PR

- Issue 하나에는 독립적으로 확인하거나 구현할 결함, 작업 또는 실험 하나를 둔다. 같은 fingerprint가 이미 있으면 새 issue를 만들지 말고 기존 issue를 갱신한다. Fingerprint는 컴포넌트, 진입점, 깨진 불변 조건, 관측 증상으로 구성한다.
- Issue는 측정 분모와 단위, 실패·미판정·미측정 수, 근거 상태, 검증 명령과 결과, 영향, 되돌리기 또는 복구, 실행하지 않은 점검을 기록한다. 적용되지 않는 항목은 이유를 적는다.
- PR은 리뷰와 검증이 가능한 논리 단위 하나로 묶는다. 완전히 해결하는 issue에는 `Closes #N`, 후속 작업이 남거나 부분 해결이면 `Refs #N`을 쓴다.
- PR 본문 validator는 issue 연결, 측정 분모, 검증 명령, 실패·미판정·미측정, 근거 상태, 영향, 되돌리기, 미실행 항목을 확인한다. 이 검사는 본문 형식과 누락만 검사하며 결과의 기술적 타당성을 판정하지 않는다.
- PR 본문 검사는 가벼운 metadata workflow에서 `opened`, `reopened`, `synchronize`, `edited`, `ready_for_review`마다 다시 실행한다. 이 workflow는 PR merge checkout의 validator를 실행하며 `contents: read`만 부여하고 checkout credential을 남기지 않는다. main/develop ruleset의 required context `validate-pr-metadata`가 이 검사를 병합 gate로 만든다. 무거운 CI의 branch-flow 검사는 별도 유지한다.

## CI 보고와 공개 범위

- 필수 CI job 결과를 고정된 job ID 허용 목록으로 정제해 JSON과 Markdown으로 만든다. 공개 보고에는 Git SHA, 이벤트 종류, job 상태, 생성 시각, 계산 가능한 안전한 수치만 둔다.
- `needs` 입력에서는 `result`만 읽고 임의 출력, argv, 환경변수, 표준 출력·오류, 로컬 경로, 원문 설명을 보고에 복사하지 않는다. 보고 스키마는 알 수 없는 필드와 job 상태를 거부한다.
- 측정할 수 없는 값은 `null`과 `unavailable`, `not_enabled`, `not_applicable` 또는 `failed` 상태 및 사유 코드로 표시한다. `null`을 0으로 바꾸지 않는다. 값마다 단위와 실제 source를 함께 둔다.
- CI는 수집과 managed-run finalization이 성공하고 JSON·Markdown artifact가 모두 존재하며 JSON 검증까지 통과해야 report guard가 성공한다. 생성, summary, artifact 또는 검증이 실패하면 report guard는 실패한다. 워크플로가 취소되거나 강제 종료되어 report guard가 완료되지 않으면 GitHub의 `cancelled` 상태가 남고 완료로 처리되지 않는다.
- PR comment는 이 publisher workflow가 기본 브랜치에 배포된 뒤, 신뢰된 `workflow_run` publisher가 정제 artifact와 실제 workflow conclusion을 대조하고 현재 열린 PR의 head SHA와 base branch를 당시 실행값과 대조한 뒤 stable marker comment를 갱신한다. non-success workflow가 `completed` report를 내면 게시를 거부한다. PR head나 base가 바뀌었거나 PR이 닫혔으면 오래된 comment를 건너뛴다. PR 코드를 checkout하거나 실행하지 않으며 `pull_request_target`을 사용하지 않는다. 게시가 실패하면 publisher workflow의 실패 상태를 확인한다. PR 연결 정보나 쓰기 권한이 없으면 comment는 게시되지 않으며 CI summary와 artifact만 근거로 삼는다.
- 최근 CI에 추가하는 품질 검사 비용은 5분 이내를 목표로 한다. 실제 비용을 측정하기 전에는 목표를 보장값으로 말하지 않는다.

## 관리형 로컬 실행

- 기본 managed run은 ignored private artifact harness 아래에서 시작·종료 이벤트, JSON 보고서, Markdown 보고서를 만든다. 아카이브 무결성 검증은 이벤트와 manifest가 같은 종결 상태인지 확인하며, intact한 `failed`, `cancelled` 등도 무결성 검증을 통과할 수 있다. 무결성 검증 통과는 작업 성공 판정이 아니다. 성공 실행은 status가 `completed`인 경우뿐이다. 강제 종료되어 `running`에 남은 실행은 종결 검증을 통과하지 못한다.
- 종료 상태는 `completed`, `failed`, `cancelled`, `timed_out`, `interrupted`, `report_failed`, `publish_failed`를 구분한다. 보고 생성 실패는 `report_failed`로 남기거나 manifest 기록도 불가능하면 종결되지 않은 상태로 두며, 둘 다 완료 검증을 통과하지 못한다.
- `complete()`와 `fail()`은 이름이 붙은 `metrics`를 받아 JSON 숫자형 실측값을 기록할 수 있다. `bool`, 문자열, NaN, 무한대는 실측값으로 받지 않으며, 측정하지 않은 값은 `null`과 사유 코드로 기록한다. 아카이브 무결성 검증은 성공 판정과 분리하고, 성공한 실행은 status가 `completed`인 경우뿐이다.
- 새 manifest는 버전 2를 쓴다. 기존 버전 1의 완료 manifest는 보고서 파일이 없어도 읽을 수 있으며, 이전 raw 결과와 manifest를 다시 쓰지 않는다. 버전 1을 포함해 `running` 상태는 완료 검증에서 거부한다.
- 전체 argv, 표준 출력·오류 등 private 진단은 명시된 private archive에만 둔다. 자동 보고서와 CI artifact에는 포함하지 않는다. 명시적 output directory를 사용한 도구 실행은 managed 기본 실행과 별도다.

## 품질과 자원 수치

- 분모는 처리 블록 수가 아니라 원본에서 평가 대상으로 확정한 필수 source unit을 기준으로 한다. source와 결과의 대응, 누락·추가·중복·분할·합침, 오삭제, 실패, 미판정, 미측정을 각각 남긴다.
- 번역·OCR·렌더 결과를 선택할 때는 필수 내용 보존과 원본 대응을 먼저 확인한다. 품질과 처리 성능은 서로 다른 축으로 기록하고 Pareto 선택을 설명한다. 블록 수는 진단 수치일 뿐 품질의 유일한 점수나 분모가 아니다.
- 메모리 점검에는 측정 대상과 시각, process RSS 현재·최대값과 sample 수, GPU별 VRAM·util sample, CPU 또는 WSL swap 사용 여부, 정리 뒤 관측 결과를 적는다. 계측이 꺼졌거나 지원되지 않으면 0 대신 `null`과 상태·source·이유를 남긴다.
- 제품 코드에서 기존 Factory, Adapter, Command 경로와 resource lifecycle을 먼저 확인하고 재사용한다. 보고·순위·preset·raw 결과 처리는 제품 business logic에 넣지 않는다.

## 검증 선택

- 변경 위험에 맞춰 contract, differential, metamorphic, concurrency, fault-injection, resource-leak 검사를 고른다. 같은 불변 조건을 이미 검증하는 테스트를 중복 추가하지 않는다.
- 구현 전 깨지는 계약 반례를 확인하고, 최소 수정 후 같은 반례가 통과하는지 확인한다. 실패·취소·미측정·쓰기 실패를 정상 완료와 분리한다.
- 앱의 현재 run report는 성공 렌더 경로 중심이며 취소·예외를 모두 포괄하지 않는다. 이 N0 계약은 앱 내부 모든 page 작업이나 사용자 조작의 완료 증거를 대신하지 않는다.
