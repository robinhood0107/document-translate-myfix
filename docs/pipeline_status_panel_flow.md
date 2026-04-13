# Pipeline Status Panel Flow

## Automatic Run Start

1. 사용자가 `Translate All`, `One-Page Auto`, `Retry Failed` 중 하나를 실행한다.
2. 설정 검증과 실행 확인이 끝나면 상태 패널을 `running` 상태로 연다.
3. `PipelineInteractionOverlay`를 활성화한다.
4. 기존 `AutomaticProgressDialog`는 띄우지 않는다.

## Runtime Progress

1. 파이프라인은 기존 `report_runtime_progress` 흐름으로 이벤트를 보낸다.
2. 컨트롤러는 이벤트를 상태 패널 payload로 정규화한다.
3. 패널은 요약 정보와 펼침 로그를 갱신한다.
4. 페이지 완료 이벤트에 `preview_path`가 있으면 최신 완료 이미지 미리보기를 교체한다.

## Completion

1. 마지막 페이지가 끝나면 패널을 `done` 상태로 전환한다.
2. 성공 메시지를 패널에 남긴다.
3. system sound 또는 설정된 `music/*.wav`를 1회 재생한다.
4. 사용자는 `Report`, `Open Output`, `Close` 중 하나를 선택할 수 있다.
5. overlay는 완료 상태 전환 후 해제한다.

## Failure

1. 치명적 실패가 발생하면 패널을 `failed` 상태로 전환한다.
2. 상세 메시지는 펼침 영역에 남긴다.
3. 패널 버튼을 `Retry`, `Settings`, `Close`로 바꾼다.
4. traceback 복사가 필요한 경우에만 modal error dialog를 유지한다.

## Cancel

1. 사용자가 `Cancel`을 누르면 취소 요청을 worker에 전달한다.
2. 취소 완료 시 패널을 `cancelled` 상태로 바꾼다.
3. overlay를 해제한다.

## Future Hook

- 완료 지점에는 향후 `ntfy` notifier를 붙일 수 있는 stub hook을 둔다.
- 실제 전송 구현은 이번 단계 범위 밖이다.
