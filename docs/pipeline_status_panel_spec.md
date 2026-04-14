# Pipeline Status Panel Spec

## Goal

기존 자동번역 modal dialog와 흩어진 비차단 메시지를 좌하단 고정 상태 패널 하나로 정리한다.

## Main Components

- `PipelineStatusPanel`
  - 좌하단 고정
  - 직사각형 패널
  - 최대 크기: 화면의 1/4
  - 드래그 이동, 드래그 리사이즈
  - 최소화/복원 시 직전 geometry 유지
- `PipelineInteractionOverlay`
  - 자동 파이프라인 중 메인 앱 위에 약한 반투명 레이어 표시
  - 일반 입력 차단
  - 상태 패널은 계속 상호작용 가능

## Panel Layout

- 좌측
  - 상태 제목
  - 서비스명
  - 페이지 진행률
  - 파일명
  - ETA/경과 시간
  - 최근 메시지
- 우측
  - 최신 완료 결과 미리보기 1장
- 하단
  - 상태별 액션 버튼
- 펼침 영역
  - 로그/경고/완료 내역

## State Buttons

- 실행 중: `Cancel`, `Report`
- 실패: `Retry`, `Settings`, `Close`
- 완료: `Report`, `Open Output`, `Close`

## Passive Message Routing

메인 작업 창의 비차단 알림은 상태 패널로 모은다.

- 자동 파이프라인 진행/완료
- 완료 성공 메시지
- 일반 `info`, `warning`, `success`
- batch skipped summary
- TXT/MD 자동 export 실패
- 다운로드 진행 안내

아래는 계속 modal로 유지한다.

- 확인이 필요한 질문형 dialog
- 저장/복구/덮어쓰기 confirm dialog
- traceback 복사가 필요한 치명적 오류 dialog

## Preview Rules

- 최신 완료 결과 1장만 유지
- 기본 미리보기 경로는 `processing_summary["translated_image_path"]`
- 새 preview cache를 만들지 않는다
- 클릭 시 OS 기본 뷰어로 연다

## Notifications

- 자동 파이프라인 정상 종료 시 완료 알림음 1회
- 기본값은 system sound
- 사용자 정의는 저장소 `music/*.wav`만 허용
