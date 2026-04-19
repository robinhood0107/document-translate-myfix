# 시리즈 프로젝트 UI/UX 흐름

## 시작점

- Startup Home에 `New Series Project` 카드 추가
- 기존 open flow에서 `.seriesctpr` 인식

## 시리즈 생성

1. 폴더 선택
2. recursive scan
3. reorder/select popup
4. `.seriesctpr` 생성
5. series workspace 진입

## series workspace

- 좌측 미리보기 리스트 제거
- 중앙: queue board/list
- 우측 상단: `Queue Status`
  - 현재 상태
  - 현재 실행 중 항목
  - 다음 예정 항목
  - 마지막 실패 항목
  - 남은 재시도 횟수
  - 마지막 실행 시각
  - `Pause` / `Resume` / `실패 항목 열기`
- 우측 하단: `Last Queue Run`
  - 완료/실패/건너뜀 수
  - 총 실행 시간
  - 마지막 실행 시작/종료 시각
- 우측 패널 하단: 빠른 전역 실행 설정
- 좌상단: 뒤로가기 / 앞으로가기 / 트리 이동
- 우상단: `대기열대로 자동번역`
- running 상태 배지와 queue 상태 표시 제공
- 자동 번역 실행 중에는 queue 순서 변경, 항목 추가/제거, 시리즈 전역 설정 변경을 잠금
- 자동 번역 실행 중에는 뒤로가기 / 앞으로가기 / 트리 이동도 잠금
- 잠긴 상태는 비활성/흐림 처리와 안내 문구로 명확히 표시
- `Pause`는 현재 항목이 끝난 뒤에만 `paused` 상태로 전환한다.
- `paused` 상태에서는 board에서 `Resume`으로 이어서 실행하고, 실패 항목은 `실패 항목 열기`로 수동 진입한다.
- 시리즈 화면에는 `복구본 열림`, `세부 프로젝트 변경 미반영` 배지를 별도로 표시한다.

## child 진입

- row/card click -> child project 열기
- child는 기존 `.ctpr` workspace와 동일하게 동작
- 복귀 시 series queue 상태와 child 저장 상태를 동기화
- running 중 종료 후 recovery로 다시 열리면 자동 재개하지 않고 board 기반 `paused` 상태로 복구한다.

## 시각적 구분

- `Series Project` 배지
- child 진입 시 breadcrumb 또는 scope badge
