# 시리즈 프로젝트 하네스

## 목표

- 새 저장 확장자 `.seriesctpr`를 도입한다.
- 폴더 스캔으로 찾은 원본 파일과 기존 `.ctpr`를 하나의 시리즈 컨테이너에 묶는다.
- 시리즈 화면에서 대기열 순서 관리, child 프로젝트 진입, 자동번역 순차 실행을 지원한다.
- `.ctpr`와 `.seriesctpr` 저장 원자성을 함께 강화한다.

## 핵심 결정

- `.seriesctpr`는 SQLite 단일 컨테이너다.
- 각 시리즈 항목은 독립 child `.ctpr` blob으로 저장한다.
- 원본 파일은 시리즈 생성 시 즉시 child `.ctpr`로 변환해 내장한다.
- 기존 `.ctpr`도 시리즈 항목으로 import 가능하다.
- 시리즈 우측 패널은 언어/OCR/번역기/workflow/GPU/자동진행 빠른 설정만 노출한다.
- detector/inpainter/텍스트/렌더 상세 설정은 child 프로젝트에서만 조정한다.
- 자동 대기열은 순차 실행만 지원한다.
- 대기열 실패 정책은 `Settings > Series`에서 설정한다.
- 일반 프로젝트 히스토리와 시리즈 히스토리는 분리한다.
- 실행 중에는 queue 순서 변경, 항목 추가/제거, 시리즈 전역 설정 변경을 잠근다.
- 실행 중 live reorder는 제품 계획에서 제외한다.
- `Pause`는 현재 항목 완료 후 정지로만 지원한다.
- `Resume`은 paused 상태에서 남은 pending만 이어서 순차 실행한다.
- paused 또는 stop-on-failure 상태가 되면 board에서 `Resume`과 `실패 항목 열기`를 제공한다.
- 중복 원본 파일/`.ctpr` 추가는 차단하고 경고만 보여준다.
- recovery에서 `running` 상태로 복원되면 자동 재개하지 않고 `paused` 상태로 정규화한다.
- 시리즈 board 변경은 save-through이므로 `시리즈 자체 미저장` 배지는 기본적으로 두지 않는다.
- 대신 `세부 프로젝트 변경 미반영`, `복구본 열림` 상태만 명시적으로 드러낸다.
- 우측 패널에는 `Queue Status`와 `Last Queue Run` 요약을 함께 둔다.

## 구현 범위

- `.seriesctpr` 저장/복원
- folder scan + 선택/reorder popup
- series workspace
- child project 진입/복귀
- series queue 실행
- pause / resume
- queue status + last run summary
- duplicate import guard
- 시리즈 전용 settings
- `.ctpr`/`.seriesctpr` atomic save
- startup/home/open/save/recovery/recent 확장

## 제외 범위

- webtoon 전용 흐름 변경
- benchmark 전용 자산
- MangaLMM hybrid 재도입
- 실행 중 live reorder
