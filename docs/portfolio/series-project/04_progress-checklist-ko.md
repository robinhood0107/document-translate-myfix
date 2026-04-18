# 시리즈 프로젝트 진행 체크리스트

- [x] docs/portfolio/series-project 문서 세트 생성
- [x] `.seriesctpr` state store 추가
- [x] `.ctpr` atomic save 전환
- [x] `.seriesctpr` atomic save 전환
- [x] folder scan helper 추가
- [x] series creation popup 추가
- [x] Startup Home `New Series Project` 카드 추가
- [x] `.seriesctpr` open path 연결
- [x] series workspace 추가
- [x] series row/card reorder/remove 구현
- [x] child `.ctpr` materialize/import 구현
- [x] child open/back sync 구현
- [x] series history 분리 구현
- [x] `Settings > Series` 추가
- [x] queue runner 구현
- [x] recovery/recent/title bar 확장
- [x] tests 추가
- [x] translations + compiled qm 갱신

## 2026-04-19 구현 메모

- 현재 브랜치 `feature/series-project`에서 series storage/UI/controller 1차 연결을 완료했다.
- `.seriesctpr`는 SQLite 단일 컨테이너이며 child `.ctpr` blob을 내장한다.
- 제품 open/save/save-as/recent/home/title-bar 경로는 `.seriesctpr`를 인식하도록 확장했다.
- 시리즈 대기열 자동 실행은 순차 실행/stop-skip-retry 정책까지 연결했다.
- 남은 후속 작업은 실제 사용자 피드백 기반 UX 다듬기와 추가 회귀 테스트 확대다.
