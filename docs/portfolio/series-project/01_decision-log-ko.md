# 시리즈 프로젝트 결정 로그

## 2026-04-19

### 저장 구조

- `.seriesctpr`는 단일 SQLite 컨테이너로 고정한다.
- 내부 child는 정규화된 새 페이지 스키마로 풀지 않고 기존 `.ctpr` blob을 그대로 저장한다.
- 향후 series 해체 시 child `.ctpr`를 그대로 추출할 수 있게 한다.

### child 생성 시점

- 폴더 스캔에서 선택된 원본 파일은 시리즈 생성 시 즉시 child `.ctpr`로 만든다.
- lazy child 생성은 채택하지 않는다.

### 항목 입력원

- 시리즈 항목은 `원본 파일 + 기존 .ctpr` 혼합 입력을 지원한다.

### 전역 설정 범위

- 시리즈 우측 패널은 `언어 + OCR/번역기/workflow/GPU + 자동진행`만 가진다.
- detector/inpainter/텍스트/렌더/세부 export는 child 프로젝트로 내린다.

### 자동 대기열 정책

- 기본은 순차 실행이다.
- 실패 정책은 시리즈 전용 settings에서 선택한다.
- 후보: stop / skip / retry

### 이동 기록

- 일반 프로젝트 히스토리와 시리즈 히스토리를 분리한다.

### 저장 무결성

- `.ctpr`와 `.seriesctpr` 저장은 temp file 완성 후 replace 방식으로 통일한다.
- series child 편집 중 canonical 데이터는 temp child 파일이 아니라 `.seriesctpr` 내부 blob으로 본다.
- child 저장 시에는 temp child `.ctpr`를 먼저 serialize한 뒤 series container로 save-through 한다.

### 현재 제품 연결 상태

- Startup Home, 최근 프로젝트, 파일 브라우저, drag/drop에서 `.seriesctpr`를 인식한다.
- title bar project target popup은 현재 프로젝트 타입에 맞춰 `.ctpr`/`.seriesctpr` suffix를 바꾼다.
- `Settings > Series`는 새 시리즈의 기본 queue 정책을 저장하며, 시리즈별 settings dialog로도 재사용한다.

### 실행 중 대기열 변경 정책

- 현재 제품은 실행 시작 시점 snapshot 기반 queue runner를 사용한다.
- 따라서 실행 중 live reorder는 지원하지 않는다.
- 실행 중에는 queue 순서 변경, 항목 추가/제거, 시리즈 전역 설정 변경을 UI와 controller 양쪽에서 잠근다.
- 사용자가 오해하지 않도록 running 상태에서는 잠긴 컨트롤을 비활성화하고 안내 문구를 함께 보여준다.
- 다음 단계 live reorder는 `현재 running 고정`, `남은 pending만 재계산`, `실행 중 add/remove 금지` 계약을 기준으로만 검토한다.
