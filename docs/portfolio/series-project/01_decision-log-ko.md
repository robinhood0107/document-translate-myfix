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
- 따라서 실행 중 live reorder는 제품 계획에서 제외한다.
- 실행 중에는 queue 순서 변경, 항목 추가/제거, 시리즈 전역 설정 변경을 UI와 controller 양쪽에서 잠근다.
- 사용자가 오해하지 않도록 running 상태에서는 잠긴 컨트롤을 비활성화하고 안내 문구를 함께 보여준다.

### Pause / Resume

- `Pause`는 현재 항목 완료 후 정지로 고정한다.
- 즉시 중단형 pause는 배치 취소/partial child 상태/무결성 처리 비용이 커서 채택하지 않는다.
- `Resume`은 paused 상태에서만 허용한다.
- paused 상태에서는 running 항목이 없으므로 queue reorder, add/remove, 전역 설정 변경을 다시 허용한다.
- paused 또는 stop-on-failure 상태가 되면 series board로 복귀해 `Resume`과 `실패 항목 열기`를 일관되게 제공한다.

### 중복 입력 정책

- 같은 원본 파일 또는 같은 `.ctpr`를 다시 추가하려 하면 차단하고 경고한다.
- 자동 병합/덮어쓰기는 이번 단계에 넣지 않는다.
- 중복 판정은 `source_origin_path`의 정규화된 절대 경로 기준으로 한다.

### recovery 정책

- recovery에서 `running` 상태를 발견하면 자동 재개하지 않고 `paused`로 정규화한다.
- active item이 남아 있으면 `pending`으로 되돌려 맨 앞에 복귀시킨다.
- recovery 이후 기본 동작은 “사용자가 확인 후 Resume”이다.

### 실행 상태 가시성

- 시리즈 우측 패널은 `Queue Status`와 `Last Queue Run`으로 나눈다.
- 최근 한 번의 실행 요약은 완료/실패/건너뜀/총 시간/시작-종료 시각만 저장한다.
- `실패 항목 열기`는 board 기반 수동 진입을 기본으로 한다.

### dirty 표시 정책

- 시리즈 board의 reorder/add/remove/settings 변경은 save-through이므로 `시리즈 자체 미저장` 배지를 기본으로 두지 않는다.
- 대신 아래 둘만 명시적으로 표시한다.
  - `세부 프로젝트 변경 미반영`
  - `복구본 열림`
