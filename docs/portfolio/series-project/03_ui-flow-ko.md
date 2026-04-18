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
- 우측: 빠른 전역 실행 설정
- 좌상단: 뒤로가기 / 앞으로가기 / 트리 이동
- 우상단: `대기열대로 자동번역`
- running 상태 배지와 queue 상태 표시 제공
- 자동 번역 실행 중에는 queue 순서 변경, 항목 추가/제거, 시리즈 전역 설정 변경을 잠금
- 잠긴 상태는 비활성/흐림 처리와 안내 문구로 명확히 표시

## child 진입

- row/card click -> child project 열기
- child는 기존 `.ctpr` workspace와 동일하게 동작
- 복귀 시 series queue 상태와 child 저장 상태를 동기화

## 시각적 구분

- `Series Project` 배지
- child 진입 시 breadcrumb 또는 scope badge
