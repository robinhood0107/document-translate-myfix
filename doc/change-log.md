# 변경 이력

## 체크포인트 1

- 헤더 버튼 흐름 분석 문서를 작성했다.
- 렌더 설정의 기존 동작을 문서화했다.
- 코드 변경 전 구현 전략을 정리했다.

## 체크포인트 2

- 자동 색상 강제와 상하 정렬 계산을 담당하는 공통 렌더 정책 헬퍼를 추가했다.
- `TextRenderingSettings`에 아래 필드를 확장했다.
  - `force_font_color`
  - `smart_global_apply_all`
  - `vertical_alignment_id`
- 직렬화되는 텍스트 상태에 아래 필드를 확장했다.
  - `vertical_alignment`
  - `source_rect`
  - `block_anchor`

## 체크포인트 3

- 렌더 패널에 아래 UI를 추가했다.
  - `Apply All SMART Globally`
  - `Force Color`
  - `Top / Center / Bottom`
  - 명시적 윤곽선 `OFF | ON`
- 기존 헤더 버튼 연결 구조는 그대로 유지했다.
- 기존 윤곽선 체크박스는 내부 호환용으로 남기고 화면에서는 숨겼다.

## 체크포인트 4

- 수동 렌더와 수동 번역 후 재래핑이 같은 공통 렌더 정책을 쓰도록 바꿨다.
- 일반 배치와 웹툰 배치의 렌더 상태 생성도 같은 정책을 쓰도록 바꿨다.
- 텍스트와 블록 매칭이 보이는 `position`보다 `block_anchor`를 우선 사용하도록 바꿨다.
- 웹툰 좌표 변환과 내보내기 렌더에서 `position`, `source_rect`, `block_anchor`를 각각 독립적으로 변환하도록 수정했다.

## 체크포인트 5

- 변경 파일에 대해 `./.venv/bin/python -m py_compile` 검증을 수행했다.
- 오프스크린 환경에서 새 렌더 패널 컨트롤 생성 여부를 확인했다.
- `TextBlockItem` 스모크 테스트로 상하 정렬 이동/재정렬 드리프트가 없는지 확인했다.

## 체크포인트 6

- 렌더 패널에서 보이던 `SMART / GLOBAL / ITEM` 용어를 사용자 화면에서 제거했다.
- 혼란을 주던 SMART 마스터 체크박스를 화면에서는 숨기고 설명형 문구로 대체했다.
- 색상 강제 옵션을 `Always Use This Color`로 노출했다.
- 예전 `smart_global_apply_all` 저장값은 로드시 보이는 색상 강제 동작으로 자연스럽게 이어지게 했다.

## 체크포인트 7

- `doc/` 아래 문서를 전부 한국어로 번역했다.
- 새로 추가한 렌더 패널 UI 문자열을 각 언어 `ts` 파일에 반영했다.
- `lrelease`로 `resources/translations/compiled/*.qm`을 다시 생성해 실제 프로그램에서도 번역이 적용되도록 정리했다.

## 체크포인트 8

- 오른쪽 렌더 패널을 `색상`, `정렬`, `스타일/윤곽선` 3개 의미 그룹으로 재배치했다.
- 좁은 패널에서도 컨트롤이 한 줄에 몰리지 않도록 그룹별 카드형 배치와 더 큰 클릭 영역을 적용했다.
- `Always Use This Color` 문구를 `Use Selected Color`로 줄여 시인성을 높였다.
- 상하 정렬과 윤곽선 `OFF / ON` 버튼은 렌더 패널 내부 전용 체크 스타일을 사용하도록 바꿨다.
- 오른쪽 패널 최소 폭을 늘려 한국어와 독일어 같은 긴 번역에서도 잘림 가능성을 줄였다.
