# Performance And Bugfix Audits

이 폴더는 제품 코드와 직접 맞물린 성능개선, 안정화, 버그수정 감사 문서를 프로젝트 단위로 보관한다.

`docs/repo/`는 저장소 전체 규칙과 운영 문서를 위한 위치다. 특정 성능개선 프로젝트나 버그 헌팅 장부는 여기처럼 별도 프로젝트 폴더로 분리한다.

## 폴더 규칙

- 성능개선 프로젝트는 `docs/performance-and-bugfix-audits/<project-slug>/` 아래에 둔다.
- 버그 헌팅 또는 안정화 감사도 독립 주제라면 별도 `<project-slug>/`를 만든다.
- 각 성능개선 프로젝트의 최상단 문서는 `00-truth-specification-ko.md`다.
- 이후 문서는 번호 prefix로 정렬한다.
  - `01-...-audit-ko.md`: 관측, 로그, 계산 근거
  - `02-...-implementation-spec-ko.md`: AST/코드 검토 기반 구현 명세
  - `03-...`: 최종 실행 계획, PR별 설계, 실험 결과, 사용자 검토 기록
- 테스트 로그, 이미지 산출물, benchmark raw output은 repo 밖 validation log 또는 `benchmarking/lab` 정책 위치에 둔다. 이 폴더에는 추적 가능한 문서와 링크만 둔다.

## 현재 프로젝트

- `auto-translation-performance/`: 자동번역 runtime, OCR, inpaint, translation, render/export 성능개선 프로젝트
- `runtime-ui-bug-hunt/`: 런타임/UI 사소 버그와 의도치 않은 동작을 전수 검사하는 감사 장부

## 변경 원칙

- root cause와 현재 동작을 확인하기 전에는 구현 PR로 들어가지 않는다.
- 성능개선은 작은 PR로 나누고, 각 PR은 회귀 테스트와 before/after 근거를 가진다.
- 이미지 결과물, 번역 품질, OCR 결과, inpaint 결과가 바뀌는 변경은 테스트 이미지와 실제 산출물을 남기고 사용자 검토를 요청한다.
- 큰 구조 변경이나 데이터 손실 가능성이 보이면 바로 별도 보고 후 계획을 다시 세운다.
