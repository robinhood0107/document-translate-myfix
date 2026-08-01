# 성능·버그수정 감사 문서

이 디렉터리는 제품 코드와 직접 맞물린 성능개선, 안정화, 버그수정 감사의
공개 가능한 요약을 프로젝트 단위로 보관한다. `docs/repo/`는 저장소 전체
규칙과 운영 문서를 위한 위치이며, 특정 성능개선 프로젝트나 버그 감사 장부는
이 디렉터리 아래에서 분리한다.

실행 이미지, 원문 OCR·번역, 좌표, raw 응답, 수동 검수표, 로컬 경로와 실행
로그는 추적하지 않는다. 그런 자료는 저장소 루트의 ignore된 검증 보관소에서만
관리한다.

## 폴더 규칙

- 성능개선·안정화 주제마다 독립된 `<project-slug>/`를 만든다.
- 각 성능개선 프로젝트의 최상단 문서는 `00-truth-specification-ko.md`다.
- 이후 문서는 `01-...`, `02-...`처럼 번호 prefix로 정렬한다.
- 공개 문서에는 확정 사실, 불변 조건, 코드 계약과 비민감 통계만 기록한다.
- benchmark runner·preset은 `benchmarking/lab`, raw output은 비추적 검증
  보관소에 둔다.

## 현재 감사 프로젝트

- [Optimal++ v1.3.0 최종 감사](optimal-plus-v1.3.0/README-ko.md)
  - 관리형 llama.cpp 전용 전환, 세 OCR 전략, Gemma 최종값, 캐시·인페인트
    품질 게이트, CUDA 검증과 릴리스 종료 상태를 정리한다.
- [극한 파이프라인 성능 최종화](extreme-pipeline-performance-finalization/README-ko.md)
  - 새 폴더·시리즈 full-auto critical path, runtime scheduler와 calibrated ETA
    후속 작업을 추적한다.
- `auto-translation-performance/`
  - 기존 자동번역 runtime, OCR, inpaint, translation, render/export 성능 기록이다.
- `runtime-ui-bug-hunt/`
  - 런타임/UI 버그와 의도하지 않은 동작을 전수 검사한 감사 장부다.

## 읽는 방법

1. 각 프로젝트의 `00-truth-specification-ko.md`에서 확정 사실과 금지선을
   확인한다.
2. `01-...` 이후 문서에서 분야별 측정·채택·퇴역 근거를 확인한다.
3. 이미지·원문 또는 개별 검수가 필요하면 비추적 검증 보관소에서 확인한다.

## 변경 원칙

- root cause와 현재 동작을 확인하기 전에는 구현 PR로 들어가지 않는다.
- 성능개선은 작은 PR로 나누고 회귀 테스트와 before/after 근거를 남긴다.
- 이미지, 번역, OCR, inpaint 또는 render가 바뀌는 변경은 실제 산출물을
  비추적 보관하고 사용자 검토를 요청한다.
- 큰 구조 변경이나 데이터 손실 위험이 보이면 별도 보고 후 계획을 다시 세운다.
