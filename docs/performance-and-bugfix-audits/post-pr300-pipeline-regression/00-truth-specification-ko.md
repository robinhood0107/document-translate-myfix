# PR #300 이후 pipeline 회귀 진실명세서

## 기준선과 확정 원인

- 제품 pipeline의 비교 기준은 PR #300 merge `95f698e`다.
- `e51ce03`에서 추가된 전체 페이지 예상 메모리 합산이 workflow 선택보다 먼저
  평가되면서, 사용자가 Stage-Batched를 선택해도 legacy page pipeline으로
  전환되는 회귀가 생겼다.
- 이 전환 때문에 한 페이지 안에서 HunyuanOCR 다음 Gemma가 시작되었고,
  HunyuanOCR GPU lease가 남아 Gemma lease와 충돌했다.
- CUDA 버전, `server-cuda` 이미지, 모델 설치 누락 또는 PC 성능 차이는 이
  workflow 회귀의 원인이 아니다.

## 제품 불변 조건

1. `stage_batched_pipeline`은 페이지 수나 합산 예상 메모리와 무관하게 항상
   `StageBatchedProcessor`를 사용한다.
2. `legacy_page_pipeline`은 사용자가 명시적으로 선택한 경우에만 사용한다.
3. 이미지 안전 정책은 workflow를 선택하지 않는다. 2억 픽셀 hard cap과
   단일 페이지 예상 peak가 현재 가용 RAM의 70%를 넘는지 여부만 검사한다.
4. 아카이브와 PDF preflight는 페이지 헤더 또는 PDF page plan만 검사하며
   선택된 모든 페이지를 출력 파일로 materialize하지 않는다.
5. 앱과 run launcher는 모델 다운로드, Docker image pull, volume repair,
   reseal을 수행하지 않는다. 설치 상태가 불완전하면 page 1 전에 setup을
   안내하고 중단한다.
6. Stage-Batched는 `detect-all → ocr-all → OCR/Gemma handoff → translate-all
   → inpaint-all → render-all` 순서를 보존한다. OCR과 Gemma GPU lease는
   동시에 활성화될 수 없다.
7. 이번 교정은 Stage-Batched 내부의 자동 page reload 정책을 추가하지 않는다.

세부 stage 순서, Router의 단일 모델 상주 계약, OCR 중 Gemma page-cache
prefetch의 의미는 [Stage-Batched 파이프라인 작동 원리](../../architecture/codebase-map-ko.md#stage-batched-파이프라인-작동-원리)에
정리한다.

## PR #300 이후 변경 감사

| 변경 | 판정 | 근거 |
|---|---|---|
| `b5caf75`, PR #301 `13e50fe` | 유지 | setup의 관리형 Docker 준비 중복 제거이며 workflow 선택을 바꾸지 않는다. |
| `da68063` | 유지 | host CUDA와 image 요구값을 container 생성 전에 비교한다. |
| `7a2d74a` | 유지 | setup과 run 진입점을 분리한다. |
| `e51ce03` 합산 peak·2GiB budget·자동 legacy 전환 | 제거 | 이미지 보호가 사용자 workflow를 덮어쓴 직접 회귀다. |
| `e51ce03` 2억 픽셀 hard cap·단일 페이지 RAM 검사 | 유지 | 단일 입력의 decode/OOM 위험을 page 작업 전에 차단한다. |
| `e51ce03` setup seal·앱 무다운로드 경계 | 유지 | 실행 중 provisioning을 막는 설치 계약이다. |
| `cdaab38` optional 사전 차단 | 강화 후 유지 | tier 문자열 대신 실제 model/runtime seal 레코드를 확인한다. |
| `ef7104b` | 유지 | setup 다운로드 진행률을 표시하며 pipeline workflow와 무관하다. |
| `b39199b`, `57f2649` | 유지 | setup/runtime 검증 경계 수정이며 pipeline 순서를 바꾸지 않는다. |
| `d68b158`, `4f393d0` | 유지 | 단일 `server-cuda` 계약과 setup 다운로드 재개를 고친다. |
| `2fa94c5`, `fd8fb82`, `3f93ca4` | 유지 | 완료 대기와 부모 CMD 설정 상속만 다룬다. |
| `8eac138` | 유지 | 사용자 취소, canonical inpainter, probe cleanup의 correctness 수정이다. |
| `43d26e6` | rollback/temp 보강 후 유지 | 파일 열기를 worker에서 staging하고 embedded overlay를 사용한다. |

## 이번 교정의 최소 범위

- 합산 메모리, retention budget, `streaming` workflow 신호를 삭제한다.
- header-only `ImageResourcePlan`에는 page count, 최대 픽셀, 최대 단일-page
  예상 peak와 hard-cap 통과 여부만 둔다.
- setup `full` 상태는 full application model 전체와 core/full 관리형 runtime
  집합을 검사한다.
- Gemma runtime contract는 image ID, volume, model identity가 같은 동안
  process-local로 재사용한다.
- workspace commit 실패 시 기존 상태와 `FileHandler`를 복원하고, 성공 전에는
  이전 temp를 정리하지 않는다.
- 범용 scheduler, 새로운 cache 계층, 자동 workflow fallback, 자동 page reload는
  추가하지 않는다.

## 검증 증거

- 공개 검증은 회귀 단위 테스트, 전체 Python/headless 검사, 번역 검사,
  Windows launcher source contract와 deterministic source ZIP 검사로 남긴다.
- 실제 입력, OCR/번역 결과, GPU/runtime 로그는 저장소에 넣지 않고 ignored
  private validation archive에만 보관한다.
- 실제 검증은 네트워크 모델 다운로드를 금지한 상태에서 작은 2페이지 입력으로
  HunyuanOCR 전체 sweep, OCR release, Gemma 번역, LaMa 인페인트 순서를 확인한다.
- 93페이지와 366페이지 stress telemetry는 후속 메모리 변경의 판단 자료이며,
  이번 교정의 workflow 선택 조건으로 사용하지 않는다.

## 로컬 실측 결과

- CUDA13 core seal과 2페이지 private 입력에서 HunyuanOCR, Gemma, LaMa,
  render까지 두 페이지 모두 완료했다.
- 각 페이지에서 `detect-all`, `ocr-all`, `translate-all`, `inpaint-all`,
  `render-all`이 한 번씩 실행됐고 page-level legacy processor는 실행되지 않았다.
- Router가 HunyuanOCR instance를 정상 종료한 뒤 Gemma instance를 생성한 순서를
  container 로그로 확인했다. 모델 다운로드, image pull, repair, reseal은 0회였다.
- 이 PC의 cold Gemma 적재는 Router transition 제한시간보다 길어 기존 동기
  startup 경로로 한 번 재시도됐고 전체 실행은 완료됐다. 이 적재 시간 문제는
  workflow 회귀와 분리해 후속 runtime 성능 항목으로 남긴다.
- VRAM 관측값을 얻지 못해 비강제 정책으로 계속한 경우에도 logical inpainter
  lease를 반환하도록 보정했다. 강제 환경변수가 켜진 경우의 fail-closed 계약은
  그대로 유지한다.
