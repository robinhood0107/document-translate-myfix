# 공개 가능한 근거와 비공개 원시 증거의 경계

이 문서는 v1.3.0 최적화·품질 판정의 **증거가 어디에 어떤 형태로 보존되는지**를 설명한다. 공개 문서는 결론을 재현 가능하게 설명하되, 실제 작품·이미지·원문·번역문·로컬 환경 식별정보를 노출하지 않는다.

## 세 층의 보관 원칙

| 층 | 보관 내용 | Git 상태 | 목적 |
|---|---|---|---|
| 제품 문서 | 확정 정책, 정제된 통계, 검증 계약, 사용자 운영 방법 | tracked | 공개 가능한 제품 기준 |
| benchmark 문서 | runner 구조, protocol, 결과 이력의 요약, 재현 절차 | `benchmarking/lab` | 실험 방법의 유지 |
| 비공개 증거 보관소 | 원시 OCR·번역·inpaint·render, 이미지, review, telemetry, cleanup manifest | 저장소 루트의 ignore된 보관소와 별도 validation log | 품질 판정과 문제 재현 |

비공개 보관소의 실제 상대경로·실제 corpus 이름·원본 파일명·로컬 절대경로는 tracked 문서에 기록하지 않는다. 이 문서에서 `&lt;validation-log-root&gt;`는 그 보관 위치를 가리키는 중립 표기다.

## 주장별 근거 체인

| 공개 주장 | 공개 문서의 근거 | 비공개 증거 |
|---|---|---|
| 기본 OCR이 세 경로 중 품질 우세 | [OCR 품질·routing 판정](02-ocr-quality-and-routing-decision-ko.md) | source-first truth, blind review, 세 경로 raw response와 overlay |
| 인페인트는 불확실한 투명·free-text 영역을 보존 | [인페인트 품질 판정](05-inpaint-quality-and-safe-preservation-ko.md) | raw/final mask, cleaned crop, diff, 최종 render 비교 |
| Gemma 기본값을 유지 | [Gemma 번역·모델 판정](03-gemma-translation-and-model-decision-ko.md) | round별 입력 계약, timing, candidate-only 의미 검수 |
| cache와 prewarm을 유지 또는 보류 | [cache·checkpoint 판정](04-cache-checkpoint-and-cold-path-decision-ko.md) | isolated DB, cache hit/miss telemetry, checkpoint sidecar 검증 |
| llama.cpp 전용 및 자산 정리 | [runtime 자산·llama.cpp 전환](06-runtime-assets-and-llamacpp-retirement-ko.md) | Docker ownership manifest, model integrity manifest, live process inspection |
| v1.3.0 release 계약 | [릴리스 검증](07-release-verification-and-branch-closure-ko.md) | release-preflight output, bundle verification, launcher smoke 기록 |

## 비공개 증거에 포함되는 자료

다음은 품질 판정에 필요하지만 공개 Git에 넣지 않는다.

- source page와 확대 crop, mask, cleaned artifact, diff, 최종 render
- 원시 OCR 및 번역 응답, blind key, 완성된 검수표, 사람이 기록한 판정 메모
- timing, VRAM, RAM, swap, HTTP retry, runtime process telemetry
- SQLite DB, checkpoint sidecar, cache export, object storage
- 모델·image·volume의 세부 manifest, container ID, cleanup 대상 목록
- 실제 작품명, 원본 filename, 사용자 계정·로컬 경로, endpoint 및 secret

원시 자료는 신규 문서 작성 때 복제하지 않는다. 기존 증거를 보존하고, 새 접근에는 새 run ID·input identity·protocol version을 연결한다. 이는 서로 다른 실행의 이미지를 섞어 결론을 만드는 일을 막는다.

## 공유·검수할 때의 절차

1. 먼저 공개 문서의 지표와 판정 기준을 읽는다.
2. 품질 분쟁이나 사용자 검수가 필요한 경우에만 비공개 보관소에서 원본·결과·diff를 같은 순서로 연다.
3. 외부 공유본에는 작품명, 원문, 이미지, 로컬 path, model 위치, container ID를 제거한다.
4. 요약 통계만으로 의미 품질을 단정하지 않고, source-first 정답과 candidate-only 회귀를 함께 확인한다.
5. 비공개 evidence를 실수로 staged하지 않았는지 `git status`와 정책 검사를 먼저 확인한다.

## 무결성·보존 원칙

- 준비된 모델과 runtime은 SHA-256·manifest·fingerprint로 식별한다.
- benchmark protocol과 입력 identity는 run마다 고정한다.
- DB lock, 손상, schema mismatch는 cache만 fail-open하고 원본 evidence를 자동 삭제하거나 덮어쓰지 않는다.
- cache cleanup은 명시적 사용자 동작이며 debug와 민감 diagnostics를 자동 삭제하지 않는다.
- 비공개 evidence의 보존 한도·수명 정책은 공개 source code가 아니라 사용자 데이터 관리 정책에 따라 별도로 적용한다.

## 관련 문서

- [최적화·품질 감사 문서 색인](README-ko.md)
- [공통 진실 명세](00-truth-specification-ko.md)
- [릴리스 검증과 브랜치 정리](07-release-verification-and-branch-closure-ko.md)
- [저장소 규칙](../../../rules.md)
