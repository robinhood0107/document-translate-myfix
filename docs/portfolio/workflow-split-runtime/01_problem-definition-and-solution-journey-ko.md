# Workflow Split Runtime Problem Definition And Solution Journey

## 문제 정의

사용자는 현재 페이지 단위 파이프라인이 Docker 기동 대기와 healthcheck 지연 때문에 전체 성능을 제대로 끌어내지 못하고 있다고 판단했다. 특히 OCR runtime과 Gemma4 runtime이 동시에 오래 상주하는 구조는 VRAM을 보수적으로 쓰게 만들 수 있으며, 페이지 수가 늘수록 고정비와 단계 전환 비용이 사용자 체감 시간에 불리하게 작용할 수 있다.

## 사용자 착안

사용자는 전체 프로젝트를 다음 순서로 재구성해 보자고 제안했다.

1. 텍스트 감지 전체 수행
2. OCR 전체 수행
3. 번역 전체 수행
4. 인페이팅 전체 수행

이때 OCR 단계에는 OCR 컨테이너만, 번역 단계에는 Gemma4만 올려서 VRAM을 단계별로 재배치하면 더 공격적인 `ngl`과 병렬성이 가능할지 검증하고자 했다.

## 우리가 확인한 현재 상태

- 제품 배치 파이프라인은 페이지 단위 `detect -> ocr -> inpaint -> translate -> render` 구조다.
- OCR runtime은 한 번에 하나의 엔진만 활성화한다.
- Gemma runtime은 별도 lifecycle로 동작한다.
- 설정 UI에는 workflow mode가 없다.
- benchmark full docs는 `develop`에 실을 수 없는 저장소 정책이 있다.

## 설계 판단

1. Requirement 1과 Requirement 2를 분리한다.
2. Requirement 1이 성공하기 전까지 Requirement 2의 자동 선택기 제품 구현은 잠근다.
3. benchmark full docs/assets는 `benchmarking/lab`에 둔다.
4. `develop`에는 포트폴리오형 요약 문서만 둔다.
5. 사고과정 문서는 내부 추론 재현이 아니라, 문제 정의 -> 측정 설계 -> 설계 판단 -> 구현 계획의 결정 로그로 남긴다.
6. `candidate_stage_batched_dual_resident`는 `candidate_stage_batched_single_ocr`보다 비교 실험 결과가 덜 좋아도 Requirement 1 자체를 무효화하지 않는 정식 신규 전체 플로우 승격 대상으로 잠근다.
7. 즉 Requirement 2가 새로 추가하는 것은 dual-resident 상주 자체가 아니라, 그 위에서 동작하는 사용자 승인 기반 자동 선택 규칙이다.

## 실제 운영에서 새로 알게 된 사실

원격 push 정책을 확인한 결과, benchmark 자산이 포함된 브랜치는 사실상 `benchmarking/lab` 이름으로 publish해야 했다. 즉 benchmark family 문서와 raw evidence는 `benchmarking/lab`에 직접 반영하고, `develop`은 summary + product promotion만 담당하는 구조가 저장소 현실과 가장 잘 맞는다.

## 현재까지의 결과

- `benchmarking/lab`에 Requirement 1 family 문서 기준선을 생성했다.
- `benchmarking/lab`에 Requirement 1 실측 패키지와 baseline smoke 근거를 확보했다.
- `benchmarking/lab`과 `feature/workflow-split-runtime` 문서에 `candidate_stage_batched_dual_resident` 정식 승격 정책을 반영하기 시작했다.
- `develop`에는 이 작업을 포트폴리오와 제품 승격 관점에서 읽을 수 있도록 별도 문서 묶음을 시작했다.
- `feature/workflow-split-runtime`에는 포트폴리오 승격 문서를 원격까지 publish했고 upstream도 연결했다.

## 브랜치 전략

1. full benchmark evidence와 raw 결과는 `benchmarking/lab`에 유지한다.
2. 제품 승격 문서와 실제 제품 코드는 `develop` 기준 feature 브랜치에서 정리한다.
3. Requirement 1 실측 성공 후 `feature/workflow-split-runtime -> develop` 머지를 목표로 한다.
4. Requirement 2는 `benchmarking/lab` 검수 패키지 확정 후 `feature/hybrid-ocr-selector -> develop` 순서로 진행한다.

## 운영 중 확인한 해결 사례

- 로컬 pre-push 훅은 upstream이 없는 첫 push에서 ref 매칭을 엄격하게 검사했다.
- 해결 방법은 원격에 동일 이름 브랜치를 먼저 만들고 upstream을 연결한 뒤 다시 push하는 것이었다.
- 이 과정을 통해 "benchmarking/lab에서 문서 기준선 확보 -> feature 브랜치 publish -> 제품 코드 구현" 순서를 안정적으로 재현할 수 있게 됐다.

## 다음 단계

1. `feature/workflow-split-runtime` 포트폴리오 문서에 dual-resident 정식 승격 방향을 반영
2. Requirement 1 13장 full baseline 누적
3. experimental candidate runner 실측과 비교표 작성
4. `workflow_mode` 제품 설계를 `legacy` + `candidate_stage_batched_dual_resident` 기준으로 확정
5. Requirement 1 실측 결과를 바탕으로 제품 승격 범위를 결정
