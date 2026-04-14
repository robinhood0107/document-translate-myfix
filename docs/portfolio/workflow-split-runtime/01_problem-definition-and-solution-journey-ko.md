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
2. Requirement 1이 성공하기 전까지 Requirement 2 제품 구현은 잠근다.
3. benchmark full docs/assets는 `benchmarking/lab`에 둔다.
4. `develop`에는 포트폴리오형 요약 문서만 둔다.
5. 사고과정 문서는 내부 추론 재현이 아니라, 문제 정의 -> 측정 설계 -> 설계 판단 -> 구현 계획의 결정 로그로 남긴다.

## 실제 운영에서 새로 알게 된 사실

원격 push 정책을 확인한 결과, benchmark 자산이 포함된 브랜치는 사실상 `benchmarking/lab` 이름으로 publish해야 했다. 즉 benchmark family 문서와 raw evidence는 `benchmarking/lab`에 직접 반영하고, `develop`은 summary + product promotion만 담당하는 구조가 저장소 현실과 가장 잘 맞는다.

## 현재까지의 결과

- `benchmarking/lab`에 Requirement 1 family 문서 기준선을 생성했다.
- `develop`에는 이 작업을 포트폴리오와 제품 승격 관점에서 읽을 수 있도록 별도 문서 묶음을 시작했다.

## 다음 단계

1. `workflow_mode` 제품 설계 반영
2. runtime lifecycle / telemetry 설계
3. Requirement 1 실측 결과를 바탕으로 승격 여부 결정
