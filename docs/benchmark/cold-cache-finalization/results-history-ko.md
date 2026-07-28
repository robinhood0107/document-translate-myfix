# cold/cache 최종화 결과 이력

## protocol v1 도구 추가

- frozen 제품 기준선과 후보 matrix를 추가했습니다.
- 실제 offscreen stage-batched pipeline을 최대 6페이지로 실행합니다.
- cold 후보는 3회 순서 교차, cache는 disabled/empty miss overhead와
  all-hit를 분리 측정합니다.
- project checkpoint는 기존 output, missing output materialization,
  한 페이지 OCR downstream 무효화를 검증합니다.
- stopped Docker container의 불필요한 300초 health wait를 제거했습니다.
- wrapper 사전 부팅을 제거하고 staged runtime을 제품 manager에 주입해
  cold startup과 all-hit runtime 0을 실제 제품 경로에서 측정합니다.
- 첫 실제 스모크에서 staged Compose의 암시적 project 이름이 달라 기존
  stopped 컨테이너와 이름 충돌하는 문제를 재현했습니다. 제품과 같은
  Compose project 이름을 명시해 원인을 수정했습니다.
- 같은 stopped container 조건의 후속 1페이지·20블록 스모크는 페이지
  실패 0, Paddle runtime 시작 1회, OCR HTTP 20회로 완료됐습니다.
  runtime 시작은 82.276초, 전체 pipeline wall time은 105.318초였으며
  종료 뒤에도 제품 Compose project identity가 유지됐습니다.
- staged Gemma command에 no-spec/F16 계약을 직접 고정했습니다.
- full baseline의 실제 stage share를 검증해 stage 5% 또는 예상 전체 1%
  게이트를 판정할 수 있게 했습니다.
- 의미 검수 대상은 모든 라운드를 외부 private review JSON으로 묶습니다.
- raw 결과는 Git 밖에만 기록합니다.

이 문서는 도구 구현 이력입니다. GPU 실측 우승 결과는 외부 suite가
완료된 뒤 검증된 수치만 추가합니다. 현재 문서만으로 제품 기본값을
변경하지 않습니다.
