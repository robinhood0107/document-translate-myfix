# Comic Translate 극한 파이프라인 성능 감사

이 디렉터리는 새 폴더와 series full-auto 시간을 줄이는 후속 성능 프로젝트의
공개·비민감 기록이다.

- [00-truth-specification-ko.md](./00-truth-specification-ko.md): 목표, 불변
  품질 계약, 금지선, PR 순서와 승격 조건
- [01-performance-telemetry-v2-ko.md](./01-performance-telemetry-v2-ko.md):
  범용 계측 schema와 첫 구현 검증 계약
- [02-serving-scheduler-matrix-ko.md](./02-serving-scheduler-matrix-ko.md):
  pinned llama.cpp capability, residency preflight와 lab 후보 matrix 계약
- [03-runtime-resource-arbiter-ko.md](./03-runtime-resource-arbiter-ko.md):
  GPU model exclusive lease, 취소 세대와 release 실패 안전 계약
- [04-llamacpp-router-product-promotion-ko.md](./04-llamacpp-router-product-promotion-ko.md):
  Paddle crop·Spotting Router의 제품 승격 계약과 비민감 검증 상태

후속 PR은 같은 디렉터리에 번호 순서로 추가한다. 원시 이미지, local path,
OCR·번역 응답, runtime log와 benchmark 순위는 이 디렉터리에 넣지 않는다.
