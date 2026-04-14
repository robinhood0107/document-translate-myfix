# 05 Product Promotion And Runtime Notes

## 목적

subset benchmark winner를 제품 runtime 승격 후보로 올릴 때 필요한 조건과 메모를 남긴다.

이 문서는 benchmark 결과를 곧바로 제품 기본값으로 연결하지 않기 위한 안전장치이기도 하다. 이번 1차는 `hidden flag runtime promotion`까지만 다루며, 제품 기본값 전환은 의도적으로 범위 밖에 둔다.

## 승격 조건

- subset winner가 품질 게이트를 통과할 것
- hidden flag 상태로 `develop` PR이 구성되어 있을 것
- 22장 full corpus 승격 검증 계획이 이어질 것
- 제품 코드에는 benchmark-specific preset, runner, raw report가 포함되지 않을 것

즉, develop에 들어가는 것은 “검증된 runtime surface와 hidden mode”이지, benchmark 실험실 전체가 아니다.

## develop에 남기는 것

- runtime code
- tests
- 짧은 운영 설명 문서
- hidden scheduler mode contract
- generic telemetry surface

## develop에 남기지 않는 것

- raw benchmark outputs
- candidate presets
- generated charts
- portfolio narrative
- benchmark family runner
- generated latest/history asset trees

이 분리는 저장소 정책 차원에서 중요하다. benchmarking/lab은 실험실이고, develop은 제품 후보 통합 브랜치다. 두 영역을 섞으면 이후 유지보수와 PR 리뷰 비용이 불필요하게 커진다.

## runtime 메모

- 기본 mode는 계속 `fixed`다.
- `fixed_area_desc`, `auto_v1`는 hidden flag가 있을 때만 활성화된다.
- `parallel_workers`는 hidden scheduler가 켜질 때 cap으로 동작한다.
- `OCRProcessor`는 engine의 `last_page_profile`을 전달해야 pipeline benchmark event에서 page profile을 읽을 수 있다.

## promotion 메모

- subset winner는 `develop`에서 hidden flag 상태로만 승격한다.
- `default on`은 별도 브랜치 `fix/ocr-paddleocr-vl-default-auto-workers`에서 다룬다.
- 그 단계에서는 22장 full corpus, 더 긴 soak, 필요 시 weighted concurrency 후보 비교까지 포함한다.

## 현재 smoke 기준 판단

최신 smoke 결과에서는 `fixed_w8`만 품질 게이트를 통과했다. 즉, 현재 시점에서 speed-only 개선 후보였던 `fixed_area_desc_w8`, `auto_v1_cap4`는 제품 승격 winner가 아니다.

다만 이 결과는 이번 구현이 무효라는 뜻은 아니다. 오히려 다음 두 가지를 분명하게 보여준다.

- local VRAM headroom이 낮은 환경에서 `auto_v1`는 실제로 보수적으로 worker를 낮춘다.
- `job_order=area_desc`와 `auto_v1` 모두 latency 개선 방향성은 있으나, 품질 게이트를 유지하려면 추가 조정이 필요하다.

따라서 현재 제품 승격 메모는 다음과 같다.

- runtime surface와 hidden scheduler mode는 제품 후보로 유지
- 기본값은 계속 `fixed`
- benchmark 결과는 `fixed_area_desc`와 `auto_v1`를 후속 조정 대상으로 기록
- develop PR은 hidden flag 상태로만 유지하고, winner promotion은 보류

## 문서화 메모

사용자 제안, 측정 설계, 구현 근거, benchmark 결과, develop 승격 메모는 분리된 문서로 남긴다. 이렇게 해야 포트폴리오 문서와 제품 운영 문서가 서로 역할을 침범하지 않는다.

## 저자 및 기여

- 핵심 문제 해결 방향은 사용자가 착안했다.
- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
