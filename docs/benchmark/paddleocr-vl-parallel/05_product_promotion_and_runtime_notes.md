# 05 Product Promotion And Runtime Notes

## 목적

subset benchmark winner를 제품 runtime 승격 후보로 올릴 때 필요한 조건과 메모를 남긴다.

이 문서는 benchmark 결과를 곧바로 제품 기본값으로 연결하지 않기 위한 안전장치이기도 하다. 이번 단계는 `PaddleOCR VL 단독 상주 상한선 benchmark` 결과를 사용자 검수 가능한 형태로 정리하고, 승격 여부를 잠그는 것까지를 다룬다.

## 승격 조건

- 사용자 검수로 승인된 winner가 존재할 것
- hidden flag 공통 런타임 인프라가 `develop` 승격 브랜치에 포함될 것
- benchmark-specific asset이 develop PR에 섞이지 않을 것
- 제품 코드에는 benchmark-specific preset, runner, raw report가 포함되지 않을 것

즉, develop에 들어가는 것은 “검증된 runtime surface와 승인된 기본값”이지, benchmark 실험실 전체가 아니다.

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

- 이번 승격 승인 후 제품 기본 mode는 `fixed_area_desc`다.
- 제품 기본 `parallel_workers`는 `8`이다.
- `fixed`, `auto_v1`는 hidden override/diagnostic/benchmark mode로 계속 유지한다.
- `parallel_workers`는 hidden scheduler가 켜질 때 cap으로 동작한다.
- `OCRProcessor`는 engine의 `last_page_profile`을 전달해야 pipeline benchmark event에서 page profile을 읽을 수 있다.
- 이번 family는 `runtime_services=ocr-only` 계약을 사용하므로 Gemma preset 필드는 남아 있어도 실제 runtime/VRAM 점유에 참여하지 않는다.

## promotion 메모

- 사용자 승인으로 `fixed_area_desc_w8`가 최종 winner로 잠겼다.
- `fix/ocr-paddleocr-vl-final-promote`에서는 공통 런타임 인프라 + 승인된 winner 기본값만 develop에 올린다.
- 이번 단계에서는 `default on`의 추가 후보 재평가나 22장 full corpus 재검증을 하지 않는다.

## 현재 메모

최신 single-tenant smoke 결과와 OCR diff review 결과를 합치면 다음 결론이 나온다.

- `fixed_area_desc_w8`가 속도 1위다.
- baseline `fixed_w8` 대비 OCR changed block이 없다.
- `auto_v1_cap4` 역시 changed block은 없지만 속도는 `fixed_area_desc_w8`보다 느리다.
- 따라서 이번 라운드의 최종 승격 대상은 `fixed_area_desc_w8`다.

현재 제품 승격 메모는 다음과 같다.

- runtime surface와 hidden scheduler mode를 develop에 승격
- 제품 기본값을 `fixed_area_desc_w8`로 변경
- benchmark 자산과 narrative는 benchmarking/lab에 유지
- `auto_v1`는 차후 mixed-runtime 조건에서 다시 검토

## 문서화 메모

사용자 제안, 측정 설계, 구현 근거, benchmark 결과, develop 승격 메모는 분리된 문서로 남긴다. 이렇게 해야 포트폴리오 문서와 제품 운영 문서가 서로 역할을 침범하지 않는다.

## 저자 및 기여

- 핵심 문제 해결 방향은 사용자가 착안했다.
- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
