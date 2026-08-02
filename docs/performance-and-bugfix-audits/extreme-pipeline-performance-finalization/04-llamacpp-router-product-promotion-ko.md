# Paddle Router 제품 승격 기록

## 범위

이 변경은 기존 `develop`의 GPU ownership arbiter와 `benchmarking/lab`의 단일
llama.cpp Router handoff 결과를 제품 runtime에 연결한다. 대상은 기본 loopback
endpoint를 사용하는 다음 두 OCR·Gemma 조합뿐이다.

- PaddleOCR VL crop: `PaddleOCR-VL-1.6-0.9B`, OCR `18000`, Gemma `18080`
- PaddleOCR VL Spotting: `PaddleOCR-VL-1.6-Spotting`, OCR `18002`, Gemma `18080`

HunyuanOCR, MangaLMM, custom endpoint와 UI endpoint 설정은 이 승격에서 바꾸지
않는다. 기존 lab의 품질 `REJECT` 기록은 수정하거나 삭제하지 않는다. lab에서
확인된 속도 우위와 사용자 검토에 따른 두 Paddle pair 승격 승인을 별도 근거로
유지한다.

## 제품 계약

Controller는 하나의 `LocalLlamaRouterCoordinator`를 두 local runtime manager에
주입한다. Coordinator는 pair 판별, Router process 준비, 명시적 model load/unload,
pair 종료, 상태 snapshot만 담당한다.

각 product Router는 Docker built-in `bridge`에서 OCR와 Gemma volume 및 `models.ini`를
read-only로 마운트하고, 두 host port를 내부 `8080`으로 연결한다. `--models-max 1`,
`--no-models-autoload`, `load-on-startup=false`를 고정하며, inference HTTP는
Coordinator command lock 밖에서 실행한다.

```text
Arbiter → Coordinator command gate: prepare / load / status / unload
       → OCR 또는 Gemma inference HTTP
       → 기존 driver-global VRAM 반환 gate
```

Pair fingerprint에는 pinned image ref/digest와 실제 image ID, OCR·Gemma model 및
manifest SHA-256, read-only volume, Compose와 preset SHA-256, Router command SHA-256을
포함한다. readiness cache에는 Coordinator generation을 포함해 unload 뒤의 stale
ready 판정을 차단한다. release 실패 시 owner를 유지하고 후속 model load를
거부한다.

정상 stage handoff는 model을 unload한 뒤 loaded count `0`인 Router process를
재사용한다. 취소·실패·앱 종료·pair 전환에서는 owned container까지 정리한다.

## 검증 상태

- Coordinator, manager, Arbiter generation, stage cleanup 단위 테스트 및 기존
  OCR/Gemma/배치 취소 회귀 테스트 통과
- Python static, headless smoke, i18n, Windows launcher, repository policy 검사
  통과
- 실제 Docker smoke에서 Crop load → OCR HTTP → unload → Gemma load → translation
  HTTP → unload → Crop 재-load → loaded count `0` 및 orphan 없음 확인
- 실제 Docker smoke에서 Spotting load → Spotting HTTP → unload → loaded count `0`
  및 orphan 없음 확인

Raw request·response, 원문 이미지, local path와 GPU 실행 로그는 공개 문서에
기록하지 않으며 ignore된 private validation archive에만 둔다.
