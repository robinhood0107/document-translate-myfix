# MangaLMM llama.cpp runtime

이 번들은 MangaLMM의 관리형 full-page OCR 서버를 실행합니다.

## 모델 준비

모델은 host bind mount가 아니라 versioned named volume에 보관합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_mangalmm_llamacpp_runtime.ps1 `
  -Mode Prepare `
  -ModelDirectory 'C:\ExampleWorkspace\models\MangaLMM'
```

전체 SHA-256 재검증:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_mangalmm_llamacpp_runtime.ps1 `
  -Mode Verify
```

기본 volume은 `comic-translate-mangalmm-models-v2`이며 서비스에는 read-only로
mount됩니다.

- `/models/MangaLMM.Q8_0.gguf`
- `/models/MangaLMM.mmproj-Q8_0.gguf`

## 런타임 계약

- digest로 고정된 `ghcr.io/ggml-org/llama.cpp`
- `pull_policy: missing`
- image ID, compose command, volume, ready manifest, model/mmproj SHA를 포함한
  runtime fingerprint
- fingerprint가 정확히 같은 stopped container만 `docker start`로 재사용
- 정상 종료는 `docker compose stop`
- 페이지 전체 PNG spotting만 사용
- block crop·page tile·숨은 Paddle fallback·Paddle 동시 상주 금지

앱의 관리형 endpoint는 `http://127.0.0.1:28081/v1`입니다.
