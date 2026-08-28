# HunyuanOCR Docker Bundle

이 폴더는 현재 프로젝트에서 사용하는 `HunyuanOCR` 로컬 OCR 런타임의 기준 번들입니다.

## 기준 파일

- `docker-compose.yaml`

이 파일은 `llama.cpp` 기반 `HunyuanOCR` 서버를 같은 방식으로 다시 올리기 위한 tracked 기준입니다.

## 요구 모델 파일

현재 제품 계약은 external named volume
`comic-translate-hunyuanocr-models-v2`에 아래 두 파일을 준비합니다.

- `HunyuanOCR.Q8_0.gguf`
- `HunyuanOCR.mmproj-Q8_0.gguf`

`run_comic.bat`과 `run_comic_cuda13.bat`의 첫 실행이 등록된 원본을 자동으로
내려받아 크기·SHA-256·CUDA model-load smoke를 검증합니다. volume은 앱에서
read-only로 마운트합니다.

## 수동 준비와 서버 실행

정상 launcher 실행에는 수동 명령이 필요하지 않습니다. 수동으로 준비하려면:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_hunyuanocr_llamacpp_runtime.ps1 `
  -Mode Auto -AllowDownload
```

준비된 volume으로 서버만 직접 실행하려면 저장소 루트에서:

```bash
docker compose -f hunyuanocr_docker_files/docker-compose.yaml pull --policy always
docker compose -f hunyuanocr_docker_files/docker-compose.yaml up -d --force-recreate
```

앱 설정:

- OCR: `HunyuanOCR`
- Server URL: `http://127.0.0.1:28080/v1`

## 기준 요약

- 기본 image: `ghcr.io/ggml-org/llama.cpp:server-cuda` (기존 `:server-cuda13` 봉인도 지원)
- pull policy: local image가 없을 때만 pull
- OpenAI-compatible endpoint: `/v1/chat/completions`
- health endpoint: `/health`
- OCR request defaults: `temperature=0`, `top_k=1`, `repetition_penalty=1.0`
- prompt cache: disabled with `--cache-ram 0`
- purpose: block-crop OCR for the app's existing `TextBlock` pipeline
- note: moving tag 최신 상태는 pull 후 실제 runtime digest/version 기록으로 확인

## 참고

- 이 런타임은 전체 페이지 spotting이 아니라 현재 앱 구조에 맞춘 block-crop OCR 용도입니다.
- 루트 `docker-compose.yaml`은 Gemma 번역 서버용으로 유지하고, HunyuanOCR는 이 별도 번들에서 관리합니다.
