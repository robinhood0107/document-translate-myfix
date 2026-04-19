# HunyuanOCR Docker Bundle

이 폴더는 현재 프로젝트에서 사용하는 `HunyuanOCR` 로컬 OCR 런타임의 기준 번들입니다.

## 기준 파일

- `docker-compose.yaml`

이 파일은 `llama.cpp` 기반 `HunyuanOCR` 서버를 같은 방식으로 다시 올리기 위한 tracked 기준입니다.

## 요구 모델 파일

아래 두 파일이 저장소 루트의 `testmodel/` 폴더에 있어야 합니다.

- `HunyuanOCR-BF16.gguf`
- `mmproj-HunyuanOCR-BF16.gguf`

현재 compose는 `../testmodel:/models:ro`를 마운트하고 아래 경로를 사용합니다.

- `/models/HunyuanOCR-BF16.gguf`
- `/models/mmproj-HunyuanOCR-BF16.gguf`

## 서버 실행

저장소 루트에서 실행:

```bash
docker compose -f hunyuanocr_docker_files/docker-compose.yaml pull --policy always
docker compose -f hunyuanocr_docker_files/docker-compose.yaml up -d --force-recreate
```

앱 설정:

- OCR: `HunyuanOCR`
- Server URL: `http://127.0.0.1:28080/v1`

## 기준 요약

- image: `ghcr.io/ggml-org/llama.cpp:server-cuda`
- pull policy: `always`
- OpenAI-compatible endpoint: `/v1/chat/completions`
- health endpoint: `/health`
- OCR request defaults: `temperature=0`, `top_k=1`, `repetition_penalty=1.0`
- prompt cache: disabled with `--cache-ram 0`
- purpose: block-crop OCR for the app's existing `TextBlock` pipeline
- note: moving tag 최신 상태는 pull 후 실제 runtime digest/version 기록으로 확인

## 참고

- 이 런타임은 전체 페이지 spotting이 아니라 현재 앱 구조에 맞춘 block-crop OCR 용도입니다.
- 루트 `docker-compose.yaml`은 Gemma 번역 서버용으로 유지하고, HunyuanOCR는 이 별도 번들에서 관리합니다.
