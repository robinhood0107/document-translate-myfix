# Gemma 로컬 서버 설정 가이드

이 문서는 `Custom Local Server(Gemma)`를 현재 저장소 기준으로 설정하는 방법을 정리합니다.

## 준비

- 모델 파일을 `testmodel/` 폴더에 둡니다.
- 앱은 `Settings > Credentials > Model`에 적은 GGUF 파일명을 그대로 사용합니다.

## 서버 실행

저장소 루트에서 실행:

```bash
docker compose pull --policy always
docker compose up -d --force-recreate
```

앱 설정:

- Endpoint URL: `http://127.0.0.1:18080/v1`
- Model: `testmodel/` 안에 둔 실제 GGUF 파일명과 정확히 같아야 합니다.

## 현재 활성 요청값

- `temperature=0.6`
- `top_k=64`
- `top_p=0.95`
- `min_p=0.0`
- `Chunk Size=6`
- `Max Completion Tokens=512`
- `Request Timeout=180`
- `response_format=json_schema`

## 현재 compose 기준값

- `ctx-size=4096`
- `n_gpu_layers=23`
- `threads=12`
- `--swa-full=enabled`
- `reasoning=off`
- `reasoning-budget=0`
- `reasoning-format=none`

## 참고 이미지 버전

- Image tag: `ghcr.io/ggml-org/llama.cpp:server-cuda`
- Pull policy: `always`
- 현재 관측된 내부 `llama-server --version`은 구현 시점 기준 `8740`이며, 최신 상태는 실행 로그나 benchmark summary의 digest/version 기록을 확인하세요.

## 관련 문서

- [README.md](../../paddleocr_vl_docker_files/README.md)

벤치마크 preset, 보고서, 차트, 실험 문서는 제품 브랜치가 아니라 `benchmarking/lab` 브랜치에서만 관리합니다.
