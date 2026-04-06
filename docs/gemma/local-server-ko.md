# Gemma 로컬 서버 설정 가이드

이 문서는 `Custom Local Server(Gemma)`를 현재 저장소 기준으로 설정하는 방법을 정리합니다.

## 준비

- 모델 파일을 `testmodel/` 폴더에 둡니다.
- 현재 compose 기준 모델 파일은 `gemma-4-26b-a4b-it-heretic.q3_k_m.gguf`입니다.

## 서버 실행

저장소 루트에서 실행:

```bash
docker compose up -d
```

앱 설정:

- Endpoint URL: `http://127.0.0.1:18080/v1`
- Model: `gemma-4-26b-a4b-it-heretic.q3_k_m.gguf`

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

- Image tag: `local/llama.cpp:server-cuda-b8665`
- Pull policy: `never`
- `llama.cpp` 계열 빌드 기준: `b8665`

## 관련 문서

- [README.md](../../paddleocr_vl_docker_files/README.md)

벤치마크 preset, 보고서, 차트, 실험 문서는 제품 브랜치가 아니라 `benchmarking/lab` 브랜치에서만 관리합니다.
