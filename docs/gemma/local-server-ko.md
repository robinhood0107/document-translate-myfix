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
- `Chunk Size=4`
- `Max Completion Tokens=512`
- `Request Timeout=180`
- `response_format=json_object`

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
- Digest: `ghcr.io/ggml-org/llama.cpp@sha256:0d60155f9cbd5118d02568d90f505638259d85f6f1cc4ac98d0f1002001e1f7a`
- `llama.cpp --version`: `8660 (d00685831)`

재현용 pull:

```bash
docker pull ghcr.io/ggml-org/llama.cpp@sha256:0d60155f9cbd5118d02568d90f505638259d85f6f1cc4ac98d0f1002001e1f7a
```

## 관련 문서

- [profiles-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/gemma/profiles-ko.md)
- [translation-optimization-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/gemma/translation-optimization-ko.md)
- [resource-strategy-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/resource-strategy-ko.md)
- [workflow-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/benchmark/workflow-ko.md)
