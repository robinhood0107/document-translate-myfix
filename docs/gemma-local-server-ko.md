# Gemma 로컬 서버 설정 가이드

이 문서는 `Custom Local Server(Gemma)` 번역기를 현재 저장소의 `docker-compose.yaml` 기준으로 설정하는 방법을 설명합니다.

현재 문서에서는 아래 두 기준을 구분합니다.

- `merged baseline`: 이미 `develop`에 머지된 기준
- `live-ops baseline`: 현재 로컬 운영/실험 기준

## 1. 준비

- 모델 파일을 `testmodel/` 폴더에 둡니다.
- 현재 `docker-compose.yaml` 기준 모델 파일은 `gemma-4-26b-a4b-it-heretic.q3_k_m.gguf`입니다.

## 2. Docker 서버 실행

저장소 루트에서 아래 명령을 실행합니다.

```bash
docker compose up -d
```

정상 실행 후 앱에서는 아래 값을 사용합니다.

- Endpoint URL: `http://127.0.0.1:18080/v1`
- Model: `gemma-4-26b-a4b-it-heretic.q3_k_m.gguf`

## 3. 앱 설정

`Settings > Tools > Translator`에서 `Custom Local Server(Gemma)`를 선택합니다.

`Settings > Credentials`에서는 아래 값을 입력합니다.

- `Endpoint URL`
- `Model`

`Settings > Gemma Local Server Settings`에서는 아래 값을 조절할 수 있습니다.

- `Chunk Size`: 한 번에 번역할 블록 수
- `Max Completion Tokens`: Gemma 응답 최대 토큰 수
- `Request Timeout (sec)`: 요청 타임아웃
- `Raw Response Log`: 원시 응답 로그 출력 여부

## 4. 현재 적용된 번역 요청값

- `temperature`: `1.0`
- `top_k`: `0`
- `top_p`: `1.0`
- `min_p`: `0.05`
- `Chunk Size` 기본값: `4`
- `Max Completion Tokens` 기본값: `512`
- `Request Timeout (sec)` 기본값: `180`
- `response_format`: `json_object`

## 4-1. 현재 live-ops baseline

현재 로컬 운영 기준 compose 값은 아래와 같습니다.

- `ctx-size`: `4096`
- `n_gpu_layers`: `20`
- `--swa-full`: `enabled`
- `reasoning`: `off`
- `reasoning-budget`: `0`
- `reasoning-format`: `none`

## 5. 응답이 잘릴 때

아래 순서로 조정하는 것을 권장합니다.

1. 앱의 `Chunk Size`를 더 작게 낮춥니다.
2. `docker-compose.yaml`의 `LLAMA_CTX_SIZE`를 더 크게 올립니다.
3. 필요하면 `Max Completion Tokens`도 함께 조정합니다.

## 6. 참고

- 현재 로컬 번역기는 generic local server가 아니라 Gemma Docker 서버에 맞춰져 있습니다.
- `Custom Service`는 별도의 인증형 OpenAI 호환 서비스용입니다.
- 현재 `docker-compose.yaml`은 `--swa-full`이 켜진 상태입니다.

## 7. 현재 확인된 llama.cpp Docker 이미지 버전

아래 값은 이 워크스페이스에서 `2026-04-05`에 실제 실행 중이던 `gemma-local-server` 컨테이너 기준으로 확인한 값입니다.

### 확인된 이미지 정보

- Image tag: `ghcr.io/ggml-org/llama.cpp:server-cuda`
- Exact digest: `ghcr.io/ggml-org/llama.cpp@sha256:0d60155f9cbd5118d02568d90f505638259d85f6f1cc4ac98d0f1002001e1f7a`
- Image ID: `sha256:0d60155f9cbd5118d02568d90f505638259d85f6f1cc4ac98d0f1002001e1f7a`
- Image label version: `24.04`
- Image created at: `2026-04-04T06:17:58.828077388Z`
- `llama.cpp --version`: `version: 8660 (d00685831)`
- Build info: `built with GNU 14.2.0 for Linux x86_64`

### 나중에 동일한 이미지를 다시 받는 방법

가장 안전한 방법은 digest를 직접 지정해서 pull하는 것입니다.

```bash
docker pull ghcr.io/ggml-org/llama.cpp@sha256:0d60155f9cbd5118d02568d90f505638259d85f6f1cc4ac98d0f1002001e1f7a
```

태그만 쓰면 나중에 다른 빌드로 바뀔 수 있으므로, 재현성이 중요하면 위처럼 digest를 고정하는 것을 권장합니다.

### 참고 명령

현재 실행 중인 컨테이너에서 같은 정보를 다시 확인하고 싶다면 아래 명령을 사용할 수 있습니다.

```bash
docker inspect gemma-local-server --format '{{.Config.Image}} {{.Image}}'
docker image inspect ghcr.io/ggml-org/llama.cpp:server-cuda
docker run --rm ghcr.io/ggml-org/llama.cpp:server-cuda --version
```

## 8. 관련 문서

- [pipeline-resource-strategy-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/pipeline-resource-strategy-ko.md)
- [pipeline-benchmarking-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/pipeline-benchmarking-ko.md)
