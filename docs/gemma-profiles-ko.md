# Gemma 프로필 메모

기준 날짜: `2026-04-05`

이 문서는 현재 워크스페이스에서 실제로 적용한 번역 설정과, 참고용 Usage 메모만 남긴 문서입니다.

이 문서에서는 기준선을 둘로 나눠 기록합니다.

- `merged baseline`: 이미 `develop`에 머지된 기준
- `live-ops baseline`: 현재 로컬 운영/실험 기준

## 현재 적용한 번역 설정

### 앱 요청값

현재 `Custom Local Server(Gemma)` 번역 요청은 아래 값으로 보냅니다.

| 항목 | 현재 값 |
| --- | --- |
| temperature | `0.5` |
| top_k | `64` |
| top_p | `0.95` |
| min_p | `0.0` |
| response_format | `json_object` |
| image input | `off` |

추가로 현재 코드 기준 기본값은 아래와 같습니다.

| 항목 | 현재 값 |
| --- | --- |
| Chunk Size 기본값 | `4` |
| Max Completion Tokens 기본값 | `512` |
| Request Timeout 기본값 | `180` |
| Raw Response Log 기본값 | `False` |

### 현재 docker-compose 설정

현재 로컬 `live-ops baseline`의 `docker-compose.yaml` 기준 값입니다.

| 항목 | 현재 값 |
| --- | --- |
| image | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| model path | `/models/gemma-4-26b-a4b-it-heretic.q3_k_m.gguf` |
| ctx-size | `4096` |
| threads | `12` |
| n_gpu_layers | `22` |
| reasoning | `off` |
| reasoning-budget | `0` |
| reasoning-format | `none` |
| `--swa-full` | `enabled` |
| `paddleocr-server --device` | `cpu` |

### merged baseline 참고

이미 `develop`에 머지된 기준 compose 기본값은 `n_gpu_layers=8`이었습니다. 이번 브랜치부터는 문서에 `merged baseline`과 `live-ops baseline`을 분리해서 기록합니다.

## 설정 이력

앞으로 Gemma 관련 설정이 바뀌면 이 섹션에 커밋 기준으로 계속 추가합니다.

### `9285b71` `chore(docker): add local llama server compose`

Gemma 로컬 `llama.cpp` Docker compose가 처음 들어온 시점입니다.

| 항목 | 당시 값 |
| --- | --- |
| image | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| service/container | `llama-server-gpu` / `llama-cpp-server-cuda` |
| model path | `/models/gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf` |
| ctx-size | `2048` |
| parallel | `1` |
| threads | `8` |
| n_gpu_layers | `8` |
| reasoning flags | 없음 |
| `--swa-full` | 없음 |

### `b8712ee` `feat(translation): specialize gemma local server`

`Custom Local Server(Gemma)` 전용 번역 경로가 들어온 첫 시점입니다.

#### 당시 앱 기본값

| 항목 | 당시 값 | 비고 |
| --- | --- | --- |
| Endpoint 기본 placeholder | `http://127.0.0.1:18080/v1` | Credentials UI |
| Model 기본 placeholder | `gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf` | Credentials UI 기본 표시값 |
| Chunk Size 기본값 | `4` | Gemma Settings UI |
| Max Completion Tokens 기본값 | `512` | Gemma Settings UI |
| Request Timeout 기본값 | `180`초 | Gemma Settings UI |
| Raw Response Log 기본값 | `False` | Gemma Settings UI |
| temperature | `0.0` | 앱이 요청마다 직접 보냄 |
| top_p | `1.0` | 앱이 요청마다 직접 보냄 |
| response_format | `json_object` | JSON 강제 |
| image input | `False` | 현재 Gemma 번역 경로는 이미지 입력 비활성 |
| thinking tag | 없음 | 시스템 프롬프트 맨 위에 `<|think|>`를 넣지 않음 |

#### 당시 앱이 보내지 않던 샘플러 값

- `top_k`
- `min_p`
- `top-n-sigma`
- `adaptive-target`
- `adaptive-decay`

당시 `llama.cpp server --help` 기준 기본값은 아래처럼 문서에 기록됐습니다.

| 항목 | 당시 유효 기본값 | 비고 |
| --- | --- | --- |
| top_k | `40` | 앱이 override하지 않음 |
| top_p | 서버 기본 `0.95`지만 앱이 `1.0`으로 override | 현재는 앱 값이 우선 |
| min_p | `0.05` | 앱이 override하지 않음 |
| top-n-sigma | `-1.0` | 비활성 |
| adaptive-target | `-1.0` | 비활성 |
| adaptive-decay | `0.90` | adaptive-target이 꺼져 있어 사실상 영향 없음 |

#### 당시 docker-compose 값

| 항목 | 당시 값 | 비고 |
| --- | --- | --- |
| model path | `/models/gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf` | 초기 Gemma 기본값 |
| ctx-size | `4096` | `-c ${LLAMA_CTX_SIZE:-4096}` |
| parallel | `1` | `-np ${LLAMA_N_PARALLEL:-1}` |
| threads | `8` | `-t ${LLAMA_THREADS:-8}` |
| n_gpu_layers | `8` | `--n-gpu-layers ${LLAMA_N_GPU_LAYERS:-8}` |
| reasoning | `off` | 서버 thinking 비활성 |
| reasoning-budget | `0` | 즉시 답변 모드 |
| reasoning-format | `none` | thought를 별도 필드로 분리하지 않음 |
| predict | 설정 안 함 | 서버 기본 `-1` |
| reasoning-budget-message | 설정 안 함 | 없음 |
| image-min-tokens | 설정 안 함 | 모델 기본값 사용 |
| image-max-tokens | 설정 안 함 | 모델 기본값 사용 |
| `--swa-full` | 꺼짐 | 당시 없음 |

### `370b335` `docs(gemma): add profile tuning guide`

이 커밋에서는 설정 자체를 바꾸지는 않았고, 위 `b8712ee` 시점의 값들을 문서로 정리해 남겼습니다. 당시 문서에는 번역용 추천값과 creative/chat용 추천값도 함께 적혀 있었습니다.

### `c750008` `docs(gemma): pin tested llama image version`

이 커밋에서는 실행 중이던 `llama.cpp` Docker 이미지를 재현 가능하도록 문서에 고정했습니다.

| 항목 | 기록된 값 |
| --- | --- |
| image tag | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| exact digest | `ghcr.io/ggml-org/llama.cpp@sha256:0d60155f9cbd5118d02568d90f505638259d85f6f1cc4ac98d0f1002001e1f7a` |
| image label version | `24.04` |
| `llama.cpp --version` | `8660 (d00685831)` |

### `8f60916` `chore(gemma): align local sampler settings`

현재 사용 중인 값으로 맞춘 커밋입니다.

| 항목 | 현재 값 |
| --- | --- |
| model placeholder | `gemma-4-26b-a4b-it-heretic.q3_k_m.gguf` |
| docker model path | `/models/gemma-4-26b-a4b-it-heretic.q3_k_m.gguf` |
| temperature | `1.0` |
| top_k | `0` |
| top_p | `1.0` |
| min_p | `0.05` |
| response_format | `json_object` |
| reasoning | `off` |
| reasoning-budget | `0` |
| reasoning-format | `none` |
| `--swa-full` | `enabled` |

### `6b6b15e` `feat(benchmark): add stable gemma translation presets`

현재 브랜치에서 Gemma 번역 안정화 실험을 시작한 커밋입니다.

핵심 변경:

- 앱 기본 sampler를 번역 안정화 기준으로 조정
  - `temperature=1.0`
  - `top_k=64`
  - `top_p=0.95`
  - `min_p=0.0`
- benchmark preset 추가
  - `gemma-translation-stable-22`
  - `gemma-translation-stable-24`
  - `gemma-translation-stable-24-ctx3072`
  - `gemma-translation-stable-22-t07`
  - `gemma-translation-stable-22-t05`
- Gemma benchmark 지표 추가
  - `gemma_json_retry_count`
  - `gemma_chunk_retry_events`
  - `gemma_truncated_count`
  - `gemma_empty_content_count`

### `로컬 미푸시 후속 조정`

현재 브랜치에서는 representative benchmark 결과를 반영해 기본 운영값도 아래처럼 조정했습니다.

| 항목 | 현재 운영값 |
| --- | --- |
| temperature | `0.5` |
| top_k | `64` |
| top_p | `0.95` |
| min_p | `0.0` |
| Gemma threads | `12` |
| Gemma n_gpu_layers | `22` |
| PaddleOCR front device | `cpu` |

## Usage 참고 메모

```text
Usage
Google recommends the following sampler settings:

temperature = 1.0
top_k = 64
top_p = 0.95
min_p = 0.0

For creative writing, I use:

temperature = 1.0
top_k = 0
top_p = 1.0
min_p = 0.05
top-n-sigma = 1.0
adaptive-target = 0.7
adaptive-decay = 0.9

For the image encoder:

image-min-tokens: 70
image-max-tokens: 1120

While it's reasoning is not excessive, sometimes I do want to limit it:

predict: 16384
reasoning-budget: 8192
reasoning-budget-message: "... I think I've explored this enough, time to respond."

Within llama.cpp and koboldcpp, ensure that --swa-full is enabled as this model uses Sliding Window Attention (SWA).

Thinking
In order to enable thinking, add <|think|> at the top of your system prompt:

<|think|>
You are a helpful assistant.

Conversely, to disable thinking simply omit <|think|> from your system prompt:

You are a helpful assistant.

To parse the thinking, use the following in SillyTavern or your platform of choice:

prefix: <|channel>thought
postfix: <channel|>
```
