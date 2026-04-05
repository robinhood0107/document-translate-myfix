# Gemma 프로필 메모

기준 날짜: `2026-04-05`

이 문서는 현재 워크스페이스에서 실제로 적용한 번역 설정과, 참고용 Usage 메모만 남긴 문서입니다.

## 현재 적용한 번역 설정

### 앱 요청값

현재 `Custom Local Server(Gemma)` 번역 요청은 아래 값으로 보냅니다.

| 항목 | 현재 값 |
| --- | --- |
| temperature | `1.0` |
| top_k | `0` |
| top_p | `1.0` |
| min_p | `0.05` |
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

현재 워크스페이스의 `docker-compose.yaml` 기준 값입니다.

| 항목 | 현재 값 |
| --- | --- |
| image | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| model path | `/models/gemma-4-26b-a4b-it-heretic.q3_k_m.gguf` |
| ctx-size | `4096` |
| n_gpu_layers | `8` |
| reasoning | `off` |
| reasoning-budget | `0` |
| reasoning-format | `none` |
| `--swa-full` | `enabled` |

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
