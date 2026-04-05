# Gemma 로컬 서버 프로필 정리

이 문서는 현재 `Comic Translate`의 `Custom Local Server(Gemma)` 경로를 기준으로, 번역용 프로필과 creative/chat용 프로필을 어떻게 나누어 생각해야 하는지 정리한 문서다.

기준 날짜: `2026-04-05`

## 결론 요약

- 번역용과 creative/chat용은 같은 프로필로 쓰지 않는 편이 좋다.
- 번역용은 `JSON 안정성`, `낮은 랜덤성`, `reasoning off`가 중요하다.
- creative/chat용은 `자유도`, `긴 출력`, `thinking 허용`이 중요하다.
- 현재 `Comic Translate` 앱은 번역기이므로, 기본 운영은 번역용 프로필에 맞추고 creative/chat은 별도 프로필이나 별도 클라이언트로 분리하는 것이 맞다.

## 현재 내가 쓰고 있는 값

이 섹션은 현재 워크스페이스의 코드와 로컬 `docker-compose.yaml`을 기준으로 정리했다.

### 1. 현재 앱 기본값

출처:

- [custom_local_gemma.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/modules/translation/llm/custom_local_gemma.py)
- [gemma_local_server_page.py](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/app/ui/settings/gemma_local_server_page.py)

현재 앱에서 실제로 사용하는 값은 아래와 같다.

| 항목 | 현재 값 | 비고 |
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

### 2. 현재 앱이 보내지 않는 샘플러 값

현재 앱은 아래 항목을 요청 payload에 넣지 않는다.

- `top_k`
- `min_p`
- `top-n-sigma`
- `adaptive-target`
- `adaptive-decay`

즉, 이 값들은 현재 앱 안에서는 조절할 수 없고, `llama.cpp` 서버 기본값에 의존한다.

현재 `llama.cpp server --help` 기준 기본값은 아래와 같다.

| 항목 | 현재 유효 기본값 | 비고 |
| --- | --- | --- |
| top_k | `40` | 앱이 override하지 않음 |
| top_p | 서버 기본 `0.95`지만 앱이 `1.0`으로 override | 현재는 앱 값이 우선 |
| min_p | `0.05` | 앱이 override하지 않음 |
| top-n-sigma | `-1.0` | 비활성 |
| adaptive-target | `-1.0` | 비활성 |
| adaptive-decay | `0.90` | adaptive-target이 꺼져 있어 사실상 영향 없음 |

### 3. 현재 로컬 docker-compose 값

출처:

- [docker-compose.yaml](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docker-compose.yaml)

현재 워크스페이스의 `docker-compose.yaml` 기준 값은 아래와 같다.

| 항목 | 현재 값 | 비고 |
| --- | --- | --- |
| model path | `/models/gemma-4-26b-a4b-it-heretic.q3_k_m.gguf` | 현재 로컬 compose 기준 |
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
| swa-full | 설정 안 함 | 현재 꺼져 있음 |

## 현재 상태에서 보이는 중요한 포인트

### 1. 앱 기본 모델명과 현재 compose 모델명이 다르다

- 앱 placeholder 기본값은 `gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf`
- 현재 로컬 compose는 `gemma-4-26b-a4b-it-heretic.q3_k_m.gguf`

즉, UI에 보이는 기본 설명과 실제 로컬 서버에서 띄우는 모델 파일이 현재는 일치하지 않는다.

### 2. 현재 번역 경로는 creative/chat용으로 설계되어 있지 않다

현재 `Custom Local Server(Gemma)`는 아래 특성을 가진다.

- 전체 목적이 `번역`
- 출력 형식이 `정확한 JSON`
- reasoning이 켜지면 파싱이 흔들릴 수 있음
- chunk retry와 block key 검증이 들어감

즉, creative/chat용 샘플러를 그대로 번역 파이프라인에 넣으면 품질보다 `형식 붕괴`와 `JSON 실패`가 먼저 생길 가능성이 높다.

## 추천 프로필 분리

## 번역용 추천값

목표:

- 자연스러운 번역
- JSON 파싱 안정성
- block key 누락 최소화
- reasoning으로 인한 `message.content` 오염 방지

### 추천값

| 항목 | 추천값 | 이유 |
| --- | --- | --- |
| temperature | `0.0 ~ 0.2` | 번역 일관성과 JSON 안정성 |
| top_k | `64` | 표현 다양성은 조금 주되 흔들림은 제한 |
| top_p | `0.90 ~ 0.95` | 자연스러움과 안정성 균형 |
| min_p | `0.0` | 과한 필터링 방지 |
| top-n-sigma | `비활성` | JSON 작업에선 이득보다 흔들림 가능성 |
| adaptive-target | `비활성` | 번역용에서는 불필요 |
| adaptive-decay | `기본값 유지` | adaptive를 안 쓰면 사실상 무관 |
| max_completion_tokens | `512 ~ 768` | 페이지 번역 JSON에 충분 |
| predict | `크게 중요하지 않음` | 앱은 `max_completion_tokens`로 제어 중 |
| reasoning | `off` | JSON 파싱 안정성 |
| reasoning-budget | `0` | 즉답 |
| reasoning-budget-message | `없음` | reasoning 자체를 안 씀 |
| system prompt 상단 `<|think|>` | `넣지 않음` | thinking 방지 |
| response_format | `json_object` | 필수 |
| Chunk Size | `4` | 현재 기본값으로 무난 |
| Chunk Size fallback | `2` 또는 `3` | truncation이 날 때 |
| image-min-tokens / image-max-tokens | 사용 안 함 | 현재 번역 경로는 이미지 입력 비활성 |
| `--swa-full` | `켜기 권장` | Gemma SWA 대응 |

### 번역용 운영 메모

- 현재 `Comic Translate` 앱은 본질적으로 이 프로필에 맞게 설계되어 있다.
- 번역용에서는 `thinking`을 켜지 않는 편이 맞다.
- reasoning이 켜지면 `reasoning_content` 또는 thought tag가 섞여 JSON 파싱 실패 가능성이 올라간다.
- 창의성보다 `형식 유지`가 중요하므로 creative/chat 세팅을 그대로 가져오지 않는 것이 좋다.

## creative/chat용 추천값

목표:

- 자연스러운 자유 응답
- 긴 대화
- 생각 과정을 어느 정도 허용
- 번역 JSON 제약 해제

### 추천값

| 항목 | 추천값 | 이유 |
| --- | --- | --- |
| temperature | `1.0` | 자유도 확보 |
| top_k | `0` | 상한 제거 |
| top_p | `1.0` | 최대 자유도 |
| min_p | `0.05` | 너무 낮은 후보 억제 |
| top-n-sigma | `1.0` | creative 쪽 다양성 보강 |
| adaptive-target | `0.7` | 확률 분포 적응 |
| adaptive-decay | `0.9` | 완만한 적응 |
| predict | `16384` | 긴 응답 허용 |
| reasoning | `on` 또는 `auto` | thinking 허용 |
| reasoning-budget | `8192` | 과도한 thinking 제한 |
| reasoning-budget-message | `" ... I think I've explored this enough, time to respond."` | thinking 종료 유도 |
| system prompt 상단 `<|think|>` | `필요할 때만 추가` | thinking 활성화 |
| response_format | 사용 안 함 | 일반 채팅에서는 JSON 불필요 |
| image-min-tokens | `70` | 비전 입력 시 |
| image-max-tokens | `1120` | 비전 입력 시 |
| `--swa-full` | `켜기 권장` | Gemma SWA 대응 |

### creative/chat 운영 메모

- 이 프로필은 현재 `Comic Translate` 번역 파이프라인과는 성격이 다르다.
- creative/chat은 `JSON object 강제`, `chunk retry`, `block key 검증`과 잘 맞지 않는다.
- 따라서 creative/chat은 별도 클라이언트나 별도 모드로 분리하는 것이 좋다.

## 실제 운영 권장안

## 권장안 A: 가장 현실적인 운영

### 번역은 Comic Translate

- 번역용 프로필 유지
- reasoning off
- JSON strict
- chunked translation 유지

### creative/chat은 별도 클라이언트

- OpenAI-compatible client
- SillyTavern
- Open WebUI
- 별도 `curl` 또는 테스트 UI

이 방식이 가장 단순하고 안정적이다.

## 권장안 B: 나중에 앱 안에 두 프로필을 모두 넣기

앱 안에 아래 선택지를 추가하면 된다.

- `Gemma Profile: Translation (JSON)`
- `Gemma Profile: Creative / Chat`

그리고 UI에 아래 항목을 추가한다.

- `temperature`
- `top_k`
- `top_p`
- `min_p`
- `top_n_sigma`
- `adaptive_target`
- `adaptive_decay`
- `predict`
- `reasoning`
- `reasoning_budget`
- `reasoning_budget_message`
- `enable_thinking_tag`
- `image_min_tokens`
- `image_max_tokens`

하지만 현재 `Comic Translate`는 번역 앱이므로, 우선순위는 `번역용 프로필 완성`이 더 높다.

## 현재 앱에서 바로 되는 것 / 안 되는 것

### 지금 바로 되는 것

- `temperature`
- `top_p`
- `max_completion_tokens`
- `chunk_size`
- `request_timeout`
- `raw_response_logging`
- `reasoning off` 상태 유지

### 지금 바로 안 되는 것

- `top_k`
- `min_p`
- `top-n-sigma`
- `adaptive-target`
- `adaptive-decay`
- `predict`
- `reasoning-budget-message`
- `image-min-tokens`
- `image-max-tokens`
- `creative/chat 전용 프로필 분기`

즉, creative/chat 추천값을 제대로 쓰려면 앱 코드와 UI를 추가로 확장해야 한다.

## 실전 추천

현재 프로젝트 구조를 기준으로 한 추천은 아래와 같다.

### 번역용

- 앱:
  - `temperature = 0.1`
  - `top_p = 0.95`
  - `max_completion_tokens = 512`
  - `chunk_size = 4`
  - `reasoning off`
- 서버:
  - `--swa-full` 추가
  - `ctx-size = 4096`
  - `n-gpu-layers = 8`에서 시작

### creative/chat용

- 별도 클라이언트 사용
- 서버:
  - `--swa-full`
  - `--reasoning on` 또는 `auto`
  - `--reasoning-budget 8192`
  - `--reasoning-budget-message "... I think I've explored this enough, time to respond."`
- 요청:
  - `temperature = 1.0`
  - `top_k = 0`
  - `top_p = 1.0`
  - `min_p = 0.05`
  - `top-n-sigma = 1.0`
  - `adaptive-target = 0.7`
  - `adaptive-decay = 0.9`
  - `predict = 16384`

## 마지막 정리

- 번역용과 creative/chat용은 같은 프로필로 운영하지 않는 것이 좋다.
- 현재 `Comic Translate`는 번역용 프로필에 맞게 유지하는 것이 맞다.
- creative/chat용은 별도 모드 또는 별도 클라이언트로 분리하는 것이 안정적이다.
- 현재 기준에서 가장 먼저 손볼 만한 것은 `--swa-full` 추가와, 앱 쪽에 `top_k/min_p` 같은 샘플러 확장 여부를 결정하는 일이다.
