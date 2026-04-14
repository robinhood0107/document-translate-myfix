# MangaLMM 로컬 서버 설정 가이드

이 문서는 `MangaLMM`를 현재 저장소 기준으로 설정하는 방법을 정리합니다.

## 준비

- `testmodel/` 폴더에 아래 두 파일을 둡니다.
  - `MangaLMM.Q8_0.gguf`
  - `MangaLMM.mmproj-Q8_0.gguf`
- 현재 기준 Docker image는 `ghcr.io/ggml-org/llama.cpp:server-cuda`입니다.
- 최신 이미지를 실제로 반영하려면 실행 전에 `docker compose pull --policy always`를 먼저 수행한 뒤 `up -d --force-recreate`를 사용하세요.

## 서버 실행

저장소 루트에서 실행:

```bash
docker compose -f mangalmm_docker_files/docker-compose.yaml pull --policy always
docker compose -f mangalmm_docker_files/docker-compose.yaml up -d --force-recreate
```

앱 설정:

- OCR: `MangaLMM`
- Server URL: `http://127.0.0.1:28081/v1`

## 현재 요청 형식

앱은 현재 파이프라인에 맞춰 `페이지 전체 1장`만 아래 OpenAI-compatible 형식으로 전송합니다.
핵심 계약은 `PNG data URL + image -> text 순서 + full-page single-shot`입니다.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,..."
          }
        },
        {
          "type": "text",
          "text": "Please perform OCR on this image and output the recognized Japanese text along with its position (grounding)."
        }
      ]
    }
  ],
  "temperature": 0.1,
  "top_k": 1,
  "top_p": 0.001,
  "repeat_penalty": 1.05,
  "max_completion_tokens": 2048
}
```

`dense` 페이지는 아래 하이브리드 프롬프트를 사용합니다.

```text
Please perform OCR on this image and output the recognized Japanese text along with its position (grounding) as a JSON array. Each item must contain "bbox_2d" and "text_content". Do not translate.
```

## 현재 기준값

- port: `28081 -> 8080`
- `ctx-size=4096`
- `n_parallel=1`
- `threads=12`
- `n_gpu_layers=99`

## 참고

- 현재 제품 파이프라인은 `RT-DETR-v2`가 먼저 텍스트 블록을 만들고, `MangaLMM`는 `페이지 전체 1회 OCR`의 JSON 결과를 각 `TextBlock`에 다시 매칭합니다.
- 일본어 `Optimal+`에서는 `MangaLMM` 계약을 내부 고정값으로 사용합니다.
  - `standard`: `1224 x 1728`, grounding prompt, `2048 tokens`
  - `dense`: `900 x 1270`, hybrid JSON grounding prompt, `1024 tokens`
  - 실패 시에만 `900 x 1270 + dense prompt + 4096 tokens` 방어 재시도를 허용합니다.
- detector의 `bubble_xyxy`는 말풍선 소유권 게이트로, `xyxy`는 같은 말풍선 안에서 최종 블록을 고르는 정밀 박스로 사용합니다.
- `bbox_2d`는 실제 요청 크기 기준 `scale_x`, `scale_y`를 사용해 원본 페이지 좌표로 역매핑합니다.
- resize는 비율 유지 downscale만 허용하며, `standard`와 `dense` 프로파일 중 하나를 페이지 단위로 한 번만 고릅니다.
- `standard`는 대체로 `1224 x 1728` 상한, `dense`는 대체로 `900 x 1270` 상한을 사용합니다.
- 제품 기본 경로에서는 tile, crop, rescue macro를 사용하지 않습니다.
- 재시도는 방어 목적일 때만 최대 2단계까지 허용하며, `bbox_2d`가 없는 text-only 결과는 성공으로 보지 않습니다.
- 루트 `docker-compose.yaml`은 Gemma 번역 서버용으로 유지하고, MangaLMM는 `mangalmm_docker_files/`에서 별도로 관리합니다.
