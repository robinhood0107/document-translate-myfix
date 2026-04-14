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

앱은 현재 파이프라인에 맞춰 검출된 텍스트 블록 crop을 잘라 아래 OpenAI-compatible 형식으로 전송합니다.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Please perform OCR on this image and output only a JSON array of recognized text regions, where each item has \"bbox_2d\" and \"text_content\". Do not translate. Do not explain. Do not add markdown or code fences."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,..."
          }
        }
      ]
    }
  ],
  "temperature": 0,
  "top_k": 1,
  "max_completion_tokens": 256
}
```

## 현재 기준값

- port: `28081 -> 8080`
- `ctx-size=4096`
- `n_parallel=1`
- `threads=12`
- `n_gpu_layers=99`

## 참고

- 현재 제품 파이프라인은 `RT-DETR-v2`가 먼저 텍스트 블록을 만들고, `MangaLMM`는 각 block crop의 JSON OCR 결과에서 `text_content`를 읽습니다.
- `bbox_2d`는 앱에서 원본 crop 좌표와 원본 페이지 좌표로 역매핑합니다.
- resize는 항상 적용하지 않고, 큰 crop만 조건부로 줄여 안정성을 높입니다.
- 루트 `docker-compose.yaml`은 Gemma 번역 서버용으로 유지하고, MangaLMM는 `mangalmm_docker_files/`에서 별도로 관리합니다.
