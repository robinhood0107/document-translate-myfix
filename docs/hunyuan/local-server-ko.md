# HunyuanOCR 로컬 서버 설정 가이드

이 문서는 `HunyuanOCR`를 현재 저장소 기준으로 설정하는 방법을 정리합니다.

## 준비

- `testmodel/` 폴더에 아래 두 파일을 둡니다.
  - `HunyuanOCR-BF16.gguf`
  - `mmproj-HunyuanOCR-BF16.gguf`
- 현재 기준 Docker image는 `local/llama.cpp:server-cuda-b8672`입니다.
- `b8672`를 사용하는 이유는 `HunyuanOCR` 지원이 포함된 `llama.cpp` 서버 빌드를 맞추기 위해서입니다.

## 서버 실행

저장소 루트에서 실행:

```bash
docker compose -f hunyuanocr_docker_files/docker-compose.yaml up -d
```

앱 설정:

- OCR: `HunyuanOCR`
- Server URL: `http://127.0.0.1:28080/v1`

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
          "text": "OCR"
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
  "max_completion_tokens": 256
}
```

## 현재 기준값

- port: `28080 -> 8080`
- `ctx-size=4096`
- `n_parallel=1`
- `threads=12`
- `n_gpu_layers=80`

## 참고

- 현재 제품 파이프라인은 페이지 전체 spotting 결과로 좌표를 다시 받지 않습니다.
- `RT-DETR-v2`가 먼저 텍스트 블록을 만들고, `HunyuanOCR`는 각 block crop의 텍스트만 읽습니다.
- 루트 `docker-compose.yaml`은 Gemma 번역 서버용으로 유지하고, HunyuanOCR는 `hunyuanocr_docker_files/`에서 별도로 관리합니다.
