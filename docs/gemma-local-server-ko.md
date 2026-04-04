# Gemma 로컬 서버 설정 가이드

이 문서는 `Custom Local Server(Gemma)` 번역기를 현재 저장소의 `docker-compose.yaml` 기준으로 설정하는 방법을 설명합니다.

## 1. 준비

- 모델 파일을 `testmodel/` 폴더에 둡니다.
- 기본 모델명은 `gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf`입니다.

## 2. Docker 서버 실행

저장소 루트에서 아래 명령을 실행합니다.

```bash
docker compose up -d
```

정상 실행 후 앱에서는 아래 값을 사용합니다.

- Endpoint URL: `http://127.0.0.1:18080/v1`
- Model: `gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf`

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

## 4. 권장 시작값

- `Chunk Size`: `4`
- `Max Completion Tokens`: `512`
- `Request Timeout (sec)`: `180`

## 5. 응답이 잘릴 때

아래 순서로 조정하는 것을 권장합니다.

1. 앱의 `Chunk Size`를 더 작게 낮춥니다.
2. `docker-compose.yaml`의 `LLAMA_CTX_SIZE`를 더 크게 올립니다.
3. 필요하면 `Max Completion Tokens`도 함께 조정합니다.

## 6. 참고

- 현재 로컬 번역기는 generic local server가 아니라 Gemma Docker 서버에 맞춰져 있습니다.
- `Custom Service`는 별도의 인증형 OpenAI 호환 서비스용입니다.
