# 빠른 설치 가이드

이 문서는 처음 체크아웃한 뒤 가능한 가장 짧은 경로로 앱을 실행하고, 로컬 런타임까지 붙이는 방법을 정리합니다.

## 1. 준비물

- Windows 10/11
- Python 3.11
- Git
- GPU 지원이 켜진 Docker Desktop
- 로컬 Gemma / HunyuanOCR / PaddleOCR VL 가속을 쓰려면 NVIDIA 드라이버와 CUDA 호환 GPU

## 2. 저장소 실행

저장소 루트에서 아래 런처 중 하나를 실행합니다.

기본 경로:

```bat
run_comic.bat
```

CUDA13 경로:

```bat
run_comic_cuda13.bat
```

이 런처들은 `.venv-win`, `.venv-win-cuda13` 환경을 자동 bootstrap합니다.

## 3. 선택 로컬 런타임

### Gemma 로컬 번역 런타임

- compose 파일: `/docker-compose.yaml`
- Docker 이미지: `ghcr.io/ggml-org/llama.cpp:server-cuda`
- 참고 링크:
  - [llama.cpp](https://github.com/ggml-org/llama.cpp)
  - [Gemma](https://ai.google.dev/gemma)

실행:

```bash
docker compose pull --policy always
docker compose up -d --force-recreate
```

앱에서는 `Custom Local Server(Gemma)`를 선택합니다.

### HunyuanOCR 로컬 런타임

- compose 파일: `/hunyuanocr_docker_files/docker-compose.yaml`
- Docker 이미지: `ghcr.io/ggml-org/llama.cpp:server-cuda`
- 참고 링크:
  - [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)
  - [llama.cpp](https://github.com/ggml-org/llama.cpp)

필수 로컬 모델 파일:

- `HunyuanOCR-BF16.gguf`
- `mmproj-HunyuanOCR-BF16.gguf`

실행:

```bash
docker compose -f hunyuanocr_docker_files/docker-compose.yaml pull --policy always
docker compose -f hunyuanocr_docker_files/docker-compose.yaml up -d --force-recreate
```

### PaddleOCR VL 로컬 런타임

- compose 파일: `/paddleocr_vl_docker_files/docker-compose.yaml`
- Docker 이미지: `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-offline`
- 참고 링크:
  - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
  - [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

실행:

```bash
docker compose -f paddleocr_vl_docker_files/docker-compose.yaml pull --policy always
docker compose -f paddleocr_vl_docker_files/docker-compose.yaml up -d --force-recreate
```

bundle 파일 설명은 [/paddleocr_vl_docker_files/README.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/paddleocr_vl_docker_files/README.md)를 참고하세요.

## 4. 권장 앱 설정

- 워크플로 모드: `Stage-Batched Pipeline (Recommended)`
- OCR: `Optimal (HunyuanOCR / PaddleOCR VL)`
- 번역기: 로컬 Gemma 런타임을 켰다면 `Custom Local Server(Gemma)`

기본 OCR 라우팅:

- 중국어 -> `HunyuanOCR`
- 일본어 / 기타 언어 -> `PaddleOCR VL`

## 5. 선택 알림 설정 (ntfy)

열기:

- `Settings > Notifications`

설정 항목:

- ntfy 알림 사용
- 서버 URL
- topic
- 선택 access token
- 완료 / 실패 / 취소 전송 여부

이 앱은 ntfy로 **텍스트만** 보내며, 본문은 ntfy 기본 텍스트 제한을 넘지 않도록 줄입니다. 첨부파일은 보내지 않습니다.

공식 ntfy 문서:

- [ntfy publish docs](https://docs.ntfy.sh/publish/)
- [ntfy config docs](https://docs.ntfy.sh/config/)

## 6. 현재 제품 코드가 실제로 참고하는 모델/런타임 링크

검출 / 마스킹:

- [RT-DETR v2](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)
- [ComicTextDetector (CTD)](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3)
- [Font Detector](https://huggingface.co/gyrojeff/YuzuMarker.FontDetection)

OCR:

- [MangaOCR](https://huggingface.co/kha-white/manga-ocr-base)
- [MangaOCR ONNX](https://huggingface.co/mayocream/manga-ocr-onnx)
- [Pororo OCR](https://huggingface.co/ogkalu/pororo)
- [RapidOCR / PPOCRv5](https://www.modelscope.cn/models/RapidAI/RapidOCR)
- [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)

인페인팅:

- [AOT](https://huggingface.co/ogkalu/aot-inpainting)
- [AnimeMangaInpainting / LaMa legacy](https://github.com/Sanster/models/releases/tag/AnimeMangaInpainting)
- [lama_large_512px](https://huggingface.co/dreMaz/AnimeMangaInpainting)
- [lama_mpe](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3)
- [MI-GAN](https://github.com/Sanster/models/releases/tag/migan)

## 7. 같이 보면 좋은 문서

- [/README.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/README.md)
- [/README_ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/README_ko.md)
- [/docs/gemma/local-server-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/gemma/local-server-ko.md)
- [/docs/hunyuan/local-server-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/hunyuan/local-server-ko.md)
