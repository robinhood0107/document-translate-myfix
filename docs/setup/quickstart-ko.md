# 빠른 설치 가이드

이 문서는 처음 체크아웃한 뒤 가능한 가장 짧은 경로로 앱을 실행하고, 로컬 런타임까지 붙이는 방법을 정리합니다.

## 1. 준비물

- Windows 10/11
- Python 3.12 이상
- Git
- GPU 지원이 켜진 Docker Desktop
- 로컬 Gemma / HunyuanOCR / PaddleOCR VL 가속을 쓰려면 NVIDIA 드라이버와 CUDA 호환 GPU
- 최초 Gemma volume 준비 검사에 필요한 `C:` 여유 공간 60 GiB 이상

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
CUDA13 경로는 공식 ONNX Runtime CUDA13 nightly feed를 사용합니다. CUDA13 GPU wheel은 아직 기본 PyPI `onnxruntime-gpu` 패키지 경로가 아니기 때문입니다.

수동으로 venv를 만들고 pip 한 번으로 설치하려면 아래 중 하나를 사용합니다.

```bat
py -3.12 -m venv .venv-win
.venv-win\Scripts\python.exe -m pip install -r requirements-cuda12.txt
```

```bat
py -3.12 -m venv .venv-win-cuda13
.venv-win-cuda13\Scripts\python.exe -m pip install -r requirements-cuda13.txt
```

`requirements.txt`는 기본 Windows CUDA12 런타임용 별칭이며, 공통 의존성은 `requirements-base.txt`에 모아둡니다.

## 3. 선택 로컬 런타임

### Gemma 로컬 번역 런타임

- compose 파일: `/docker-compose.yaml`
- Docker 이미지: `ghcr.io/ggml-org/llama.cpp@sha256:22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb`
- 참고 링크:
  - [llama.cpp](https://github.com/ggml-org/llama.cpp)
  - [Gemma](https://ai.google.dev/gemma)

Windows PowerShell에서 버전이 지정된 external model volume을 한 번 준비합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_gemma_runtime.ps1 -Mode Prepare `
  -CandidateModelPath 'C:\ExampleWorkspace\models\Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ4_XS.gguf' `
  -LegacyModelPath 'C:\ExampleWorkspace\models\gemma-4-26B-IQ4_NL.gguf'
```

앱에서는 `Custom Local Server(Gemma)`를 선택합니다. 관리 런타임이 준비된 volume을 read-only로 마운트하고 정확히 준비된 컨테이너를 자동으로 시작합니다.

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
- Docker 이미지: `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server@sha256:d0d32c04a2119613d25a0a4c292e165ccc107954b74580613cf59e378037f8f5`
- 참고 링크:
  - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
  - [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

실행:

```bash
docker compose -f paddleocr_vl_docker_files/docker-compose.yaml pull
docker compose -f paddleocr_vl_docker_files/docker-compose.yaml up -d --force-recreate
```

첫 번째 명령은 고정된 이미지를 준비할 때 한 번만 실행합니다. 평상시 앱
시작은 구성이 정확히 같은 중지 컨테이너를 재사용하며 이미지를 다시 pull하지
않습니다. Stage-Batched 폴더 처리는 `Settings > PaddleOCR VL Settings`에서
관리하는 exact 영구 OCR 결과 캐시도 사용할 수 있습니다.

bundle 파일 설명은 [/paddleocr_vl_docker_files/README.md](/paddleocr_vl_docker_files/README.md)를 참고하세요.

## 4. 권장 앱 설정

- 워크플로 모드: `Stage-Batched Pipeline (Recommended)`
- OCR: `Optimal (HunyuanOCR / PaddleOCR VL)`
- 번역기: Gemma volume 준비 후 `Custom Local Server(Gemma)`

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

- [/README.md](/README.md)
- [/README_ko.md](/README_ko.md)
- [/docs/gemma/local-server-ko.md](/docs/gemma/local-server-ko.md)
- [/docs/hunyuan/local-server-ko.md](/docs/hunyuan/local-server-ko.md)

## 8. 공식 Windows 릴리스 패키지

공식 Windows 릴리스 패키지는 `main`에 포함된 커밋에만 `vX.Y.Z` 태그를 달았을 때 생성됩니다.

- 릴리스 트리거: Git 태그 push
- 허용 태그 형식: `vX.Y.Z`
- 빌드 대상: `Nuitka` 기반 Windows exe/portable 패키지
- 릴리스 순서: Windows 로컬 Nuitka 빌드 검증, `main` 승격, Windows CI preflight, 태그 기반 release CI
- 포함 범위: 앱 본체, Python 런타임, PySide6, torch/onnxruntime 런타임, 번역/resources
- 미포함 범위: 모델, 체크포인트, Docker 런타임, NVIDIA 드라이버

릴리스 후보를 `main`으로 승격하기 전에는 Windows PowerShell에서 필요한 Nuitka 빌드 스크립트를 직접 실행하고, 성공한 명령과 `build/nuitka-*` 산출물 경로를 PR에 기록합니다. WSL 전용 확인은 Windows 로컬 빌드 검증을 대체하지 않습니다.

릴리스 패키지만으로 전체 로컬 런타임이 완성되지는 않으므로, 내려받은 뒤 이 가이드의 런타임 설정 단계를 이어서 진행하면 됩니다.
