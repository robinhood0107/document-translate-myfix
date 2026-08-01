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
  -ModelPath 'C:\ExampleWorkspace\models\gemma-4-26B-IQ4_NL.gguf'
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
- 추론 backend: 공식 PaddleOCR-VL 1.6 GGUF와 multimodal projector를
  사용하는 고정 llama.cpp
- 요청 경로: detector crop -> direct llama.cpp `OCR:` 요청. 관리형 crop
  경로에서는 PaddleX 전체 layout pipeline이나 vLLM process를 시작하지 않는다.
- 참고 링크:
  - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
  - [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

아래 두 파일을 같은 모델 폴더에 둡니다.

- `PaddleOCR-VL-1.6-GGUF.gguf`
- `PaddleOCR-VL-1.6-GGUF-mmproj.gguf`

Windows PowerShell에서 versioned external model volume을 한 번 준비하고
검증합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_paddleocr_llamacpp_runtime.ps1 `
  -Mode Prepare `
  -ModelDirectory 'C:\ExampleWorkspace\models\PaddleOCR-VL-1.6-GGUF'
```

이후 앱이 PaddleOCR VL이 필요할 때 관리 런타임을 자동 시작합니다. 준비된
model volume은 read-only로 마운트하며, fingerprint가 정확히 같은 stopped
컨테이너만 재사용하고 오래된 컨테이너는 force-recreate합니다. 앱이 OCR stage
종료를 명시한 뒤 모델을 해제하며, 해제를 확인하지 못하면 정상 `stop`으로
전환합니다. 자동 경로는 `down`을 사용하지 않습니다.

Stage-Batched 폴더 처리는 `Settings > PaddleOCR VL Settings`에서 관리하는
exact 영구 OCR 결과 캐시도 사용할 수 있습니다.

bundle 파일 설명은 [/paddleocr_vl_docker_files/README.md](/paddleocr_vl_docker_files/README.md)를 참고하세요.

### 선택형 PaddleOCR-VL full-page Spotting 경로

Full-page Spotting은 기본 detector + crop OCR과 분리된 OCR 선택지입니다.
기존 crop projector를 바꾸지 않으며, 다음 명령으로 전용 named volume을
준비합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_paddleocr_spotting_llamacpp_runtime.ps1 `
  -Mode Prepare `
  -ModelDirectory 'C:\ExampleWorkspace\models\PaddleOCR-VL-1.6-GGUF'
```

Spotting 경로는 공식 `Spotting:` prompt, `--special` 좌표 token 모드,
`1,605,632` projector pixel budget을 고정합니다. crop 경로는 기존
`1,003,520` projector를 그대로 유지합니다. 자세한 내용은
[/paddleocr_vl_spotting_docker_files/README.md](/paddleocr_vl_spotting_docker_files/README.md)를
참고하세요.

## 4. 권장 앱 설정

- 워크플로 모드: `Stage-Batched Pipeline (Recommended)`
- OCR: `Optimal (HunyuanOCR / PaddleOCR VL)`
- 번역기: Gemma volume 준비 후 `Custom Local Server(Gemma)`

프로젝트 stage checkpoint는 `Settings > Project`에서 검증된 one-time migration으로
한 번 활성화되며, 이후 사용자의 선택은 그대로 보존합니다. cache 관리 기능을
사용하기 전에 `.ctpr`를 먼저 저장해야 합니다. 옆의 `.ctpr.cache` 폴더는
재계산 가능한 데이터이며 프로젝트를 여는 데 필수적이지 않습니다. 유효할 때는
감지, 사전 적용 전 OCR, 인페인트 mask·cleaned artifact, 렌더 출력을 복원합니다.
번역 내용은 프로젝트 파일에만 남고 sidecar 서명이 정확히 일치할 때만 재사용합니다.

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
- 공식 자산:
  `comic-translate-vX.Y.Z-windows-launcher-source.zip`과
  `SHA256SUMS.txt`
- 릴리스 순서: Windows 로컬 deterministic bundle·추출 launcher 검증,
  `main` 승격, Windows CI preflight, 태그 기반 release CI
- 포함 범위: allowlist 제품 source, launcher, CUDA12/CUDA13 requirements,
  Docker Compose/config, Gemma/PaddleOCR 준비 도구, 번역/resources, README, LICENSE
- 미포함 범위: venv, 모델, 체크포인트, 캐시, benchmark 도구/raw 결과,
  Python/CUDA runtime, NVIDIA 드라이버, 로컬 경로, secret

릴리스 후보를 `main`으로 승격하기 전에는 `HEAD`에서 ZIP을 생성하고
manifest와 SHA-256을 검증합니다. 새 폴더에 압축을 푼 뒤
`COMIC_VERIFY_ONLY=1`로 두 launcher를 실행하고, Windows 명령·ZIP hash·
두 결과를 PR에 기록합니다.

일반 첫 실행에서는 launcher가 고정 환경을 bootstrap합니다. 기존
Nuitka 스크립트는 비공식 수동 도구이며 공식 릴리스 gate가 아닙니다.
