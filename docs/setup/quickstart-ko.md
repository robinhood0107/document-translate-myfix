# 빠른 설치 가이드

이 문서는 처음 체크아웃한 뒤 가능한 가장 짧은 경로로 앱을 실행하고, 로컬 런타임까지 붙이는 방법을 정리합니다.

## 1. 준비물

- Windows 10/11 x64
- [공식 Python 3.12.10 Windows installer (64-bit)](https://www.python.org/downloads/release/python-31210/)
- WSL2 backend와 GPU 지원이 켜진 Docker Desktop
- NVIDIA 드라이버와 CUDA 호환 GPU
- 선택한 venv용 약 6~8 GiB와, 아직 없는 모델 파일의 실제 크기만큼의 여유 공간
- 소스 checkout을 사용할 때만 Git

현재 pinned `mahotas` Windows wheel이 CPython 3.12까지만 제공되므로 Python
3.13/3.14만 설치된 PC는 지원 대상이 아닙니다. 다른 Python과 나란히 3.12 x64를
설치하면 launcher가 `py -3.12`를 우선 사용하며 전역 package는 가져오지 않습니다.
Python installer에서는 `py` launcher를 포함하세요. 시스템 PATH 추가는 선택이며,
launcher는 `py -3.12`를 우선 탐색합니다.

## 2. 최초 1회 준비

설치와 실행은 진입점이 다릅니다. 오래 걸리는 작업은 전부 설치 쪽에 있고,
실행 런처는 그 작업을 하지 않습니다.

CUDA12 경로(Python cu128 + llama.cpp `server-cuda`):

```bat
setup.bat
```

CUDA13 Python 경로(cu130, Docker llama.cpp는 CUDA12 setup과 같은 호환 범위가
넓은 `server-cuda` 사용):

```bat
setup_cuda13.bat
```

두 BAT는 같은 PowerShell bootstrap 엔진을 사용합니다. 시스템 Python은 venv
생성에만 쓰고 전역 package와 Python 환경변수는 무시합니다. Docker Desktop이
꺼져 있으면 자동으로 시작하고, 선택한 CUDA 계열의 llama.cpp server image가
로컬에 없을 때만 가져옵니다.

설치는 질문 없이 아래 항목을 순서대로 준비합니다.

- 선택한 `.venv-win` 또는 `.venv-win-cuda13`과 pinned Python package
- RT-DETR v2 ONNX, font-detector ONNX, CTD Torch/ONNX 및 positive-claim ONNX
- LaMa large와 LaMa MPE 앱 모델
- HunyuanOCR Q8 model/mmproj
- PaddleOCR VL 1.6 model/mmproj
- `gemma-4-26B-IQ4_NL.gguf`(약 13.58 GiB)

`setup_full.bat`과 `setup_full_cuda13.bat`은 여기에 MangaLMM과 PaddleOCR VL
Spotting까지 추가합니다. core 단계는 full 단계의 부분집합이므로
`setup_full.bat` 뒤에 `setup.bat`을 실행해도 추가로 만든 볼륨이 사라지지 않습니다.

Docker 원본 모델은 설치 폴더의 `models\managed-runtime-sources`에서 재사용하고,
중단된 다운로드는 다음 실행에서 이어받습니다. 모든 항목은 크기·SHA-256과
실제 model-load smoke를 통과해야 준비 완료로 인정합니다. 읽기 전용 상태 검사는
`setup.bat --doctor` 또는 `setup_cuda13.bat --doctor`를 사용합니다.

BAT은 클래식 명령 프롬프트를 그대로 사용하고 현재 창에만 UTF-8과 Consolas
16px를 적용하며 레지스트리는 바꾸지 않습니다. 화면에는 CUDA DLL을 불러오지 않는 package metadata 하위 단계,
모델별 시작·완료, 다운로드 10% 단위 진행률, runtime 준비 경계와 최종 결과만
표시합니다. 자식 명령의 전체 출력은 시간별 `logs\bootstrap\*-detail.log`에
남습니다.

## 3. 실행

```bat
run_comic.bat
```

```bat
run_comic_cuda13.bat
```

실행 런처는 venv와 원자적 install-state를 확인하고 setup이 선택한 정확한
llama.cpp image를 전달한 뒤 앱을 띄웁니다. package/model/image/volume을 설치,
다운로드, pull, 생성, 재봉인하지 않습니다. core 상태가 없으면 해당 setup BAT,
MangaLMM/Spotting 상태가 없으면 페이지 처리 전에 setup_full을 요구합니다.

CUDA13 launcher는 `server-cuda`를 사용하더라도 컨테이너를 만들기 전에 이미지의 `NVIDIA_REQUIRE_CUDA`와
드라이버 호환 버전을 비교합니다. 준비 완료 뒤에는 ready manifest, image ID,
모델 크기가 그대로인 볼륨을 재사용하므로 전체 해시와 GPU smoke를 반복하지 않습니다.

CUDA13 경로는 공식 ONNX Runtime CUDA13 nightly feed를 사용합니다.

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

### 설치 위치

- CUDA12 venv: 설치 폴더의 `.venv-win`(현재 패키지 기준 약 5.9GB)
- CUDA13 venv: 설치 폴더의 `.venv-win-cuda13`(약 4.3GB)
- GGUF 원본 cache: 설치 폴더의 `models\managed-runtime-sources`
- bootstrap log/state: 설치 폴더의 `logs\bootstrap`, `.comic-bootstrap`
- 준비된 GGUF: Docker external named volume

따라서 저장소가 `D:`에 있으면 venv, 원본 cache, log/state도 `D:`에 생깁니다.
Docker named volume의 실제 물리 위치는 Docker Desktop의 disk image location을
따르므로 Docker 설정이 `C:`이면 그 복사본은 `C:`를 사용합니다. 기본 3종 모델
원본 합계는 약 16.50GiB이며 Docker volume에도 비슷한 크기의 준비본이 저장됩니다.
bootstrap은 고정 60GiB를 요구하지 않고, 빠진 파일마다 정확한 크기 + 512MiB만
검사합니다.

별도 CUDA Toolkit 설치는 필요하지 않습니다. PyTorch/ONNX Runtime의 CUDA
사용자 공간 DLL은 선택한 venv에 설치되고 launcher가 그 경로를 우선합니다.
llama.cpp의 CUDA 사용자 공간은 선택한 Docker image 안에 있습니다. 단, 실제 GPU를
구동하는 NVIDIA display driver와 Docker Desktop/WSL2 GPU passthrough는 시스템
준비물이라 venv 안에 넣을 수 없습니다.

## 4. 로컬 런타임 수동 관리

정상적인 첫 실행에서는 아래 준비 명령을 직접 실행할 필요가 없습니다. 전체 hash
재검증, 별도 volume 또는 선택 runtime을 수동 관리할 때만 사용합니다.

### Gemma 로컬 번역 런타임

- compose 파일: `/docker-compose.yaml`
- Docker 이미지: `ghcr.io/ggml-org/llama.cpp:server-cuda`
- 참고 링크:
  - [llama.cpp](https://github.com/ggml-org/llama.cpp)
  - [Gemma](https://ai.google.dev/gemma)

런처는 이 exact 모델을 자동으로 준비합니다. 수동 준비가 필요하면:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_gemma_runtime.ps1 -Mode Prepare `
  -ModelPath 'C:\ExampleWorkspace\models\gemma-4-26B-IQ4_NL.gguf'
```

앱에서는 `Custom Local Server(Gemma)`를 선택합니다. 관리 런타임이 준비된 volume을 read-only로 마운트하고 정확히 준비된 컨테이너를 자동으로 시작합니다.

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

### HunyuanOCR 로컬 런타임

최적값의 중국어 분기가 쓰는 엔진입니다. 다른 관리형 엔진과 같은 규약으로
versioned external model volume을 한 번 준비합니다. 필요한 파일은

- `HunyuanOCR.Q8_0.gguf`
- `HunyuanOCR.mmproj-Q8_0.gguf`

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_hunyuanocr_llamacpp_runtime.ps1 `
  -Mode Prepare `
  -ModelDirectory 'C:\ExampleWorkspace\models\HunyuanOCR-GGUF'
```

준비 도구는 volume에 이미 있는 파일의 SHA-256이 일치하면 다시 복사하지 않고
그대로 재사용합니다. 준비가 끝나면 CUDA 모델 적재 스모크를 실행하고 그 결과를
ready manifest에 기록합니다. 검증만 다시 하려면 `-Mode Verify`를 씁니다.

### 준비 볼륨이 갑자기 거부될 때

모든 준비 스크립트는 `-Mode Auto`를 받습니다. 유효한 봉인은 즉시 재사용하고,
볼륨이 비어 있으면 준비합니다. 업스트림이 llama.cpp 태그를 갱신해 image digest가
움직이면 모델이 멀쩡한데도 manifest만 어긋나는데, 이때만 `Auto`가 `Reseal`을
선택해 원본 파일 없이 복구합니다. 실행 중인 앱은 어긋난 봉인을 보고하고 해당
setup BAT을 요구할 뿐 volume을 복구하지 않습니다. 자세한 내용은
[관리형 llama.cpp 볼륨 복구 가이드](../runtime/managed-volume-repair-ko.md)를
참고하세요.

`-ModelDirectory`를 생략하면 저장소의 gitignore된 `testmodel/`과 그 바로 아래
하위 폴더를 먼저 찾습니다. launcher 경로는 설치 폴더의 ignored
`models\managed-runtime-sources` cache를 명시합니다.

과거 HunyuanOCR은 `testmodel` 폴더를 bind mount하고 Gemma와 이름이 겹치는
`LLAMA_CTX_SIZE` 같은 일반 환경변수를 읽었습니다. 이제는 준비된 volume과
`HUNYUAN_OCR_LLAMA_*` 전용 이름을 사용하므로, 한쪽을 조정해도 다른 엔진이
바뀌지 않습니다.

## 5. 권장 앱 설정

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

## 6. 선택 알림 설정 (ntfy)

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

## 7. 현재 제품 코드가 실제로 참고하는 모델/런타임 링크

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

## 8. 같이 보면 좋은 문서

- [/README.md](/README.md)
- [/README_ko.md](/README_ko.md)
- [/docs/gemma/local-server-ko.md](/docs/gemma/local-server-ko.md)
- [/docs/hunyuan/local-server-ko.md](/docs/hunyuan/local-server-ko.md)

## 9. 공식 Windows 릴리스 패키지

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
