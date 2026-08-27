[English](README.md) | [한국어](README_ko.md)

# Comic Translate 포크

이 저장소는 upstream `comic-translate` `v2.6.7` 코드베이스에서 시작한 뒤, 로컬 런타임/OCR/워크플로/Windows 환경 쪽으로 제품화 수정을 누적한 local-first 포크입니다.

현재 포크의 제품 릴리스 버전은 `1.5.1`입니다. upstream `2.7.1`은
마지막 selective backport 계보로 별도 기록하며, 이 포크의 제품 버전과
같은 의미로 사용하지 않습니다.

이 포크는 아래 워크플로를 중심으로 유지됩니다.

- 로컬 Gemma 번역 런타임
- `PaddleOCR VL`, `HunyuanOCR` 같은 로컬 OCR 런타임
- Windows 중심 설치/실행 도구
- upstream `v2.7.0`, `v2.7.1`의 selective manual backport
- benchmark 작업과 제품 브랜치의 분리

## 중요 기능

- 데스크톱 중심 번역 워크플로를 위한 로컬 Gemma 번역 런타임.
- 전체 identity 기반 Gemma 결과 캐시와 사용자 승인형 정확 일치 번역 메모리.
- `HunyuanOCR`와 `PaddleOCR VL`을 상황에 맞게 고르는 로컬 OCR 최적 라우팅.
- 저장/불러오기까지 유지되는 인페인팅 Add / Exclude / Restore 도구.
- OCR/번역 교정 사전이 포함된 TXT/MD 원문 export 및 번역 import.
- lazy page materialization 기반의 CBZ/CBR 만화 아카이브 입력.
- PDF 입력은 페이지마다 작업 이미지 하나를 유지하고, 안전한 내장 이미지는 재인코딩 없이 복사하며, 복합 페이지는 OCR 전에 고해상도로 렌더링합니다.
- 좌하단 상태 패널, 오버레이 잠금, 최신 결과 미리보기가 결합된 자동 파이프라인 UI.

## 서브 기능

- 이미 실행 중인 로컬 OCR 컨테이너를 재기동하지 않는 reuse-only preflight 검사.
- 다음 실행에서 재사용할 수 있도록 로컬 Docker 런타임을 삭제하지 않고 중지하는 생명주기.
- 자동 실행 중 페이지가 끝날 때마다 최신 번역 완료 이미지를 바로 갱신하는 미리보기.
- 시스템 알림음 또는 저장소 `music/*.wav`를 쓰는 완료 알림음.
- `.venv-win`, `.venv-win-cuda13`을 자동 bootstrap하는 Windows 런처.
- UI 변경과 함께 유지되는 다국어 툴팁, 도움말, Qt 번역 자산.

## 원점과 upstream 출처 고지

이 저장소는 [ogkalu2/comic-translate](https://github.com/ogkalu2/comic-translate) 에서 시작된 downstream 포크/파생 작업입니다. upstream `v2.6.7` 코드베이스에서 출발했고, 이후 로컬 런타임, OCR, Windows 실행 환경, 제품 워크플로 방향으로 분기하며 확장되었습니다.

## 라이선스와 재배포 기준

upstream 프로젝트는 Apache License 2.0으로 배포되며, 이 포크도 upstream에서 파생된 코드에 대해서는 그 라이선스 기반을 유지합니다.

이 포크나 이 포크를 수정한 빌드를 공개 재배포할 때의 최소 기준은 아래입니다.

- 재배포물과 함께 Apache 2.0 라이선스 전문을 포함할 것
- 아직 적용되는 upstream 저작권, 특허, 출처, 고지 사항을 유지할 것
- 이 저장소가 원본이 아니라 수정된 downstream 포크/파생 작업임을 명확히 밝힐 것
- 소스를 재배포할 때는 수정한 파일에 변경 고지를 분명히 남길 것
- 코드 라이선스와 별개로 서드파티 자산 라이선스를 따로 검토할 것

## 서드파티 모델 및 런타임 고지

이 프로젝트는 여러 외부 모델, 체크포인트, 런타임 이미지를 사용하거나 자동 다운로드하거나 연동합니다. 해당 자산의 저작권, 라이선스, 사용 조건은 원 저작권자와 원 배포처에 귀속되며, 이 저장소는 그 소유권을 주장하지 않습니다. 사용자는 각 upstream 모델/런타임의 라이선스와 이용 조건을 직접 확인하고 준수해야 합니다.

### 현재 제품 코드가 사용하는 모델 및 런타임

검출 / 마스킹:
- [RT-DETR v2](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)
- [ComicTextDetector (CTD)](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3) (`comictextdetector.pt`, `comictextdetector.pt.onnx`)
- [Font Detector](https://huggingface.co/gyrojeff/YuzuMarker.FontDetection)

OCR:
- [MangaOCR](https://huggingface.co/kha-white/manga-ocr-base)
- [MangaOCR ONNX](https://huggingface.co/mayocream/manga-ocr-onnx)
- [Pororo OCR](https://huggingface.co/ogkalu/pororo)
- [PPOCRv5 / RapidOCR](https://www.modelscope.cn/models/RapidAI/RapidOCR)
- [PaddleOCR VL](https://github.com/PaddlePaddle/PaddleOCR)
- [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)

인페인팅:
- [AOT](https://huggingface.co/ogkalu/aot-inpainting)
- [LaMa legacy runtime](https://github.com/Sanster/models/releases/tag/AnimeMangaInpainting)
- [lama_large_512px](https://huggingface.co/dreMaz/AnimeMangaInpainting)
- [lama_mpe / manga-image-translator 인페인팅 체크포인트](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3)
- [MI-GAN](https://github.com/Sanster/models/releases/tag/migan)

로컬 번역 / 런타임:
- [Gemma](https://ai.google.dev/gemma) 로컬 GGUF 런타임
- [llama.cpp](https://github.com/ggml-org/llama.cpp) Docker 런타임 이미지

### 자동 다운로드 자산과 사용자 준비 자산 구분

앱이 누락 시 자동 다운로드하는 자산:
- CTD 모델 파일 (`comictextdetector.pt`, `comictextdetector.pt.onnx`)
- `AOT`, `lama_large_512px`, `lama_mpe` 같은 인페인팅 체크포인트
- `MangaOCR`, `Pororo OCR`, `PPOCRv5` 같은 OCR 체크포인트

Windows 런처가 첫 실행에 자동 다운로드하고 검증된 external volume에 준비하는 자산:
- `gemma-4-26B-IQ4_NL.gguf` Gemma 번역 모델
- HunyuanOCR Q8 GGUF 및 mmproj
- PaddleOCR VL 1.6 GGUF 및 mmproj

PaddleOCR VL Spotting과 MangaLMM은 기본 bootstrap에 포함하지 않으며, 해당
선택 기능을 처음 사용할 때 기존 관리형 준비 경로가 처리합니다.

## 릴리스 정책

현재 저장소는 엄격한 `main + develop + tag` 모델을 사용합니다.

- `develop`: 다음 제품 작업을 통합하는 브랜치
- `main`: 실제 출하 기준선
- 공식 릴리스: `main`에 포함된 커밋에만 `vX.Y.Z` 버전 태그를 달아 GitHub Release 생성
- 공식 Windows 자산:
  `comic-translate-vX.Y.Z-windows-launcher-source.zip`과
  `SHA256SUMS.txt`
- ZIP 포함 범위: allowlist에 든 제품 source, CUDA12/CUDA13 첫 실행
  launcher·requirements, 공통 bootstrap, Docker Compose/config,
  Gemma/HunyuanOCR/PaddleOCR 준비 스크립트,
  번역/resources, README, LICENSE
- ZIP 제외 범위: venv, 모델, 캐시, benchmark runner/raw 결과,
  로컬 절대경로, secret
- launcher는 첫 실행 때 지원 환경을 설치하며, 릴리스 후보는 추출 후
  `COMIC_VERIFY_ONLY=1`로 두 launcher의 무설치 계약을 확인
- 기존 Nuitka PowerShell 스크립트는 비공식 수동 도구로만 유지
- `release/*` 브랜치는 사용하지 않음

저장소 운영 기준 문서는 [rules.md](rules.md)입니다.

## upstream `v2.6.7` 이후 포크 개선 축

`v2.6.7` 기반에서 출발한 뒤, 이 포크는 몇 개의 기술 축을 중심으로 개선됐습니다.

### 렌더링과 수동 편집

- 공통 렌더 정책 동작을 먼저 문서화하고 이후 코드로 통합했습니다.
- 강제 색상, 블록 앵커, source rect, 세로 정렬 메타데이터를 렌더 상태에 확장했습니다.
- 오른쪽 렌더 패널의 레이아웃과 문구, 선택 affordance를 정리했습니다.
- 수동 렌더와 배치/웹툰 렌더가 같은 공통 정책을 쓰도록 맞췄습니다.

### Windows 런타임과 저장소 워크플로

- Windows 실행기와 CUDA13 전용 실행 경로를 추가했습니다.
- Windows 진입점을 분리했습니다. `setup*.bat`이 venv와 모델 volume을 준비하고, `run_comic*.bat`은 실행만 합니다.
- 로컬 Git hook과 CI 검증 체계를 강화했습니다.
- 브랜치 정책을 정리해 `feature/*`, `fix/*`, `chore/*`, `hotfix/*`, `benchmarking/lab` 체계로 표준화했습니다.

### OCR 품질과 진단

- block-local OCR fallback과 suspicious-result retry 흐름을 추가했습니다.
- bubble residue cleanup과 잔여 글자 제거용 mask 보정을 추가했습니다.
- one-page auto와 batch OCR의 parity/diagnostics를 개선했습니다.
- 로컬 PaddleOCR VL 지원과 기본값 튜닝을 추가했습니다.
- 로컬 HunyuanOCR 지원을 추가했습니다.
- `Optimal (HunyuanOCR / PaddleOCR VL)` OCR 라우팅, 실행 전 언어 확인, on-demand 로컬 런타임 관리까지 추가했습니다.

### 로컬 번역 런타임

- 로컬 Gemma 번역 서버 경로를 분리/정교화했습니다.
- custom translator 모드를 분리하고 keyless local endpoint 지원을 보강했습니다.
- Gemma 입력 정규화와 문제 glyph 정리를 추가했습니다.
- 로컬 sampler/runtime 기본값을 benchmark 결과에 맞춰 조정했습니다.
- 전체 hit에서 Gemma 시작을 생략하는 SQLite 결과 캐시 fast path를 추가했습니다.
- 보수적 일치와 명시적 승인·가져오기·내보내기를 사용하는 Exact Translation Memory를 별도로 추가했습니다.

### 벤치마크와 브랜치 분리

- 전용 benchmark toolkit과 one-click runner를 추가했습니다.
- benchmark harness/report 자산을 제품 브랜치와 분리했습니다.
- `benchmarking/lab` 승격 경계를 문서화했습니다.

## Selective Backport 기록

이 포크는 upstream 릴리스를 통째로 merge하지 않고, compare 기반으로 필요한 변경만 골라 현재 제품 구조에 맞게 적응 이식합니다.

### `v2.6.7 -> v2.7.0`

`v2.7.0` 라운드에서는 아래 사용자 가치 기능을 선별 반영했습니다.

- configurable keyboard shortcuts
- PSD export / PSD import
- chapter-aware export
- project rename/move
- startup recent-project copy path / delete file
- multi-select text block formatting
- undo text render as one undo step
- custom translator extra context unlimited
- target language 확장과 RTL 개선
- webtoon/list-view 관련 선택 이식 수정

### `v2.7.0 -> v2.7.1`

`v2.7.1` 라운드에서는 이 포크에 의미 있는 upstream 수정만 선택 적용합니다.

- PSD import 안정화: 폰트 카탈로그 준비 헬퍼, decode fallback 로깅, thread-safe font catalog guard
- 비동기 UI 콜백의 main-thread-safe `QTimer.singleShot(...)` 정리
- 리스트 썸네일 로더를 worker `QImage` + main-thread `QPixmap` 구조로 안정화
- import 메뉴에서 `Project File` 옆 `PSD` 정리
- upstream selective-backport 계보를 `2.7.1`로 기록

## 빠른 사용법

조금 더 자세한 설치/실행 경로는 아래 문서를 같이 보세요.

- [docs/setup/quickstart.md](docs/setup/quickstart.md)
- [docs/setup/quickstart-ko.md](docs/setup/quickstart-ko.md)

### 1. 앱 실행

준비물은 Windows 10/11 x64,
[공식 Python 3.12.10 Windows x64](https://www.python.org/downloads/release/python-31210/),
WSL2 기반 Docker Desktop,
NVIDIA 드라이버와 CUDA 호환 GPU입니다. Docker Desktop은 설치되어 있어야 하며,
꺼져 있으면 런처가 시작하고 준비될 때까지 기다립니다.
Python installer에서는 `py` launcher를 포함하세요. 시스템 Python은 전용 venv
생성에만 사용하고 전역 package는 가져오지 않습니다.


설치와 실행은 분리되어 있습니다. 설치는 한 번만 하고, 이후에는 계속 실행만
하면 됩니다. 실행 런처는 모델 volume을 내려받지 않으므로 수 초 안에 뜹니다.

**1단계 - 최초 1회 준비**(CUDA12는 Python cu128 + llama.cpp `server-cuda`,
CUDA13은 cu130이며 Docker가 이미지의 CUDA 요구 조건을 먼저 검사한 뒤
`server-cuda13` 또는 호환 `server-cuda`를 선택):

```bat
setup.bat
```

```bat
setup_cuda13.bat
```

전용 venv, 앱 필수 모델, 그리고 핵심 런타임 3종(HunyuanOCR, PaddleOCR VL,
Gemma IQ4_NL)을 준비합니다. MangaLMM과 PaddleOCR VL Spotting까지 미리 받아
두려면 `setup_full.bat` 또는 `setup_full_cuda13.bat`을 사용하세요. 설치는 질문
없이 진행되고 중단되어도 이어집니다. 다운로드는 설치 폴더의
`models/managed-runtime-sources` cache에서 이어받고, 크기와 SHA-256을 통과한
파일만 Docker volume에 들어갑니다. 다시 실행하면 ready manifest, 현재 image ID,
모델 크기가 모두 맞을 때 전체 해시와 GPU smoke를 건너뜁니다.

**2단계 - 실행:**

```bat
run_comic.bat
```

```bat
run_comic_cuda13.bat
```

설치를 건너뛰어도 동작하지만 느립니다. 이 경우 앱이 처음 쓰는 런타임을 GUI
안에서 진행률 표시 없이 그때그때 준비합니다. 설치 상태만 읽기 전용으로
확인하려면 `setup.bat --doctor` 또는 `setup_cuda13.bat --doctor`를 실행합니다.

### 2. 로컬 번역 서버 사용

앱 기본값인 `Custom Local Server(Gemma)`는 런처가 준비한 read-only volume의
`gemma-4-26B-IQ4_NL.gguf`를 사용합니다. 전체 SHA-256을 수동으로 다시 검사할
때만 `scripts/prepare_gemma_runtime.ps1 -Mode Verify`를 사용합니다.

**사용자 사전** 설정에서는 영구 블록 결과 캐시와 정확 일치 번역 메모리도 관리합니다. 결과 캐시는 번역과 runtime의 전체 identity가 같은 경우에만 재사용합니다. 원문→번역 쌍은 사용자가 명시적으로 승인해야 Gemma를 우회하며, 승인 항목이 포함된 파일을 가져올 때도 확인을 요구합니다. DB에는 민감한 로컬 텍스트가 저장되며 앱 user-data 디렉터리에만 남습니다. 잠금·손상 오류가 나도 자동 삭제하지 않습니다. 자세한 내용은 [번역 메모리 가이드](docs/gemma/translation-memory-ko.md)를 참고하세요.

### 3. 로컬 OCR 서버 사용

권장 `Optimal (HunyuanOCR / PaddleOCR VL)` 경로의 두 runtime도 런처가
자동 준비합니다. 상세 계약은 [HunyuanOCR 문서](docs/hunyuan/local-server-ko.md)와
[PaddleOCR VL 문서](paddleocr_vl_docker_files/README.md)에 정리돼 있습니다.

선택형 full-page `Spotting:` 경로는 projector·container·named volume·cache
identity를 crop OCR과 분리합니다. 준비 방법은
[paddleocr_vl_spotting_docker_files/README.md](paddleocr_vl_spotting_docker_files/README.md)에
정리돼 있습니다.

관리 대상 컨테이너는 단계 완료, 취소, 앱 종료 때 삭제되지 않고 중지 상태로 보존됩니다. 런타임을 명시적으로 초기화하거나 제거할 때만 `docker compose down`을 사용합니다.

### 4. 권장 OCR 설정

Settings에서 아래 중 하나를 선택합니다.

- `Default (existing auto: MangaOCR / PPOCR / Pororo...)`: 기존 자동 OCR 경로 유지
- `Optimal (HunyuanOCR / PaddleOCR VL)`: 중국어는 `HunyuanOCR`, 일본어/기타 언어는 `PaddleOCR VL`로 라우팅

Stage-Batched 폴더 처리에서는 `Settings > PaddleOCR VL Settings`의 관리형
exact 영구 OCR 캐시도 사용할 수 있습니다. crop 이미지는 저장하지 않고 사전
적용 전 OCR 결과와 진단만 저장하며, 사용자 지정 endpoint에서는 자동으로
비활성화됩니다.

`Settings > Project`에는 검증된 프로젝트 checkpoint 기능도 있습니다. one-time
migration이 한 번 활성화한 뒤에는 이후 사용자의 선택을 그대로 보존합니다. 감지 좌표,
사전 적용 전 PaddleOCR-VL 결과, lossless 인페인트 결과·final mask, 인코딩된 렌더
출력을 각 runtime 시작 전에 복원할 수 있습니다. 번역문은 `.ctpr`에만 두고
sidecar에는 검증용 stage 서명만 기록하므로 project all-hit에서는 detector, Paddle,
Gemma, inpainter, renderer 추론을 모두 건너뜁니다. 현재 OCR·번역 사전은 hit와
miss 모두 정확히 한 번 적용하고, 사전 변경 시 소비 결과와 downstream stage만
무효화합니다. 재사용 가능한 stage manifest와 content-addressed artifact는
`.ctpr` 옆 `<project>.ctpr.cache` 폴더에 저장합니다. cache가 없거나 잠겼거나
손상돼도 프로젝트 열기와 처리는 계속되며 해당 stage를 다시 계산합니다.

### 5. 선택 알림 설정 (ntfy)

`Settings > Notifications`에서 아래를 설정할 수 있습니다.

- 완료 알림음
- ntfy 서버 URL / topic / 선택 access token
- 완료 / 실패 / 취소 시 전송 여부

앱은 ntfy로 텍스트만 보내며, 본문은 ntfy 기본 텍스트 제한을 넘지 않도록 줄입니다.

공식 ntfy 문서:

- [Publish notifications](https://docs.ntfy.sh/publish/)
- [Server configuration](https://docs.ntfy.sh/config/)

## 이 저장소가 사용하는 Docker 이미지

현재 추적 중인 compose/runtime 이미지:

관리형 llama.cpp 런타임(Gemma, PaddleOCR VL, PaddleOCR VL Spotting, HunyuanOCR,
MangaLMM, Router)은 모두 하나의 CUDA 서버 이미지를 씁니다.

- 기본값: `ghcr.io/ggml-org/llama.cpp:server-cuda13`
- 함께 지원: `ghcr.io/ggml-org/llama.cpp:server-cuda`

## 참고 설치 문서

- [docs/setup/quickstart.md](docs/setup/quickstart.md)
- [docs/setup/quickstart-ko.md](docs/setup/quickstart-ko.md)
- [docs/gemma/local-server-ko.md](docs/gemma/local-server-ko.md)
- [docs/hunyuan/local-server-ko.md](docs/hunyuan/local-server-ko.md)
- [paddleocr_vl_docker_files/README.md](paddleocr_vl_docker_files/README.md)

## 저장소 문서

- [rules.md](rules.md)
- [코드 구조와 OCR 전략 경계](docs/architecture/codebase-map-ko.md)
- [관리형 llama.cpp 전용 런타임 정책](docs/runtime/managed-llamacpp-only-ko.md)
- [docs/gemma/local-server-ko.md](docs/gemma/local-server-ko.md)
- [docs/hunyuan/local-server-ko.md](docs/hunyuan/local-server-ko.md)
- [docs/repo/github-rulesets-public-free-ko.md](docs/repo/github-rulesets-public-free-ko.md)

## Legacy Localized README

`docs/i18n/` 아래 예전 localized README는 더 이상 source of truth가 아닙니다.

현재 기준 문서는 아래 둘입니다.

- 영문 기준: [README.md](README.md)
- 한글 기준: [README_ko.md](README_ko.md)
