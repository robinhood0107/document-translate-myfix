[English](README.md) | [한국어](README_ko.md)

# Comic Translate 포크

이 저장소는 upstream `comic-translate` `v2.6.7` 코드베이스에서 시작한 뒤, 로컬 런타임/OCR/워크플로/Windows 환경 쪽으로 제품화 수정을 누적한 local-first 포크입니다.

이 포크는 아래 워크플로를 중심으로 유지됩니다.

- 로컬 Gemma 번역 런타임
- `PaddleOCR VL`, `HunyuanOCR` 같은 로컬 OCR 런타임
- Windows 중심 설치/실행 도구
- upstream `v2.7.0`, `v2.7.1`의 selective manual backport
- benchmark 작업과 제품 브랜치의 분리

## 원점과 upstream 출처 고지

이 저장소는 [ogkalu2/comic-translate](https://github.com/ogkalu2/comic-translate) 에서 시작된 downstream 포크/파생 작업입니다. upstream `v2.6.7` 코드베이스에서 출발했고, 이후 로컬 런타임, OCR, Windows 실행 환경, 제품 워크플로 방향으로 분기하며 확장되었습니다.

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

사용자가 별도로 준비하거나 로컬 런타임 번들이 제공하는 자산:
- 로컬 Gemma 번역 런타임에 마운트하는 Gemma GGUF 파일
- HunyuanOCR GGUF 및 mmproj 파일
- PaddleOCR VL Docker/runtime bundle 파일

## 릴리스 정책

현재 저장소는 엄격한 `main + develop + tag` 모델을 사용합니다.

- `develop`: 다음 제품 작업을 통합하는 브랜치
- `main`: 실제 출하 기준선
- 릴리스: `main` 커밋에 버전 태그를 달아 GitHub Release 생성
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
- `run_comic.bat`, `run_comic_cuda13.bat` 자체가 로컬 venv/runtime을 자동 bootstrap하도록 바꿨습니다.
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

검수 문서:

- [docs/history/v267-to-v270-backport-audit.md](docs/history/v267-to-v270-backport-audit.md)
- [docs/history/v267-to-v270-backport-audit-ko.md](docs/history/v267-to-v270-backport-audit-ko.md)

### `v2.7.0 -> v2.7.1`

`v2.7.1` 라운드에서는 이 포크에 의미 있는 upstream 수정만 선택 적용합니다.

- PSD import 안정화: 폰트 카탈로그 준비 헬퍼, decode fallback 로깅, thread-safe font catalog guard
- 비동기 UI 콜백의 main-thread-safe `QTimer.singleShot(...)` 정리
- 리스트 썸네일 로더를 worker `QImage` + main-thread `QPixmap` 구조로 안정화
- import 메뉴에서 `Project File` 옆 `PSD` 정리
- 앱 버전 `2.7.1` 반영

검수 문서:

- [docs/history/v270-to-v271-backport-audit.md](docs/history/v270-to-v271-backport-audit.md)
- [docs/history/v270-to-v271-backport-audit-ko.md](docs/history/v270-to-v271-backport-audit-ko.md)

## 빠른 사용법

### 1. 앱 실행

런처는 첫 실행 시 필요한 로컬 runtime 환경을 스스로 생성하거나 갱신합니다.


기본 Windows 런타임:

```bat
run_comic.bat
```

CUDA13 런타임:

```bat
run_comic_cuda13.bat
```

### 2. 로컬 번역 서버 사용

저장소 루트에서 Gemma 서버 실행:

```bash
docker compose up -d
```

앱에서는 `Custom Local Server(Gemma)`를 선택합니다.

### 3. 로컬 OCR 서버 사용

HunyuanOCR 실행:

```bash
docker compose -f hunyuanocr_docker_files/docker-compose.yaml up -d
```

PaddleOCR VL 런타임 기준 파일은 [paddleocr_vl_docker_files/README.md](paddleocr_vl_docker_files/README.md)에 정리돼 있습니다.

### 4. 권장 OCR 설정

Settings에서 아래 중 하나를 선택합니다.

- `Default (existing auto: MangaOCR / PPOCR / Pororo...)`: 기존 자동 OCR 경로 유지
- `Optimal (HunyuanOCR / PaddleOCR VL)`: 중국어는 `HunyuanOCR`, 일본어/기타 언어는 `PaddleOCR VL`로 라우팅

## 저장소 문서

- [rules.md](rules.md)
- [docs/history/change-log.md](docs/history/change-log.md)
- [docs/history/change-log-ko.md](docs/history/change-log-ko.md)
- [docs/gemma/local-server-ko.md](docs/gemma/local-server-ko.md)
- [docs/hunyuan/local-server-ko.md](docs/hunyuan/local-server-ko.md)
- [docs/repo/github-rulesets-public-free-ko.md](docs/repo/github-rulesets-public-free-ko.md)

## Legacy Localized README

`docs/i18n/` 아래 예전 localized README는 더 이상 source of truth가 아닙니다.

현재 기준 문서는 아래 둘입니다.

- 영문 기준: [README.md](README.md)
- 한글 기준: [README_ko.md](README_ko.md)
