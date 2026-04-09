[English](README.md) | [한국어](README_ko.md)

# Comic Translate 포크

이 저장소는 upstream `comic-translate` `v2.6.7` 코드베이스에서 시작한 뒤, 로컬 런타임/OCR/워크플로/Windows 환경 쪽으로 제품화 수정을 누적한 local-first 포크입니다.

이 포크는 아래 워크플로를 중심으로 유지됩니다.

- 로컬 Gemma 번역 런타임
- `PaddleOCR VL`, `HunyuanOCR` 같은 로컬 OCR 런타임
- Windows 중심 설치/실행 도구
- upstream `v2.7.0`, `v2.7.1`의 selective manual backport
- benchmark 작업과 제품 브랜치의 분리

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
- `setup.bat`으로 `.venv-win`, `.venv-win-cuda13` 생성/검증 흐름을 만들었습니다.
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

### 1. Windows 환경 준비

```bat
setup.bat
```

이 명령은 아래 환경을 생성하거나 갱신합니다.

- `.venv-win`
- `.venv-win-cuda13`

### 2. 앱 실행

기본 Windows 런타임:

```bat
run_comic.bat
```

CUDA13 런타임:

```bat
run_comic_cuda13.bat
```

### 3. 로컬 번역 서버 사용

저장소 루트에서 Gemma 서버 실행:

```bash
docker compose up -d
```

앱에서는 `Custom Local Server(Gemma)`를 선택합니다.

### 4. 로컬 OCR 서버 사용

HunyuanOCR 실행:

```bash
docker compose -f hunyuanocr_docker_files/docker-compose.yaml up -d
```

PaddleOCR VL 런타임 기준 파일은 [paddleocr_vl_docker_files/README.md](paddleocr_vl_docker_files/README.md)에 정리돼 있습니다.

### 5. 권장 OCR 설정

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
