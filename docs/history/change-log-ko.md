# 변경 이력

## 기준선: upstream `v2.6.7`

이 포크는 upstream `comic-translate` `v2.6.7` 코드에서 시작했고, 이후 로컬 제품 요구에 맞춘 수정이 단계적으로 누적됐습니다.

## 로컬 개선 축

### 렌더링과 수동 워크플로

- 헤더/렌더 흐름을 먼저 문서화한 뒤 렌더 정책 변경을 진행했습니다.
- 강제 색상, 정렬, 윤곽선 상태를 다루는 공통 render policy helper를 도입했습니다.
- 직렬화되는 텍스트 상태에 `vertical_alignment`, `source_rect`, `block_anchor`를 추가했습니다.
- 렌더 패널을 더 명확한 사용자 그룹으로 재배치했습니다.
- 수동 렌더, 배치 렌더, 웹툰 렌더가 같은 공통 정책 경로를 쓰도록 맞췄습니다.

### Windows 런타임과 저장소 운영

- 기본 런타임과 CUDA13 런타임용 Windows launcher를 추가했습니다.
- `run_comic.bat`, `run_comic_cuda13.bat`가 로컬 venv/runtime을 스스로 bootstrap하도록 바꿨습니다.
- 로컬 hook과 CI 검증을 강화했습니다.
- 브랜치 정책을 표준화했고, 이후 `codex/` 접두사도 제거했습니다.
- 최종적으로 저장소 정책을 `main + develop + tag`로 정리했습니다.

### OCR 안정성과 진단

- one-page auto와 batch OCR의 parity를 개선했습니다.
- block-local detection fallback을 추가했습니다.
- suspicious short-result retry를 추가했습니다.
- 잔여 글자 제거를 위해 text mask를 넓히고 bubble residue cleanup을 추가했습니다.
- OCR diagnostics와 runtime selection 테스트를 확장했습니다.

### 로컬 모델/런타임 통합

- 로컬 Gemma 서버 경로와 런타임 튜닝을 추가했습니다.
- PaddleOCR VL 통합과 기본값 튜닝을 추가했습니다.
- HunyuanOCR 통합을 추가했습니다.
- `Optimal (HunyuanOCR / PaddleOCR VL)` 라우팅과 언어 기반 런타임 선택을 추가했습니다.

### 벤치마크 도구와 브랜치 분리

- benchmark toolkit 스크립트와 one-click runner를 추가했습니다.
- benchmark 자산을 제품 브랜치에서 분리했습니다.
- `benchmarking/lab`를 benchmark 전용 브랜치로 명문화했습니다.

## Selective Backport 축

### `v2.6.7 -> v2.7.0`

upstream `v2.7.0`의 필요한 사용자 기능만 선택적으로 현재 포크 구조에 맞게 적응 이식했습니다.

문서:

- [v267-to-v270-backport-audit.md](v267-to-v270-backport-audit.md)
- [v267-to-v270-backport-audit-ko.md](v267-to-v270-backport-audit-ko.md)

### `v2.7.0 -> v2.7.1`

upstream `v2.7.1`의 버그 수정 중 이 포크에 의미 있는 항목만 선택 반영했습니다. 핵심은 아래입니다.

- PSD import 안정화
- main-thread callback 안전성 정리
- 리스트 썸네일 로더 안정화
- PSD 메뉴 정리
- 앱 버전 `2.7.1` 반영

문서:

- [v270-to-v271-backport-audit.md](v270-to-v271-backport-audit.md)
- [v270-to-v271-backport-audit-ko.md](v270-to-v271-backport-audit-ko.md)
