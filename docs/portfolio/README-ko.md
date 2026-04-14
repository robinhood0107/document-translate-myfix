# Portfolio Docs

## 목적

- `develop`에는 benchmark raw 결과 대신, 사용자 제안과 설계/문제 해결 과정을 재사용 가능한 문서로 남긴다.
- full benchmark docs/assets/history는 `benchmarking/lab`에 남긴다.
- 아이디어 착안자: 사용자

## 현재 문서 묶음

### 1. Workflow Split Runtime

- 전체 워크플로우를 단계형으로 분리해 Docker/VRAM 병목을 줄일 수 있는지 검증하는 작업
- 제품 승격 문서와 진행 체크리스트 포함

### 2. Hybrid OCR Selector

- Requirement 1 성공 후 MangaLMM과 PaddleOCR VL의 페이지별 전환 기준을 만드는 작업
- 사용자 검수 기반 selector rule 설계 문서 포함

## 문서 원칙

1. 내부 추론 원문 대신 결정 로그를 남긴다.
2. 사용자 제안, 문제 정의, 측정 설계, 구현 방향, 효과, 남은 리스크를 빠짐없이 적는다.
3. raw benchmark 결과는 링크만 하거나 별도 branch 존재를 설명하고, `develop`에는 직접 들고 오지 않는다.
