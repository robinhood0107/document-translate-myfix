# Portfolio Docs

## 목적

- `develop`에는 benchmark raw 결과 대신, 사용자 제안과 제품 승격에 필요한 요약 문서만 남긴다.
- canonical benchmark docs와 full evidence는 `benchmarking/lab`에 둔다.
- 아이디어 착안자: 사용자

## 현재 문서 묶음

### 1. Workflow Split Runtime

- 전체 워크플로우를 단계형으로 분리해 시간 이득을 확보한 작업
- 현재 제품 승격 대상:
  - `Stage-Batched Pipeline (Recommended)`
  - `Legacy Page Pipeline (Legacy)`
- 마지막 기술 blocker였던 CTD 마스킹 경로 연결까지 포함해 정리

### 2. Hybrid OCR Selector

- `MangaLMM` hybrid selector 실험 기록
- 현재 상태: `failed_closed`

## 문서 원칙

1. benchmark canonical 문서는 `benchmarking/lab` 기준으로 본다.
2. `develop`에는 제품 코드와 직접 연결되는 요약만 남긴다.
3. raw benchmark 결과, chart asset, review pack은 `develop`에 들고 오지 않는다.
