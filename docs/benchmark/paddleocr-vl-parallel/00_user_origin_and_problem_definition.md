# 00 User Origin And Problem Definition

## 배경과 문제 상황

현재 `PaddleOCR VL` 경로는 `1 block -> 1 crop -> 1 request` 구조다. 이 구조는 OCR 정확도를 안정적으로 유지하는 대신, 페이지마다 block 수와 crop 크기 분포가 크게 달라질 때 전체 OCR 시간이 불규칙하게 흔들리는 문제가 있었다.

특히 이 프로젝트는 로컬 `vLLM`과 로컬 OCR runtime을 함께 사용하므로, 단순히 평균 속도만 보는 것이 아니라 `요청 직전의 실제 GPU headroom`, `동시 요청 수`, `큰 crop tail latency`가 서로 어떻게 영향을 주는지를 함께 봐야 한다.

문제는 단순했다. 번역 단계는 이번 의사결정에 필요하지 않다. OCR 품질만 현재와 같다면, 가능한 한 많은 block을 가장 빠르게 처리해야 한다.

## 사용자가 제안한 핵심 발상

핵심 문제 해결 방향은 사용자가 착안했다. 사용자는 다음 두 가지를 문제의 중심으로 짚었다.

- `parallel_workers`를 고정값으로 두지 말고, 페이지별 block/crop 상황에 맞게 worker 수를 계산해야 한다.
- 큰 crop과 작은 crop을 같은 비용으로 취급하면 안 되므로, 큰 crop이 많은 페이지와 작은 crop이 많은 페이지를 다르게 다뤄야 한다.

이 발상은 단순한 파라미터 튜닝이 아니라, `요청 스케줄링 자체를 페이지 특성 기반으로 바꾸자`는 제안이었다. 이후의 계측 설계, telemetry surface, benchmark family, winner selection 기준은 모두 이 문제 정의에서 출발한다.

## 왜 기존 접근이 실패했는가

- full-page single-shot `PaddleOCR VL`은 구조화 출력 순응성이 낮았다.
- 좌표와 OCR 텍스트를 직접 prompt로 강제해도 반복 토큰, 빈 JSON, malformed 좌표가 발생했다.
- 따라서 현 구조에서는 full-page 전환보다 block-crop 병렬 최적화가 현실적인 해법이었다.

여기서 중요한 포인트는 `full-page로 바꾸는 것`이 단순히 요청 수를 줄이는 문제가 아니었다는 점이다. 실제 실험에서는 좌표와 텍스트를 함께 안정적으로 뽑아내지 못했고, 결과적으로 후처리 비용과 품질 리스크가 현재 block-crop 방식보다 훨씬 컸다.

즉, 이번 문제는 OCR 모델을 바꾸는 문제가 아니라, `이미 맞는 구조를 얼마나 효율적으로 돌릴 것인가`의 문제로 재정의되었다.

## 측정 설계와 기준

- 절대 정답 OCR이 아니라 `baseline shipping 결과와 품질이 같으면 된다`는 운영 목표를 사용한다.
- detector geometry는 baseline seed로 고정하고, 이후 비교는 OCR만 한다.
- 속도 지표는 `전체 OCR 시간`, `OCR page p95`, `request-level elapsed`, `GPU free floor`를 함께 본다.
- 품질 게이트는 CER, exact match, empty block delta, page failure delta로 정의한다.

이 기준은 “더 정확한 OCR”이 아니라 “지금과 같은 품질을 더 빠르게 제공하는 OCR 스케줄러”를 뽑기 위한 것이다. 운영 관점에서 이게 더 중요하다고 판단했다.

## 구현 방식과 설계 선택

- hidden scheduler mode만 제품 코드에 넣고 기본값은 바꾸지 않는다.
- `fixed`, `fixed_area_desc`, `auto_v1` 세 모드만 1차 범위에 둔다.
- benchmark는 `detect-ocr-only` 실행으로 translation 비용을 배제한다.
- request-level telemetry는 제품 코드에 generic surface로 남기고, benchmark-specific preset/ranking/report는 `benchmarking/lab`에만 둔다.

이 분리는 이후 제품 승격과 benchmark 자산 보존을 동시에 만족시키기 위한 저장소 정책이기도 하다. 제품 코드에는 일반화 가능한 runtime surface만 두고, 후보 비교 로직과 narrative report는 benchmark branch에 남긴다.

## 결과와 효과

이 문서는 문제 정의 문서다. 실측 결과는 `04_candidate_results_and_decision_log.md`, `docs/banchmark_report/paddleocr-vl-parallel-report-ko.md`, 그리고 latest asset summary에서 관리한다.

현재까지의 가장 중요한 효과는 다음과 같다.

- full-page single-shot 시도 대신 block-crop 병렬 최적화라는 현실적인 방향을 확정했다.
- 사용자 제안이 benchmark family와 제품 runtime contract로 구체화되었다.
- 이후 결과가 쌓여도 설명 가능한 구조로 문서 체계를 먼저 고정했다.

## 남은 한계와 다음 단계

- subset winner만으로는 `default on` 승격을 하지 않는다.
- 2차에서는 22장 full corpus와 weighted concurrency를 검토한다.
- 1차 gold는 baseline seed 기반이므로, 사람이 교정한 definitive gold와는 분리해 관리해야 한다.
- 현재 `auto_v1`는 page-start 수준 의사결정이다. request-time weighted scheduling은 후속 단계다.

## 저자 및 기여

- 핵심 문제 해결 방향은 사용자가 착안했다.
- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
