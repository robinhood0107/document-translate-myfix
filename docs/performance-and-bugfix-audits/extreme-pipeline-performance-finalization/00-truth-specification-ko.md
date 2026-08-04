# Comic Translate 극한 파이프라인 성능 진실 명세서

## 목표

캐시가 없는 새 폴더를 `Detection → OCR → Inpaint → Gemma → Render → 저장`까지
처리하는 실제 전체시간과, 여러 하위 프로젝트를 연속 처리하는 시리즈 시간을
줄인다. 후보별 stage replay가 아니라 최종 full-auto 시간이 빨라져야 제품에
승격한다.

## 변경할 수 없는 품질 계약

- Detection의 box, class, 순서와 mask를 보존한다.
- OCR raw text, diagnostics와 페이지·block 대응을 보존한다.
- Inpaint와 render의 decoded pixel 결과를 보존한다.
- 번역 prompt, schema, sampler, seed와 요청 순서를 보존한다.
- 출력이 달라지는 후보는 speed 개선으로 병합하지 않고 별도 품질 실험으로
  분리한다.
- 최소 개선율은 두지 않지만 AB/BA 반복 측정으로 실제 이득을 입증하지 못하면
  현행을 유지한다.

## 기준선

현재 제품 기준선은 Stage-Batched, direct PaddleOCR-VL llama.cpp crop OCR,
IQ4_NL Gemma contextual-single, CUDA inpaint, CPU render다. 캐시가 없는 S6 warm
실측 174.200초에서 번역 73.866초, 인페인트 53.628초, Inpaint→Gemma 전환
17.789초가 가장 큰 구간이었다.

## 허용된 최적화 축

1. 범용 performance telemetry와 critical-path work graph
2. GPU runtime exclusive lease와 상태 전이 직렬화
3. Paddle handoff, global queue와 llama.cpp slot/worker 조합
4. CPU translation 준비·render와 GPU stage의 안전한 overlap
5. 설정이 같은 series child의 stage super-batch
6. 동일 tensor shape에 한정한 exact GPU execution
7. 실측 우승 시에만 router 또는 sm89 전용 llama.cpp image
8. 환경·stage fingerprint별로 검증된 관측만 사용하는 ETA

모델 출력 계약을 바꾼 과거 후보, 강제 page-cache read-ahead, `mlock`,
DirectIO, unified memory, Docker pause와 GPU 75% overlap은 다시 시험하지 않는다.

## 실행 순서

1. `feature/pipeline-critical-path-telemetry-v2`
2. lab serving/scheduler matrix
3. runtime resource arbiter
4. Paddle handoff와 throughput
5. series stage super-batch
6. translation/render overlap
7. exact GPU execution
8. 조건부 router/residency/sm89
9. calibrated critical-path ETA
10. 누적 S1/S6/series/S22 full-auto 검증

제품 코드는 `develop`, benchmark runner와 순위는 `benchmarking/lab`, 원시
이미지·응답·hardware 결과는 Git에서 제외된 private validation archive에 둔다.
telemetry v2의 공개 구현 계약은
[01-performance-telemetry-v2-ko.md](./01-performance-telemetry-v2-ko.md)에
기록한다. GPU runtime lease의 공개 구현 계약은
[03-runtime-resource-arbiter-ko.md](./03-runtime-resource-arbiter-ko.md)에
기록한다. 실제 실행 로그는 private archive의 `cold-pipeline-speed`와 향후
managed performance run 아래에서 보존한다.

## 공통 환경 게이트

- 시작 GPU background 2GiB 이하
- Windows available RAM 최저 6GiB
- 실행 시작 대비 새로운 WSL swap 증가 0
- OOM과 shared GPU memory fallback 0
- 실행 사이 정상 `stop`; `down`과 광범위 prune 금지
- inpainter와 OCR/Gemma의 동시 GPU 상주 금지
- Paddle와 Gemma의 동시 활성 residency는 물리 VRAM 90% 이하, shared GPU
  memory와 swap 증가 0을 모두 만족한 lab 후보만 허용

현재 12GB GPU에서 active Paddle+Gemma 예상치는 114.3%, sleeping
Paddle+Gemma 관측치는 95.4%다. 두 조합 모두 active-residency 사전검사에는
통과하지 못한다. 따라서 현 제품은 GPU model exclusive lease를 유지한다.

## ETA 원칙

페이지 수에 고정 시간을 곱하는 기존 추정은 폐기한다. stage workload,
runtime 상태, 코드·모델 fingerprint와 최근 호환 관측을 사용하고 backtest 품질이
부족하면 숫자 대신 `측정 중`을 표시한다. ETA 저장소에는 raw text, 파일명,
이미지와 로컬 경로를 저장하지 않는다.
