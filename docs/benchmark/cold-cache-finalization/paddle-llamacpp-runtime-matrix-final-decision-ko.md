# PaddleOCR-VL llama.cpp batch/ubatch CUDA 최종 판정

## 결론

제품 기본값 `batch=2048`, `ubatch=512`를 유지한다. 실제 CUDA Stage-Batched 경로에서 출력은 모두 같았지만 어떤 후보도 반복 측정상 양수 속도 이득을 입증하지 못했다.

## batch 최종 비교

`batch=512`, `ubatch=512` 후보를 기준선과 7회 AB/BA로 비교했다.

| 지표 | 평균 개선 | 중앙값 개선 | 단측 95% bootstrap 하한 | 승패 |
|---|---:|---:|---:|---:|
| 전체 elapsed | -0.160% | +0.521% | -4.900% | 4승 3패 |
| pipeline run wall | -0.650% | -2.693% | -5.565% | 3승 4패 |
| OCR request 합계 | -0.081% | +0.196% | -0.548% | 4승 3패 |
| OCR stage | -0.031% | +0.042% | -0.392% | 5승 2패 |

- 7/7 page snapshot exact
- 페이지 실패 0, HTTP retry 0
- 컨테이너 실제 명령과 runtime fingerprint로 후보값 확인
- 최소 개선율 미적용

`batch=1024/4096`은 단일 스크리닝에서도 request 시간을 줄이지 못했다. `ubatch=256/384/768`도 `batch=2048` 고정 스크리닝에서 기준선보다 request 합계가 길었다.

초기 후보 스크리닝 세 건은 WSL 환경변수가 Windows Python에 전달되지 않아 실제 명령이 기준선으로 유지된 것을 사후 발견했다. 해당 결과는 폐기하고 `WSLENV`와 Docker `Config.Cmd`를 확인한 유효 실행만 판정에 사용했다.
