# PaddleOCR-VL folder-global queue CUDA 최종 판정

## 결론

`folder-global queue + worker 4`는 제품에 승격하지 않는다. 과거 relay 경로와
현재 direct llama.cpp 경로를 각각 7회 paired CUDA 실행했지만, 두 환경 모두
OCR 결과는 기준선과 완전히 같고 반복 측정상 속도 우위는 입증되지 않았다.
제품은 기존 페이지 단위 장벽과 worker 8 설정을 유지한다.

최종 상태 코드는 `reject_no_proven_speed_gain`이다.

이 판정에는 최소 개선율을 적용하지 않았다. 작은 이득도 실제 양수라는 증거가 있으면 채택하는 정책을 그대로 사용했다.

## 고정 조건

- 실제 `.venv-win-cuda13` Stage-Batched 제품 파이프라인
- RTX 4070 SUPER, 실행 시작 GPU 사용량 2 GiB 이하
- 일본어 3페이지, 실행마다 실제 OCR HTTP 요청 60회
- persistent OCR cache는 실행별 빈 격리 DB 사용
- translation cache, Exact TM, project checkpoint 비활성
- 각 실행 사이 Docker Compose `stop`; `down` 미사용
- 기준선: 기존 페이지 단위 처리, worker 8
- 후보: 폴더 전역 queue, worker 4

## 7회 결과

| 회차 | 기준선 (초) | 후보 (초) | 후보 개선율 |
|---:|---:|---:|---:|
| 1 | 72.602 | 66.268 | +8.724% |
| 2 | 66.695 | 66.888 | -0.289% |
| 3 | 66.613 | 66.442 | +0.257% |
| 4 | 66.437 | 66.402 | +0.053% |
| 5 | 66.828 | 66.775 | +0.079% |
| 6 | 65.683 | 67.272 | -2.419% |
| 7 | 65.189 | 65.714 | -0.805% |

- 후보 승패: 4승 3패
- 기준선 중앙값: 66.613초
- 후보 중앙값: 66.442초
- 중앙값 기준 명목 개선: 0.257%
- paired 평균 개선: 0.800%
- 단측 95% bootstrap 하한: -0.891%
- 첫 cold outlier를 제외한 paired 평균: -0.521%
- 첫 cold outlier를 제외한 단측 95% bootstrap 하한: -1.202%

## 품질과 자원

- 7/7 snapshot 비교 통과
- 페이지·블록 수, OCR 품질 카운터, normalized raw OCR 완전 동일
- 페이지 실패 0, HTTP retry 0
- 실행 중 신규 WSL swap 증가 0
- GPU peak에서 일관된 감소 없음

## 최종 해석

첫 실행의 +8.724%는 cold 상태 차이로 보이며 이후 재현되지 않았다. warm 6회만 보면 후보가 평균 0.521% 느렸다. 전체 7회에서도 신뢰구간 하한이 0보다 작고 라운드별 승패가 뒤집혔으므로 속도 동률로 처리한다.

초기 소규모 화면에서 관측된 2.8904% 개선은 제품 근거로 재현되지 않았다. 후보 제품 코드는 병합하지 않고, 이 문서와 Git 밖 원시 실행 결과만 역사적 증거로 보존한다.

## Direct llama.cpp transport 재검증

relay 제거가 병합된 제품 commit `e354106`을 기준으로, 같은 3페이지 corpus를
direct `18000/v1/chat/completions` 경로에서 다시 7회 실행했다. 실행별
`LOCALAPPDATA`와 persistent OCR SQLite는 격리했고 각 실행 사이에는 Compose
`stop`만 사용했다.

| 회차 | page barrier w8 | folder-global w4 | 후보 개선율 |
|---:|---:|---:|---:|
| 1 | 18.257934초 | 16.697639초 | +8.545846% |
| 2 | 16.681376초 | 16.465158초 | +1.296164% |
| 3 | 16.458521초 | 16.479327초 | -0.126415% |
| 4 | 16.405624초 | 16.639376초 | -1.424828% |
| 5 | 16.549862초 | 16.511537초 | +0.231573% |
| 6 | 16.331019초 | 16.389623초 | -0.358851% |
| 7 | 16.603216초 | 16.485230초 | +0.710621% |

- 후보 승패: 4승 3패
- 기준선 pipeline 중앙값: 16.549862초
- 후보 pipeline 중앙값: 16.485230초
- 중앙값 명목 개선: 0.390529%
- paired 평균 개선: 1.267730%
- 단측 95% bootstrap 하한: -0.276427%
- 첫 cold 회차 제외 평균 개선: 0.054711%
- 7/7 snapshot, 84/84 block, normalized OCR 완전 동일
- 실행당 OCR HTTP 60회, retry 0, page failure 0
- 실행 시작 대비 WSL swap 증가 0

첫 회차의 큰 차이는 container/model startup 편차가 섞인 cold outlier이며,
후속 6회는 사실상 동률이다. direct transport에서도 신뢰구간 하한이 0보다
작으므로 최종 상태는 계속 `reject_no_proven_speed_gain`이다.

원시 결과는 Git 밖
`<validation-log-root>/paddle-folder-global-queue/20260801_direct_v3/`에 보존한다.
