# Paddle 현재-폴더 source-aware prewarm 최종 판정

## 결론

`feature/paddle-source-aware-prewarm`을 제품에 승격한다.

기존 경로는 persistent OCR DB에 다른 입력의 row가 하나라도 있으면 현재 폴더도 cache hit일 수 있다고 가정해 Paddle 시작을 OCR 단계까지 미뤘다. 새 경로는 detection이 끝난 페이지의 exact crop cache를 먼저 조회하고, 현재 폴더에서 첫 miss가 확인되는 즉시 PaddleOCR llama.cpp prewarm을 시작한다.

## CUDA AB/BA 결과

고정 조건:

- Windows CUDA13, RTX 4070 SUPER
- WSL memory 20GB, swap 8GB
- 일본어 6페이지, 73 OCR block
- 실제 offscreen Stage-Batched OCR stage ceiling
- 각 cold 실행은 이번 입력과 무관한 cache row 7개만 가진 격리 DB에서 시작
- 실행 사이 관리형 컨테이너는 정상 `stop`; `down` 미사용

| 순서 | 기준선 | 후보 | 후보 이득 |
|---|---:|---:|---:|
| baseline → candidate | 79.638958초 | 76.450467초 | 4.003682% |
| candidate → baseline | 79.594652초 | 77.238041초 | 2.960766% |

- 평균/중앙 이득: 3.482224%
- 단측 95% bootstrap 하한: 2.960766% (100,000 resamples, seed 20260801)
- 두 라운드 모두 후보 승리
- 모든 유효 cold 실행은 73/73 HTTP 완료, 실패 0, WSL swap 증가 0
- 기준선·후보 normalized page snapshot SHA-256은 모두 `6b6ea449602d4f22f7df142c92b721a27ee1d99d0ffd917417cc1846d1269be1`

## all-hit와 부분 hit

all-hit:

- 73 hits, 0 misses
- Paddle runtime start 0
- OCR HTTP 0
- wall 11.053760초
- 출력 snapshot exact

부분 hit:

- 첫 페이지 20개 crop hit, 나머지 53개 miss
- 첫 페이지는 `defer/persistent_cache_hit`
- 두 번째 페이지에서 `start/persistent_cache_miss`
- Paddle runtime start 1, OCR HTTP 53
- wall 64.682185초
- 출력 snapshot exact

따라서 신규 이미지의 첫 OCR을 단축하면서도 all-hit의 zero-runtime 계약을 유지한다.

## 제외 실행

첫 기준선·후보 실행 한 쌍은 Windows Python launcher가 WSL 호출에서 분리된 뒤 결과 summary가 기록되기 전에 컨테이너를 정지한 운영 오류가 있었다. 두 raw 실행은 외부 validation log에 그대로 보존하지만 성능 통계에서는 제외했다. 이후 실행은 summary 파일 생성을 확인한 뒤 컨테이너를 정지했다.

## 자동 검증

- `.venv-win` 관련 140 tests + 8 subtests 통과
- `.venv-win-cuda13` 관련 187 tests 통과, 1 skipped
- Python 511 files validation 통과
- headless smoke, Qt translation check, Windows launcher verification 통과
- pre-landing review unresolved finding 0

raw 입력·DB·실행 로그·페이지 결과는 Git 밖에만 보존한다.
