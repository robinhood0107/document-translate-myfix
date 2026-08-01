# Serving·scheduler matrix 구현 계약

## 목적

제품 코드를 임시 수정하지 않고 현재 pinned llama.cpp와 prepared named volume을
사용해 Paddle crop OCR의 server slot, client worker, HTTP thread, polling과 idle
handoff를 한 축씩 비교한다. 원시 결과는 관리형 비공개 archive에만 기록한다.

## 현재 환경에서 확인된 사실

- pinned image: llama.cpp `b10133`, revision `ff067f76d`
- 이 build는 `parallel`, `threads-http`, `poll`, `poll-batch`, `metrics`, `slots`,
  idle sleep와 multi-model router를 지원한다.
- Paddle 단독 model-ready GPU 사용량: 2,407 MiB
- Gemma 단독 model-ready GPU 사용량: 11,634 MiB
- 두 model을 동시에 올린 합계: 14,041 MiB / 12,282 MiB, 114.3%
- 따라서 사용자가 허용한 95% residency gate에서도 dual-model R3는 실제 OOM
  실행 없이 탈락한다.
- Paddle가 idle sleep으로 model을 내리고 process만 유지한 상태에서 Gemma를
  올리면 11,715 MiB, 95.4%였고 shared-memory fallback 없이 동작했다. 현재 제품
  handoff도 이미 이 R1 계열이다.

## Lab runner

`scripts/benchmark_serving_scheduler_matrix.py`는 다음을 강제한다.

- 제품 container를 삭제하지 않고 별도 lab container와 별도 port 사용. 실패
  로그를 보존한 뒤 생성한 lab container만 정확한 이름으로 제거함
- Paddle·Gemma prepared named volume은 read-only mount
- Paddle volume의 label·ready manifest SHA·smoke 성공·GGUF/mmproj 크기를
  실행 전에 빠르게 검증하며, 대형 파일 전체 재해시는 하지 않음
- llama.cpp image/build/revision과 필수 option 재검증
- 시작 GPU background 2 GiB 이하, Windows available RAM 6 GiB 이상
- 실행 중 Windows available RAM 최저값을 1초 간격으로 기록하고, 6 GiB
  미만이면 승격을 금지한다. 1 GiB 미만이 3회 연속이면 child run을
  종료한다.
- 실행 시작 대비 WSL swap 증가 0. 새 lab container에서는 cgroup v2
  `memory.swap.peak`도 0이어야 하며, cgroup을 읽을 수 없을 때만 전역 WSL
  delta를 fallback으로 사용
- cache OFF 실제 offscreen 제품 pipeline에서 detector workload를 한 번 고정한
  뒤, 모든 serving 후보에 같은 페이지·bbox·crop 정책을 재생
- `/props`, `/slots`, `/metrics`, 실제 Docker command·mount·label 보존
- OCR page/block/raw text/diagnostics canonical snapshot exact 비교
- baseline/candidate 순서를 바꾼 paired 실행과 단측 95% bootstrap
- 이득이 불확실하면 최대 7회, 끝까지 하한이 0 이하면 현행 유지
- `np > 1`에서는 total context를 `4096 × np`로 늘리고 `/props`와 `/slots`로
  각 slot의 context가 4096 이상인지 재검증한다.
- OCR ceiling의 E2E에는 llama.cpp container start-to-health도 포함한다.

`idle 5초 → 1초`는 OCR request 시간이 아니라 OCR→인페인트 handoff를 바꾸므로
이 후보만 관리형 제품 runtime의 실제 full-auto S1/S6로 측정한다. 나머지
slot/worker/HTTP/poll 후보는 별도 포트의 OCR ceiling replay로 빠르게 선별한다.

full-auto 비교의 제품 sampler는 변경하지 않는다. `temperature=0.7`의 자연
샘플링 때문에 후보와 무관한 번역 차이가 생기지 않도록 benchmark HTTP
payload에만 고정 seed를 추가한다. 이 seed는 benchmark copy에만 들어가며 제품
설정과 사용자 요청에는 저장되지 않는다.

## 2026-08-01 live preflight 결과

첫 S1 진단 쌍에서는 OCR·좌표가 같았고 Paddle release가
`5.242초 → 1.037초`로 4.205초 줄었다. 전체 wall은 `86.253초 → 68.265초`였지만
Gemma 자연 page-cache의 cold/warm 순서와 고정 seed 이전의 번역 변동이 함께
섞였으므로 이 수치는 승격 근거로 사용하지 않는다.

고정 seed를 적용한 다음 실행에서는 20GB WSL profile이 Gemma page cache와
Windows CUDA 앱 메모리를 동시에 유지하는 동안 Windows available RAM이
0.393GiB까지 내려갔다. 합의한 6GiB 조건을 위반해 즉시 중단했으며 Paddle와
Gemma container는 정확한 이름으로 `stop`했다. 계획의 fallback에 따라 다음
재시작용 `.wslconfig`는 18GB로 조정했다. 현재 WSL을 종료하지 않았으므로 새
상한은 아직 적용 전이다.

따라서 현재 판정은 다음과 같다.

- `idle=1`: 직접 handoff 이득은 재현됐지만 전체 E2E·RAM gate 검증 미완료
- Paddle-only `np/worker`, HTTP, poll matrix: 현재 WSL에서 실행 완료
- dual-active R3: 합산 114.3%이므로 95% VRAM gate에서도 탈락
- Paddle process를 sleep 상태로 남기고 Gemma만 활성화한 R1은 95.4%로 경계값을
  약간 넘지만 shared-memory fallback은 없었다. 18GB WSL의 full-auto RAM gate를
  통과하기 전에는 제품 후보로 자동 승격하지 않는다.

## Paddle CUDA matrix 결과

모든 숫자는 cache OFF, 같은 frozen detector/crop workload, baseline/candidate
AB/BA 조건에서 측정했다. `E2E`는 lab llama.cpp start-to-health와 OCR replay
wall을 합친 serving ceiling이고, `request`는 동시 요청들의 누적 request wall이다.

### S6 73-block 선별

| 후보 | E2E 중앙 개선 | request 중앙 개선 | 두 라운드 snapshot | 판정 |
|---|---:|---:|---|---|
| `np2-w2` | 3.09% | 69.13% | 비대사 edge/noise 1개 변동 | 탈락 |
| `np2-w4` | 11.41% | 47.15% | 73/73 exact | 확대 검증 |
| `np2-w6` | 10.13% | 26.50% | 비대사 edge/noise 1개 변동 | 탈락 |
| `np2-w8` | 9.59% | 8.75% | 비대사 edge/noise 1개 변동 | 탈락 |
| `np4-w4` | 14.48% | 49.83% | 반복 중 비대사 1개 변동 | 탈락 |
| `np4-w6` | 18.54% | 29.86% | 73/73 exact | 확대 검증 |
| `np4-w8` | 16.01% | 12.98% | 비대사 edge/noise 1개 변동 | 탈락 |

`np2-w4`와 `np4-w6`을 각각 7회 확대하자 두 후보 모두 모든 라운드에서
baseline보다 빨랐지만, 같은 non-text edge crop이 일부 라운드에서 기존의 한
글자 false positive 대신 긴 비문자 응답으로 흔들린 뒤 제품 guard에 의해 빈
값으로 거부됐다.

| 확대 후보 | 7회 E2E 중앙 개선 | 단측 95% 하한 | request 중앙 개선 | 변동 |
|---|---:|---:|---:|---|
| `np2-w4` | 12.59% | 10.20% | 47.26% | 7회 중 1회 |
| `np4-w6` | 17.69% | 15.82% | 30.52% | 7회 중 2회 |

원본 페이지를 직접 확인한 결과 해당 좌표에는 의미 있는 글자가 없었다. 따라서
candidate-only 의미 회귀는 0이고 빈 값 거부는 기존 false positive를 줄이는
방향이다. 다만 결과가 byte-exact가 아니므로 이 lab PR에서 제품 기본값을
자동 변경하지 않고 `np4-w6`을 별도 품질 검수 후보로 분류한다.

### Japan 22페이지 확대

현재 detector snapshot의 293개 block을 두 라운드 재생했다.

- startup 포함 serving ceiling: 19.723초 → 16.163초, 중앙 18.043% 개선
- request 누적 wall: 106.099초 → 76.191초, 중앙 28.177% 개선
- 292개 block은 모든 실행에서 완전히 동일
- 유일한 차이는 S6와 같은 non-text edge/noise block의 안전 거부
- swap·Windows RAM·GPU resource gate 통과

따라서 현재 최속 품질 후보는 `np=4`, client worker `6`, 기본 HTTP/poll이다.
제품 승격은 18GB WSL에서 실제 full-auto S1/S6가 비회귀이고 최종 이미지가
동일하거나 해당 false positive만 제거됨을 다시 확인한 뒤 별도 PR로 한다.

최종 runner 변경 뒤 S1 재검증에서도 15/15 snapshot exact, E2E 8.812%,
request 34.251% 개선을 기록했다. 네 baseline/candidate 실행 모두 global WSL
swap delta와 container cgroup swap peak가 0이었다.

### HTTP·poll 축

- `threads-http=2`: 전체 중앙 +0.04%였으나 라운드가 -3.83%~+3.91%로
  뒤집히고 request는 1.89% 느려짐
- `threads-http=4`: 전체 3.53%, request 1.57% 느려짐
- `threads-http=8`: 전체 2.58%, request 0.26% 느려짐
- `poll=0`: 전체 0.77% 느리고 승패가 뒤집힘
- `poll=100`: 전체는 1.29% 빨랐지만 request가 1.25% 느리고 승패가 뒤집힘
- `poll-batch=0`: request는 1.35% 빨랐지만 전체 신뢰구간 하한이 -0.37%

HTTP thread와 polling은 실제 순이득을 입증하지 못해 llama.cpp 기본값을
유지한다.

## 중복 시험 방지

다음 후보는 direct llama.cpp CUDA 환경에서 이미 최대 7회까지 판정됐으므로 새
기본 matrix에 넣지 않는다.

- folder-global queue worker 4: 명목 +0.390529%, CI 하한 -0.276427%
- Paddle completion 768: 명목 +0.421%, CI 하한 -0.520%
- Paddle batch/ubatch 후보: 반복 측정상 양수 우위 없음

이번 matrix는 아직 끝나지 않은 `np × worker`, HTTP thread, polling과 idle
handoff만 실행한다. 한 축의 우승값을 다음 축의 기준선에 누적하는 제품 승격은
실제 full-auto E2E 검증 후 별도 제품 PR에서만 수행한다.

## Router 판정 방식

공식 INI preset은 Paddle와 Gemma의 모델별 command를 분리하고
`models-max=1`, `no-models-autoload`로 구성한다. Router는 최신 기능이 존재한다는
이유만으로 채택하지 않는다. 별도 process 두 개를 유지하는 현행 R1보다 model
load·handoff·전체 E2E가 반복 측정상 빨라야만 후속 제품 후보가 된다.
