# Runtime resource arbiter 구현 기록

## 목적

OCR, CUDA inpainter와 Gemma가 서로 다른 코드 경로에서 시작·종료되더라도 GPU
model ownership은 하나의 직렬화된 계약으로 관리한다. 이 변경은 모델, prompt,
OCR 결과, 인페인트 픽셀과 render 결과를 바꾸지 않는 구조 리팩터링이다.

## 상태와 lease 계약

관리형 OCR·Gemma model 전이는 하나의 command lock을 통과한다.

```text
STOPPED/SLEEPING → MODEL_LOADING → MODEL_READY
MODEL_READY → RELEASING → STOPPED/SLEEPING
RELEASING → RELEASE_FAILED
```

- 다른 model이 `MODEL_READY`이면 새 model load를 거부한다.
- release 실패 시 기존 owner를 유지하고 다음 GPU model 시작을 막는다.
- CUDA inpainter는 process 밖 runtime manager가 아니므로 model load 직전에
  external lease를 획득하고 VRAM 반환 gate가 확인된 뒤에만 해제한다.
- CPU inpainter는 GPU lease를 획득하지 않는다.
- process 시작·health 상태는 기존 runtime-manager telemetry가 계속 기록하며,
  arbiter는 GPU model ownership만 담당한다.

관리형 Docker OCR/Gemma의 GPU 반환은 UI Python PID만으로 증명하지 않는다. model
load 직전의 driver 전체 GPU memory 기준선을 기록하고, stop 또는 idle sleep 뒤 같은
GPU의 driver-total memory가 model-load delta의 충분한 비율만큼 감소했는지 확인한다.
sleeping llama.cpp는 model을 내린 뒤에도 재사용 process·CUDA context가 남을 수 있어
512MiB 이내의 명시적인 process-only residue만 허용한다. stop은 이 allowance를 받지
않는다. GPU measurement가 없거나 반환이 모호하면 fail-closed로
`RELEASE_FAILED`를 유지한다. 이 상태에서는 다음 GPU model을 시작할 수 없다.

## 취소와 prewarm

- background runtime command executor는 한 worker만 사용한다.
- 각 command는 run generation token을 가진다.
- 취소된 generation은 아직 시작하지 않은 command를 거부한다.
- model load 도중 취소되면 manager의 targeted cleanup을 실행한다.
- 취소 직후 늦게 끝난 command도 `MODEL_READY`를 게시하지 않고 정리한다.
- telemetry observer 오류는 실제 runtime ownership을 깨뜨리지 않는다.

## 90% residency 판정

물리 VRAM의 90% 이하이고 shared GPU memory와 새 swap 증가가 0인 경우에만
활성 동시 residency를 lab 후보로 허용한다.

- active Paddle+Gemma 합계: 물리 VRAM의 114.3%
- sleeping Paddle process+Gemma 관측치: 물리 VRAM의 95.4%

sleeping 조합도 90% 사전검사를 초과하고 실제 활성 model 조합은 114.3%로 더 크게
초과한다. 따라서 활성 동시 model residency를 실제 OOM 실험으로 밀어붙이지
않고 exclusive lease를 제품 계약으로 유지한다. process 재사용과 model
residency는 구분하며, sleeping process는 실제 model 해제 확인을 통과해야 다음
stage로 넘어간다.

## 검증과 범위

- cancellation generation과 stale cleanup
- OCR/Gemma 상호 배제와 sleeping release
- CUDA inpainter external lease와 VRAM gate 실패 차단
- Docker OCR/Gemma driver-total model-sized VRAM 반환 확인
- sleeping Paddle은 model-load delta의 85% 이상 반환 및 512MiB 이내 process-only
  residue 확인
- release 실패 후 owner 보존
- observer failure의 fail-open 격리
- 기존 stage cancellation·inpainter artifact 보존 테스트

최종 전체 suite는 `.venv-win`, `.venv-win-cuda13`을 순차 실행했으며 각 환경에서
`914 tests`, `7 skipped`로 통과했다. raw validation log와 hardware evidence는
private archive에만 보관한다.

이 PR은 serving 설정이나 기본 profile을 승격하지 않는다. 실제 full-auto 성능
판정은 18GB WSL memory profile이 적용된 뒤 후속 Paddle handoff·throughput PR에서
S1/S6와 series AB/BA로 수행한다.
