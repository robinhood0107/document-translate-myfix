# Gemma llama.cpp 프로필 토너먼트 구조

## 데이터 경계

Git에는 runner, 중립 테스트, 계약 문서만 둔다. 다음은 외부 validation
root에만 존재한다.

- 로컬 모델 절대 경로를 가진 model manifest
- sensitive-15와 다국어 54블록 원문·reference
- 모델 SHA-256 inventory lock
- raw 번역, llama.cpp log, GPU telemetry
- 후보 비교와 전수 품질 검수 결과

runner는 model manifest, corpus, result output이 저장소 안을 가리키면
거부한다.

## 모델 저장

기존 제품 model volume은 read-only로 마운트한다. 신규 후보만 별도
관리형 lab volume으로 준비한다.

```text
host GGUF
  -> <volume_filename>.partial
  -> size + SHA-256 검증
  -> atomic rename
  -> .ready-v1.json을 마지막에 기록
```

서로 다른 로컬 GGUF가 같은 공개 파일명을 가져도 manifest의
`volume_filename`으로 볼륨 목적지를 고유하게 지정한다. 같은 볼륨에
목적지 이름이 중복되면 복사를 시작하기 전에 manifest를 거부한다.
파일이 이미 있으나 계약이 다르면 덮어쓰거나 삭제하지 않고 중단한다.

## runtime identity

container name은 아래 계약의 SHA-256 prefix로 결정한다.

- llama.cpp image ID
- target model ID·크기·SHA-256
- MTP draft ID·크기·SHA-256
- 전체 server command
- read-only Docker volume mapping

정확히 같은 stopped container만 `start`한다. 다른 fingerprint는 별도
이름으로 생성하며 실행 사이에는 `stop`만 사용한다.

## tokenizer와 pairing

GGUF metadata에서 architecture, tokenizer model/pre, token 목록과
type/merge, chat template fingerprint를 읽는다. manifest가 허용한
target→draft 관계여야 하고, target과 draft 양쪽의 tokenizer model/pre,
token 문자열, merge table은 같아야 한다. 일부 MTP가 생략한 항목과
assistant draft에서 달라질 수 있는 token type·BOS/EOS flag는 증거로
기록한다. 최종 호환성은 실제 load·generation smoke로 확정한다.

## 계측

각 실행은 다음을 분리한다.

- runtime start→ready
- 제품 계약 warm-up
- request-only
- 전체 합계
- chunk별 wall/prompt/decode
- streaming probe TTFT
- Prometheus token/draft/accepted metric delta
- peak dedicated VRAM, GPU utilization, shared GPU memory
- llama.cpp 컨테이너 cgroup `memory.swap.current`와 `memory.swap.peak`
- 전역 WSL swap 증가는 진단값으로 별도 기록

swap hard gate는 실행 중인 llama.cpp 컨테이너의 cgroup peak를 우선한다.
이 값은 해당 프로필의 실제 swap만 포함한다. cgroup 파일을 읽을 수 없는
환경에서만 전역 WSL swap 증가량으로 fail-safe 판정한다.

랭킹은 실제 제품 번역 요청 시간을 paired 비교한다. startup 회귀와
메모리 안정성은 별도 hard gate다.

## 실패 안전성

- 외부 GPU 컨테이너나 높은 idle VRAM: 시작 전 중단
- 포트의 비관련 컨테이너: 중단
- OOM·health 실패·컨테이너 swap 한도 초과: 후보 실패
- Docker `down`과 제품 volume 변경: 사용하지 않음
- 결과 파일: checksum을 넣고 기존 파일을 덮어쓰지 않음
- 실패 컨테이너: 삭제하지 않고 stopped/exited 증거로 보존
