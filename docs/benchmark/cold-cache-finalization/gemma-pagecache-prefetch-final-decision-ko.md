# Gemma page-cache read-ahead 최종 판정

## 결론

- named volume과 OS page cache의 자연 재사용: **유지**
- 인페인트 중 14.59GB GGUF 명시적 read-ahead: **제품 미채택**
- S6 확대 실행: **S1 속도·swap 게이트에서 정상 취소**

자연 page cache는 이미 제품의 versioned named volume, 정상 `docker stop`, cache를
강제로 지우지 않는 정책으로 자동 적용됩니다. 별도 제품 코드가 필요하지 않습니다.

## 격리 방법

시스템 전체 cache나 WSL을 종료하지 않고, Gemma GGUF 한 파일에만
`POSIX_FADV_DONTNEED`를 적용했습니다. `mincore`로 3,560,899 page 전체의 residency를
확인했고 매 실행 직전 resident page가 정확히 0인지 검사했습니다.

후보는 인페인트 시작과 동시에 다음 조건으로 모델 파일만 읽었습니다.

- 동일한 검증된 Gemma named volume과 image digest
- read-only volume mount
- network none
- GPU 미사용
- `ionice -c 3`, `nice -n 19`
- 인페인트 종료 시 미완료 read를 정상 `docker stop`
- `down`, 광범위 prune, `drop_caches`, WSL 종료 미사용

독립 read smoke에서는 14,585,439,872 bytes를 9.406초에 읽고 99.962% resident로
만들었으며, 모델 파일만 다시 0 page로 비울 수 있었습니다.

## S1 실제 Stage-Batched AB/BA

| round | baseline 전체 | read-ahead 전체 | baseline Gemma 시작 | read-ahead Gemma 시작 | 판정 |
|---:|---:|---:|---:|---:|---|
| 1 | 156.961초 | 97.554초 | 59.741초 | 16.022초 | deep-cold 후보 우위 |
| 2 | 103.656초 | 103.806초 | 20.402초 | 20.746초 | 후보 0.145% 느림 |
| 3 | 99.008초 | 99.234초 | 20.240초 | 18.341초 | 전체 후보 0.229% 느림 |

- 후보 median: 99.234초
- 기준선 median: 103.656초
- median 표면 개선: 4.265%
- paired 평균 개선: 12.492%
- paired median 개선: -0.145%
- 단측 95% bootstrap 하한: -0.201%
- 세 라운드 모두 페이지·detector·OCR 구조 계약 통과
- candidate swap delta: +0.164, +0.121, +0.121 MiB

1라운드에서는 Linux와 Windows/VHD 하위 cache가 모두 차가워 명시적 read-ahead가
Gemma 시작을 크게 줄였습니다. 그러나 한 번 읽은 뒤에는 Linux resident page를 0으로
만들어도 Windows 하위 파일 cache가 남아 자연 baseline도 18~20초에 시작했습니다.
이 상태에서 명시적 read는 중복 I/O가 되어 전체시간 이득이 반복되지 않았습니다.

## 제품 판정

첫 deep-cold 1회만 보면 가치가 크지만, 현재 제품은 한 번의 정상 시작 이후 자연
page cache를 이미 활용합니다. 명시적 후보는 반복 측정상 0%보다 빠르다고 입증되지
않았고 strict per-run swap 증가 0 계약도 통과하지 못했습니다.

따라서 제품에는 read-ahead thread, 별도 helper container, cache residency probe를
추가하지 않습니다. 자연 page cache를 보존하기 위해 다음 현행 정책만 유지합니다.

- Gemma GGUF를 versioned named volume에 보관
- 관리형 컨테이너 종료는 `stop`
- 정상 실행에서 `down`, model volume 재생성, cache drop, WSL 종료 금지
- translation all-hit에서는 Gemma 시작 0 유지

원시 결과와 모델 residency 자료는 Git 밖 validation log에 보존합니다.

