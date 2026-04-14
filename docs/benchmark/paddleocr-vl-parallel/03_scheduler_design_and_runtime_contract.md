# 03 Scheduler Design And Runtime Contract

## 배경과 문제 상황

현재 구조는 `1 block -> 1 crop -> 1 request`이며, 단순 고정 worker 수만으로는 페이지마다 다른 crop 분포를 감당하기 어렵다. 예를 들어 작은 crop 8개를 동시에 보내는 것과 매우 큰 crop 8개를 동시에 보내는 것은 local vLLM에 전혀 다른 부하를 준다.

따라서 이번 설계의 목적은 “고정 worker”를 완전히 버리는 것이 아니라, 기존 동작을 보존하면서도 hidden mode에서만 더 똑똑한 스케줄링을 실험할 수 있게 만드는 것이다.

## 사용자가 제안한 핵심 발상

사용자는 `페이지별 동적 worker 계산 + 큰 crop 우선 취급`을 명확한 방향으로 제안했다.

## 구현 방식과 설계 선택

- mode:
  - `fixed`
  - `fixed_area_desc`
  - `auto_v1`
- `parallel_workers`는 hidden scheduler가 켜졌을 때 cap으로 사용한다.
- `auto_v1`는 local server에서만 GPU headroom을 반영한다.
- remote server는 local `nvidia-smi`와 무관하므로 crop 통계 기반 fallback만 적용한다.
- 기본 mode는 `fixed`이며, 제품 기본값은 바꾸지 않는다.

이 구조의 핵심은 `기본 behavior를 절대 깨지 않는 것`이다. 제품 승격 전까지는 hidden flag로만 candidate를 비교하고, runtime surface가 충분히 안정적이라고 판단될 때만 default-on 여부를 논의한다.

## worker 계산 규칙

- base:
  - `<2500MB -> 1`
  - `<4500MB -> 2`
  - `<6500MB -> 3`
  - `<9000MB -> 4`
  - `>=9000MB -> 5`
- penalty:
  - `p90_area_ratio >= 0.03 -> -2`
  - `0.02 <= p90 < 0.03 -> -1`
  - `large_crop_ratio >= 0.35 -> -1`
  - `max_new_tokens >= 1024 -> -1`
  - `gpu_util >= 85 -> -1`

이 규칙은 1차 휴리스틱이다. 즉, 모델링된 최적 해답이 아니라 `subset benchmark로 검증 가능한 명시적 규칙 집합`으로 먼저 고정했다. 그래야 결과가 좋지 않을 때 어떤 기준이 과했는지 또는 약했는지를 분해해서 볼 수 있다.

또한 `fixed_area_desc`를 별도 후보로 둔 이유는, 단순한 정렬만으로도 tail latency가 줄어드는지 확인하기 위함이다. 이 후보는 auto worker 계산 없이 `큰 crop 먼저` 효과만 따로 측정하는 통제군 역할을 한다.

## telemetry contract

- page profile:
  - `scheduler_mode`
  - `requested_cap`
  - `chosen_workers`
  - `block_count`
  - `p50_area_ratio`
  - `p90_area_ratio`
  - `large_crop_ratio`
  - `request_records`
- request record:
  - `job_index`
  - `bbox`
  - `crop_area_px`
  - `crop_area_ratio`
  - `enqueue_ts`
  - `start_ts`
  - `end_ts`
  - `elapsed_ms`
  - `status`

추가로 page profile에는 `page_width`, `page_height`, `job_order`, `max_new_tokens`, `local_server`, `gpu_metrics`, `started_at`, `completed_at`, `page_status`, `elapsed_ms`가 들어간다. 이 필드들은 benchmark 분석뿐 아니라 향후 운영 진단에도 도움이 된다.

이 runtime contract의 목적은 두 가지다.

- benchmark branch에서는 request-level 재구성과 GPU join 분석이 가능해야 한다.
- develop 쪽 제품 코드에서는 benchmark-specific 로직 없이도 generic telemetry surface만 남길 수 있어야 한다.

## 남은 한계와 다음 단계

- `weighted concurrency`
- next-page adaptive feedback
- default-on promotion

1차에서는 page-start 계산과 job ordering까지만 다루고, request-time weighted scheduling은 의도적으로 제외했다. 먼저 단순하고 설명 가능한 스케줄러를 benchmark로 검증한 뒤, 필요하면 2차에서 가중치 세마포어나 피드백 조정을 확장한다.

## 저자 및 기여

- 핵심 문제 해결 방향은 사용자가 착안했다.
- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
