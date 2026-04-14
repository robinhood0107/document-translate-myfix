# 01 Subset Curation And Coverage

## 배경과 문제 상황

`Sample/japan` 전체 22장을 모든 candidate마다 여러 번 반복하면, local OCR runtime tuning 속도가 지나치게 느려진다. 특히 이번 과제는 warmup, measured repeat, GPU sampling, candidate matrix 비교가 함께 들어가므로, representative subset 없이 진행하면 실험 회전율이 급격히 떨어진다.

따라서 이번 1차는 “속도 결정을 내리기에 충분히 representative 하면서도, 반복 실험이 가능한 크기”의 subset을 먼저 만드는 것이 필요했다.

## 사용자가 제안한 핵심 발상

사용자는 `block 수`, `crop 크기`, `가장 지저분한 p_016.jpg 포함`을 기준으로 실험 세트를 따로 모으자는 방향을 제안했다. 이는 단순한 샘플 축소가 아니라, 실제 runtime에 큰 영향을 주는 request shape를 우선적으로 남기자는 의도였다.

## 측정 설계와 기준

- corpus는 `Sample/japan_vllm_parallel_subset`
- 선정 파일 수: `13 / 22`
- block coverage: `212 / 305 = 69.5%`
- 필수 포함: `p_016.jpg`
- 해상도 family:
  - `094-101.png`
  - `i_099-i_105.jpg`
  - `p_015-p_021.jpg`

subset은 파일 수가 아니라 `block coverage`와 `request shape diversity`를 기준으로 설계했다. 따라서 단순히 랜덤 13장을 뽑는 것보다 훨씬 높은 대표성을 가진다.

## 구현 방식과 설계 선택

- dense page, sparse large-crop page, bubble-heavy page, free-text-heavy page를 모두 남긴다.
- `p_016.jpg`는 mandatory hard anchor로 유지한다.
- `p_021.jpg`는 large crop pressure를 보는 anchor로 유지한다.
- `i_102.jpg`, `p_020.jpg`는 다수 block 페이지의 throughput pressure를 대표한다.
- `p_019.jpg`는 상대적으로 low-pressure high-resolution 케이스로 남긴다.

이 구성은 `요청 수 스트레스`, `큰 crop tail latency`, `page density`, `family별 해상도 차이`를 동시에 확인하기 위한 것이다. 즉, subset은 단순한 축소판이 아니라 “스케줄러 설계에서 반드시 흔들리면 안 되는 페이지 집합”으로 취급한다.

## 결과와 효과

subset만으로도 local vLLM scheduling에 중요한 요청 수 스트레스와 단일 crop VRAM 스트레스를 동시에 볼 수 있다. 또한 smoke나 candidate sweep을 빠르게 반복할 수 있으므로, 실험-해석-수정 루프의 속도가 훨씬 좋아진다.

결과적으로 이 subset은 1차 winner selection을 위한 operational corpus로 사용하고, 22장 full corpus는 2차 promotion 검증용으로 남긴다.

## 남은 한계와 다음 단계

subset winner가 나와도 22장 full promotion 검증이 추가로 필요하다.

또한 subset coverage가 높다고 해도 모든 edge case를 대표하지는 않는다. 향후 definitive gold 단계에서는 `subset winner`가 full corpus에서도 동일하게 유지되는지 별도 검증이 필요하다.

## 저자 및 기여

- 핵심 문제 해결 방향은 사용자가 착안했다.
- Idea Origin: User
- Planning / Measurement Design / Implementation Detailing / Validation: Collaborative
- Execution Support: Codex
