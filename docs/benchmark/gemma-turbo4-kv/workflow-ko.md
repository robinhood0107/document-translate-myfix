# Gemma Turbo4 KV-V workflow

> Retired: TurboQuant/Turbo4는 최종 폐기됐으며 이 workflow는 과거 검증 문서다. 새 replay,
> output-limit tuning, 제품 반영을 실행하지 않는다.

이 lab은 동일 IQ4_NL 모델의 TurboQuant fork KV-V `turbo4`만 검증했다. 새 GGUF, QAT,
MTP/draft, n-gram/speculative 후보는 포함하지 않았다.

1. 독립 lab 컨테이너와 runtime identity를 준비한다. 실행 중인 product/lab GPU 컨테이너는
   격리 실패로 중단한다. GPU background·RAM·shared GPU·swap은 1초 샘플로 기록한다.
2. fork F16/F16과 F16 K+Turbo4 V의 fixed-seed 구조 계약을 먼저 확인한다. 이어 shipping
   b10133 F16 control과 비교한다. `finish_reason`, JSON, 요청 수/순서, model/runtime identity
   중 하나라도 다르면 즉시 REJECT다.
3. raw non-exact만 남으면 기존 private 73-request ledger를 text-first로 전수 검수한다.
   삭제·검열·민감 표현 약화·부정/동의/강제/화자/대상/행동/관계/숫자 변화는 REJECT다.
   애매한 행은 사용자 확인 전 `REVIEW_REQUIRED`다. 의미 검수용 GPU replay는 하지 않는다.
4. 의미 PASS인 후보만 fork F16 ↔ Turbo4 ABBA를 하고, 이어 shipping b10133 F16 ↔ Turbo4
   ABBA를 한다. one-sided 95% 하한이 0 이하이거나 A/B와 B/A 승패가 갈리면 최대 7
   pair-round까지 측정한다.
5. 통과 후보만 S1 ABBA, S6 ABBA, true series 3+3으로 진행한다. 각 E2E arm의 실제 번역
   response ledger도 text-first 의미 검수를 거쳐야 한다. Turbo4 E2E는 upstream exact와
   render 완료를 확인하되 후보 간 final decoded-pixel SHA는 진단값이다.
6. 결과 보고 후 반드시 멈춘다. 사용자 승인 전에는 제품 브랜치나 다음 성능 단계를 시작하지
   않는다.

R3 active Paddle+Gemma 동시 상주는 실행하지 않는다. Turbo4 peak로 90% 추정치만 갱신한다.
속도 저하, OOM, container/runtime 불안정, orphan, GPU 반환 실패는 즉시 REJECT다. true series
adapter 또는 fixture가 없을 때 flat batch를 series PASS로 바꾸지 않는다.
