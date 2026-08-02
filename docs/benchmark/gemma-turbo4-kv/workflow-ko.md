# Gemma Turbo4 KV-V workflow

이 lab은 동일 IQ4_NL 모델에서 TurboQuant fork의 KV-V `turbo4`만 검증한다. 새 GGUF, QAT, MTP/draft, n-gram/speculative 후보는 이 workflow에 포함하지 않는다.

순서는 고정이다.

1. GPU background와 실행 중인 product GPU container를 fail-closed로 검사한다. Windows RAM, WSL/container swap, shared GPU counter는 1초 샘플로 기록하지만 단독 탈락 조건은 아니다.
2. 40자리 TurboQuant fork commit을 확인하고 SM89 CUDA image를 빌드한다.
3. fork F16/F16, fork F16 K + Turbo4 V, shipping b10133 F16을 동일 fixed-seed replay로 비교한다. fork F16이 shipping output과 다르면 즉시 탈락하며, 이어 fork F16 대 Turbo4 ABBA가 출력 exact·속도 판정을 통과해야 한다.
4. 통과 후보만 S1 ABBA, S6 ABBA, true series 3+3 순으로 실제 offscreen pipeline을 실행한다. 모든 arm은 product Gemma 컨테이너가 아닌 독립 lab container·loopback port·read-only model volume을 쓴다.
5. 각 단계가 통과해야 다음 단계로 간다. 결과가 나온 뒤에는 제품 반영 전에 사용자 승인을 기다린다.

실행 중 Paddle+Gemma active co-residency(R3)는 절대 시작하지 않는다. Turbo4 peak VRAM으로 R3 90% 계산만 다시 한다. host-memory·swap 관측치는 실제 E2E 시간과 함께 보고하며, OOM·GPU 반환 미확인·container orphan은 즉시 REJECT다.

S1/S6/series E2E는 finalized Arbiter가 benchmark base에 동기화된 뒤에만 Arbiter 검증으로 기록한다. true series adapter 또는 fixture가 없을 때 flat batch를 series PASS로 바꾸지 않는다.
