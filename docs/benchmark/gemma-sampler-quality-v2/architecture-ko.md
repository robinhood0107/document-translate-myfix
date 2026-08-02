# Gemma sampler quality v2 구조

이 benchmark family는 제품 경로가 아니다. `protocol`, `corpus`, `execution`,
`judgment`, `report` 모듈과 얇은 CLI로 구성하며, 원문·정답·요청·응답·판정은
ignored private managed validation archive에만 저장한다.

`execution`은 Router v2의 정확한 Crop + 기본 Gemma pair를 준비한 뒤 Gemma만
명시적으로 load한다. HTTP는 inference lease 안에서만 보내고, timeout 뒤에는
active request 0과 Router slot idle을 확인한 경우에만 재시도한다. phase가 끝나면
Gemma unload, Router container stop, GPU 반환 검증까지 성공해야 한다.

정답지는 758 occurrence를 `language + source_text + context_after_text`로 묶은
478 case다. 382 tuning / 96 holdout은 source 순서와 무관한 case-id hash로 고정한다.
holdout 출력은 tuning provisional winner가 기록되기 전에는 판정 packet으로 열 수 없다.
