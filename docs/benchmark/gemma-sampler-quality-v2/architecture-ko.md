# Gemma sampler quality v2 구조

이 benchmark family는 제품 경로가 아니다. `protocol`, `corpus`, `execution`,
`judgment`, `report` 모듈과 얇은 CLI로 구성하며, 원문·정답·요청·응답·판정은
ignored private managed validation archive에만 저장한다.

Windows용 `gemma-sampler-launcher.exe`는 CUDA13 BAT supervisor와 Go/Bubble Tea
monitor를 한 창으로 연결한다. monitor는 managed run의 원자 `progress.json`, completion
index, manifest만 짧게 열어 읽고 즉시 닫으며, runner·Router·Docker에 쓰기 요청을 보내지
않는다. 새 실행·재사용 증거 진행률을 분리하고, 초기에는 r6 실측 ETA를, 500응답 뒤에는
recent/overall 속도를 결합한 ETA를 표시한다. interruption이나 stale progress에서는
중복 runner를 만들지 않고, 끊긴 worker checkpoint만 같은 campaign ID로 resume한다. EXE와
runner log도 private archive에만 둔다.

`campaign`은 Router v2의 정확한 Crop + 기본 Gemma pair를 준비한 뒤 Gemma를 한 번만
명시적으로 load한다. r6는 immutable provenance로만 읽고, 새 joint/min-p slot은 하나의
Router session에서 순서대로 실행한다. HTTP는 inference lease 안에서만 보내고, timeout
뒤에는 active request 0과 Router slot idle을 확인한 경우에만 재시도한다. 모든 새 slot이
끝난 뒤 Gemma unload, Router container stop, GPU 반환 검증까지 성공해야만
`WAITING_FOR_FINAL_JUDGMENT`가 된다.

정답지는 758 occurrence를 `language + source_text + context_after_text`로 묶은
478 case다. tuning / holdout 구분은 provenance로 보존하지만, 이번 단일 campaign의
최종 semantic judgment와 순위는 처음부터 478 case 전체를 사용한다.
