# Single llama.cpp Router handoff workflow

이 lab은 OCR 조합 하나와 Gemma 하나를 **단일 llama.cpp router 컨테이너**가 순차로 관리할 때, 현재 별도 OCR/Gemma container보다 실제 E2E가 빨라지는지만 확인한다.

각 pair는 `baseline → router → router → baseline`을 정확히 한 번만 실행한다. 추가 반복, adaptive 확장, R3 동시 상주는 하지 않는다.

- router는 `--models-max 1`, `--no-models-autoload`, explicit `/models/load`·`/models/unload`를 사용한다.
- OCR load/request/unload 후 Gemma load/request/unload, 그리고 OCR re-load/request/unload을 한 번 확인한다.
- 기존 Paddle crop/Spotting의 prompt, model alias, image 처리, 재시도, parser, client scheduling은 바꾸지 않는다. router는 기존 요청을 검증·라우팅만 한다.
- MangaLMM도 현재 full-page 설정을 그대로 사용한다. completion·resize budget을 낮춰 OCR 결과를 바꾸는 것은 handoff 실험이 아니다.
- detection은 boxes/class/order/mask, OCR은 raw result/diagnostics/block mapping, inpaint는 decoded-pixel/diagnostics를 exact로 검증한다. 번역 hard contract의 exact 범위는 model/runtime identity, prompt/schema/seed, 요청 순서·개수, JSON 완결성, `finish_reason`, 누락·중복이다. raw 번역 문구 차이는 그 자체로 실패가 아니며 private text-first 의미 검수 대상으로 남긴다.
- raw 번역이 exact하면 decoded pixel SHA도 exact여야 한다. 의미 동등한 다른 번역이면 baseline pixel SHA를 품질 실패로 삼지 않고, router 내부 재현성·render 완료·오류 없음을 확인한다.
- OCR raw result는 run-local UUID가 아니라 detection 순서에 묶어 private archive에 보관한다. 텍스트만으로 애매한 경우에만 해당 페이지를 확인한다.
- OOM, hang, wrong route, identity mismatch, orphan, cancellation cleanup failure, GPU return failure는 즉시 REJECT다. RAM/shared GPU/swap은 관측값이다.
- 두 baseline이 모두 번역 전 OCR quality gate에서 실패하면 해당 fixture는 `NOT ELIGIBLE`이다. router 속도 수치는 채택 근거가 될 수 없다.

MangaLMM 사전조건 불충족 시도는 성능 비교에 포함하지 않으며, baseline-gated 최종 결과만 보고한다. 결과가 생성되면 제품 Compose나 product runtime을 바꾸지 않고 사용자 승인에서 중지한다.
