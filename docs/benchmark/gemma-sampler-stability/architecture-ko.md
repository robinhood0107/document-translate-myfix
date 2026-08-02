# Gemma sampler stability architecture

이 family는 Gemma 번역의 sampler tuple만 비교한다. OCR, 이미지 렌더링, prompt, schema, context, model, 출력 한도, KV 설정은 benchmark 중 고정한다.

```text
fixed text corpus + canonical reference
          |
          v
product request builder
          |
          v
Crop Router: Gemma explicit load (one model)
          |
          v
direct chat replay -> atomic private response record
          |
          v
schema gate -> exact cluster -> semantic review -> ranking
```

runner는 제품의 단일 블록 request builder를 재사용하되 OCR 요청과 render를 호출하지 않는다. 시작 시 Crop pair의 Router에 OCR 모델을 준비하고 Gemma로 전환한 뒤, 모든 요청을 동일한 loaded Gemma에 직접 replay한다. inference HTTP 동안 Router command lock을 잡지 않으며, 종료 시 Gemma unload와 container stop을 확인한다.

제품 추론 정책은 `reasoning=off`로 고정한다. 인접 텍스트는 화자·대상·관계·행동을 보존하기 위한 contextual-single 입력일 뿐, 모델에게 새 사실을 추론하거나 설명을 생성시키는 체인이 아니다. Router protocol token이 content에 섞이면 제품 parser와 같은 규칙으로 제거하고 `channel_token_sanitized` 통계로 기록한다. 그 token이나 내부 사고 문자열을 번역문으로 판정하지 않는다.

단계는 다음과 같다.

- temperature: `0.0..1.0`/0.1, `top_p=0.95`, `top_k=64`, `min_p=0`, 공통 seed 3개
- top-p: 선택 temperature에서 `0.90/0.95/1.00`; 기존 `0.95` identity는 재사용
- top-k: 선택 temperature/top-p에서 `32/64/128`; 기존 `64` identity는 재사용

sampler identity가 같은 응답 파일은 phase가 달라도 재사용한다. 그래서 전체 계획은 중복을 제외한 최대 990 response다. 원자 저장과 resume 검사는 response id, request contract hash, complete 상태를 함께 확인한다.

raw request/response, 원문, canonical reference, 이미지 locator, judgment 원본은 ignored private archive에만 둔다. Git에는 runner, 단위 테스트, private 자료를 노출하지 않는 이 문서와 집계 기준만 보관한다.
