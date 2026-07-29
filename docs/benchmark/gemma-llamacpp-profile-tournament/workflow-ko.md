# Gemma llama.cpp 프로필 토너먼트 워크플로

## 목적

현재 제품 기준선과 여러 26B GGUF를 동일한 제품 번역 계약으로 비교한다.
비교 축은 target 양자화, llama.cpp ngram speculative decoding, 전용 MTP
draft model, target/draft GPU offload다. Gemma 런타임은 llama.cpp만
사용한다.

## 고정 계약

- request mode: `contextual-single`
- shared context chunk: 6
- context: 4096
- parallel slot: 1
- threads: 10
- target KV: F16
- draft KV: F16
- completion limit: 512
- prompt, sampler, JSON Schema, sanitizer: 현재 제품 구현
- translation result cache, Exact TM, project checkpoint: 모두 OFF
- llama.cpp prompt cache RAM: 0
- 실행 전 idle dedicated VRAM: 2048 MiB 이하
- 실행 사이 후보 컨테이너는 정상 `stop`

`contextual-grouped`, Q8 KV, vLLM Gemma, prompt 변경은 이 토너먼트에
포함하지 않는다.

## 단계

1. 외부 model manifest의 pairing과 cache contract를 검증한다.
2. 모든 target/draft의 SHA-256, 크기, GGUF architecture와 tokenizer
   fingerprint를 한 번 잠근다.
3. 관리형 lab volume에만 신규 모델을 `.partial`로 복사하고, 크기와
   SHA-256을 확인한 뒤 atomic rename한다. 제품 volume은 변경하지 않는다.
4. 외부 GPU 컨테이너, 높은 idle VRAM, 포트 충돌이 있으면 중단한다.
5. target NGL 23과 draft full GPU에서 시작해 안전 경계를 찾는다.
6. 모든 프로필을 smoke와 sensitive-15에 통과시킨다.
7. 18블록을 정방향·역방향으로 실행한다. 확실히 느리거나 품질에 실패한
   후보만 탈락시킨다.
8. 생존 후보를 일본어·중국어·영어 각 18블록으로 두 번 실행한다.
9. paired bootstrap의 단측 95% 하한이 0을 넘지 않으면 해당 후보만
   3회차를 실행한다.
10. MTP가 작은 입력에서 느리고 큰 입력에서 빠르면 6·15·30·54블록
    손익분기점을 측정한다.

최소 속도 향상률은 없다. 품질이 유지되고 반복 측정으로 실제 이득이
확정되면 작은 이득도 채택한다.

## 품질 게이트

- 모든 item과 원래 순서 보존
- 빈 값, 잘림, 중첩·중복·후행 JSON, 잘못된 타입 0
- unresolved fallback과 `finish_reason=length` 0
- 후보에서만 발생한 화자, 관계, 부정, 의무, 행동, 주체·대상, 숫자,
  고유명사 회귀 0
- 명시적 성적·폭력적 의미의 누락·순화·거부 0
- 라운드 간 심각한 의미 반전 0
- OOM과 비정상 shared GPU memory 증가 0
- llama.cpp 컨테이너 cgroup swap peak가 manifest 한도 이내
- cgroup 계측을 사용할 수 없을 때만 전역 WSL swap 증가량으로 안전 판정

구조 게이트는 runner가 판정한다. 의미 품질은 외부 raw 결과를 원문
기준으로 전수 검수한다.
