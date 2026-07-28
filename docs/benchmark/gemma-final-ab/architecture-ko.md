# Gemma 최종 A/B blind 품질 검수 architecture

## 경계

`scripts/benchmark_gemma_final_ab.py`는 benchmark report layer에만 있다.
제품 번역 코드, prompt, JSON Schema, sampler, runtime 설정을 변경하지
않는다.

이 도구가 실행할 수 있는 외부 프로세스는 output 경로가 저장소 안에 있을
때 Git ignore 여부를 확인하는 명령뿐이다. Docker CLI, HTTP client,
Gemma 모델은 호출하지 않는다.

## 데이터 흐름

```text
locked protocol v3 suite
  |-- suite_state + source result SHA-256
  |-- frozen input/OCR contracts
  |-- model/runtime/translation contracts
  `-- seven source run files
              |
              v
       protocol v4 verifier
        |             |
        |             `--> Q8 elimination evidence only
        v
 baseline r1/r2 + grouped F16 r1/r2
              |
              v
 one private A/B mapping shared across rounds
        |                         |
        v                         v
 public HTML/CSV            private key/payload/audit
        |
        v
 complete 292-row review validator
        |
        v
 explicit-confirmation unblind
        |
        +--> grouped regression 0: quality_approved
        `--> grouped regression >0: quality_rejected
```

## 고정 source 계약

protocol v4는 다음 값을 코드에 고정한다.

- source protocol version과 suite canonical fingerprint
- 22페이지·292블록 input/OCR snapshot 계약
- IQ4_XS model 이름, 크기, SHA-256
- llama.cpp image ID와 세 runtime fingerprint·전체 명령
- prompt/profile/schema/sampler/cache 비활성 상태를 포함한 번역 동작 계약
- source 일곱 run의 순서, 상대 경로, SHA-256
- 검수 산출물을 만든 protocol v4 report tool의 SHA-256

source state가 결과 SHA-256을 함께 바꾸더라도 코드에 고정된 digest와
다르면 거부한다.

## blind 불변 조건

- A와 B는 baseline/grouped F16의 전단사다.
- Q8은 mapping에 들어갈 수 없다.
- 같은 label은 1·2라운드에서 같은 후보다.
- HTML·CSV·public state에는 후보명, 속도, mapping이 없다.
- review의 원문·A1·A2·B1·B2는 private payload와 완전히 같아야 한다.
- 292행의 네 판정이 모두 `yes` 또는 `no`여야 한다.
- unblind는 완전한 review와 `292-ROWS-REVIEWED` 확인문이 모두 필요하다.
- 기존 output 디렉터리는 빈 상태여도 재사용하지 않는다.

## 실패 처리

source 변조, 누락, 순서 오류, Q8 혼입, 계약 drift는 산출물 생성 전에
실패한다. review 오류는 mapping을 공개하지 않고 validation 오류만
기록한다. 기존 source suite와 사용자가 작성한 CSV를 수정하지 않는다.
