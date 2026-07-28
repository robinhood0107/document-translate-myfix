# Gemma 최종 번역 전용 비교 architecture

## 경계

runner는 benchmark 계층에만 있으며 제품 기본값, preset 선택, ranking 정책을
바꾸지 않는다. 번역은 `CustomLocalGemmaTranslation.translate()`를 페이지별로
직접 호출하므로 grouped decoder, contextual fallback, 반복 guard, telemetry는
제품과 같은 코드를 통과한다.

## 데이터 흐름

```text
22-image manifest
  + 292-block OCR snapshot
  + model/image/runtime contracts
        |
        v
counterbalanced candidate rounds
        |
        v
page-ordered product translation
        |
        +--> structural/performance summary
        |
        +--> randomized blind A/B/C review
```

TM과 exact TM은 후보 비교 중 비활성화한다. 각 후보가 292블록 전체를 실제
Gemma 서버에 보내도록 보장하기 위한 통제다.

## 안전장치

- 입력 파일 manifest SHA-256
  `63cdfa53fc7c48efa9e6f1f11aae3e86bb5ea0aadcba361491ee34bc94cc9b1e`와
  OCR snapshot SHA-256
  `22fd706b63da75a5a4c7f4175cec6d23f9b9e9a831e82365329fca42e1c84605`를
  알려진 최종 corpus 값으로 고정
- snapshot 보관본의 raw SHA-256이 다르면 decoded pixel을 비교하고, lossless
  equality 또는 `MAE <= 1.25`, `p99 <= 5`, `max <= 12`, `PSNR >= 43 dB`를
  모두 만족한 JPEG 재인코딩만 허용
- 매 suite 시작마다 host와 volume 모델 전체 SHA-256을 다시 계산하고, digest로
  고정한 helper image의 image ID까지 다르면 중단
- image ID, `/models/<model>` 경로, 전체 llama.cpp command, 후보별 config
  fingerprint가 역할과 다르면 중단하고, `/models` 아래를 가리는 추가 mount도
  허용하지 않음
- 컨테이너는 `127.0.0.1:18080`에만 publish하며, API URL도 같은 loopback
  endpoint만 허용. `bridge`, non-privileged, no auto-remove, restart policy
  `no`도 고정
- suite 전체에 배타적 lock을 잡고, 시작·중지는 이름이 아니라 사전검사한
  container ID로 실행
- 실행 중 예외가 나도 시작한 후보는 `stop`
- 결과 위치가 Git tracked 경로이면 중단
- resume 결과는 `runs/` 아래 경로, 파일 SHA-256, protocol/code/prompt 계약,
  candidate/round, 292개 원문 hash·순서와 gate를 모두 다시 검증
- 사용자 품질 gate는 자동 통과시키지 않음

속도 타이머 안에서는 resource subprocess나 sampling thread를 실행하지 않는다.
대신 시작 전, warmup 후, 번역 직후, stop 후에 driver VRAM, Docker
memory/process, runner RSS, WSL memory/swap, Windows GPU adapter
shared/dedicated counter를 점검한다. VRAM gate는 시작 전에는 없고 warmup 후와
번역 직후에 계속 존재하는 새 GPU process가 정확히 하나이며, 그 PID가 후보
Docker process에 속하고 외부 GPU process 목록이 그대로일 때만 사용한다.
이 귀속을 입증하지 못하면 VRAM 값은 unavailable이며 Q8은 3% 속도 조건으로만
통과할 수 있다. Docker server 또는 NVIDIA driver identity 자체를 얻지
못하면 서로 다른 측정 환경을 같은 resume 결과로 합치지 않도록 suite를
시작하지 않는다.
