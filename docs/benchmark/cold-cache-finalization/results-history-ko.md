# cold/cache 최종화 결과 이력

## protocol v1 도구 추가

- frozen 제품 기준선과 후보 matrix를 추가했습니다.
- 실제 offscreen stage-batched pipeline을 최대 6페이지로 실행합니다.
- cold 후보는 3회 순서 교차, cache는 disabled/empty miss overhead와
  all-hit를 분리 측정합니다.
- project checkpoint는 기존 output, missing output materialization,
  한 페이지 OCR downstream 무효화를 검증합니다.
- stopped Docker container의 불필요한 300초 health wait를 제거했습니다.
- wrapper 사전 부팅을 제거하고 staged runtime을 제품 manager에 주입해
  cold startup과 all-hit runtime 0을 실제 제품 경로에서 측정합니다.
- 첫 실제 스모크에서 staged Compose의 암시적 project 이름이 달라 기존
  stopped 컨테이너와 이름 충돌하는 문제를 재현했습니다. 제품과 같은
  Compose project 이름을 명시해 원인을 수정했습니다.
- 같은 stopped container 조건의 후속 1페이지·20블록 스모크는 페이지
  실패 0, Paddle runtime 시작 1회, OCR HTTP 20회로 완료됐습니다.
  runtime 시작은 82.276초, 전체 pipeline wall time은 105.318초였으며
  종료 뒤에도 제품 Compose project identity가 유지됐습니다.
- 첫 project checkpoint cold control에서는 staged Gemma command에
  `-ctk/-ctv`와 같은 의미의 긴 옵션이 중복돼 제품 runtime의 명령 완전
  일치 계약이 실패했습니다. staging이 기존 `-ctk/-ctv` 값을 직접
  교체하도록 수정하고 중복 옵션 금지 회귀 검사를 추가했습니다.
- staged Gemma command에 no-spec/F16 계약을 직접 고정했습니다.
- full baseline의 실제 stage share를 검증해 stage 5% 또는 예상 전체 1%
  게이트를 판정할 수 있게 했습니다.
- 의미 검수 대상은 모든 라운드를 외부 private review JSON으로 묶습니다.
- raw 결과는 Git 밖에만 기록합니다.
- provisional global OCR cache 1페이지 검증은 miss overhead 0.036%,
  all-hit 99.413% 단축, runtime 시작 0, HTTP 0, raw OCR 완전 동일로
  통과했습니다.
- provisional project checkpoint 1페이지 검증은 all-stage hit,
  runtime 시작 0, Gemma HTTP 0, render-only 복원, 페이지 단위 재계산,
  all-hit 91.406% 단축을 확인했습니다. 다만 checkpoint에서 복원한
  과거 OCR profile을 현재 telemetry가 다시 합산해 Paddle HTTP 30회로
  보이던 오류와 최초 runtime 준비 편차 때문에 정식 판정은 보류했습니다.
- 과거 OCR telemetry 재합산은 제품 PR #156에서 수정했습니다.
- cache runner는 cache-disabled/enabled-empty를 각각 한 번 비채점
  안정화한 뒤 격리된 측정 3회를 실행하도록 보강했습니다. 안정화 결과도
  보존하고 실패 시 즉시 중단하지만 miss overhead와 variance에는 넣지
  않습니다.
- 새 project 검증에서 cold median miss overhead -4.375%, all-hit
  92.230% 단축, 모든 stage hit, runtime/HTTP 0, render-only 복원,
  한 페이지 OCR downstream 재계산이 통과했습니다.
- cold wall variance는 disabled 8.676%, enabled-empty 15.284%였습니다.
  outlier는 각각 inpainter 시간 증가와 Paddle/Gemma runtime wait 증가로,
  cache I/O와 무관했습니다. cache 보고서에는 이를 비차단 diagnostic으로
  유지하고 승인된 median miss overhead 3% gate와 분리합니다.
- 첫 partial 재계산은 프로젝트 복원 UI callback이 창 종료 뒤 실행되는
  benchmark race로 실패했습니다. 프로젝트 UI 작업을 pipeline 시작 전에
  drain하도록 수정한 뒤 동일 partial smoke와 정식 재계산이 통과했습니다.
- commit `47b6480`의 clean checkout에서 protocol v1
  (`7dfc2baa57ebdafc0e9ca2944e20397ccc9cd008d92f366f2a046dbffd3af04f`)
  cache suite를 다시 실행했습니다.
- global OCR exact cache는 disabled cold 중앙값 20.101초,
  enabled-empty 중앙값 19.999초로 miss overhead -0.510%였고,
  all-hit는 0.120초로 99.401% 단축됐습니다. raw OCR은 완전히 같았고
  Paddle runtime 시작, 논리 요청, HTTP 요청은 모두 0이었습니다.
- project checkpoint는 disabled cold 중앙값 129.793초,
  enabled-empty 중앙값 129.657초로 miss overhead -0.105%였고,
  all-hit는 9.551초로 92.633% 단축됐습니다. 모든 project stage hit,
  detector/Paddle/Gemma/inpainter inference 0, Paddle/Gemma runtime 시작
  및 HTTP 0, missing-output render-only 복원, 단일 페이지 OCR downstream
  재계산, exact output과 비영향 페이지 보존을 모두 통과했습니다.
- 최종 cold variance는 global OCR disabled/enabled 0.885%/0.494%,
  project disabled/enabled 1.587%/2.971%로 diagnostic 기준 5% 안에
  들어왔습니다.
- pre-landing review에서 cache protocol state가 입력 content SHA를
  기록하지 않던 재현성 누락을 찾았습니다. 입력 SHA 계약과 standalone
  Windows runner import bootstrap을 추가하고 commit `cbeac5d`의 clean
  checkout에서 최종 suite를 새 출력 디렉터리로 다시 실행했습니다.
- 최종 입력 계약 SHA-256은
  `9d85fe106bf54a658b517ad9d5c5eba145d010a0a0cadef2a26d9bb614f06ea1`입니다.
- global OCR exact cache 최종 결과는 disabled cold 중앙값 19.873초,
  enabled-empty 중앙값 19.763초, miss overhead -0.554%였습니다.
  all-hit는 0.110초로 99.445% 단축됐고 raw OCR exact, Paddle runtime
  시작 0, 논리/HTTP 요청 0을 모두 통과했습니다. cold variance도
  disabled/enabled 2.561%/2.253%였습니다.
- project checkpoint 최종 결과는 exact output, 모든 stage hit,
  inference/runtime/HTTP 0, render-only 복원, 단일 페이지 downstream
  재계산과 all-hit 92.106% 단축을 모두 통과했습니다. 그러나 disabled
  cold 중앙값 122.272초 대비 enabled-empty 128.974초로 miss overhead가
  5.481%여서 3% 승격 gate를 통과하지 못했습니다. enabled cold variance
  7.941%와 runtime wait·번역·인페인트 변동이 관찰됐지만 정식 결과를
  자동 재실행으로 덮지 않습니다.
- 따라서 global OCR exact cache는 승격 근거를 확보했고, project
  checkpoint 기능은 보존하되 기본 OFF를 유지합니다.
- Gemma telemetry에서 prefill이 번역 요청 시간의 약 30%를 차지해
  conditional spike 조건을 충족했습니다. 제품 runtime의 명시적
  batch/ubatch identity hook을 사용해 logical batch를 먼저 선별하고,
  통과한 경우에만 physical ubatch를 이어서 비교하는 제한 matrix를
  추가했습니다.
- clean commit `956dc64`에서 IQ4_NL·single·chunk6·no-spec·F16을
  고정하고 logical batch `512/1024/2048/4096`를 54블록·3회 교차
  실행했습니다. 모든 후보가 54/54 구조, runtime identity, severe
  telemetry 0 계약을 통과했습니다.
- `batch=1024`가 60.573초로 `batch=2048` 기준 63.674초보다 가장
  빨랐지만 개선율은 4.871%로 고정된 5% gate에 0.129%p 못 미쳤습니다.
  `batch=4096`은 3.057%, `batch=512`는 0.826% 개선에 그쳤습니다.
  기준선 분산도 13.439%였으므로 결과를 상향 해석하거나 자동
  재실행하지 않았습니다.
- 따라서 batch 후보는 전부 탈락했고 제품 기본 2048/512를 유지합니다.
  통과한 batch 값을 누적해야만 실행할 수 있는 ubatch 축과 54블록 의미
  검수, 292행 blind 검수는 정상 gate 종료로 취소했습니다. raw 결과와
  판정문은 Git 밖 validation log에만 보존합니다.
- RT-DETR-v2 ONNX page microbatch 2/4를 실제 제품 640 preprocess,
  threshold, block 생성, render-area 계산, 일본어 RTL 정렬까지 포함해
  6페이지·3회 순서 교차로 비교했습니다. batch2는 감지 중앙값이
  1.807% 개선돼 5% gate에 미달했고 canonical boxes·class·order가
  1/6페이지에서 달라졌습니다. batch4는 5.047% 느렸고 2/6페이지가
  달라졌습니다. 두 후보 모두 탈락하고 단건 ONNX 감지를 유지합니다.
- LaMa Large 512px·CUDA13·bf16 인페인트 조건부 spike에서는
  channels-last가 2.232% 개선에 그쳤고 SSIM 0.999744로 품질 gate
  0.9999를 통과하지 못했습니다. 기본 `torch.compile`은 working
  Triton 부재와 complex FFT 비지원으로 실패했고, cudagraphs는 고정
  shape에서는 빨랐지만 실제 가변 crop shape 전환에서 실패했습니다.
- 같은 448×448 입력의 microbatch 2/4는 각각 40.157%/44.051%
  빨랐지만 최소 SSIM이 0.999596/0.999565로 필수 품질 gate를
  위반했습니다. 실제 blockwise 경로는 앞 블록 결과가 뒤의 겹치는
  블록 입력에 반영되는 순차 의미도 있으므로 제품 scheduler를 구현하지
  않고 현행 sequential eager NCHW 경로를 유지합니다.
- crop 준비·JPEG encode·base64·응답 parse·guard가 OCR stage에서
  차지한 비중은 약 0.25%로, 중복 page I/O/hash/encode 제거 후보를
  여는 조건에 미달했습니다. Paddle wrapper overhead도 OCR의 10%에
  미달해 direct vLLM endpoint를 검토하지 않았고, encode/base64는
  약 0.05%로 JPEG quality 95/90/85 비교 조건 5%에 미달했습니다.
- HTTP Session, Paddle workers·max-num-seqs·batched-tokens·completion
  tokens·vLLM cache, IQ4_XS, single chunk 2~12, `np=2`, Gemma
  batch, detection microbatch, 인페인트 후보까지 모든 cold-path
  후보가 사전 속도·구조·품질·호환성 gate 중 하나에서 탈락했습니다.
  누적 예상 전체 3% 자격 후보가 없으므로 최종 Gemma 292행 blind와
  22페이지 cold AB/BA는 실행하지 않습니다. 이는 미완료가 아니라
  사전 계약에 따른 정상 gate 종료입니다.
- 제품 cold 기준선은 `IQ4_NL + contextual-single + chunk 6 +
  no-spec + F16`, Gemma batch/ubatch `2048/512`, Paddle workers 8,
  max-num-seqs 32, max-num-batched-tokens 98,304, OCR completion
  tokens 1024, prefix cache ON, multimodal cache default로 유지합니다.
  이번 최적화에서 기본 승격하는 성능 경로는 검증을 통과한 global
  exact OCR result cache뿐이며 project checkpoint는 기능을 보존하되
  miss-overhead gate 실패로 기본 OFF를 유지합니다.

이 문서는 도구 구현 이력과 검증된 cache gate 결과입니다. raw 입력,
OCR/번역 결과, DB와 sidecar는 Git 밖에만 보존합니다. cold 후보의 제품
기본값은 별도 family 선별과 품질 게이트를 통과하기 전에는 변경하지
않습니다.

## protocol v2 순이득 재검증

- 역사적 `miss overhead 3% 이하`와 `all-hit 50% 이상` 하드 게이트를
  제거했습니다. 기존 결과와 판정 기록은 그대로 보존합니다.
- 새 판정은 cache-disabled 두 번과 `cache cold + all-hit`의 누적시간을
  비교합니다.
- project checkpoint는 `cache cold + 한 페이지 수정` 누적시간도
  cache-disabled 두 번보다 빨라야 합니다.
- 최소 개선율은 없고, exact output·runtime/HTTP zero·페이지 단위
  downstream 무효화 계약은 그대로 유지합니다.
- 한 페이지 수정의 순이득을 측정할 수 있도록 project 검증은 최소
  2페이지를 요구합니다.

## 2026-08-01 Paddle llama.cpp completion 768 재검증

- 일본어·중국어·영어 corpus를 각각 3회, 총 9회 paired CUDA 실행했습니다.
- 모든 실행에서 페이지·블록·OCR quality·normalized raw OCR가 1024 기준선과 완전히 같았습니다.
- 전체 elapsed 평균 개선은 +0.057%였지만 단측 95% bootstrap 하한은 -1.674%였습니다.
- pipeline run wall 평균 개선은 +0.857%, 하한은 -0.150%였습니다.
- OCR request 합계 평균 개선은 +0.499%, 하한은 -0.520%였습니다.
- 과거 작은 표본의 +4.178%는 재현되지 않았습니다.
- 최소 개선율 없이 판정했지만 실제 양수 이득을 입증하지 못해 `reject_no_proven_speed_gain`으로 확정했습니다.
- 제품 기본값은 `max_new_tokens=1024`를 유지합니다.

상세 근거: [PaddleOCR-VL llama.cpp completion token 최종 판정](./paddle-token-llamacpp-final-decision-ko.md)
