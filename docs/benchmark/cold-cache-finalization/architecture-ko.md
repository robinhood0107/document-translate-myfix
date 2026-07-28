# cold/cache 최종화 벤치 아키텍처

## 목적

이 벤치 패밀리는 제품 코드를 실제 offscreen 앱 파이프라인으로 실행해
두 종류의 결정을 분리합니다.

- cold-path 후보: 현행 출력과 같으면서 측정된 속도 이득이 있는가
- persistent cache: miss 비용이 작고 exact hit에서 inference를 실제로
  생략하는가

후보 순위, 게이트, 보고서 생성은 `benchmarking/lab`에만 둡니다. 제품
코드에는 범용 telemetry와 stage/cache API만 남습니다.

## 고정 기준선

기준선은
`IQ4_NL + contextual-single + chunk 6 + no-spec + F16`입니다. 프롬프트,
sampler, JSON Schema, 안전장치는 후보가 바꾸지 않습니다. 기준 preset과
protocol 파일의 SHA-256을 매 실행에 기록하고 계약 drift가 있으면 실행을
거부합니다. staged Gemma Compose에도 `--spec-type none`을 직접 기록하므로
호스트의 `LLAMA_SPEC_TYPE` 환경값이 cold 결과를 바꿀 수 없습니다.
공식 run은 clean Git checkout만 허용하고 runner·pipeline runner·공통
runtime helper·protocol을 묶은 code-contract SHA-256도 기록합니다.

## 실행 경계

- 입력은 최대 6페이지 또는 54블록입니다.
- 번역은 일본어·중국어·영어 각 18블록을 사용합니다.
- cold 후보는 캐시와 DB를 실행별로 격리합니다.
- 후보 순서를 바꿔 3회 실행합니다.
- 컨테이너는 실행 사이 `docker stop`만 사용합니다.
- 후보별 Compose/config는 외부 run 폴더에 staging하지만 Docker 시작은
  벤치 wrapper가 아니라 제품 runtime manager가 결정합니다.
- staged Compose의 project 이름은 제품과 같은 `comic-translate`,
  `paddleocr_vl_docker_files`로 고정해 외부 run 경로가 달라도 기존
  stopped 컨테이너를 정확히 인수합니다.
- stopped 컨테이너는 제품 fingerprint가 같을 때만 `start`, 다르면
  `--force-recreate`됩니다.
- `docker down`, 광범위 삭제, worktree는 사용하지 않습니다.

stage 5% 미만 후보를 예상 전체 1% 기준으로 판단할 때는 임의 비율을
사용하지 않습니다. 먼저 동일 commit·입력·baseline으로 완료한 full
pipeline suite를 `--full-reference-suite`로 제공하고, 그 결과의 실제
full wall time을 분모로 사용합니다. stage-only screening run에서 줄어든
wall time을 이 full 기준선에 대입해 예상 전체 개선율을 계산하므로 runtime
시작 차이도 포함됩니다. 참조 계약이 다르면 후보 실행 전에 거부합니다.

## 데이터와 개인정보

raw 이미지, OCR, 번역, stdout/stderr, SQLite, 프로젝트 sidecar와 로컬
경로는 Git 저장소 밖의 새 output 디렉터리에만 기록합니다. 추적 가능한
요약에는 원문 대신 canonical SHA-256과 카운트만 사용합니다. 같은
corpus를 재사용할 때는 파일 크기와 SHA-256이 모두 같아야 합니다.
의미 검수가 필요한 family는 외부 `private/pipeline-review.json` 또는
`private/translation-review.json`에 원문과 모든 라운드 출력을
page/block 또는 language/index 순서로 모읍니다.

## cache 검증 모델

global OCR cache는 cache-disabled cold와 enabled-empty cold를 각각 3회
교차 실행합니다. miss overhead 중앙값은 3% 이하여야 합니다. all-hit는
Paddle runtime start와 OCR HTTP가 모두 0이어야 하며 raw OCR 계약이
cold와 완전히 같아야 합니다. runtime은 사전 부팅하지 않으므로 이 0은
제품 경로가 실제로 시작을 생략했다는 뜻입니다.

project checkpoint도 disabled cold와 enabled-empty cold를 각각 3회
비교합니다. 이후 아래를 실제 프로젝트로 확인합니다.

1. 기존 render output이 있는 all-hit
2. output만 삭제한 뒤 render artifact materialization
3. 한 페이지의 OCR checkpoint와 downstream만 무효화

temperature가 0이 아닌 번역을 강제 재계산하면 같은 입력도 문구가 달라질
수 있으므로 서로 독립적인 cold run의 최종 PNG SHA를 동일성 근거로
사용하지 않습니다. checkpoint seed와 기존-output/missing-output all-hit는
render SHA가 완전히 같아야 하고, 부분 무효화에서는 미변경 페이지만 SHA가
같아야 합니다. 모든 cold·hit·부분 실행의 detection/OCR 계약은 별도로
완전 동일해야 합니다.

감지·OCR·번역·인페인트·렌더 hit 여부는 private `metrics.jsonl` 이벤트를
페이지 인덱스에 매핑해 판정합니다. `stage_status`만으로 추정하지 않습니다.

## 자동 승격 금지

exact-output 후보만 속도 게이트를 통과할 수 있습니다. 번역 모델·문맥·병렬
후보는 짧은 선별을 통과해도 자동 승격되지 않으며, 별도의 292행 blind
의미 검수와 사용자 승인이 필요합니다.

`np=2` 후보는 총 context를 8192로 고정해 slot당 4096을 보장합니다.
concurrency가 slot 수를 넘거나 slot당 context가 4096 미만인 protocol은
실행 전에 거부합니다.
