# cold/cache 최종화 실행 흐름

## 1. protocol 고정 확인

`describe`는 Docker나 모델을 실행하지 않습니다. frozen baseline,
후보군, 제한과 게이트를 검증해 외부 디렉터리에 기록합니다.

## 2. 짧은 cold-path 선별

한 번에 한 family 또는 generated axis만 실행합니다. baseline과 후보를
순서를 바꿔 3회 실행하고, stage median 5% 또는 예상 전체 1% 이상인
후보만 다음 축의 base preset으로 넘깁니다.

순서는 다음을 권장합니다.

1. HTTP session/pool
2. Paddle workers
3. Paddle `max_num_seqs`
4. Paddle `max_num_batched_tokens`
5. Paddle token limit
6. vLLM prefix/multimodal processor cache
7. 측정상 조건을 충족한 conditional spike

출력이 바뀌는 후보는 속도와 무관하게 탈락합니다. direct vLLM, JPEG,
prefill batch, detector/inpainter microbatch는 protocol에 적힌 병목
비율을 실측한 경우에만 엽니다.

첫 full HTTP family의 완료된 `suite_state.json`은 이후 OCR stage
family에 `--full-reference-suite`로 전달합니다. 그러면 stage 5% 또는
실측 stage share로 환산한 예상 전체 1% 중 하나를 통과한 후보만 남습니다.
참조 없이 실행하면 보수적으로 stage 5% 게이트만 사용합니다.

## 3. 번역 선별

`run-translation`은 54블록 multilingual snapshot을 사용합니다.
IQ4_NL/IQ4_XS, single chunk, `np=2`를 각각 한 변수씩 비교합니다.
구조 오류가 없어도 모델 또는 runtime 후보는 `requires-292-blind`로
남습니다. 짧은 선별 원문과 모든 라운드 출력은 Git 밖의
`private/translation-review.json`에 합쳐져 선별 검수를 먼저 할 수
있지만, 이 파일이 최종 292행 blind 검수를 대체하지는 않습니다.

prefill이 실측 병목이면 `gemma-batch`의 `batch_size` 축을 먼저
실행합니다. 그 축에서 통과 후보가 있을 때만 해당 값을
`--base-batch-size`로 넘겨 `ubatch_size` 축을 실행합니다. 첫 축이
게이트를 통과하지 못하면 ubatch 축은 실행하지 않는 것이 정상 종료입니다.

## 4. cache 통합 검증

`run-cache --scenario global-ocr`은 cache-disabled/empty-cache 비채점
안정화 한 쌍, 측정 3회, all-hit를 실행합니다.

`run-cache --scenario project`는 disabled/enabled 비채점 안정화 한 쌍,
측정 cold 3회, 기존 output all-hit, output materialization, 한 페이지
OCR downstream 재계산을 실행합니다.

두 cache 시나리오는 다음을 모두 만족해야 합니다.

- exact output
- inference/runtime/HTTP zero 계약
- 페이지 단위 downstream 무효화
- `cache cold + all-hit` 누적시간이 cache-disabled 두 번보다 짧음
- project는 `cache cold + 한 페이지 수정` 누적시간도
  cache-disabled 두 번보다 짧음

최소 개선율은 두지 않습니다. miss overhead, all-hit 단축률, cache cold
3회의 5% variance는 보고서에 diagnostic으로 남기되 승격 gate에는
포함하지 않습니다. project 부분 수정은 최소 2페이지에서 검증합니다.
cold 후보와 최종 22페이지 비교에서는 기존 5% variance gate를 계속
적용합니다.

## 5. 승격

짧은 후보를 누적했을 때 예상 전체 3% 이상인 조합만 22페이지 검증 자격을
얻습니다. 22페이지 cold 비교는 baseline과 winner를 AB/BA 순서로
실행하고 전체 중앙값 10% 이상일 때만 제품 기본값을 바꿉니다.

캐시는 cold 후보와 독립적으로 승격할 수 있습니다. global OCR exact
cache는 최종 gate를 통과했습니다. project checkpoint의 기존 판정은
miss overhead 기준을 사용한 역사적 결과이며, protocol v2의 두 순이득
시나리오를 다시 통과한 뒤 기본값을 결정합니다.
