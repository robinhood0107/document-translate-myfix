# PaddleOCR-VL-1.5 속도 개선 검토 메모

## 목표

- `PaddleOCR-VL-1.5`의 인식 품질은 유지한다.
- 이 저장소의 만화 말풍선 OCR 경로에서 체감 지연과 처리량을 개선한다.
- benchmark-specific preset, runner, report는 제품 브랜치에 넣지 않고, 제품 코드/런타임에서 바로 써먹을 수 있는 개선안만 정리한다.

## 현재 저장소 기준 관찰

- 앱 OCR 엔진은 페이지 전체를 보내지 않고, 텍스트 블록별 crop 이미지를 `PaddleOCR VL` 서비스의 `/layout-parsing` 엔드포인트로 개별 전송한다.
- Docker 구성은 `paddleocr-vllm`과 `paddleocr-layout` 2단 구조다.
  - `paddleocr-vllm`: 실제 `PaddleOCR-VL-1.5-0.9B` 추론 백엔드
  - `paddleocr-layout`: PaddleX/PaddleOCR pipeline 프런트 서비스
- 현재 pipeline 설정은 이미 `use_layout_detection: false`, `use_doc_preprocessor: false`라서, 앱이 직접 crop을 만들고 있는 현재 경로에서는 문서 전체 레이아웃 분석 이점이 거의 없다.
- 현재 vLLM 설정은 `bfloat16`, `gpu_memory_utilization: 0.84`, `max_num_seqs: 32`, `max_num_batched_tokens: 98304`로 이미 꽤 공격적인 편이다.

## 공식 문서에서 확인한 핵심 포인트

- PaddleOCR 공식 문서는 production 환경에서는 기본 pipeline 추론 대신 `vLLM`, `SGLang`, `FastDeploy` 같은 가속 프레임워크 사용을 권장한다.
- PaddleOCR 공식 문서는 client 측 동시성(`vl_rec_max_concurrency`)과 service 측 동시성(`VLRecognition.genai_config.max_concurrency`)을 별도로 조정하라고 안내한다.
- PaddleOCR 공식 문서는 일반 service deployment 경로가 동시 요청 처리에 최적화된 경로가 아니며, 동시 처리 요구가 있으면 고성능 배포 경로를 보라고 명시한다.
- vLLM의 PaddleOCR-VL 레시피는 OCR 작업에서는 prefix caching, image reuse 효과가 크지 않으므로 이를 꺼서 불필요한 hashing/caching 오버헤드를 줄이라고 권장한다.
- vLLM 공식 튜닝 문서는 throughput 위주라면 `max_num_batched_tokens`를 크게 두되, preemption이 발생하면 `gpu_memory_utilization`을 올리거나 `max_num_seqs`/`max_num_batched_tokens`를 낮추라고 권장한다.
- Hugging Face 모델 카드는 `transformers` + `flash_attention_2` 경로를 소개하지만, 공식 추론 경로가 더 빠르고 page-level parsing까지 지원하므로 기본적으로는 공식 경로를 권장한다.

## 왜 현재 경로가 느릴 가능성이 큰가

### 1. 앱은 crop OCR만 필요한데, 현재는 page parser 프런트 서비스를 거친다

현재 앱은 말풍선 또는 텍스트 블록 단위로 이미지를 잘라낸 뒤 `/layout-parsing`에 보낸다. 이 경로는 원래 문서 페이지 전체를 받아서 Markdown, `prunedResult`, 시각화 결과까지 만드는 pipeline 서비스다. 하지만 앱은 최종적으로 plain text만 꺼내 쓴다.

즉, 현재 경로에는 아래 추가 비용이 있다.

- `crop -> /layout-parsing` HTTP hop
- pipeline 서비스의 request/response 가공
- Markdown / `prunedResult` 생성
- 앱 측 응답 파싱

이 단계들은 `PaddleOCR-VL-1.5` 모델 품질을 높여주기보다, 현재 사용 시나리오에서는 주로 부가 오버헤드로 작동할 가능성이 높다.

### 2. 현재 앱 병렬화와 서비스 구조가 잘 안 맞을 수 있다

앱은 `parallel_workers`로 여러 crop 요청을 병렬 전송할 수 있지만, PaddleOCR 공식 문서는 일반 service deployment 경로가 동시 요청 처리용이 아니라고 설명한다. 즉, 앱에서 worker를 올려도 pipeline 서비스에서 직렬화되거나 queue 지연이 커질 수 있다.

### 3. layout 프런트 서비스가 CPU 경로라면 작은 요청 다발에 더 불리하다

현재 저장소의 tracked compose는 `paddleocr-layout`을 `--device cpu`로 띄운다. 큰 page-level batch라면 VLM 백엔드가 지배적일 수 있지만, 현재처럼 작은 crop를 많이 보내는 구조에서는 front service의 decode/prepare/postprocess 비용이 더 눈에 띄기 쉽다.

## 품질을 유지하면서 속도를 끌어올릴 우선순위

### 1. 가장 우선: `/layout-parsing` 대신 `vLLM` 직접 호출 경로를 추가

가장 큰 기대 효과가 있는 방향이다.

- 모델은 그대로 `PaddleOCR-VL-1.5`를 쓴다.
- 입력도 같은 crop 이미지를 쓴다.
- task prompt는 공식 예시와 동일하게 `OCR:`를 사용한다.
- 즉, 바꾸는 것은 모델이 아니라 "중간 파이프라인 경로"다.

이 방식이면 아래를 제거할 수 있다.

- PaddleX pipeline 프런트 서비스 hop
- Markdown / `prunedResult` 생성 비용
- `/layout-parsing` 전용 응답 파싱 비용

권장 형태:

- 앱에 `PaddleOCR VL`용 direct-vLLM 모드를 추가한다.
- 기본 엔드포인트를 `http://127.0.0.1:18000/v1`로 두고 OpenAI-compatible chat completions를 사용한다.
- 요청 본문은 공식 vLLM recipe와 동일하게 `image_url` + `text: "OCR:"` 구조를 따른다.
- 기존 `/layout-parsing` 경로는 fallback/debug 용으로 남긴다.

이 방향은 현재 저장소 시나리오에서 품질 저하 위험이 가장 낮고, 구조상 가장 큰 속도 이득을 기대할 수 있다.

### 2. vLLM 서버를 직접 쓰는 경우, OCR용으로 caching 성격을 다시 맞춘다

vLLM의 PaddleOCR-VL 레시피는 OCR 작업에서는 prefix caching과 image reuse 이점이 크지 않다고 명시한다. 따라서 direct vLLM 경로를 도입하면 아래 옵션을 우선 검토한다.

- `--no-enable-prefix-caching`
- `--mm-processor-cache-gb 0`

현재 tracked compose는 `paddleocr genai_server` 래퍼를 사용하므로, 이 래퍼가 해당 옵션을 backend config로 얼마나 직접 노출하는지 확인이 필요하다. 만약 노출이 애매하면, direct vLLM 경로 검증용 브랜치에서는 `vllm serve` 기반 런타임을 별도 실험하는 편이 더 명확하다.

### 3. 현재 vLLM 설정은 "더 키우기"보다 "안정적인 고처리량 sweet spot 찾기"가 중요하다

현재 값:

- `gpu_memory_utilization: 0.84`
- `max_num_seqs: 32`
- `max_num_batched_tokens: 98304`

이미 꽤 높은 편이라, 무조건 더 올리는 것이 정답은 아니다. vLLM 공식 문서 기준으로는 다음 원칙이 맞다.

- throughput이 낮으면 `max_num_batched_tokens`를 충분히 크게 둔다.
- preemption이 보이면 `gpu_memory_utilization`을 올리거나 `max_num_seqs`/`max_num_batched_tokens`를 낮춘다.
- 작은 모델 + 큰 GPU에서는 큰 `max_num_batched_tokens`가 유리할 수 있지만, 12 GB급 카드에서는 preemption/latency spike를 같이 봐야 한다.

이 저장소 기준 권장 순서:

1. direct-vLLM 경로를 먼저 만든다.
2. 그 다음 아래 조합을 짧게 sweep한다.
3. 평균 latency만 보지 말고 p95/p99와 preemption 로그를 같이 본다.

후보 예시:

- `gpu_memory_utilization`: `0.84 -> 0.88 -> 0.90`
- `max_num_seqs`: `16 / 24 / 32`
- `max_num_batched_tokens`: `32768 / 65536 / 98304`

중요한 점은 "숫자를 크게 만들기"가 아니라 "preemption 없이 가장 높은 steady throughput"을 찾는 것이다.

### 4. 앱 쪽 worker 수는 서버 구조를 바꾼 뒤 다시 맞춘다

현재 앱 기본값은 `parallel_workers=2`다. direct-vLLM 경로로 바꾸면 worker 증설이 실제 처리량 증가로 이어질 가능성이 커진다. 반대로 `/layout-parsing` 경로를 계속 쓰면 worker를 늘려도 서비스 쪽 queue만 커질 수 있다.

따라서 권장 순서는 다음과 같다.

1. direct-vLLM 경로 도입
2. worker `2 -> 3 -> 4`만 짧게 확인
3. latency tail이 튀면 다시 내린다

### 5. 앱 내부에서 가능한 저위험 미세 최적화

아래는 큰 구조 변경 다음 단계에서 볼 만한 항목이다.

- `requests.post(...)` 반복 호출 대신 `requests.Session()` 재사용
- crop별 payload 크기, 응답 시간, 빈 응답률을 로그로 남겨 병목 구간 가시화
- `max_new_tokens`를 전역 고정값으로 줄이기보다, 정말 짧은 bubble에서만 보수적으로 낮추는 adaptive 정책 검토

다만 이 항목들은 주효한 1차 해법이라기보다, direct-vLLM 전환 후에 붙는 2차 최적화에 가깝다.

## 현재 단계에서 굳이 건드리지 않는 것이 좋은 것

품질 유지가 우선이라면 아래는 1차 최적화 대상에서 제외하는 편이 안전하다.

- 양자화(INT4/GGUF/저정밀 가중치) 기본 적용
- `PaddleOCR-VL-1.5`보다 작은 다른 모델로 교체
- 입력 해상도를 공격적으로 낮추는 downscale
- `max_new_tokens`를 전역적으로 크게 축소
- spotting/seal/formula/table prompt를 OCR prompt로 임의 대체

이들은 속도 이점이 있을 수는 있지만, 현재 요구사항인 "품질 유지" 관점에서는 회귀 위험이 상대적으로 크다.

## 이 저장소에 대한 추천 결론

가장 가능성이 높은 정답은 아래 조합이다.

1. `PaddleOCR-VL-1.5` 모델은 그대로 유지한다.
2. 앱 OCR 경로를 `/layout-parsing` 중심에서 `vLLM direct OCR` 중심으로 바꾼다.
3. vLLM에서 OCR용 cache 정책을 단순화한다.
4. 그 위에서 `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens`, `parallel_workers`를 짧게 sweep한다.
5. `/layout-parsing` 경로는 fallback/debug 전용으로 남긴다.

현재 구조를 유지한 채 숫자만 만지는 것보다, "중간 파이프라인 제거"가 훨씬 큰 효과를 낼 확률이 높다. 그리고 이 방향은 모델 자체와 prompt 의미를 바꾸지 않으므로 품질 유지 조건에도 가장 잘 맞는다.

## 참고 자료

- PaddleOCR 공식 Usage Tutorial: <https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html>
- Hugging Face 모델 카드: <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5>
- vLLM PaddleOCR-VL recipe: <https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html>
- vLLM Optimization and Tuning: <https://docs.vllm.ai/en/stable/configuration/optimization/>
