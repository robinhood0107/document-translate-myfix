# 자동번역 GPU 자원 전략

기준 날짜: `2026-04-05`

이 문서는 현재 프로젝트의 자동번역 파이프라인이 GPU 자원을 어떻게 쓰는지, 그리고 `Gemma`와 `PaddleOCR VL`을 어떤 전략으로 공존시키며 최적화할지 정리한 문서입니다.

## 1. 현재 파이프라인 흐름

일반 배치는 이미지마다 아래 순서로 순차 처리됩니다.

1. `detect`
2. `ocr`
3. `inpaint`
4. `translate`
5. `render/save`

핵심은 `PaddleOCR VL`과 `Gemma`가 같은 페이지에서 동시에 추론하지 않는다는 점입니다.

- `PaddleOCRVLEngine`은 OCR 단계 안에서만 블록 단위 병렬 요청을 보냅니다.
- `CustomLocalGemmaTranslation`은 번역 단계 안에서만 청크 단위 순차 요청을 보냅니다.
- 앱 내부 `RT-DETR` / `inpainter` ONNX 세션은 단계별로 GPU를 사용하지만, 페이지 기준 전체 흐름은 직렬에 가깝습니다.

즉 병목은 `동시 추론`보다 `동시에 VRAM에 상주하는 프로세스와 캐시`입니다.

## 2. 현재 상주 구성

현재 기준 서비스는 아래 3개입니다.

- `gemma-local-server`
- `paddleocr-server`
- `paddleocr-vllm`

여기에 앱 내부 GPU 사용자도 있습니다.

- RT-DETR-v2 detection
- AOT / LaMa / MI-GAN inpainter
- ONNX/Torch 런타임 캐시

## 3. 기준선 두 종류

이 작업에서는 기준선을 둘로 나눠 기록합니다.

### merged baseline

이미 `develop`에 머지된 기준값입니다.

- Gemma compose 기본 `n_gpu_layers=8`
- 앱 기본 PaddleOCR VL 값
  - `parallel_workers=2`
  - `max_new_tokens=256`

### live-ops baseline

현재 로컬 운영/실험 기준값입니다.

- Gemma compose 현재 로컬 값 `n_gpu_layers=20`
- OCR runtime 기준 번들: [paddleocr_vl_docker_files/README.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/paddleocr_vl_docker_files/README.md)
- OCR client ops snapshot 값
  - `parallel_workers=8`
  - `max_new_tokens=1024`

## 4. 이번 v1 전략

이번 단계의 기본 전략은 `Warm Balanced`입니다.

- `Gemma`, `paddleocr-server`, `paddleocr-vllm`는 모두 warm 상태 유지
- 페이지마다 컨테이너를 stop/start 하지 않음
- 목표는 `즉시 응답 가능`과 `전체 페이지 처리시간 최소`를 동시에 만족하는 것

이 전략을 택한 이유는 다음과 같습니다.

- 자동번역은 이미지마다 `추론 -> 대기 -> 추론 -> 대기` 흐름이 반복됩니다.
- per-page 재기동 전략은 cold start 비용과 실패 표면적이 큽니다.
- 사용자는 요청 즉시 반응하는 warm runtime을 원합니다.

## 5. 최적화 우선순위

튜닝 순서는 아래 고정 순서로 진행합니다.

1. `paddleocr-server` 프런트 서비스를 `gpu:0`에서 `cpu`로 옮겨도 성능 손실이 없는지 먼저 검증
2. 확보된 여유 VRAM을 `Gemma n_gpu_layers` 상향에 우선 배분
3. 그 다음 `PaddleOCR VL parallel_workers`와 `max_new_tokens`를 조정
4. 마지막 단계에서만 `paddleocr-vllm` backend 값을 조정

## 6. 현재 탐색 후보군

### Gemma

- `n_gpu_layers = 16, 18, 20, 22, 24`

### OCR client

- `parallel_workers = 2, 4, 6, 8`
- `max_new_tokens = 256, 512, 768, 1024`

### OCR runtime

- `front_device = gpu:0, cpu`
- `gpu_memory_utilization = 0.80, 0.84, 0.88, 0.90`
- `max_num_seqs = 16, 32, 48`
- `max_num_batched_tokens = 49152, 98304, 131072`

## 7. 채택 기준

아래 조건을 모두 만족해야 새 조합을 baseline으로 승격합니다.

- OOM, CUDA 오류, ONNX 오류, HTTP 실패, JSON parse failure `0회`
- `free VRAM floor >= 1.5 GiB`
- OCR quality retry count가 기존 baseline보다 증가하지 않음
- 빈 번역/잘린 응답 비율 `0%`
- one-page auto 체감 latency가 baseline보다 나빠지지 않음
- representative corpus 기준 median total page time이 개선됨

## 8. 왜 `/Sample` 30장을 기준으로 하나

검증용 이미지는 로컬 `/Sample` 폴더의 `30장`을 기준 코퍼스로 사용합니다.

- smoke 코퍼스: 처음 `5장`
- representative 코퍼스: 전체 `30장`

이 폴더는 로컬 검증 전용이며 Git에 올리지 않습니다.

- `.gitignore`에 `/Sample/`가 추가됨
- 벤치 스크립트도 기본적으로 이 경로를 기준으로 사용함

## 9. 관련 문서

- [pipeline-benchmarking-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/pipeline-benchmarking-ko.md)
- [pipeline-benchmark-results-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/pipeline-benchmark-results-ko.md)
- [pipeline-benchmark-checklist-ko.md](/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/docs/pipeline-benchmark-checklist-ko.md)
