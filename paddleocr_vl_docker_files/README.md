# PaddleOCR VL Docker Bundle

이 폴더는 현재 프로젝트에서 사용 중인 PaddleOCR VL Docker 런타임 스냅샷을 저장하는 기준 번들입니다.

## 기준 파일

- `docker-compose.yaml`
- `pipeline_conf.yaml`
- `vllm_config.yml`

이 세 파일은 실제 OCR Docker 런타임을 재현하거나 벤치 preset을 만들 때 기준으로 사용합니다.

## 참고용 스냅샷 파일

- `ocr_paddle_VL.py`
- `ocr_paddleocr_vl_15_hf_personal.py`
- `ocr_paddleocr_vl_hf.py`

이 파일들은 과거 운영/실험 코드 참고용 스냅샷입니다. 앱이 직접 import해서 실행하는 기준 런타임 파일은 아닙니다.

## 현재 기준 요약

- `paddleocr-server`
  - `/layout-parsing` 프런트 서비스
  - 현재 compose 기준 `--device gpu:0`
- `paddleocr-vllm`
  - 실제 VL 모델 추론 백엔드
  - `gpu_memory_utilization: 0.84`
  - `max_model_len: 4096`
  - `max_num_seqs: 32`
  - `max_num_batched_tokens: 98304`

벤치마킹 작업에서는 이 폴더를 복사해 preset별로 임시 runtime 디렉터리를 만들고, 원본 tracked 파일은 직접 덮어쓰지 않습니다.
