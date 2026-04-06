# PaddleOCR VL Docker Bundle

이 폴더는 Comic Translate에서 `PaddleOCR VL`을 **direct `/v1` OCR 서비스**로 실행하기 위한 공식 런타임 번들입니다.

## 제품 기본 런타임

- Docker image: `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-offline`
- Start command:
  - `paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8118 --backend vllm`
- Client endpoint:
  - `http://127.0.0.1:8118/v1`

이 프로젝트는 이미 `RT-DETR-v2`로 텍스트 블록을 검출한 뒤 crop OCR만 수행하므로, `paddlex serve + /layout-parsing` 기반의 page-level pipeline 대신 공식 `genai_server` 단일 서비스를 제품 기본 경로로 사용합니다.

## 포함 파일

- `docker-compose.yaml`
  - 공식 `genai_server`를 1컨테이너로 실행하는 기준 compose
- `vllm_config.yml`
  - 향후 서버-side 튜닝이 필요할 때만 선택적으로 mount하는 예시 설정
- `pipeline_conf.yaml`
  - 기존 `/layout-parsing` baseline 비교용 레거시 참조 파일

## 사용 방법

```bash
cd paddleocr_vl_docker_files
docker compose up -d
```

이 compose는 공식 `docker run --network host ...` 경로를 그대로 감싼 형태라서, 별도 포트 포워딩 없이 호스트의 `127.0.0.1:8118`로 직접 접근합니다.

정상 기동 후 모델 목록은 아래에서 확인할 수 있습니다.

```bash
curl http://127.0.0.1:8118/v1/models
```

앱 설정에서는 `PaddleOCR VL` 서버 URL을 `http://127.0.0.1:8118/v1`로 지정하면 됩니다.

## 서버-side 튜닝

현재 tracked compose는 공식 기본 파라미터를 그대로 사용합니다. direct `/v1` 경로의 품질 검증이 끝난 뒤에만 `vllm_config.yml`을 선택적으로 mount해서 다음 항목을 조정합니다.

- `enable_prefix_caching`
- `mm_processor_cache_gb`
- `max_num_batched_tokens`

만약 공식 기본값으로 기동할 때 `No available memory for the cache blocks` 오류가 난다면, tracked compose는 그대로 두고 로컬 검증용으로만 `--backend_config /tmp/vllm_config.yml` 경로를 추가해 사용할 수 있습니다.

제품 기본값 승격은 30장 회귀 검증에서 품질 저하 없이 elapsed 개선이 확인될 때만 진행합니다.
