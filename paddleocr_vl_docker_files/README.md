# PaddleOCR VL llama.cpp Docker Bundle

이 폴더는 앱이 관리하는 PaddleOCR-VL 1.6 Docker 런타임의 기준
번들입니다. 문서 분석 프런트는 PaddleX를 유지하고, 실제 VL 추론
백엔드는 vLLM 대신 고정된 llama.cpp 서버를 사용합니다.

## 기준 파일

- `docker-compose.yaml`
- `pipeline_conf.yaml`

`paddleocr-layout`은 `/layout-parsing` API와 PaddleX 전처리·후처리를
담당합니다. `paddleocr-llamacpp`은 OpenAI 호환 API로
`PaddleOCR-VL-1.6-0.9B` GGUF와 vision projector를 실행합니다.

## 최초 모델 준비

아래 두 파일을 같은 폴더에 준비합니다.

- `PaddleOCR-VL-1.6-GGUF.gguf`
- `PaddleOCR-VL-1.6-GGUF-mmproj.gguf`

Windows PowerShell에서 한 번 실행합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_paddleocr_llamacpp_runtime.ps1 `
  -Mode Prepare `
  -ModelDirectory 'C:\ExampleWorkspace\models\PaddleOCR-VL-1.6-GGUF'
```

스크립트는 source와 복사본의 정확한 크기·SHA-256을 검사하고,
versioned external volume
`comic-translate-paddleocr-vl-llamacpp-models-v1`에 `.partial`로 복사한
뒤 atomic rename합니다. 실제 model/mmproj load smoke가 통과한 후에만
ready manifest를 마지막에 기록합니다.

전체 SHA-256을 다시 검사하려면 다음을 실행합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_paddleocr_llamacpp_runtime.ps1 `
  -Mode Verify
```

정상 앱 시작은 큰 파일을 다시 해시하지 않습니다. read-only volume의
ready manifest, 파일 크기, 이미지 ID, Compose·pipeline·command
fingerprint만 빠르게 검사합니다.

## 고정 런타임

- llama.cpp:
  `ghcr.io/ggml-org/llama.cpp@sha256:22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb`
- PaddleX layout:
  `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server@sha256:d0d32c04a2119613d25a0a4c292e165ccc107954b74580613cf59e378037f8f5`
- llama.cpp backend: `llama-cpp-server`
- context: 4096
- parallel slots: 1
- KV type: llama.cpp default F16
- model volume: read-only external named volume

PaddleX layout 이미지는 이름에 `vllm-server`가 남아 있지만, 이
구성에서는 vLLM 프로세스를 시작하지 않습니다. 해당 고정 이미지에 포함된
PaddleX/PaddleOCR 3.6 계열의 `llama-cpp-server` client와 serving
프런트만 사용합니다.

## 시작·중지와 저VRAM 재사용

- fingerprint가 정확히 같은 stopped 컨테이너만 `docker start`로
  재사용합니다.
- 이미지·command·pipeline·volume·manifest 중 하나라도 다르면
  `docker compose up -d --force-recreate`를 사용합니다.
- OCR stage가 끝나면 llama.cpp의 `--sleep-idle-seconds 5`가 model과
  projector를 unload한 것을 확인한 뒤 컨테이너는 유지합니다.
- unload 확인에 실패하면 다음 GPU stage 전에 두 컨테이너를 정상
  `docker compose stop`합니다.
- 앱 종료 시에도 `stop`을 사용합니다. `down`은 자동 경로에서 사용하지
  않습니다.

sleep 상태는 Docker 기동과 PaddleX 프런트를 보존하지만 model wake 시
GGUF load·GPU offload는 다시 수행합니다. named volume은 파일 영속성과
읽기 경로를 제공할 뿐 RAM/VRAM 상주를 뜻하지 않습니다.

## 참고용 스냅샷

- `ocr_paddle_VL.py`
- `ocr_paddleocr_vl_15_hf_personal.py`
- `ocr_paddleocr_vl_hf.py`

이 파일들은 과거 운영·실험 코드 참고용이며 앱의 기준 런타임으로 import하지
않습니다. 벤치마크 preset과 raw 결과도 이 제품 폴더가 아니라
`benchmarking/lab` 및 Git 밖 validation log에 둡니다.
