# PaddleOCR VL llama.cpp Docker Bundle

이 폴더는 앱이 관리하는 PaddleOCR-VL 1.6 Docker 런타임의 기준
번들입니다. 앱이 crop을 공식 `OCR:` 계약으로 고정된 llama.cpp 서버에
직접 전송합니다. PaddleX 중계 프런트와 vLLM은 실행하지 않습니다.

## 기준 파일

- `docker-compose.yaml`

`paddleocr-llamacpp`은 OpenAI 호환 API로
`PaddleOCR-VL-1.6-0.9B` GGUF와 vision projector를 실행합니다. 관리형
endpoint는 `http://127.0.0.1:18000/v1/chat/completions`입니다.

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
ready manifest, 파일 크기, 이미지 ID, Compose·command·direct transport
fingerprint만 빠르게 검사합니다.

## 런타임

- llama.cpp: `ghcr.io/ggml-org/llama.cpp:server-cuda13` (`:server-cuda`도 지원)
- direct API: llama.cpp `/v1/chat/completions`, image-first PNG + `OCR:`
- context: 4096
- parallel slots: 1
- KV type: llama.cpp default F16
- model volume: read-only external named volume

## 시작·중지와 저VRAM 재사용

- fingerprint가 정확히 같은 stopped 컨테이너만 `docker start`로
  재사용합니다.
- Windows 앱에서는 WSL Compose가 만든 컨테이너를 재사용하지 않습니다.
  Docker Desktop이 Compose Stop에서 `wsl`을 다시 호출하지 않도록
  Windows 경로·Windows Compose 메타데이터로 한 번 재생성합니다.
- 이미지·command·transport·volume·manifest 중 하나라도 다르면
  `docker compose up -d --force-recreate`를 사용합니다.
- OCR stage가 끝나면 llama.cpp의 `--sleep-idle-seconds 5`가 model과
  projector를 unload한 것을 확인한 뒤 컨테이너는 유지합니다.
- unload 확인에 실패하면 다음 GPU stage 전에 컨테이너를 정상
  `docker compose stop`합니다.
- 앱 종료 시에도 `stop`을 사용합니다. `down`은 자동 경로에서 사용하지
  않습니다.

sleep 상태는 Docker 기동을 보존하지만 model wake 시 GGUF load·GPU
offload는 다시 수행합니다. named volume은 파일 영속성과
읽기 경로를 제공할 뿐 RAM/VRAM 상주를 뜻하지 않습니다.

## 참고용 스냅샷

- `modules/ocr/ocr_paddle_VL.py` (현재 `paddle_crop/engine.py`의 호환 alias)
- `ocr_paddleocr_vl_15_hf_personal.py`
- `ocr_paddleocr_vl_hf.py`

이 파일들은 과거 운영·실험 코드 참고용이며 앱의 기준 런타임으로 import하지
않습니다. 벤치마크 preset과 raw 결과도 이 제품 폴더가 아니라
`benchmarking/lab` 및 Git 밖 validation log에 둡니다.
