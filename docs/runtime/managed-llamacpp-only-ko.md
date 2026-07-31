# 관리형 로컬 추론 llama.cpp 전용 정책

Comic Translate가 직접 시작하는 관리형 로컬 추론은 모두 llama.cpp만 사용한다.

| 경로 | 관리형 runtime |
|---|---|
| Gemma 번역 | llama.cpp |
| HunyuanOCR | llama.cpp |
| MangaLMM full-page | llama.cpp |
| PaddleOCR-VL crop 인식 | llama.cpp + CPU-only PaddleX relay |
| PaddleOCR-VL full-page Spotting | llama.cpp |

사용자가 URL을 직접 바꾼 custom/unmanaged endpoint는 이 정책의 대상이 아니다.
앱은 해당 URL을 그대로 사용하며 Docker runtime을 시작하거나 backend를 추정하지
않는다.

## PaddleX relay 예외

`paddleocr-layout`의 고정 vendor image 이름에는 역사적으로
`paddleocr-genai-vllm-server`가 포함돼 있다. 현재 Compose는 이 이미지에서
`paddlex --serve --device cpu`만 시작하며 vLLM process를 시작하지 않는다.
실제 VLM 요청은 `pipeline_conf.yaml`의 `llama-cpp-server` backend를 통해
`paddleocr-llamacpp`으로 전달된다.

이 relay는 direct `OCR:` 경로와 동일 crop으로 품질·속도를 비교하기 전에는
제거하지 않는다. 이미지 이름만 보고 활성 vLLM으로 오판해서도 안 된다.

## 강제 장치

- 과거 vLLM backend QSettings 값은 version 1 마이그레이션에서 한 번만
  `llama.cpp`로 바꾼다. endpoint, token, sampler, timeout, logging은 건드리지
  않는다.
- 과거 vLLM 환경변수는 관리형 Compose에 전달하지 않고 key 이름만 warning으로
  기록한다.
- `scripts/verify_managed_llamacpp_runtime.py`는 활성 Compose command와 Paddle
  relay 설정을 검사한다. `--live`를 주면 현재 실행 중인 관리형 컨테이너의
  process tree도 검사한다.
- 기존 `paddleocr-vllm` 컨테이너는 먼저 dry-run으로 소유권을 확인하고,
  실제 immutable container ID와 label 값을 담은 resolved manifest를 만든다.
  실행 시 현재 ID·image·label이 resolved manifest와 모두 같을 때만 ID를
  대상으로 stop 후 제거한다.
- 광범위 image/volume prune과 `docker compose down`은 사용하지 않는다.

## 검증 명령

정적 계약 검사:

```powershell
.venv-win\Scripts\python.exe scripts\verify_managed_llamacpp_runtime.py
```

Docker Desktop이 켜진 상태의 process tree 검사:

```powershell
.venv-win\Scripts\python.exe scripts\verify_managed_llamacpp_runtime.py --live
```

구 컨테이너 삭제 전 dry run, Git 밖 resolved manifest 생성, 명시적 실행:

```powershell
.venv-win\Scripts\python.exe scripts\retire_legacy_vllm_runtime.py
.venv-win\Scripts\python.exe scripts\retire_legacy_vllm_runtime.py --snapshot-output <validation-log-root>\resolved-vllm-retirement.json
.venv-win\Scripts\python.exe scripts\retire_legacy_vllm_runtime.py --manifest <validation-log-root>\resolved-vllm-retirement.json --execute
```

삭제 도구는 현재 CPU relay도 공유하는 vendor image를 삭제하지 않는다.
