# 관리형 로컬 추론 llama.cpp 전용 정책

Comic Translate가 직접 시작하는 관리형 로컬 추론은 모두 llama.cpp만 사용한다.

| 경로 | 관리형 runtime |
|---|---|
| Gemma 번역 | llama.cpp |
| HunyuanOCR | llama.cpp |
| MangaLMM full-page | llama.cpp |
| PaddleOCR-VL crop 인식 | llama.cpp direct `OCR:` |
| PaddleOCR-VL full-page Spotting | llama.cpp |

MangaLMM full-page는 기본 OCR이 아니다. 사용자 화면에서는
`MangaLMM(실험용, 느림)`으로 표시해 비교·연구용 선택지임을 분명히 한다.

사용자가 URL을 직접 바꾼 custom/unmanaged endpoint는 이 정책의 대상이 아니다.
앱은 해당 URL을 그대로 사용하며 Docker runtime을 시작하거나 backend를 추정하지
않는다.

## Paddle crop direct 계약

관리형 기본 endpoint는 `http://127.0.0.1:18000/v1/chat/completions`이다.
제품 JPEG crop을 다시 PNG로 인코딩하고 image-first content 뒤에 공식 `OCR:`
프롬프트를 보낸다. 이 계약은 과거 PaddleX relay와 일본어·영어·중국어
362/362 block 결과가
동일한 검증 결과를 기준으로 고정됐다. 사용자가 직접 지정한 과거
`/layout-parsing` endpoint는 unmanaged 호환 경로로만 남는다.

## Router pair와 기본 경로

관리형 기본 구성(Gemma 번역기 + 제품 기본 endpoint·모델)에서는 Compose project
`comic-translate-llama-router-v2`의 Router pair가 기본 경로다. pair는 상호
배타적이며 한 번에 하나만 뜬다.

| OCR 선택 | Router 컨테이너 | 호스트 포트 |
|---|---|---|
| PaddleOCR VL (최적값의 일본어·한국어·영어) | `comic-translate-router-crop-v2` | 18000, 18080 |
| PaddleOCR VL Spotting | `comic-translate-router-spotting-v2` | 18002, 18080 |
| HunyuanOCR (최적값의 중국어) | `comic-translate-router-hunyuan-v2` | 28080, 18080 |
| MangaLMM(실험용, 느림) | `comic-translate-router-mangalmm-v2` | 28081, 18080 |

Router는 요청의 `model` 필드로만 대상 모델을 고른다. 따라서 OCR 엔진이 추론
요청에 넣는 모델명, pair의 `ocr_alias`, preset 섹션명이 **완전히 같아야** 한다.
어긋나면 컨테이너와 모델은 정상 적재되지만 추론이 400으로 거부되거나 GPU 귀속
증거를 찾지 못해 기동이 실패한다. `tests/test_router_model_identity_alignment.py`가
세 이름을 계약으로 고정한다.

Router는 모델을 fingerprint된 정적 preset(`router-models.ini`)으로 구성하고
Compose 환경변수를 쓰지 않는다. 따라서 separate-server 경로가 노출하는 조정
가능한 런타임 옵션이 기본값에서 벗어나면 그 pair는 Router를 쓰지 않고
separate-server 경로로 남는다. 조정값이 조용히 무시되면 기동 비용이 아니라 OCR
동작 자체가 달라지기 때문이다.

## Router와 separate-server 호스트 포트 인계

Router 컨테이너는 OCR 포트와 Gemma 포트 `18080`을 함께 publish하므로
separate-server 컨테이너와 정확히 같은 호스트 포트를 쓴다.

| 컨테이너 | 호스트 포트 |
|---|---|
| `comic-translate-router-crop-v2` | 18000, 18080 |
| `comic-translate-router-spotting-v2` | 18002, 18080 |
| `comic-translate-router-hunyuan-v2` | 28080, 18080 |
| `comic-translate-router-mangalmm-v2` | 28081, 18080 |
| `paddleocr-llamacpp` | 18000 |
| `paddleocr-spotting-llamacpp` | 18002 |
| `hunyuanocr-local-server` | 28080 |
| `mangalmm-local-server` | 28081 |
| `gemma-local-server` | 18080 |

따라서 두 경로는 상대 경로의 컨테이너를 서로 인계받아야 한다.

- separate-server 경로는 관리형 기본 포트일 때만, 그리고 Router 소유 label이
  붙은 컨테이너만 정지해 포트를 회수한다. custom 포트와 외부 프로세스는
  건드리지 않는다.
- Router 경로는 자신이 바인딩할 포트를 쥔 이 제품의 separate-server OCR·Gemma
  컨테이너를 먼저 정지한다. 제품 소유가 아닌 listener는 계속 adapter의 명시적
  ownership 오류로 남는다.
- 회수는 이전 프로세스가 남긴 컨테이너까지 대상으로 한다. in-process Router
  상태만 보고 판단하면 앱을 다시 켠 뒤 컨테이너가 계속 생성·실패를 반복한다.
- 코디네이터가 컨테이너를 소유한 동안의 포트 회수는 state machine을 깨뜨리므로
  거부한다.

## 강제 장치

- 과거 vLLM backend QSettings 값은 version 1 마이그레이션에서 한 번만
  `llama.cpp`로 바꾼다. endpoint, token, sampler, timeout, logging은 건드리지
  않는다.
- 과거 vLLM 환경변수는 관리형 Compose에 전달하지 않고 key 이름만 warning으로
  기록한다.
- `scripts/verify_managed_llamacpp_runtime.py`는 활성 Compose command와 Paddle
  direct port 설정을 검사한다. `--live`를 주면 현재 실행 중인 관리형 컨테이너의
  process tree도 검사한다.
- 기존 `paddleocr-vllm`·`paddleocr-server` 컨테이너는 먼저 dry-run으로 소유권을 확인하고,
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

삭제 도구는 `paddleocr-vllm`과 `paddleocr-server`의 immutable ID와 제품
label을 먼저 확인한다. 두 컨테이너가 제거되고 다른 컨테이너 참조가 없을
때만 retired vendor image를 삭제하며 광범위 prune은 실행하지 않는다.

## 최종 검증 기록

관리형 runtime 전환의 현재 적용 범위, 측정 결과와 보존·퇴역 기준은
[Optimal++ v1.3.0 런타임·자산 판정](../performance-and-bugfix-audits/optimal-plus-v1.3.0/06-runtime-assets-and-llamacpp-retirement-ko.md)에
정리한다. raw container 정보와 실행 로그는 추적하지 않는다.
