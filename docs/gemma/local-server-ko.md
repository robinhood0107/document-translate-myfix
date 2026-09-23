# Gemma 로컬 서버 설정 가이드

이 문서는 `Custom Local Server(Gemma)`의 제품용 Docker 런타임을 현재 저장소 기준으로 준비하고 검증하는 방법을 정리합니다.

## 한 번만 준비

Windows PowerShell에서 저장소 루트를 연 뒤 실행합니다.

기본 `Prepare` 실행은 다운로드 대상 drive에 모델 크기 + 512 MiB가 남았는지
검사합니다. `gemma-local-server` 컨테이너가 실행 중이면 앱을 정상 종료해
컨테이너를 중지한 뒤 다시 실행합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_gemma_runtime.ps1 -Mode Prepare `
  -ModelPath 'C:\ExampleWorkspace\models\gemma-4-26B-IQ4_NL.gguf'
```

준비 스크립트는 다음 순서를 지킵니다.

- source 경로는 `Prepare`에만 필요하며 준비가 끝난 뒤 앱 시작에는 필요하지 않습니다.
- 고정된 llama.cpp image digest를 확인하고, 로컬에 없을 때만 가져옵니다.
- 최종 제품 모델 IQ4_NL의 원본 SHA-256과 크기를 확인합니다.
- `comic-translate-gemma-models-v2` external volume에 파일을 `.partial`로 복사합니다.
- 복사본의 크기와 SHA-256을 확인한 뒤 같은 volume 안에서 원자적으로 이름을 바꿉니다.
- GPU에서 실제 모델 load, `/health`, `/v1/models`, chat 요청을 통과시킵니다.
- 모든 검증이 끝난 마지막 단계에서만 ready manifest를 기록합니다.

준비 또는 명시적 검증에서만 대형 GGUF 전체를 다시 해시합니다. 평상시 앱 시작은 ready manifest, 파일 크기, 설정 identity만 빠르게 확인합니다.

전체 SHA-256을 다시 확인하려면 다음을 실행합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_gemma_runtime.ps1 -Mode Verify
```

`-ModelPath`를 생략하면 저장소의 gitignore된 `testmodel/`을 먼저 찾고, 거기에도
없으면 `-AllowDownload`를 줬을 때만 등록된 Hugging Face 원본을 내려받습니다.
따라서 아무 인자 없이도 준비가 끝납니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_gemma_runtime.ps1 -Mode Auto -AllowDownload
```

업스트림이 지원 llama.cpp 태그를 갱신하면 모델은 멀쩡한데 manifest에 봉인된
image identity만 어긋납니다. 이때는 `-Mode Reseal`이 원본 없이 스모크를 다시
통과시키고 manifest만 다시 씁니다. 앱도 같은 상태를 스스로 감지해 한 번
복구합니다. 자세한 내용은
[관리형 llama.cpp 볼륨 복구 가이드](../runtime/managed-volume-repair-ko.md)를
참고하세요.

준비 스크립트의 공개 옵션은 아래와 같습니다.

- `-Mode`: `Prepare`, `Verify`, `Reseal`, `Auto`이며 기본값은 `Prepare`입니다.
  `Auto`는 볼륨 상태를 보고 `Prepare`와 `Reseal` 중 맞는 쪽을 고릅니다.
- `-ModelPath`: `Prepare`에서만 쓰는 최종 IQ4_NL GGUF 경로입니다. 비우면
  `testmodel/`과 다운로드 캐시를 차례로 찾습니다.
- `-AllowDownload`: 검증된 로컬 원본을 못 찾았을 때만 등록된 원본을 내려받습니다.
- `-DownloadDirectory`: 내려받은 원본을 둘 위치입니다. 비우면 `testmodel/`입니다.
- `-VolumeName`: 기본값은 `comic-translate-gemma-models-v2`입니다. 다른 이름을 쓰면 앱 실행 전 `GEMMA_MODEL_VOLUME`에도 같은 값을 설정해야 합니다.
- `-SmokePort`: 실제 GPU smoke 서버의 로컬 포트이며 기본값은 `18082`입니다.
- `-SmokeTimeoutSec`: smoke 준비 제한 시간이며 `30`~`900`초, 기본값은 `420`초입니다.
- `-MinimumFreeBytes`: 선택적인 추가 최소 여유 공간입니다. 기본값은 `0`이며,
  실제 다운로드 직전에 모델 파일 크기 + 512 MiB를 자동 검사합니다.
- `-SkipFreeSpaceCheck`: 자동 파일별 여유 공간 검사까지 명시적으로 건너뜁니다.

## 앱 설정

- Endpoint URL: `http://127.0.0.1:18080/v1`
- Model: 준비된 volume 안의 정확한 GGUF 파일명
- 제품 기본 모델: `gemma-4-26B-IQ4_NL.gguf`

앱은 모델 volume을 read-only로 마운트합니다. image ID, compose/command hash, volume, manifest SHA-256, model SHA-256, 준비 버전을 합친 fingerprint가 정확히 같은 중지 컨테이너만 `docker start`로 재사용합니다. 하나라도 다르면 `docker compose up -d --force-recreate`로 재생성합니다.

정상 종료는 컨테이너와 volume을 보존하는 `docker stop`만 사용합니다. 초기화나 삭제가 필요한 별도 작업에서만 `down`을 사용합니다.

## 결과 캐시와 Exact Translation Memory

`사용자 사전` 설정에는 Gemma용 두 로컬 fast path가 있습니다.

- `영구 블록 결과 캐시`: 전체 정렬 문맥, 대상 index, 언어, prompt/schema/sampler, 모델 SHA-256, runtime fingerprint 등이 모두 같은 결과만 재사용합니다.
- `정확 일치 번역 메모리`: 사용자가 승인한 원문→번역 쌍만 Gemma를 우회합니다. 승인하지 않은 항목은 후보로만 저장됩니다.

번역 전에 cache/TM hit를 먼저 판정합니다. 전체 hit이면 중지된 Gemma 컨테이너를 시작하지 않습니다. 부분 hit이면 모든 원문 marker를 문맥으로 유지하면서 누락된 `requested_blocks`만 요청합니다. 결과 사전 치환은 hit와 miss 모두에서 정확히 한 번 적용합니다.

이 기능은 앱 user-data 디렉터리의 SQLite DB에 원문과 번역문을 저장하므로 민감한 로컬 데이터로 취급해야 합니다. DB 잠금이나 손상을 감지하면 해당 실행의 cache/TM만 끄고 정상 번역을 계속하며, DB를 자동 삭제하거나 덮어쓰지 않습니다.

승인, 승인 해제, 삭제, result-cache 비우기, Exact TM JSON 가져오기·내보내기는 `사용자 사전` 화면에서 수행합니다. 가져온 승인 항목은 Gemma를 우회할 수 있으므로 신뢰하는 파일만 확인 후 가져옵니다.

세부 identity, 보존 한도, 정규화, 장애 동작은 [번역 메모리 가이드](translation-memory-ko.md)를 참고하세요.

## 명시적 host-bind rollback

versioned volume을 사용하지 않고 별도 host model directory로 수동 rollback해야 할 때만 override compose를 함께 지정합니다.

```powershell
$env:GEMMA_HOST_MODEL_DIR = 'C:\ExampleWorkspace\models'
docker compose `
  -f .\docker-compose.yaml `
  -f .\docker-compose.gemma-host-rollback.yaml `
  up -d --force-recreate
```

이 경로는 제품 기본값이 아니며, 자동 runtime fingerprint 재사용 대상도 아닙니다.

## 현재 활성 요청값

- `temperature=0.5`
- `top_k=32`
- `top_p=1.0`
- `min_p=0.0`
- `Chunk Size=6`
- `Max Completion Tokens=512`
- `Request Timeout=180`
- `response_format=json_schema`

## 현재 compose 기준값

- `LLAMA_CTX_SIZE=4096` (`1024`~`32768`)
- `LLAMA_N_PARALLEL=1` (`1`~`4`)
- `LLAMA_N_GPU_LAYERS=23` (`0`~`99`)
- `LLAMA_THREADS=10` (`1`~`64`)
- `LLAMA_BATCH_SIZE=2048` (`128`~`4096`)
- `LLAMA_UBATCH_SIZE=512` (`64`~`2048`, batch size 이하)
- `LLAMA_CACHE_TYPE_K=f16`, `LLAMA_CACHE_TYPE_V=f16` (`f16` 또는 `q8_0`)
- `LLAMA_CACHE_RAM_MIB=0` (`0` 또는 `256`)
- `LLAMA_SPEC_TYPE=none` (`none` 또는 `ngram-mod`)
- `LLAMA_SPEC_DRAFT_N_MAX=8` (`2`, `4`, `8`)
- `--fit off`
- flash attention enabled
- `--swa-full`
- reasoning disabled

환경변수로 허용하는 runtime 후보는 위 범위로 제한됩니다.
batch/ubatch 기본값은 pinned llama.cpp의 기존 암시적 기본값과 같으며,
환경변수로 값을 바꾸면 runtime fingerprint와 command identity도 달라집니다.
이 명시적 command 계약이 처음 적용될 때 기존 번역 result-cache 항목은
삭제되지 않지만 이전 runtime fingerprint라서 재사용되지 않습니다.

현재 llama.cpp 런타임 이미지에서는 `cache-ram=0`의 prompt cache 동작이 256 MiB보다 빠른 실측값을 보였으므로 `0`을 유지합니다. 이 server-side prompt reuse는 출력 token을 다시 생성하는 기능이며, SQLite 번역 결과 캐시와는 별개입니다.

## 12GB VRAM 환경과 GPU layer 수동 조정

제품 기본값은 `LLAMA_N_GPU_LAYERS=23`입니다. 모델 파일, context 4096,
F16 K/V cache, `--fit off`를 그대로 유지한 상태에서 GPU layer만 바꾸므로
번역 모델과 요청 계약은 바뀌지 않습니다. 다만 CPU로 옮긴 layer가 늘면 속도는
환경에 따라 낮아질 수 있습니다.

Windows WDDM에서는 같은 12GB GPU라도 바탕 화면, 브라우저, Qt, 다른 CUDA
프로세스의 점유량과 메모리 단편화에 따라 `23`의 성공 여부가 달라질 수 있습니다.
2026-08-29에 16GB host RAM, WSL 12GB, swap 4GB, RTX 3060 12GB 환경에서
동일한 IQ4_NL 계약을 다시 검사한 결과는 다음과 같습니다.

| GPU layers | 모델 적재 | 짧은 생성 | 생성 후 GPU 표시값 | 판정 |
|---:|---|---|---:|---|
| 23 | 성공, 3분 33.8초 | 성공 | 11,851MiB 사용 / 265MiB 여유 | 기본값이지만 이 PC에서는 경계값 |
| 22 | 성공, 2분 40.6초 | 성공 | 11,381MiB 사용 / 735MiB 여유 | 이 PC의 안정성 권장값 |

같은 PC에서 `23`은 이전 실행 중 576MiB KV buffer 또는 약 10GiB weight
buffer 할당에 실패한 기록도 있습니다. 이번 성공과 모순되는 것이 아니라,
시작 당시 약 100MiB 수준의 다른 GPU 점유 차이도 결과를 바꿀 만큼 여유가
작다는 뜻입니다. `22`는 이 측정에서 약 470MiB를 더 남겼습니다. 위 시작
시간은 page cache와 WDDM 상태에 크게 좌우되므로 성능 순위로 사용하지 않습니다.

환경별 시작점은 아래처럼 잡습니다. 물리 RAM과 WSL RAM은 GGUF 읽기와 page
cache에 영향을 주지만 CUDA VRAM 부족을 대신 해결하지는 않습니다.

| 환경 | 권장 시작점 | 설명 |
|---|---:|---|
| VRAM 16GB 이상 | 23 | 제품 기본값을 먼저 사용합니다. |
| VRAM 12GB, 적재 전 여유가 반복해서 11.7GB 이상 | 23 | 실제 smoke를 두 번 이상 통과할 때 유지합니다. |
| VRAM 12GB, Windows 표시용 GPU이거나 여유가 11.7GB 미만 | 22 | 23의 OOM이 한 번이라도 재현되면 안정성을 우선합니다. |
| VRAM 12GB 미만 | 22 이하를 실측 | 고정 보장값은 없습니다. layer를 한 단계씩 낮추고 health와 실제 생성을 확인합니다. |
| host RAM 16GB | WSL 10~12GB, swap 4GB | 모델 적재는 가능하지만 14.6GB 전체 prefetch는 보통 건너뜁니다. |
| host RAM 32GB 이상 | WSL 20GB, swap 8GB | 전체 GGUF page-cache prefetch를 시도할 여유가 생깁니다. VRAM 판단은 별도입니다. |

먼저 다른 모델 컨테이너와 앱을 종료하고 `nvidia-smi`의 적재 전 free VRAM을
확인합니다. `23`이 안정적이면 기본값을 유지하고, OOM이 반복되거나 여유가 너무
작으면 아래처럼 `22`를 적용합니다.

standalone Gemma 경로는 앱을 시작한 같은 CMD 세션에서 환경변수를 지정합니다.

```bat
set "LLAMA_N_GPU_LAYERS=22"
call run_comic_cuda13.bat
```

PowerShell에서 실행할 때는 다음과 같습니다.

```powershell
$env:LLAMA_N_GPU_LAYERS = '22'
.\run_comic_cuda13.bat
```

Stage-Batched가 관리형 OCR과 Gemma를 같은 Router에서 순차 적재하는 경우에는
Router preset의 Gemma section도 같은 값이어야 합니다. 사용하는 OCR에 해당하는
파일 하나만 열어 `[gemma-4-26B-IQ4_NL.gguf]` 아래의 값을 바꿉니다. OCR section의
`n-gpu-layers`는 건드리지 않습니다.

| 선택 OCR | 수정할 preset |
|---|---|
| HunyuanOCR | `hunyuanocr_docker_files/router-models.ini` |
| PaddleOCR VL | `paddleocr_vl_docker_files/router-models.ini` |
| PaddleOCR VL Spotting | `paddleocr_vl_spotting_docker_files/router-models.ini` |
| MangaLMM | `mangalmm_docker_files/router-models.ini` |

```ini
[gemma-4-26B-IQ4_NL.gguf]
n-gpu-layers = 22
```

앱이 종료된 상태에서 수정한 뒤 다시 실행하면 preset SHA가 달라져 Router
container가 새 fingerprint로 재생성됩니다. 기본값으로 돌아갈 때는 환경변수를
지우고 preset 값을 `23`으로 복원합니다.

```bat
set "LLAMA_N_GPU_LAYERS="
```

Stage-Batched의 OCR/Gemma handoff와 page-cache prefetch가 GPU layer 설정과 어떻게
분리되는지는 [Stage-Batched 파이프라인 작동 원리](../architecture/codebase-map-ko.md#stage-batched-파이프라인-작동-원리)를
참고하세요.

## runtime image

- image: `ghcr.io/ggml-org/llama.cpp:server-cuda`
- 관측 image label version: `b10133`
- Pull policy: `missing`
- 태그는 moving tag이므로, 이미지가 바뀌면 ready manifest에 기록된 image ID와
  달라집니다. 그때는 준비 스크립트를 다시 실행해야 합니다.

벤치마크 preset, raw 결과, 보고서, 차트는 제품 브랜치가 아니라 `benchmarking/lab` 또는 Git 밖의 검증 로그 폴더에서 관리합니다.

현재 모델·요청·runtime 후보의 최종 품질/속도 판정은
[Optimal++ v1.3.0 Gemma 결정 기록](../performance-and-bugfix-audits/optimal-plus-v1.3.0/03-gemma-translation-and-model-decision-ko.md)을
참고하세요.
