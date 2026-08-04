# Gemma 로컬 서버 설정 가이드

이 문서는 `Custom Local Server(Gemma)`의 제품용 Docker 런타임을 현재 저장소 기준으로 준비하고 검증하는 방법을 정리합니다.

## 한 번만 준비

Windows PowerShell에서 저장소 루트를 연 뒤 실행합니다.

기본 `Prepare` 실행은 `C:` 여유 공간이 30 GiB 이상인지 확인합니다. `gemma-local-server` 컨테이너가 실행 중이면 앱을 정상 종료해 컨테이너를 중지한 뒤 다시 실행합니다.

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

준비 스크립트의 공개 옵션은 아래와 같습니다.

- `-Mode`: `Prepare` 또는 `Verify`이며 기본값은 `Prepare`입니다.
- `-ModelPath`: `Prepare`에서만 필요한 최종 IQ4_NL GGUF 경로입니다.
- `-VolumeName`: 기본값은 `comic-translate-gemma-models-v2`입니다. 다른 이름을 쓰면 앱 실행 전 `GEMMA_MODEL_VOLUME`에도 같은 값을 설정해야 합니다.
- `-SmokePort`: 실제 GPU smoke 서버의 로컬 포트이며 기본값은 `18082`입니다.
- `-SmokeTimeoutSec`: smoke 준비 제한 시간이며 `30`~`900`초, 기본값은 `420`초입니다.
- `-MinimumFreeBytes`: `C:` 최소 여유 공간이며 기본값은 `32212254720` bytes(30 GiB)입니다.
- `-SkipFreeSpaceCheck`: 공간을 별도로 확인한 경우에만 `C:` 여유 공간 검사를 건너뜁니다.

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

현재 pinned llama.cpp에서는 `cache-ram=0`의 prompt cache 동작이 256 MiB보다 빠른 실측값을 보였으므로 `0`을 유지합니다. 이 server-side prompt reuse는 출력 token을 다시 생성하는 기능이며, SQLite 번역 결과 캐시와는 별개입니다.

## 고정 runtime image

- Image: `ghcr.io/ggml-org/llama.cpp@sha256:22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb`
- 관측 image label version: `b10133`
- Pull policy: `missing`

벤치마크 preset, raw 결과, 보고서, 차트는 제품 브랜치가 아니라 `benchmarking/lab` 또는 Git 밖의 검증 로그 폴더에서 관리합니다.

현재 모델·요청·runtime 후보의 최종 품질/속도 판정 기록과 sampler 캠페인 결과도
`benchmarking/lab`의 벤치마크 문서 세트에서 관리합니다.
