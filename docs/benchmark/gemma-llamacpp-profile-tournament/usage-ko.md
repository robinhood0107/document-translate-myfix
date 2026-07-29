# Gemma llama.cpp 프로필 토너먼트 사용법

## 준비 파일

모델 manifest, corpus, inventory lock, raw 결과는 모두 Git 밖에 둔다.
아래 placeholder는 실제 절대 경로로 바꾼다.

```powershell
$Python = ".venv-win\Scripts\python.exe"
$Runner = "scripts\benchmark_gemma_llamacpp_profile_tournament.py"
$Manifest = "<validation-log-root>\model-manifest.json"
$Corpus = "<validation-log-root>\corpus.json"
$Lock = "<validation-log-root>\inventory.lock.json"
$Results = "<validation-log-root>\runs"
```

manifest에는 정확히 하나의 baseline target, target별 허용 MTP draft,
Docker volume, 고정 image ID, cache OFF 계약을 넣는다. 서로 다른
디렉터리의 draft를 임의 연결하면 검증 단계에서 거부된다. 서로 내용은
다르지만 로컬 파일명이 같은 GGUF는 `volume_filename`을 고유하게
지정해야 하며, 동일 볼륨의 목적지 이름 중복도 검증 단계에서 거부된다.
WSL의 기존 swap 점유가 프로필 순서에 섞이지 않게 하려면 preflight에
`container_memory_limit_mib`와 `container_swap_disabled: true`를 함께
지정한다. swap을 끄면서 양수 memory limit을 생략하면 manifest가
거부된다.

## 검증·잠금·준비

```powershell
& $Python $Runner validate-manifest --model-manifest $Manifest
& $Python $Runner inventory `
  --model-manifest $Manifest `
  --output $Lock
& $Python $Runner prepare-managed-volumes `
  --model-manifest $Manifest `
  --inventory-lock $Lock
& $Python $Runner verify-volumes `
  --model-manifest $Manifest `
  --inventory-lock $Lock `
  --full-hash
& $Python $Runner list-profiles `
  --model-manifest $Manifest `
  --inventory-lock $Lock
```

`inventory`는 큰 GGUF를 실제로 읽으므로 최초 한 번만 수행한다.
일상 preflight는 volume의 파일명·크기만 확인한다.

## 환경 preflight

```powershell
& $Python $Runner preflight `
  --model-manifest $Manifest `
  --inventory-lock $Lock `
  --corpus $Corpus
```

외부 GPU 컨테이너, idle GPU memory 초과, 포트 충돌, 모델 volume 누락이
있으면 실행하지 않는다. runner에는 이를 무시하는 formal benchmark
옵션이 없다.

`container_swap_disabled`가 켜진 실행은 Docker에 memory limit과 같은
`--memory-swap` 값을 전달한다. cgroup v2의 `memory.swap.max=0`이 되어
해당 llama.cpp 컨테이너는 호스트의 기존 WSL swap을 사용할 수 없다.
물리 RAM 한도를 넘으면 swap으로 느려지는 대신 명시적 OOM/health
실패로 기록된다.

## NGL 경계

no-spec 대표와 target별 MTP draft 4 대표에서 먼저 실행한다.

```powershell
& $Python $Runner tune-ngl `
  --model-manifest $Manifest `
  --inventory-lock $Lock `
  --profile "<target-id>__none" `
  --max-ngl 40 `
  --output "<validation-log-root>\ngl\<target-id>-none.json"

& $Python $Runner tune-ngl `
  --model-manifest $Manifest `
  --inventory-lock $Lock `
  --profile "<target-id>__mtp-4" `
  --max-ngl 40 `
  --output "<validation-log-root>\ngl\<target-id>-mtp.json"
```

MTP full draft GPU가 전부 실패하면 runner가 draft NGL 0을 별도로
검사한다. 각 결과의 `screen_comparison_target_ngls`에 안전 최대값과
바로 아래 값이 기록된다.

컨테이너 cgroup swap은 NGL에 따라 비단조적으로 변할 수 있다. 따라서
상향 탐색 중 swap 한도만 넘긴 지점은 첫 실패에서 중단하지 않고
`max-ngl`까지 계속 확인한다. 최대 NGL에서 swap-only 실패로 시작했는데
상향 후보가 없으면 하향 탐색으로 안전 지점을 찾는다.

ngram 2·4·8은 pinned llama.cpp의
`--spec-ngram-mod-n-min/--spec-ngram-mod-n-max`에 각각 같은 값을
전달한다. MTP 2·4·8만 `--spec-draft-n-max`를 사용한다.

## 단일 stage 실행

```powershell
& $Python $Runner run-profile `
  --model-manifest $Manifest `
  --inventory-lock $Lock `
  --corpus $Corpus `
  --profile "<profile-id>" `
  --stage sensitive15 `
  --round 1 `
  --target-ngl 23 `
  --output-dir $Results
```

stage는 `smoke`, `sensitive15`, `screen18`, `final54`,
`breakeven6/15/30/54`다. 두 번째 라운드는 그룹과 item 순서를
역방향으로 실행한다.

## paired 비교

```powershell
& $Python $Runner compare `
  --baseline "<baseline-round-1.json>" `
  --baseline "<baseline-round-2.json>" `
  --candidate "<candidate-round-1.json>" `
  --candidate "<candidate-round-2.json>" `
  --output "<validation-log-root>\comparisons\<profile-id>.json"
```

`one_sided_95_lower_percent > 0`이면 속도 이득이 확인된 것이다.
0을 걸치면 3회차를 실행한다. 원문·번역·reference가 들어간 raw result는
커밋하지 않는다.
