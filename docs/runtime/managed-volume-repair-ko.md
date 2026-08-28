# 관리형 llama.cpp 볼륨 복구 가이드

관리형 llama.cpp 런타임 다섯 개(Gemma, HunyuanOCR, MangaLMM, PaddleOCR-VL,
PaddleOCR-VL Spotting)는 모두 준비된 Docker 볼륨 안의 ready manifest로 자신을
증명합니다. 이 문서는 그 manifest가 깨졌을 때 무엇이 자동으로 복구되고 무엇이
사람의 판단을 요구하는지 정리합니다.

## 왜 멀쩡하던 볼륨이 갑자기 거부되는가

manifest는 준비 당시의 llama.cpp image identity를 함께 봉인합니다. 기본 image
참조는 고정 digest가 아니라 움직이는 태그
(`ghcr.io/ggml-org/llama.cpp:server-cuda13`)이므로, 업스트림이 그 태그를 갱신하고
로컬에서 새 image를 받으면 digest가 바뀝니다.

그러면 모델 파일이 계약과 완전히 같은데도 manifest의 image identity만 어긋나
런타임 계약이 깨집니다. 증상은 다음과 같습니다.

- Docker Desktop에 짧게 뜨는 프로브 컨테이너
  (`comic-translate-gemma-cache-warm` 등)만 보이고, 정작 `gemma-local-server`
  같은 관리 컨테이너는 끝까지 뜨지 않는다.
- 로그나 오류에 `ready manifest mismatch for source_image_digest`가 남는다.

## 준비 스크립트의 네 가지 모드

`scripts/prepare_*_runtime.ps1`은 모두 같은 `-Mode`를 받습니다.

| 모드 | 원본 파일 | 하는 일 |
| --- | --- | --- |
| `Prepare` | 필요 | 원본을 볼륨에 복사하고, 해시를 검증하고, 실제 GPU 스모크를 통과시킨 뒤 manifest를 씁니다. |
| `Verify` | 불필요 | 읽기 전용 검사만 합니다. 아무것도 바꾸지 않습니다. |
| `Reseal` | **불필요** | 볼륨 내용을 그대로 두고, 해시를 다시 검증하고, 현재 image로 스모크를 다시 통과시킨 뒤 manifest만 다시 씁니다. |
| `Auto` | 상황에 따라 | 유효한 봉인은 즉시 재사용하고, 봉인이 낡으면 `Reseal`, 모델이 빠졌으면 `Prepare`를 고릅니다. |

image identity drift는 `Reseal`로 복구됩니다. 수십 GB를 다시 복사하거나 다시
내려받지 않습니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_gemma_runtime.ps1 -Mode Auto
```

## 앱의 읽기 전용 검증

앱은 관리형 런타임을 시작할 때 계약을 읽기 전용으로 검사합니다. image identity,
manifest, 파일, volume 중 하나라도 setup 봉인과 다르면 다운로드, pull, `Auto`,
reseal을 실행하지 않고 페이지 처리 전에 실패합니다. core 런타임은 해당
`setup*.bat`, MangaLMM/Spotting은 `setup_full*.bat`을 다시 실행해야 합니다.

준비 스크립트의 `Auto`/`Reseal`은 setup과 명시적인 운영자 복구 명령에서만
사용합니다. 이 경계 때문에 GUI 파이프라인이 장시간 실행 중 갑자기 모델을
내려받거나 volume을 바꾸지 않습니다.

## 원본 파일 해결 순서

`Prepare`와 `Auto`는 원본을 다음 순서로 찾습니다.

1. `-ModelPath` 또는 `-ModelDirectory`로 명시한 경로
2. 저장소의 gitignore된 `testmodel/`과 그 바로 아래 하위 폴더
3. `-DownloadDirectory`로 지정한 다운로드 캐시
4. 등록된 원본에서 내려받기 — `-AllowDownload`를 줬을 때만

크기와 SHA-256이 계약과 정확히 같은 파일만 원본으로 인정합니다. 내려받기는
`.partial`로 받아 검증한 뒤에만 최종 이름으로 옮기므로, 중단된 다운로드가 다음
실행에서 정상 원본으로 오인되지 않습니다. 이미 받다 만 파일이 있으면 Range
요청으로 이어받습니다.

### 등록된 원본

각 파일의 SHA-256이 계약값과 정확히 같음을 확인한 출처입니다.

| 런타임 | Hugging Face 저장소 |
| --- | --- |
| Gemma | `Vastopian/gemma-4-26B-A4B-it-abliterated-GGUF` |
| HunyuanOCR | `ggml-org/HunyuanOCR-GGUF` |
| MangaLMM | `mradermacher/MangaLMM-GGUF` |
| PaddleOCR-VL | `PaddlePaddle/PaddleOCR-VL-1.6-GGUF` |
| PaddleOCR-VL Spotting | `PaddlePaddle/PaddleOCR-VL-1.6-GGUF` (대상 GGUF는 crop VLM과 동일 파일, projector는 공식 crop projector에서 로컬 파생) |

HunyuanOCR은 업스트림 파일명(`HunyuanOCR-Q8_0.gguf`,
`mmproj-HunyuanOCR-Q8_0.gguf`)이 볼륨 안 이름과 다릅니다. 받는 쪽에서 계약
이름으로 저장하므로 사용자가 이름을 바꿀 필요는 없습니다.

## 스스로 고쳐지지 않는 상태

| 상태 | 대응 |
| --- | --- |
| 볼륨 안 모델의 SHA-256이 다르다 | `-Mode Prepare`로 원본에서 다시 복사합니다. `Reseal`은 모델 데이터를 절대 복사하지 않습니다. |
| 볼륨 자체가 없다 | `-Mode Auto`(또는 `Prepare`)가 볼륨을 만들고 채웁니다. `Reseal`은 없는 볼륨을 만들지 않습니다. |
| 볼륨 라벨이 준비 계약과 다르다 | 새 versioned 볼륨 이름을 씁니다. 라벨만 고쳐 쓰지 않습니다. |
| Docker가 실행 중이 아니다 | Docker Desktop을 켠 뒤 다시 시도합니다. |
| llama.cpp image가 없다 | 준비 스크립트와 앱 모두 지원 태그를 한 번 자동으로 `pull` 합니다. |

## 관련 문서

- [Gemma 로컬 서버 설정 가이드](../gemma/local-server-ko.md)
- [관리형 llama.cpp 전용 런타임](managed-llamacpp-only-ko.md)
- [빠른 시작](../setup/quickstart-ko.md)
