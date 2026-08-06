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
| `Auto` | 상황에 따라 | 볼륨이 이미 계약된 파일을 담고 있으면 `Reseal`, 아니면 `Prepare`를 고릅니다. |

image identity drift는 `Reseal`로 복구됩니다. 수십 GB를 다시 복사하거나 다시
내려받지 않습니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_gemma_runtime.ps1 -Mode Auto
```

## 앱의 자가복구

앱은 관리형 런타임을 시작할 때 계약을 세웁니다. 계약이 깨졌고 **어긋난 것이
봉인된 image identity 하나뿐**이라고 판정되면, 해당 준비 스크립트를 `Auto`로
한 번 실행한 뒤 계약을 다시 세웁니다. 진행 상황은 `runtime_repair` 단계로
표시되며 취소할 수 있습니다.

판정은 보수적입니다. manifest가 스스로 기록한 image identity로 되돌렸을 때
전체 계약이 통과해야만 drift로 봅니다. 모델 SHA-256 불일치, 스키마 위반,
파일 누락처럼 볼륨을 실제로 신뢰할 수 없는 상태는 자동으로 다시 봉인하지 않고
그대로 실패시킵니다. 신뢰할 수 없는 볼륨에 유효 도장을 찍지 않기 위해서입니다.

자가복구는 한 번만 시도합니다. 다시 봉인한 뒤에도 계약이 서지 않으면 원래
오류를 그대로 올립니다.

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
