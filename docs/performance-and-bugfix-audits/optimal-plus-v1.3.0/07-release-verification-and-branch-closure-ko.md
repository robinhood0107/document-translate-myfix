# v1.3.0 릴리스 검증과 브랜치 정리

이 문서는 v1.3.0을 내보내기 전에 확인한 제품 계약과, 실험 브랜치·원시 산출물을 제품 이력과 분리한 방법을 기록한다. 이 문서는 benchmark raw output이나 실제 작품 결과를 포함하지 않는다.

## 릴리스 기준

| 항목 | 확정값 |
|---|---|
| 공식 태그 | `v1.3.0` |
| main release commit | `d7bcb58f5721927f42c26a589f547f8799c64e3f` |
| 공식 Windows 산출물 | launcher-source ZIP + `SHA256SUMS.txt` |
| ZIP SHA-256 | `dced54a33ad3f46cd0d5625836d6fe1bf9e85ff0dfd9767e960fbf4a038f5758` |
| 배포 방식 | 첫 실행에서 지원 환경을 준비하는 source launcher |
| 비공식 도구 | 기존 Nuitka script는 수동 도구로만 유지 |

공식 ZIP에는 allowlist 제품 source, launcher, CUDA12/13 requirements, Docker/설정, 준비 script, 번역 asset, README·LICENSE만 포함한다. virtual environment, 모델, cache, benchmark runner, raw 결과, 로컬 경로와 secret은 포함하지 않는다.

## 자동·실제 환경 검증

| 범주 | CUDA12 | CUDA13 |
|---|---:|---:|
| Python unit suite | 876 passed, 6 skipped | 876 passed, 6 skipped |
| CUDA inpainting contract | 13/13 통과 | 13/13 통과 |
| launcher contract | 통과 | 통과 |
| headless smoke·번역 asset 검사 | 통과 | 통과 |

추가로 다음 공통 검증을 통과했다.

- `validate_changed_python.py --all`
- `headless_smoke.py`
- `compile_translations.py --check`
- `verify_windows_launchers.py`
- managed llama.cpp runtime의 정적 계약과 live process-tree 검사
- deterministic source bundle의 build·verify·새 폴더 추출 후 launcher 확인

CUDA13은 실제 GPU runtime에서 ONNX Runtime CUDA와 TensorRT 계약도 별도로 확인했다. CUDA12와 CUDA13은 같은 checkout에서 동시에 검사하지 않고 순차적으로 실행해 cache 및 bytecode 잠금 충돌을 피한다.

## 릴리스에 포함한 판정

- 관리형 로컬 inference를 llama.cpp 전용으로 통일했다.
- 기본 OCR은 detector + Paddle crop direct llama.cpp 경로로 유지했다.
- Spotting·MangaLMM은 Experimental로 보존하고 자동 선택이나 숨은 fallback을 넣지 않았다.
- Gemma는 IQ4_NL, contextual-single, chunk 6, no-spec, F16, batch/ubatch 2048/512, NGL 23 계약을 유지했다.
- 검증된 result cache, exact OCR cache, source-aware prewarm은 유지했다.
- project checkpoint는 검증된 one-time migration으로 한 번 활성화하고 이후 사용자 선택을 보존한다.

릴리스에 포함하지 않은 후보도 명확히 기록했다. 품질 회귀가 있거나 반복 속도 이득이 입증되지 않은 grouped, Q8/QAT/MTP, explicit GGUF read-ahead, unsafe inpaint 후보와 여러 병렬화 후보는 기본값으로 승격하지 않았다.

## 브랜치와 산출물의 경계

장기 유지 브랜치는 아래 세 개다.

- `main`: 배포 기준
- `develop`: 통합 기준
- `benchmarking/lab`: benchmark runner·preset·설명 문서 기준

짧은 작업 브랜치는 PR이 병합된 뒤 정리한다. `main`과 `develop`에는 직접 push하지 않고 PR을 통해서만 반영한다. 원시 benchmark 결과, 검수표, 이미지, OCR·번역·인페인트 산출물은 이 세 브랜치에 넣지 않는다.

제품 코드의 기본값을 바꾸려면 먼저 benchmark branch와 Git 밖 validation log에서 동일한 품질 계약을 검증하고, 통과한 제품 변경만 별도 PR로 `develop`에 올린다.

## 사후 점검 절차

1. GitHub Release에서 tag, ZIP, checksum을 확인한다.
2. 새 폴더에 ZIP을 풀어 checksum과 두 launcher의 verify-only 계약을 확인한다.
3. managed runtime이 llama.cpp만 시작하는지 검사한다.
4. test corpus나 원시 출력 없이 headless smoke와 launcher 계약을 다시 확인한다.
5. 발견한 문제는 `develop` 대상 작업 브랜치에서 재현·수정하고, 배포 기준에는 hotfix 절차만 사용한다.

## 관련 문서

- [공통 진실 명세](00-truth-specification-ko.md)
- [runtime 자산과 llama.cpp 전용 정책](06-runtime-assets-and-llamacpp-retirement-ko.md)
- [공개 가능한 근거와 비공개 원시 증거의 경계](08-sanitized-evidence-index-ko.md)
- [저장소 Git·릴리스 규칙](../../../rules.md)
