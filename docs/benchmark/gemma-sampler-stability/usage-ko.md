# Gemma sampler stability usage

이 runner는 `benchmarking/lab` 전용이다. private corpus manifest와 결과 root는 저장소 밖 또는 ignored `banchmark_result_log/` 아래에 둔다. raw 결과를 stage하거나 PR에 첨부하지 않는다.

## 실행

Windows CUDA13 환경에서 전체 단계는 다음처럼 실행한다.

```powershell
.venv-win-cuda13\Scripts\python.exe scripts\benchmark_gemma_sampler_stability.py `
  --corpus-manifest <private-corpus-manifest> `
  --output-dir <private-output-dir> `
  --phase temperature
```

선택한 temperature를 이어 비교할 때는 다음과 같이 실행한다.

```powershell
.venv-win-cuda13\Scripts\python.exe scripts\benchmark_gemma_sampler_stability.py `
  --corpus-manifest <private-corpus-manifest> `
  --output-dir <same-private-output-dir> `
  --phase top-p `
  --selected-temperature <selected-temperature>

.venv-win-cuda13\Scripts\python.exe scripts\benchmark_gemma_sampler_stability.py `
  --corpus-manifest <private-corpus-manifest> `
  --output-dir <same-private-output-dir> `
  --phase top-k `
  --selected-temperature <selected-temperature> `
  --selected-top-p <selected-top-p>
```

`--dry-run`은 request 수만 확인하고, `--limit N`은 준비·cleanup을 포함한 짧은 smoke에 사용한다. 이미 완료된 response는 기본적으로 재사용한다. 계약이 바뀐 경우에는 새 output directory를 사용한다.

## 판정

구조 검사는 choice 수, finish reason, JSON key 순서·개수, translation 문자열을 확인한다. hard fail은 검열·삭제·민감 표현 약화, 부정/질문/선언 변화, 화자·대상·행동·관계·숫자·정체성 변화다. 일반적인 존댓말·어순·구두점·동의어 차이는 canonical과 문맥을 대조해 직접 판정한다.

자동 집계가 `review_required`를 남긴 unique cluster는 private judgment schema로만 확정한다. clean candidate가 없으면 sampler 기본값을 제품에 적용하지 않는다. 최종 제품 변경은 이 runner나 benchmark preset을 복사하지 않고 별도 product PR에서 수행한다.

## 확인 항목

- `.venv-win`과 `.venv-win-cuda13`에서 runner 단위 테스트
- Router 실행 중 loaded model이 항상 1개 이하인지 확인
- 마지막에 `loaded_count=0`, `container_running=false`, `release_failed=false` 확인
- 전체 실행량이 990을 넘지 않는지 확인
- private output이 Git status에 나타나지 않는지 확인
