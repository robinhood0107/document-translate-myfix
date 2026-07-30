# GPU 인페인트 품질 게이트 사용법

모든 실행 결과는 반드시 Git 밖의 새 폴더에 쓴다. 입력 예시는
`benchmarks/inpaint_quality_gates/case-manifest.example.json`에 있다.

```powershell
.\.venv-win-cuda13\Scripts\python.exe scripts\benchmark_inpaint_quality_gates.py capture `
  --manifest C:\external-validation\cases.json `
  --output C:\external-validation\frozen-v1

.\.venv-win-cuda13\Scripts\python.exe scripts\benchmark_inpaint_quality_gates.py run `
  --frozen C:\external-validation\frozen-v1 `
  --phase mask `
  --output C:\external-validation\mask-screen-v1

.\.venv-win-cuda13\Scripts\python.exe scripts\benchmark_inpaint_quality_gates.py blind `
  --results C:\external-validation\mask-screen-v1 `
  --output C:\external-validation\mask-blind-v1
```

mask blind 검수 후 선택된 dilation 하나만 model 단계에 전달한다.

```powershell
.\.venv-win-cuda13\Scripts\python.exe scripts\benchmark_inpaint_quality_gates.py run `
  --frozen C:\external-validation\frozen-v1 `
  --phase model `
  --selected-dilation 2 `
  --include-feasibility `
  --output C:\external-validation\model-screen-v1
```

HTML에서 모든 후보를 판정하고 CSV를 내보낸 뒤 먼저 검증한다.

```powershell
.\.venv-win\Scripts\python.exe scripts\benchmark_inpaint_quality_gates.py validate-review `
  --review-root C:\external-validation\model-blind-v1 `
  --review-csv C:\external-validation\model-blind-v1\blind_review_completed.csv
```

`unblind` confirmation은 state에 기록된 정확한
`<row-count>-ROWS-REVIEWED` 값이어야 한다. 최종 full-pipeline 검수에서는
먼저 실제 offscreen 실행 render를 외부 manifest 순서대로 새 결과 계약에
붙인다.

```powershell
.\.venv-win\Scripts\python.exe scripts\benchmark_inpaint_quality_gates.py attach-renders `
  --results C:\external-validation\model-screen-v1 `
  --manifest C:\external-validation\render-attachments.json `
  --output C:\external-validation\model-screen-with-renders-v1

.\.venv-win\Scripts\python.exe scripts\benchmark_inpaint_quality_gates.py blind `
  --results C:\external-validation\model-screen-with-renders-v1 `
  --require-render `
  --output C:\external-validation\model-final-blind-v1
```

render manifest는
`benchmarks/inpaint_quality_gates/render-manifest.example.json` 형식을
사용한다. 후보×케이스 전체가 원래 순서대로 존재하고 각 파일 SHA-256이
일치해야 하며, 원본 result 폴더는 수정하지 않는다.
