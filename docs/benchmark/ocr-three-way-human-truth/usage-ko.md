# 사용법

Windows PowerShell에서 `.venv-win` 또는 `.venv-win-cuda13`의 Python을 직접
사용합니다. 아래 경로는 예시이며 모든 입력·출력은 Git 저장소 밖이어야 합니다.

## 1. corpus manifest 만들기

`corpus-build-spec.json`은
[`corpus-build-spec.example.json`](./corpus-build-spec.example.json)을 복사해 Git
밖에서 작성합니다.

```powershell
.venv-win\Scripts\python.exe scripts\benchmark_ocr_three_way_human_truth.py `
  build-manifest `
  --spec C:\comic-translate-validation\ocr-three-way\corpus-build-spec.json `
  --output C:\comic-translate-validation\ocr-three-way\corpus-manifest.json
```

manifest는 원본 이미지와 detector snapshot의 SHA-256, 실제 width/height를
기록합니다. detector 결과의 OCR text는 정답 초안에 복사하지 않습니다.

## 2. 후보를 보기 전에 정답 만들기

```powershell
.venv-win\Scripts\python.exe scripts\benchmark_ocr_three_way_human_truth.py `
  init-truth `
  --manifest C:\comic-translate-validation\ocr-three-way\corpus-manifest.json `
  --output C:\comic-translate-validation\ocr-three-way\truth
```

초기화 시 `truth\truth-entry.csv`가 함께 생성됩니다. 원본과 확대 crop만 보고 이
CSV를 작성한 뒤 다시 가져옵니다. `detector_block_ids_json`과 `bbox_xyxy_json`은 JSON
배열 형식입니다. detector가 놓친 영역은 고유한 `truth_region_id`, 빈 detector ID
배열, `region_source=human_extra`로 행을 추가합니다.

```powershell
.venv-win\Scripts\python.exe scripts\benchmark_ocr_three_way_human_truth.py `
  import-truth-csv `
  --truth-dir C:\comic-translate-validation\ocr-three-way\truth `
  --csv C:\comic-translate-validation\ocr-three-way\truth\truth-entry.csv
```

JSON을 직접 편집했다면 잠그기 전에 CSV를 다시 내보내 두 표현을 일치시킵니다.

```powershell
.venv-win\Scripts\python.exe scripts\benchmark_ocr_three_way_human_truth.py `
  export-truth-csv `
  --truth-dir C:\comic-translate-validation\ocr-three-way\truth
```

bbox를 바꾸거나 region을 추가한 뒤 crop을 다시 생성합니다.

```powershell
.venv-win\Scripts\python.exe scripts\benchmark_ocr_three_way_human_truth.py `
  refresh-truth-crops `
  --truth-dir C:\comic-translate-validation\ocr-three-way\truth
```

모든 필드가 완성되면 잠급니다.

```powershell
.venv-win\Scripts\python.exe scripts\benchmark_ocr_three_way_human_truth.py `
  lock-truth `
  --truth-dir C:\comic-translate-validation\ocr-three-way\truth
```

불완전한 transcription·role·action·confidence가 하나라도 있거나 CSV와 페이지 JSON이
다르면 실패합니다. 초기 detector 행의 삭제·병합·bbox 변경도 실패하며, detector가
놓친 영역만 `human_extra`로 추가할 수 있습니다.

## 3. 기존 세 route 결과 가져오기

route마다 Git 밖에 runtime contract를 만듭니다. 예시는
[`runtime-contract.example.json`](./runtime-contract.example.json)에 있습니다.
`backend`는 반드시 `llama.cpp`여야 하며 model/mmproj/command/image digest,
prompt mode, `special_tokens`, 공식 pixel budget을 실제 실행값으로 고정합니다.
pixel budget은 crop `1,003,520`, Paddle Spotting `1,605,632`, MangaLMM
`2,116,800`입니다. `fingerprint_sha256`은 그 필드를 제외한 runtime contract 전체의
canonical JSON SHA-256이어야 합니다. 필수 digest 누락이나 fingerprint 불일치는
가져오기 전에 거부됩니다. 예시 파일의 `000...000` digest는 의도적인 자리표시자라
그대로는 거부되며, 실제 SHA로 바꾼 뒤 fingerprint도 다시 계산해야 합니다.

각 route 결과를 처음 고정할 때 원본 SHA와 결과 폴더를 manifest에 결합합니다. 과거
raw 응답에는 source SHA가 없을 수 있으므로 이 명시적 binding 없이는 가져오지
않습니다. 같은 파일명의 다른 페이지 결과가 섞이는 것을 막기 위한 단계입니다.

```powershell
$routes = @(
  @{ Id='paddle_crop'; Source='C:\validation\paddle-crop' },
  @{ Id='paddle_spotting_full_page'; Source='C:\validation\paddle-spotting' },
  @{ Id='mangalmm_full_page'; Source='C:\validation\manga-full-page' }
)

foreach ($route in $routes) {
  .venv-win\Scripts\python.exe scripts\benchmark_ocr_three_way_human_truth.py `
    bind-source-results `
    --route $route.Id `
    --manifest C:\comic-translate-validation\ocr-three-way\corpus-manifest.json `
    --source-results $route.Source
}
```

binding은 결과를 만든 직후 생성하는 것이 원칙입니다. 원본 identity뿐 아니라 route별
primary result 파일의 상대 경로와 SHA도 함께 고정합니다. 이미 존재하면 덮어쓰지 않으며,
역사 결과는 원본을 사람이 다시 확인한 뒤 별도 복사본에서 한 번만 생성합니다.

```powershell
foreach ($route in $routes) {
  .venv-win\Scripts\python.exe scripts\benchmark_ocr_three_way_human_truth.py `
    import-existing `
    --route $route.Id `
    --manifest C:\comic-translate-validation\ocr-three-way\corpus-manifest.json `
    --source-results $route.Source `
    --runtime-contract "C:\comic-translate-validation\ocr-three-way\runtime-$($route.Id).json" `
    --output "C:\comic-translate-validation\ocr-three-way\runs\$($route.Id)"
}
```

가져온 결과에는 사용한 모든 원시 파일의 상대 경로와 SHA-256이 들어갑니다. 이후
원시 파일을 바꾸면 `validate-run`과 `make-review`가 모두 실패합니다.

## 4. 블라인드 검수 패키지

```powershell
.venv-win\Scripts\python.exe scripts\benchmark_ocr_three_way_human_truth.py `
  make-review `
  --truth-dir C:\comic-translate-validation\ocr-three-way\truth `
  --run C:\comic-translate-validation\ocr-three-way\runs\paddle_crop\normalized_run.json `
  --run C:\comic-translate-validation\ocr-three-way\runs\paddle_spotting_full_page\normalized_run.json `
  --run C:\comic-translate-validation\ocr-three-way\runs\mangalmm_full_page\normalized_run.json `
  --output C:\comic-translate-validation\ocr-three-way\review
```

`review\index.html`은 후보명과 속도를 표시하지 않습니다. route 결과가 mask,
cleaned crop, render, diff 같은 해시 고정 자산을 제공했다면 같은 카드에 함께
복사·표시합니다. 실제 판정은
`region-review.csv`에 `yes`, `no`, `uncertain`, `not_applicable` 중 하나로
기록합니다. blind key는 `review\private`에만 있습니다.
`uncertain`은 유효한 최종 판정이지만 해당 페이지를 완전 성공으로 계산하지 않습니다.

## 5. 완전성 검사와 unblind

```powershell
.venv-win\Scripts\python.exe scripts\benchmark_ocr_three_way_human_truth.py `
  finalize-review `
  --review-dir C:\comic-translate-validation\ocr-three-way\review
```

한 칸이라도 누락되거나 행·열·후보 원문·좌표·자산·blind key가 변조되면 unblind를
거부합니다. 성공하면 사람 기준 품질 통계뿐 아니라 route별 구조 성공, parser/length,
retry, 요청 시간까지 포함한 `final_metrics.json`과 `final_report-ko.md`를 생성합니다.
과거 결과에서 실제 attempt 이력이 없는 페이지는 횟수를 추정하지 않고 telemetry
불완전 페이지로 따로 집계합니다.
