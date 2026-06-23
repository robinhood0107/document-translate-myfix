# CTD 정밀 마스크 + Torch CUDA 인페인팅 이식 계획

## 목표
현재 앱의 `RT-DETR-v2 -> bbox mask -> AOT/legacy LaMa` 경로를,
`RT-DETR-v2 -> CTD refined mask -> protect mask -> Torch CUDA inpainter` 구조로 재구성한다.

이번 이식의 핵심은 detector를 바꾸는 것이 아니라 **마스크 품질과 Torch CUDA 인페인팅 런타임**을 재구성하는 것이다.

## 최종 아키텍처
1. `RT-DETR-v2`
   - 역할: `text_bubble / text_free / bubble` 구조 proposal 생성
   - 유지 이유: 현재 앱의 블록 구조, bubble/text_free 분기, 자동번역/한페이지 자동번역 흐름과 이미 잘 연결돼 있음
2. `CTD mask refiner`
   - 역할: 페이지 전체에 대해 1회 실행하여 `raw_mask`, `refined_mask` 생성
   - 실제 채택값: `refined_mask`
   - 위치: detector replacement가 아니라 **정밀 마스크 생성기**
3. `protect mask`
   - 역할: bubble border, strong line, 구조 edge를 보호
   - 최종 적용: `final erase mask = CTD final mask - protect mask`
4. `inpainter`
   - 기본: `AOT (torch + cuda)`
   - 품질 모드: `lama_large_512px (torch + cuda, bf16)`
   - 고급/실험 모드: `lama_mpe (torch + cuda, fp32)`
5. `cleanup`
   - 유지하되, 주 마스크 생성 책임이 아니라 CTD+protect 이후의 후순위 보정으로 둠

## 설계 원칙
- `RT-DETR-v2`는 유지한다.
- `comictextdetector.pt`는 detector selector가 아니라 mask refiner 계층으로 넣는다.
- CPU 지원 코드는 남기되 제품 목표는 `torch + cuda` 전체 파이프라인이다.
- `legacy LaMa`는 코드 fallback은 남겨도 UI 실사용 선택지에서는 사실상 폐기한다.
- 자동번역(`Translate All`)과 한 페이지 자동번역(`One-Page Auto`) 모두 같은 detection/mask/inpaint pipeline을 읽게 유지한다.
- 디버그 export는 새로운 mask/refiner/backend 메타데이터를 포함해야 한다.

## 설정 정책
### Mask Refiner
- `mask_refiner`: `legacy_bbox`, `ctd`
- 기본값: `ctd`
- CTD 기본값
  - `ctd_detect_size=1280`
  - `ctd_det_rearrange_max_batches=4`
  - `ctd_device=cuda`
  - `ctd_font_size_multiplier=1.0`
  - `ctd_font_size_max=-1`
  - `ctd_font_size_min=-1`
  - `ctd_mask_dilate_size=2`
  - `keep_existing_lines=true`

### Inpainter
- 노출 선택지
  - `AOT`
  - `lama_large_512px`
  - `lama_mpe`
- 기본값
  - `AOT`: `backend=torch`, `device=cuda`, `inpaint_size=2048`, `precision=fp32`
  - `lama_large_512px`: `backend=torch`, `device=cuda`, `inpaint_size=1536`, `precision=bf16`
  - `lama_mpe`: `backend=torch`, `device=cuda`, `inpaint_size=2048`, `precision=fp32`
- 호환 정책
  - 기존 저장값 `LaMa`는 로드 시 `lama_large_512px`로 정규화한다.

## 구현 범위
### 1. CTD bridge
- source reference의 registry/UI 전체를 가져오지 않고, 필요한 CTD core만 vendor 레이어로 적응 이식한다.
- page-level 추론, torch/onnx 선택, refined mask 반환은 `modules/masking/ctd_refiner.py`가 담당한다.

### 2. Protect mask
- `modules/masking/protect_mask.py`에서 규칙 기반 protect mask를 생성한다.
- 첫 버전은 다음을 포함한다.
  - bubble border band
  - strong dark line protect
  - canny edge protect
- `Keep Existing Lines`가 사실상 protect mask on/off 역할을 한다.

### 3. Mask composition
- 기존 `bbox 기반 morphology` 경로는 `legacy_bbox` fallback으로 유지한다.
- `ctd` 선택 시에는:
  - CTD raw/refined mask 생성
  - block/bubble scope로 합성
  - protect mask 차감
  - 비어 있으면 CTD-only 또는 legacy fallback

### 4. Torch CUDA AOT
- 기존 ONNX 기본 경로를 제거하고 `AOT` 기본 backend를 `torch`로 바꾼다.
- ONNX 경로는 fallback으로만 남긴다.

### 5. Torch CUDA LaMa variants
- `lama_large_512px.ckpt`, `lama_mpe.ckpt`는 source reference wrapper까지 함께 이식한다.
- 단순 모델 교체가 아니라 bridge 계층으로 현재 앱의 `InpaintModel` 인터페이스에 맞춘다.

### 6. UI / Settings / Metadata
- Tools에 CTD mask refiner 옵션과 runtime 파라미터를 추가한다.
- inpainter runtime 설정(`device`, `inpaint_size`, `precision`)을 저장/복원한다.
- debug export metadata에 아래를 포함한다.
  - `mask_refiner`
  - `protect_mask_applied`
  - `protect_mask_*`
  - `refiner_backend`
  - `refiner_device`
  - `inpainter_backend`

### 7. 배치/웹툰/디버그 경로 통합
- `pipeline/batch_processor.py`
- `pipeline/webtoon_batch/chunk.py`
- `pipeline/webtoon_batch/flow.py`
- `pipeline/webtoon_batch/render.py`
- `scripts/export_inpaint_debug.py`
위 경로가 모두 같은 mask/inpaint contract를 사용해야 한다.

## 실제 수정 대상(핵심)
### 새 파일
- `modules/masking/ctd_refiner.py`
- `modules/masking/protect_mask.py`
- `modules/utils/integration_paths.py`
- `modules/utils/inpainting_runtime.py`
- `modules/inpainting/aot_torch_network.py`
- `modules/inpainting/lama_torch_network.py`
- `modules/inpainting/ffc_torch.py`
- `modules/inpainting/lama_variants.py`

### 주요 수정 파일
- `modules/utils/image_utils.py`
- `modules/utils/inpaint_debug.py`
- `modules/utils/pipeline_config.py`
- `modules/inpainting/aot.py`
- `pipeline/inpainting.py`
- `pipeline/batch_processor.py`
- `pipeline/webtoon_batch/chunk.py`
- `pipeline/webtoon_batch/flow.py`
- `pipeline/webtoon_batch/render.py`
- `app/ui/settings/tools_page.py`
- `app/ui/settings/settings_ui.py`
- `app/ui/settings/settings_page.py`
- `scripts/export_inpaint_debug.py`

## 검증 전략
### 정적 검증
- `python3 -m py_compile`로 주요 수정 파일 전체 확인
- `./.venv-win/Scripts/python.exe scripts/validate_changed_python.py --all`
- `./.venv-win/Scripts/python.exe scripts/headless_smoke.py`

### Torch CUDA 검증
- `venv-win`에 `torch`, `torchvision`, `einops`를 설치해 Windows Python 기준 GPU 경로 활성화
- 실제 확인 대상
  - `CTDRefiner(cuda)` 로드 및 refined mask 생성
  - `AOT(torch,cuda)` 한 장 inference
  - `lama_large_512px(torch,cuda)` 한 장 inference
  - `lama_mpe(torch,cuda)` 한 장 inference
  - `scripts/export_inpaint_debug.py`를 각 inpainter로 1장 실행

### 수동/후속 검수
- `/Sample` 5-way 비교 러너 `scripts/benchmark_inpaint_matrix.py` 추가
- 5-way 구성
  - `legacy_bbox + AOT`
  - `ctd + AOT`
  - `ctd + protect + AOT`
  - `ctd + protect + lama_large_512px`
  - `ctd + protect + lama_mpe`
- 현재는 `094.png` 기준 smoke 실행을 완료했고, 전체 코퍼스 full run은 다음 단계에서 수행

## 제품 마감 보강
- ToolsPage의 새 CTD/Inpainter runtime UI가 실제로 노출되는지 자동 테스트를 추가했다.
- `SettingsPage.load_settings()`는 `MComboBox`의 `setCurrentText()` 의존을 줄이고 `findText -> setCurrentIndex` 경로로 정규화했다.
- deprecated `LaMa` 저장값이 `lama_large_512px`로 복원되는 회귀 테스트를 추가했다.
- 한국어 번역은 새 ToolsPage 문자열을 채우고 `ct_ko.qm`를 재컴파일했다.

## 현재 상태 요약
- CTD refiner bridge 추가 완료
- protect mask 1차 규칙 기반 구현 완료
- `generate_mask()`의 CTD/legacy 분기 완료
- `AOT` Torch CUDA 기본 경로 전환 완료
- `lama_large_512px`, `lama_mpe` Torch CUDA bridge 추가 완료
- UI/설정/마이그레이션 추가 완료
- 배치/웹툰/디버그 export 메타데이터 연동 완료
- `venv-win` 기준 Torch CUDA smoke 완료
- `scripts/benchmark_inpaint_matrix.py` 추가 및 `094.png` 기준 5-way matrix smoke 완료

## 남은 후속 작업
- `/Sample` 전체 코퍼스에 대한 full 5-way run과 결과 비교 리포트
- protect mask 규칙 미세 조정
- full GUI 수동 검수
- 필요 시 `legacy_bbox`와 CTD 결과 비교 시각화 리포트 추가
