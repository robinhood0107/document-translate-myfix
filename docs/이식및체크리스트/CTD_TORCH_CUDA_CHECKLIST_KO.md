# CTD Torch CUDA 이식 체크리스트

## 사용 원칙
- 이 체크리스트는 실제 구현 상태를 기준으로 갱신한다.
- 단순 계획과 구현 완료를 구분한다.
- `torch + cuda` 기준 검증이 끝난 항목만 완료 처리한다.

## Stage 1. CTD refiner skeleton + 설정 + 메타데이터
- [x] `integration_paths` helper 추가
- [x] CTD vendor/core 적응 이식
- [x] `modules/masking/ctd_refiner.py` 추가
- [x] `mask_refiner` 설정값 추가
- [x] CTD 설정값 저장/복원 추가
- [x] debug metadata에 `mask_refiner`, `refiner_backend`, `refiner_device` 추가
- [x] 문서/체크리스트 파일 생성

## Stage 2. Protect mask + CTD mask composition
- [x] `modules/masking/protect_mask.py` 추가
- [x] `Keep Existing Lines` 옵션 추가
- [x] `generate_mask()`에 `legacy_bbox` / `ctd` 분기 추가
- [x] CTD refined mask + protect mask 차감 경로 추가
- [x] CTD empty/failure 시 legacy fallback 추가
- [x] debug export에 `protect_mask` 정보 포함

## Stage 3. AOT Torch CUDA 전환
- [x] `AOT` 기본 backend를 `torch`로 전환
- [x] ONNX 경로는 fallback으로만 유지
- [x] 자동번역 배치 파이프라인이 runtime backend를 읽도록 수정
- [x] 한페이지/웹툰 자동 경로가 runtime backend를 읽도록 수정
- [x] `AOT(torch,cuda)` 한 장 smoke 통과

## Stage 4. `lama_large_512px`, `lama_mpe` 이식
- [x] `aot_torch_network.py` 추가
- [x] `lama_torch_network.py` 추가
- [x] `ffc_torch.py` 추가
- [x] `lama_variants.py` bridge 추가
- [x] `lama_large_512px` UI/runtime 연결
- [x] `lama_mpe` UI/runtime 연결
- [x] `legacy LaMa` 저장값을 `lama_large_512px`로 정규화
- [x] `LaMaLarge512px(torch,cuda)` 한 장 smoke 통과
- [x] `LaMaMPE(torch,cuda)` 한 장 smoke 통과

## Stage 5. 자동/한페이지 자동/디버그 경로 통합
- [x] `pipeline/batch_processor.py`에 새 mask/inpaint 메타데이터 연결
- [x] `pipeline/webtoon_batch/chunk.py` 연결
- [x] `pipeline/webtoon_batch/flow.py` 연결
- [x] `pipeline/webtoon_batch/render.py` 연결
- [x] `scripts/export_inpaint_debug.py`를 새 runtime 계약으로 갱신
- [x] `scripts/export_inpaint_debug.py` 직접 실행 경로 수정
- [x] `scripts/export_inpaint_debug.py --inpainter AOT --use-gpu` 통과
- [x] `scripts/export_inpaint_debug.py --inpainter lama_large_512px --use-gpu` 통과
- [x] `scripts/export_inpaint_debug.py --inpainter lama_mpe --use-gpu` 통과

## 제품 마감 확인
- [x] ToolsPage에 `mask_refiner`, `Keep Existing Lines`, CTD 파라미터, 새 inpainter runtime 항목 노출
- [x] 한국어 번역에 새 ToolsPage 문자열 반영 및 `ct_ko.qm` 재컴파일
- [x] `tests/test_settings_tools_runtime.py`로 저장/복원 및 `LaMa -> lama_large_512px` 마이그레이션 검증

## Stage 6. 검증 / 회귀
- [x] `python3 -m py_compile` 주요 수정 파일 통과
- [x] `./.venv-win/Scripts/python.exe scripts/validate_changed_python.py --all` 통과
- [x] `./.venv-win/Scripts/python.exe scripts/headless_smoke.py` 통과
- [x] `torch`, `torchvision`, `einops`를 `venv-win`에 설치
- [x] `torch.cuda.is_available()` 확인
- [x] `/Sample` 5-way 비교 러너 `scripts/benchmark_inpaint_matrix.py` 추가
- [x] `094.png` 기준 5-way matrix smoke 통과
- [x] `/Sample` 전체 코퍼스 full 5-way 실행 (`benchmarking/lab`, suite id `20260410_091232_inpaint_ctd_suite`)
- [x] `benchmarking/lab`용 `inpaint-ctd` family 문서/리포트/spotlight 자산 생성
- [ ] full GUI 수동 회귀 검수 (사람이 직접 클릭하는 검수는 별도 필요, 현재는 GUI 경유 benchmark/자동 검증까지만 완료)

## 현재 확인된 결과
- [x] CTD는 `cuda + torch` 경로에서 실제 `refined_mask`를 생성함
- [x] AOT는 `torch + cuda` 경로에서 실제 inpaint 결과를 생성함
- [x] `lama_large_512px`는 `torch + cuda + bf16` 경로에서 실제 결과를 생성함
- [x] `lama_mpe`는 `torch + cuda` 경로에서 실제 결과를 생성함
- [x] 디버그 export는 `mask_refiner`, `protect_mask`, `inpainter_backend` 메타데이터를 남김

## 벤치마크/운영 메모
- full suite는 `.venv-win`에서 안정적으로 완료했다.
- `.venv-win-cuda13`는 `verify_cuda13_runtime.py`는 통과하지만, full GUI benchmark 경로에서는 환경에 따라 CuDNN/ORT 회귀가 있어 detector를 CPU로 고정해 운용했다.
- 최신 benchmark 결과는 `docs/banchmark_report/inpaint-ctd-report-ko.md`와 `docs/assets/benchmarking/inpaint-ctd/latest/`를 기준으로 검수한다.

## 남은 우선순위
1. protect mask 규칙 튜닝
2. full GUI 수동 검수
3. 일본어 OCR invariance FAIL 원인(마스킹 후 OCR 민감도) 추적
