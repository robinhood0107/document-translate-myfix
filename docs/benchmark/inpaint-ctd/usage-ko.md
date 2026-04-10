# Inpaint CTD Usage

- 권장 실행: `./.venv-win/Scripts/python.exe -u scripts/inpaint_ctd_benchmark.py --scope suite`
- 빠른 spotlight 비교: `./.venv-win/Scripts/python.exe -u scripts/inpaint_ctd_benchmark.py --scope spotlight`
- CUDA13 배치 래퍼: `scripts\inpaint_ctd_benchmark_pipeline_cuda13.bat`, `scripts\inpaint_ctd_benchmark_suite_cuda13.bat`
- 결과 리포트 재생성: `python scripts/generate_inpaint_ctd_report.py --manifest <manifest.yaml>`

## 현재 운영 메모
- 최신 accepted 결과는 `.venv-win`에서 생성했다.
- `.venv-win-cuda13`는 `verify_cuda13_runtime.py`는 통과하지만, full GUI benchmark 경로에서는 환경에 따라 정지/회귀가 있어 현재는 진단용 래퍼로만 본다.
