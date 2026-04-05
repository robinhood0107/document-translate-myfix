# 자동번역 벤치 체크리스트

## 실행 전

- [ ] 현재 브랜치가 benchmark 작업 브랜치인지 확인
- [ ] `/Sample`에 representative `30장`이 있는지 확인
- [ ] `/Sample`과 `/banchmark_result_log`가 Git ignore인지 확인
- [ ] `gemma-local-server`, `paddleocr-server`, `paddleocr-vllm` health 확인
- [ ] `.venv-win` 또는 `.venv-win-cuda13`가 준비됐는지 확인
- [ ] `PySide6`, `cv2`, `numpy`, `pandas`, `matplotlib` import 가능 확인
- [ ] 언어별 benchmark 폰트 폴더가 준비됐는지 확인

## 실행 중

- [ ] 런처가 올바른 venv를 잡았는지 확인
- [ ] output root가 `./banchmark_result_log`인지 확인
- [ ] `managed` 측정 시 runtime staging 로그가 정상인지 확인
- [ ] `command_stdout.txt`에 page failure 또는 HTTP failure가 없는지 확인

## 실행 후

- [ ] `summary.json`, `summary.md`, `metrics.jsonl` 생성 확인
- [ ] `docs/banchmark_report/report-ko.md` 자동 생성 확인
- [ ] `docs/assets/benchmarking/latest/*.png` 생성 확인
- [ ] 대표 후보는 `translation_audit.json`까지 확인
- [ ] `validate_changed_python.py --all` 실행
- [ ] 필요 시 `compile_translations.py --check`
- [ ] 필요 시 CUDA13 `headless_smoke.py`
