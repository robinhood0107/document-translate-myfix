# COO SFX shadow 사용법

이 도구는 report-only이며 Docker나 CUDA 모델을 직접 실행하지 않는다. 실제 원본,
prediction JSON, truth, 결과는 모두 Git 밖에 둔다.

## prediction 계약 확인

~~~powershell
.venv-win\Scripts\python.exe scripts\benchmark_coo_sfx_shadow.py validate --predictions <validation-log-root>\coo\cuda-predictions.json --manifest <validation-log-root>\ocr-three-way\corpus-manifest.json
~~~

prediction에는 source image SHA-256, 공식 model/source commit, runtime image digest,
threshold, CPU/CUDA 장치, 실제 normalization, model-load/process/inference 시간,
CUDA peak memory가 들어 있어야 한다.

## CPU/CUDA 일치성

~~~powershell
.venv-win\Scripts\python.exe scripts\benchmark_coo_sfx_shadow.py compare-devices --cpu <validation-log-root>\coo\cpu-predictions.json --cuda <validation-log-root>\coo\cuda-predictions.json --manifest <validation-log-root>\ocr-three-way\corpus-manifest.json --output <validation-log-root>\coo\device-comparison
~~~

두 결과는 동일 checkpoint, source commit, image digest, threshold, page SHA를
사용해야 한다. CPU는 역사적 SyncBatchNorm 제약 때문에 eval running-stat 기준의
일반 BN 참고 경로이고, CUDA가 공식 SyncBN 주 경로다.

## 잠긴 정답에 shadow 평가

~~~powershell
.venv-win\Scripts\python.exe scripts\benchmark_coo_sfx_shadow.py evaluate-shadow --predictions <validation-log-root>\coo\cuda-predictions.json --manifest <validation-log-root>\ocr-three-way\corpus-manifest.json --truth-dir <validation-log-root>\ocr-three-way\truth --run <validation-log-root>\ocr-three-way\runs\paddle-crop\normalized_run.json --run <validation-log-root>\ocr-three-way\runs\paddle-spotting\normalized_run.json --run <validation-log-root>\ocr-three-way\runs\manga-full-page\normalized_run.json --output <validation-log-root>\coo\shadow-evaluation
~~~

truth가 잠기지 않았거나 source/result SHA가 바뀌면 실패한다. 결과에
automatic_preserve_count와 meaningful_text_auto_hidden_count가 0이 아닌 경로는
존재할 수 없도록 계약이 고정돼 있다.
