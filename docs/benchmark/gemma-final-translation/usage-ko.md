# Gemma 최종 번역 전용 비교 사용법

## 필수 입력

- `<input-root>`: 정렬 가능한 이미지 22개
- `<ocr-snapshot>`: 페이지 순서를 보존한 292블록 JSON
- `<validation-log-root>`: Git 밖 또는 Git ignore가 적용된 결과 위치
- baseline, grouped F16, grouped Q8 컨테이너 이름
- host IQ4_XS 모델 경로와 사전에 잠근 SHA-256

실행 전에 세 컨테이너가 동일한 pinned image와 동일한 read-only model volume을
사용하고, host publish가 `127.0.0.1:18080:8080`인지 확인한다. runner가
전체 command와 후보별 config fingerprint까지 다시 검사하며, 다르면 요청을
보내지 않는다. Docker server와 NVIDIA driver identity를 완전하게 읽지
못해도 실행을 중단한다.

## CUDA13 suite

```bat
scripts\gemma_final_translation_suite_cuda13.bat ^
  --input-root "<input-root>" ^
  --ocr-snapshot "<ocr-snapshot>" ^
  --results-root "<validation-log-root>" ^
  --baseline-container "<baseline-container>" ^
  --f16-container "<grouped-f16-container>" ^
  --q8-container "<grouped-q8-container>" ^
  --model-source "<iq4-xs-model>" ^
  --expected-model-sha256 "<64-hex-sha256>"
```

CUDA12 Python 환경으로 동일한 runner 계약을 점검하려면
`gemma_final_translation_suite_cuda12.bat`를 사용한다. 실제 GPU 후보 비교는
한 환경에서만 순차 실행한다.

## 사전검사만 실행

위 명령 끝에 `--preflight-only`를 추가한다. 입력, snapshot, 모델 hash,
container 계약까지만 검사하고 번역 요청은 보내지 않는다.

기본값에는 이 최종 비교의 입력 manifest SHA-256, OCR snapshot SHA-256,
후보별 runtime fingerprint와 hash helper image ID가 잠겨 있다. 다른 corpus나
runtime을 실수로 넘기면 구조가 22페이지·292블록이어도 중단한다.

## 재개

중단된 결과 디렉터리를 `--output-dir`로 지정하고 `--resume`을 추가한다.
이미 hard gate를 통과한 round/candidate는 다시 실행하지 않는다. 재개 시
입력·snapshot·모델·runtime fingerprint, runner와 번역·GPU 측정 코드,
prompt/schema/sampler, Docker engine·NVIDIA driver identity, 판정 인자가
달라지면 중단한다. 저장된 result 파일의 경로와 SHA-256, 292개 원문
hash·순서도 다시 검증한다.

## 산출물

- `summary.md`, `summary.json`: 속도와 구조 gate
- `blind_review.md`, `blind_review.csv`: 사용자 A/B/C 품질 검수
- `blind_key.json`: 검수 완료 전 비공개
- `runs/`: 원문과 번역을 포함한 raw 결과

모든 산출물은 validation log에만 두며 Git에 추가하지 않는다.
