# Gemma 최종 A/B blind 품질 검수 사용법

## 1. 저장된 clean 결과 가져오기

source와 output은 Git에 추적되지 않는 로컬 검증 경로를 사용한다.

```powershell
.\.venv-win-cuda13\Scripts\python.exe -B `
  scripts\benchmark_gemma_final_ab.py import-clean `
  --source-suite "<validation-log-root>\gemma-final-translation-v3" `
  --output-dir "<validation-log-root>\gemma-final-ab-v4"
```

이 명령은 Docker와 모델 endpoint를 호출하지 않는다. source suite의
고정 계약과 일곱 결과 SHA-256을 검증하고, clean한 네 결과만 A/B 검수표로
가져온다.

## 2. 사용자 blind 검수

`blind_review.html`을 열어 292행의 A1·A2·B1·B2를 모두 판정한다.

- `회귀 없음`: 원문의 명시적 의미가 유지됨
- `의미 회귀`: 화자·관계·부정·행동·대상·숫자·고유명사·명시적 의미가
  달라짐

회귀가 있으면 유형을 선택하고 메모를 작성한다. 진행 상황은 브라우저의
local storage에 임시 보존되며, 완료 후 `완료 CSV 내보내기`를 누른다.
검수 완료 전에는 `private` 폴더를 열지 않는다.

## 3. 검수 완료 여부 확인

```powershell
.\.venv-win-cuda13\Scripts\python.exe -B `
  scripts\benchmark_gemma_final_ab.py validate-review `
  --suite-dir "<validation-log-root>\gemma-final-ab-v4" `
  --review-file "<review-export-root>\blind_review_completed.csv"
```

다음 중 하나라도 있으면 실패한다.

- 292행 미만 또는 초과
- 누락·중복·순서 변경
- 원문이나 A1·A2·B1·B2 수정
- 미검수 판정
- 허용되지 않은 회귀 유형
- 회귀 판정에 유형 또는 메모가 없음

## 4. 검수 완료 뒤 unblind

```powershell
.\.venv-win-cuda13\Scripts\python.exe -B `
  scripts\benchmark_gemma_final_ab.py unblind `
  --suite-dir "<validation-log-root>\gemma-final-ab-v4" `
  --review-file "<review-export-root>\blind_review_completed.csv" `
  --confirm-user-review "292-ROWS-REVIEWED"
```

불완전한 CSV, 바뀐 immutable 열, 잘못된 확인문에서는 unblind 결과를
생성하지 않는다. grouped 후보의 네 출력 중 하나라도 의미 회귀로
판정되면 `quality_rejected`로 끝나며 제품 승격을 허용하지 않는다.
`quality_approved`는 다음 제품 브랜치 구현을 시작할 수 있다는 뜻이며,
최종 승격에는 별도의 전체 파이프라인 비교가 필요하다.

## 로컬 산출물

- `blind_review.html`: 브라우저 검수 화면
- `blind_review.csv`: 직접 편집할 수 있는 초기 검수표
- `review_instructions.md`: 판정 기준
- `protocol_v4_state.json`: mapping-neutral 상태
- `private/blind_key.json`: 검수 완료 전 비공개
- `private/blind_payload.json`: immutable 비교 기준
- `private/source_import_audit.json`: source 계약 재검증 기록
- `review_validation.json`: 검수 완결성 검사 결과
- `unblind_summary.json`, `unblind_summary.md`: 완료 뒤에만 생성
