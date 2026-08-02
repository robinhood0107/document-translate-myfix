# 번역 후보 품질 판정 규칙

## 목적

LLM 번역의 raw 문자열 일치와 만화의 의미 품질은 서로 다른 결과다. 성능·상주
lab은 아래 네 축을 항상 분리해 기록한다. 한 축의 실패가 다른 축의 실패나
의미 회귀를 자동으로 뜻하지 않는다.

| 축 | PASS의 뜻 | FAIL 또는 미검수의 뜻 |
|---|---|---|
| 물리 동시 상주 | 두 runtime이 함께 health이고 actual VRAM peak·OOM·orphan·반환 계약을 통과함 | 해당 조합을 안전하게 함께 올릴 수 없음 |
| raw 응답 재현성 | 같은 fixed-seed 요청의 request/response ledger가 기준과 exact | 다른 계산 경로에서 문자열이 달라짐. 이것만으로 의미 회귀는 아님 |
| 의미 품질 | 원문·인접 대사·페이지 상황 기준으로 candidate-only 의미 회귀가 없음 | `미검수`는 PASS도 FAIL도 아님 |
| 속도 승격 | 적용된 재현성/품질 계약 뒤 AB/BA E2E 이득이 입증됨 | 제품 기본값으로 승격하지 않음 |

OCR raw 결과, detection geometry, inpaint/render decoded pixel은 기존 exact 계약을
계속 사용한다. 이 문서는 **번역 텍스트**의 품질 판정에만 적용한다.

## 의미 품질 기준

자연스러운 말투·어순·문장부호·호칭/register·의성어 표기 차이는 페이지의 관계와
상황이 유지되면 허용한다. 원문을 먼저 읽고, 인접 대사와 실제 만화 장면에 어울리는지를
함께 본다.

다음 중 하나가 candidate-only로 생기면 의미 회귀다.

- 민감한 대사·성적/폭력적 행위·상황의 삭제, 순화, 검열 또는 반대 의미화
- 화자, 관계·위계, 동의/강제, 부정, 행동·행위 방향, 대상, 숫자, 고유명사,
  명시적 사실의 변경
- 장면의 핵심 상황이나 만화 흐름을 다른 사건으로 보이게 만드는 번역

원문 OCR이 조각나 있거나 상황을 확정할 수 없으면 번역 후보의 회귀로 단정하지 않고
`REVIEW`와 별도 OCR/입력 이슈로 남긴다. 검수자는 전체 맥락이 유지되는 경우 직접
`PASS`로 승인할 수 있다.

## 실행 규칙

1. 요청 identity·순서·model·prompt·schema·sampler와 JSON 완결성을 먼저 확인한다.
   이 계약이 깨지면 hard reject다.
2. raw 응답이 다르면 `raw 재현성 FAIL`로 기록하고, 원문·기준·후보를 private review에
   나란히 두어 source-first 의미 검수를 한다.
3. 의미 검수의 PASS/REVIEW/FAIL과 raw 재현성 결과를 독립적으로 보고한다.
4. 해당 lab protocol이 raw exact를 속도 진입 조건으로 두었다면, 의미 PASS만으로 그
   조건을 우회하지 않는다. 조건을 바꾸려면 protocol·runner·test를 함께 바꾼다.

이 규칙은 기존 [Gemma blind 품질 검수](gemma-final-ab/workflow-ko.md)의 source-first 의미 기준을 재사용한다.
