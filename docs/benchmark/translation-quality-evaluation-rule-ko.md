# 번역 후보 품질 판정 규칙

## 원칙

raw 문자열 일치와 번역 품질은 다른 값이다. raw non-exact는 진단값으로 남기되,
원문·전체 대사 맥락·인접 대사·기준/후보 번역을 대조해 의미가 같고 민감한 표현이
보존되면 품질 PASS로 처리한다. 이 규칙은 번역 텍스트에만 적용하며 OCR, detection,
inpaint의 기존 exact 계약은 바꾸지 않는다.

## 의미 검수

검수는 private archive의 텍스트를 먼저 사용한다. 만화 전체를 읽지 않으며, 텍스트만으로
판정할 수 없을 때에만 해당 페이지 한 장을 추가로 확인한다. 공개 artifact에는 원문,
번역문, 페이지, local path를 넣지 않고 사용한 텍스트 범위·판정·분류와 hash/run identity만
기록한다.

다음 차이는 PASS가 될 수 있다.

- 같은 의미의 어순·어휘·문장부호·줄바꿈·말투 차이
- 장면의 관계와 위계를 바꾸지 않는 호칭 또는 표기 차이
- 의미가 보존된 이름 음역·등록 차이

다음 중 하나라도 candidate-only로 발생하면 REJECT다.

- 삭제, 검열, 성적·폭력적·기타 민감 표현의 순화 또는 약화
- 부정, 동의, 강제, 화자, 대상, 행동, 관계·위계, 숫자, 명시적 사실의 변화
- 전체 또는 인접 대사 맥락에서 다른 사건·관계로 읽히는 변화

끝까지 애매한 항목은 `REVIEW_REQUIRED`로 남기고 사용자 확인 전에는 속도 측정이나
승격에 사용하지 않는다. 필수 원문·기준·후보 텍스트가 없거나 response JSON/schema/
`finish_reason`가 불완전하면 의미 검수로 우회하지 않고 hard REJECT다.

## 공통 hard gate

- model/runtime identity, prompt/schema/seed, 요청 순서·개수
- JSON 완결성, `finish_reason=stop`, 누락·중복 요청 없음
- OOM 없음, runtime/container 안정성, orphan 없음, GPU 반환 확인

raw exact 여부와 mismatch 수는 보고서에 남기지만, 위 hard gate가 통과한 뒤에는 단독
승격/탈락 사유가 아니다.

## Turbo4 E2E 규칙

Turbo4에서 의미가 같은 다른 번역은 렌더된 최종 pixel을 바꿀 수 있다. 따라서 Turbo4
E2E는 detection/OCR/mask/inpaint snapshot exact와 render 완료·오류 없음을 hard gate로
쓴다. 각 arm의 HTTP request/response ledger는 순서·개수·seed·JSON/`finish_reason`을
hard gate로 검사하고, raw translation delta는 해당 run identity에 묶인 text-first 검수로만
허용한다. 후보 간 최종 decoded pixel SHA는 진단값일 뿐이다. 번역 입력이 고정되는 후속
render/exact-GPU 최적화에서는 decoded pixel SHA exact를 계속 요구한다.

GPU background, Windows available RAM, shared GPU memory, WSL/container swap은 모든 arm에서
1초 단위로 기록·보고한다. 이 관측만으로 탈락시키지 않으며, 실제 E2E 속도 저하·OOM·불안정·
orphan·GPU 반환 실패는 즉시 REJECT다.

v1의 과거 결과는 당시 측정 기록으로만 보존한다. 새 의미 승인 기준의 근거로 자동
재사용하지 않는다.
