# OCR Combo Ranked Workflow

- execution_scope: `full-pipeline`
- quality_gate_scope: `OCR-only`
- gold_source: `human-reviewed`

## 실행 순서

1. China frozen winner manifest를 읽는다.
2. Japan smoke를 4후보 전부 실행한다.
3. Japan default compare를 4후보 전부 실행한다.
4. 각 engine의 tuning ladder를 끝까지 실행한다.
5. engine별 best preset을 고른다.
6. Japan benchmark winner를 하나 고른다.
7. Japan winner를 `cold 3회` final confirm 한다.
8. ranked report와 history snapshot을 생성한다.

## 판단 철학

- strict pass/fail로 중간 탈락시키지 않는다.
- catastrophic가 아닌 후보도 모두 ranking에 남긴다.
- 가장 높은 `quality_band`를 우선하고, 같은 밴드 안에서는 속도가 빠른 후보를 우선한다.
- mixed routing policy는 China frozen winner와 Japan ranked winner를 합쳐서 생성한다.
