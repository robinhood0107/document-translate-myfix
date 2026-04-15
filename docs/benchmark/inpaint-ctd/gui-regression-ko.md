# Inpaint CTD GUI Regression

| 항목 | 결과 | 근거 | 비고 |
| --- | --- | --- | --- |
| Tools UI에 CTD/AOT/LaMa runtime 노출 | PASS | `tests/test_settings_tools_runtime.py` | 저장/복원 포함 |
| 한국어 번역 반영 | PASS | `ct_ko.ts`, `ct_ko.qm`, `compile_translations.py` | 새 ToolsPage 문자열 기준 |
| One-Page Auto GUI 경유 실행 | PASS | spotlight benchmark 5-way | AOT/lama_large_512px/lama_mpe 모두 완료 |
| Translate All GUI 경유 실행 | PASS | full suite 5-way | China 8장, japan 22장 완료 |
| export/debug 산출물 연결 | PASS | 로컬 전용 spotlight export 검수 | source/overlay/mask/cleaned/translated는 Git 외부에서 확인 |
| 프로젝트 재열기 후 export root 유지 | PASS | 기존 export root 회귀 수정 + benchmark 경로 확인 | Temp 저장 버그는 이미 수정됨 |
| 사람 직접 클릭 수동 회귀 | PENDING | - | 실제 운영자 검수는 별도로 남음 |

## 메모
- 여기의 PASS는 실제 GUI 파이프라인을 타는 benchmark/자동 검증 기준이다.
- 사람이 직접 눌러보는 최종 수동 회귀는 아직 별도 수행 전이다.
