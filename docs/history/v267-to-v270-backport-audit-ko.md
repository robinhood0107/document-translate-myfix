# `comic-translate 2.6.7 -> 2.7.0` 수동 백포트 검수 문서

## 요약

이 저장소는 upstream `comic-translate` `v2.6.7` 코드에서 출발했고, 이후 현재 fork 구조에 맞춰 별도 수정이 누적됐다.

이번 라운드의 `2.7.0` 업그레이드는 upstream를 직접 merge한 것이 아니라, `v2.6.7...v2.7.0` compare와 release notes를 기준으로 **필요한 기능만 수동으로 선별 이식**한 작업이다.

검수 포인트는 아래 두 가지다.

1. upstream `2.7.0`에서 무엇이 바뀌었는가
2. 그 중 현재 저장소에 무엇을 가져왔고, 무엇은 이미 있었고, 무엇은 의도적으로 제외했는가

## upstream `2.7.0` 변경점 기준 분류

### 이번에 가져온 항목

| upstream 항목 | 현재 저장소 반영 상태 | 비고 |
| --- | --- | --- |
| Configurable keyboard shortcuts | 반영 | 현재 앱 구조에 맞게 shortcuts controller + settings page로 이식 |
| PSD export | 반영 | PhotoshopAPI 기반 export 경로 도입 |
| PSD import | 반영 | 우리 exporter round-trip 중심으로 지원 |
| Chapter-aware export | 반영 | export dialog와 project export flow에 반영 |
| Rename/Move project files | 반영 | title bar popup + project controller 경로 변경 로직으로 반영 |
| Startup Home Copy Path / Delete File | 반영 | recent-project row context menu에 반영 |
| Multi-select text block formatting | 반영 | ctrl-click selection과 batch formatting 적용 |
| Undo Text Render as one undo step | 반영 | render macro 형태로 반영 |
| Unlimited extra context for custom translator | 반영 | custom translator일 때만 제한 해제 |
| Hebrew / Croatian target languages | 반영 | target language 목록에 추가 |
| Render Persian as RTL | 반영 | Hebrew, Arabic까지 포함해 RTL 판정 확장 |
| Title bar Snap Multitasking on Windows | 반영 | frameless window hit-test / caption drag 경로로 반영 |
| Improved webtoon reader | 반영 | visible-page 계산, on-demand placeholder, scene item 보존 쪽을 현재 구조에 맞게 선택 이식 |
| Fix duplication with text split between two images | 반영 | webtoon/list-view 개선 라운드에 포함해서 검수 |
| Claude 4.6 Sonnet label | 반영 | 사용자 표시 label만 갱신 |

### 이미 현재 저장소가 충족하고 있어서 별도 백포트하지 않은 항목

| upstream 항목 | 현재 상태 | 비고 |
| --- | --- | --- |
| Save the batch reports to the project file | 이미 충족 | 구현 방식은 upstream와 다르지만 현재 fork에 기능이 이미 있었음 |

### 의도적으로 그대로 가져오지 않은 항목

| 항목 | 제외 이유 |
| --- | --- |
| upstream 전체 파일 구조와 UI를 통째로 복원 | 현재 fork의 OCR/runtime/product 구조가 이미 크게 diverge되어 있어 위험 |
| upstream OCR/runtime 내부 구현 일괄 교체 | 현재 저장소의 PaddleOCR VL / Hunyuan / Gemma 경로가 더 앞서 있어 퇴보 위험이 큼 |
| `torch_autocast` 제품 반영 | 실험 결과가 merge 판단 기준을 넘지 못했고, 제품 tracked code에는 넣지 않음 |
| arbitrary external PSD 전체 완전 호환 | v1 범위에서는 우리 exporter가 만든 PSD schema round-trip 우선 지원 |

## 실제 이식 방식

이번 백포트는 아래 원칙으로 진행했다.

- upstream `2.7.0` 파일을 통째로 덮어쓰지 않음
- 기능 단위로 발췌해서 현재 fork 구조에 맞게 수동 이식
- 허브 파일은 최소 diff만 허용
- 현재 fork가 더 앞선 OCR/runtime/device/factory 영역은 되돌리지 않음

즉, 같은 기능이라도 upstream 구현을 그대로 복사한 것이 아니라:

- 현재 컨트롤러 구조에 맞는 adapter를 추가하고
- 현재 상태 저장 포맷과 viewer state에 맞게 변환하고
- 현재 번역/OCR/runtime 체계와 충돌하지 않도록 재배치했다

## 현재 저장소 기준 검수 포인트

### 기준선 기능

- shortcuts 설정/저장/복원
- PSD export/import round-trip
- chapter-aware export

### 저위험 기능

- Startup Home `Copy Path`, `Delete File`, missing-file cleanup
- custom translator extra context unlimited
- Hebrew/Croatian 노출
- Persian/Hebrew/Arabic RTL 렌더
- Undo Text Render macro
- Claude 4.6 label

### 중위험 기능

- multi-select text formatting
- title bar rename/move

### 고위험 기능

- webtoon reader 개선
- list view 성능 개선
- duplication fix
- Windows snap multitasking

## `torch_autocast` 처리 기록

`torch_autocast`는 별도 로컬 실험 트랙으로만 검토했고, 제품 tracked code에는 반영하지 않았다.

판단 이유:

- 현재 저장소의 tracked 제품 코드에는 `autocast` 변경이 없음
- 실험 결과상 엔진별 출력 차이와 속도/VRAM 변화가 안정적인 제품 채택 기준을 넘지 못함
- 따라서 이번 merge 범위에서는 완전히 제외

## 결론

현재 저장소의 `2.7.0` 업그레이드는 **upstream release를 그대로 복원한 것이 아니라**, 현재 fork 구조를 유지하면서 필요한 사용자 가치 기능만 고른 **수동 selective backport**다.

검수 시에는 아래만 확인하면 된다.

1. upstream `2.7.0`에서 언급된 사용자 기능이 현재 저장소에 필요한 범위만큼 반영되었는가
2. 현재 fork 고유의 OCR/runtime/product 구조를 훼손하지 않았는가
3. `torch_autocast` 같은 실험성 변경이 제품 diff에 섞이지 않았는가
