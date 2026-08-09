# PR·커밋별 설계 기록

시간 순서다. 각 항목은 **증상 → 원인 → 설계 판단 → 일부러 하지 않은 것** 순으로 적는다.
검증 명령과 통과 수치는 각 PR 본문에 있으므로 여기서는 결론만 옮긴다.

| PR | 커밋 | 제목 |
| --- | --- | --- |
| [#278](https://github.com/robinhood0107/comic-translate/pull/278) | `20241db3`, `0da322eb` | 관리형 llama.cpp 볼륨을 이미지 다이제스트 드리프트에서 복구 |
| [#279](https://github.com/robinhood0107/comic-translate/pull/279) | `d6f00ce9` | 시리즈 자식 breadcrumb 과 보드 복귀 경로 |
| [#280](https://github.com/robinhood0107/comic-translate/pull/280) | `b637740e` | 앱 전역 설정 복원과 미반영 자식 편집 보존 |
| [#281](https://github.com/robinhood0107/comic-translate/pull/281) | `b78b0f8e` | 자식 자동저장을 GUI 스레드 밖으로, 큐 재로드 제거 |
| [#282](https://github.com/robinhood0107/comic-translate/pull/282) | `fed8799e`, `917e5724`, `3282e711`, `c50169b2`, `a86eeacb` | 배치의 메모리 고갈·페이지 손실 차단과 실행 기록 |
| [#283](https://github.com/robinhood0107/comic-translate/pull/283) | `ee3b3d27` | 컨테이너 기동·모델 적재를 ETA 에 포함 |
| [#284](https://github.com/robinhood0107/comic-translate/pull/284) | `27ef27c1` | 실제 366장 실행이 드러낸 세 결함 |
| [#285](https://github.com/robinhood0107/comic-translate/pull/285) | `31809444` | 렌더 스윕을 짐작하지 말고 측정 |
| [#286](https://github.com/robinhood0107/comic-translate/pull/286) | `616e18ee` | 앱 로그와 실행 리포트를 저장소 안으로 |
| [#287](https://github.com/robinhood0107/comic-translate/pull/287) | `f29119ea` | 텍스트 없는 페이지도 정상 출력, 작업 총량 정정 |
| [#288](https://github.com/robinhood0107/comic-translate/pull/288) | 본 PR | 융합 단계 ETA 교정과 파이프라인 상태 패널 |

---

## #278 — 관리형 런타임이 전혀 뜨지 않는다

`20241db3` `fix(runtime): recover managed llama.cpp volumes from image digest drift`
`0da322eb` `feat(runtime): register the official model sources for every managed runtime`

### 증상

Docker 에 수명 짧은 프로브 컨테이너 `comic-translate-gemma-cache-warm` 만 뜨고 실제
컨테이너는 올라오지 않았다. prefetch 는 8.9초에 끝나므로 문제가 아니었고, 다음 단계인
계약 검증에서 죽고 있었다.

```
GemmaRuntimeContractError: source_image_digest 불일치
  expected='sha256:4db23c61…'  (설치된 이미지)
  actual  ='sha256:22e0e3bf…'  (볼륨 매니페스트에 봉인된 값)
```

다섯 개 볼륨 전부 MISMATCH. 모델 파일은 전부 바이트 정상이었다.

### 원인

#275 의 회귀다. 기본 이미지를 고정 다이제스트에서 이동 태그 `:server-cuda13` 으로
바꾸면서 `source_image_ref` 검사는 완화했지만 `source_image_digest` 와
`source_image_id` 는 엄격 동일성으로 남겨뒀다. 업스트림 태그가 한 번 갱신되자
준비된 볼륨이 **동시에 전부** 무효화됐다.

복구 경로가 없었다. `Verify` 는 읽기 전용이라 같은 검사에서 예외를 던지고, `Prepare`
는 원본 멀티 GB 파일을 요구한다.

### 설계 판단

**`Reseal` / `Auto` 모드를 다섯 준비 스크립트 전부에 추가한다.** `Reseal` 은 볼륨 내용을
그대로 두고 모델 SHA-256 을 재검증한 뒤, 현재 이미지로 실제 GPU 스모크를 다시 돌리고
매니페스트만 다시 쓴다. 멀티 GB 복사도 재다운로드도 없다. `Auto` 는 볼륨 상태를 보고
`Prepare` 와 `Reseal` 중에 고른다.

**앱 안에서 자가 복구한다.** 계약이 실패하면 "봉인된 이미지 신원만 틀렸는가"를 판정하고,
그렇다면 대응 스크립트를 `Auto` 로 **한 번만** 돌린 뒤 재시도한다. 판정은 보수적이다 —
매니페스트가 스스로 기록한 이미지 신원으로 재검증했을 때 계약 전체가 통과할 때만
복구한다. 모델 해시 불일치나 스키마 위반은 그대로 실패시킨다. **믿을 수 없는 볼륨이
유효한 봉인을 받는 일은 없어야 한다.** 볼륨이 아예 없거나 불완전하면 같은 경로로
프로비저닝된다.

**모델 소스를 한 곳에서 해석한다.** 명시 경로 → 저장소의 gitignore 된 `testmodel/` 과
바로 아래 디렉터리 → 다운로드 캐시 → 등록된 소스(`-AllowDownload`) 순. 다운로드는
이어받기가 되고 검증 후 원자적 rename 하므로, 중단된 전송이 유효한 소스로 오인되지
않는다.

**공식 소스를 다섯 런타임 전부 등록한다.** 저장소 이름이 아니라 업스트림 LFS oid 를
계약 SHA-256 과 대조해 확인했다.

| 런타임 | 저장소 |
| --- | --- |
| Gemma | `Vastopian/gemma-4-26B-A4B-it-abliterated-GGUF` |
| HunyuanOCR | `ggml-org/HunyuanOCR-GGUF` |
| MangaLMM | `mradermacher/MangaLMM-GGUF` |
| PaddleOCR-VL | `PaddlePaddle/PaddleOCR-VL-1.6-GGUF` |
| PaddleOCR-VL Spotting | 동일 저장소. projector 는 공식 crop projector 에서 로컬 파생 |

### 도중에 드러난 잠복 결함 둘

**준비 스크립트의 CRLF.** PowerShell here-string 은 CRLF 를 실어 나르는데 현재 이미지의
`/bin/sh` 는 dash 라 첫 줄에서 `set: Illegal option -` 로 죽는다. 즉 **모든 매니페스트
기록과 볼륨 복사가 실패할 수밖에 없었다.** Docker 인자를 한 곳에서 정규화했다.

**다운로더가 Windows PowerShell 5.1 에서 실행 불가.** `System.Net.Http` 를 기본 로드하지
않아 한 바이트도 전송하기 전에 `Unable to find type [System.Net.Http.HttpClient]` 로
죽었다.

### 부수 정리

쓰이지 않는 HunyuanOCR 볼륨 두 개(`-models`, `-v1`, 약 2.6 GB 중복 GGUF)를 제거해
런타임당 볼륨을 정확히 하나로 맞췄다. 남은 `-v2` 는 전체 `-Mode Verify` SHA-256
재검사를 통과했다.

---

## #279 — 시리즈 자식에서 보드로 돌아갈 방법이 없다

`d6f00ce9` `feat(series): add a child breadcrumb bar with a board return path`

### 증상

시리즈 자식 프로젝트로 들어가면 보드 복귀 UI 진입점이 하나도 없다. 복귀 로직
(`request_show_board` / `request_back`)은 컨트롤러에 완성돼 있는데 연결된 UI 가 둘뿐이고
둘 다 조건부였다 — 보드의 `←` 버튼은 보드 위젯 안에 있어 문서 화면에서 안 보이고,
파이프라인 패널의 `Series Board` 는 큐 실행 중에만 뜬다.

같은 결함이 "지금 단일 프로젝트인가, 시리즈의 한 화인가"도 구분 불가능하게 만들었다.
유일한 단서인 창 제목은 커스텀 타이틀바가 폭에 따라 숨긴다.

### 설계 판단

문서 화면 헤더에 **자식 활성 시에만** 뜨는 컨텍스트 표시줄을 둔다. 표시줄의 존재
자체가 단일/자식 구분 신호가 되고, 시리즈 이름 링크가 복귀 경로가 된다.

갱신 지점을 `_set_series_window_title` 과 `_update_queue_runtime` 두 곳으로 한정했다.
둘 다 시리즈 컨텍스트가 바뀌는 지점이라 별도 훅이 필요 없다 — **새 상태를 만들지 않고
기존 상태 전이에 얹는다.**

보드 링크는 `show_board_during_queue` 로 라우팅한다. 큐 실행 중에도 보드는 열려야 하고,
그 함수가 자식 materialization 을 유지한 채 화면만 바꾼다. 뒤로 버튼만 큐 실행 중
잠그고 사유를 툴팁으로 노출한다.

미반영 상태는 창 제목 접미사 대신 뱃지로 노출한다. 제목은 숨겨질 수 있으므로 상태
신호를 실을 자리가 아니다.

---

## #280 — 시리즈를 한 번 열면 앱 기본 설정이 바뀐 채 남는다

`b637740e` `fix(series): restore app globals and stop losing unsynced child edits`

### 1. 전역 설정이 복원되지 않았다

`_apply_series_globals_to_main` 은 OCR·workflow mode·translator·GPU·export·render 설정을
설정 페이지 위젯에 그대로 써 넣는데, `reset_series_context` 에 되돌리는 코드가 없었다.

시리즈 값이 위젯을 덮기 **전에** 스냅샷을 한 번만 찍고 컨텍스트 해제 때 복원한다.
스냅샷 스키마를 시리즈 `global_settings` 와 맞춰 복원이 같은
`_apply_global_settings_to_main` 경로를 재사용하게 했다.

두 번째 호출에서 다시 찍으면 이미 들어간 시리즈 값을 "원래 값"으로 오인하므로 최초
1회로 제한하고, **그 계약을 테스트로 고정했다.**

### 2. 미반영 자식 변경이 조용히 사라질 수 있었다

`_show_board` 는 작업 디렉터리를 지우고 `set_project_clean()` 으로 dirty 표시까지 없앤다.
`push_history=True` 경로는 가드를 타지만 큐 종료·실패·일시정지 경로의
`_show_board(push_history=False)` 는 가드가 없었다. 게다가
`on_batch_process_finished` 의 sync 실패는 로그도 없이 `return` 했다 — 방금 번역한
결과가 반영되지 않았다는 사실을 사용자도 로그도 알 수 없었다.

작업본을 버리기 전에 미반영 변경을 반영하고, 실패하면 **작업 디렉터리를 남긴 채**
경로와 함께 경고한다. 지우면 복구가 불가능하므로 실패 시 보존이 기본이다.

### 3. `project_kind` 하나로 두 질문에 답하고 있었다

자식을 열어도 `project_kind` 는 `PROJECT_KIND_SERIES` 로 고정된다. "시리즈 프로젝트인가"와
"지금 자식을 편집 중인가"는 다른 질문이고 저장 분기가 필요로 하는 것은 후자다.
`ProjectController._series_child_is_active` 로 분리했다. 동작 변경은 없고 의도를 코드에
드러내는 변경이다.

---

## #281 — 자동저장이 GUI 스레드를 멈춘다

`b78b0f8e` `perf(series): move child autosave off the GUI thread and cut queue reloads`

### 1. 자동저장이 메인 스레드에서 전체 저장을 동기 실행했다

`_autosave_series_project` 는 워커를 띄우기 **전에** `sync_active_child_to_series` 를 동기
호출했고 그 안에서 `save_state_to_proj_file` 이 통째로 돌았다. 일반 프로젝트 자동저장은
저장 자체를 워커로 던지는데 시리즈만 메인 스레드였다.

동기화를 세 단계로 쪼갰다.

| 단계 | 스레드 | 하는 일 |
| --- | --- | --- |
| `prepare_active_child_sync` | 메인 | `save_current_state` 로 UI 상태 수집 |
| `write_active_child_sync` | 워커 | 자식 직렬화 + 시리즈 임베드 |
| `finalize_active_child_sync` | 메인 | 매니페스트 재로드와 UI 반영 |

**옮기려면 전역 조작부터 없애야 했다.** 예전 코드는 `main.project_file` 을 자식 경로로
바꿔치기했다 되돌렸는데, 워커에서 그러면 메인 스레드가 그 사이 잘못된 값을 읽는다.
v2 writer 에 `source_project_file` 인자를 추가해 전역 대신 값으로 넘긴다. 인자를 생략하면
종전 동작 그대로라 기존 호출자는 영향이 없다.

기존 동기 API 는 세 단계를 순서대로 부르는 래퍼로 남겨 명시적 저장 경로를 보존했다.

### 2. 자동저장이 꺼져 있어도 원본 시리즈 파일을 썼다

복구 스냅샷 경로에서는 원본을 복사한 뒤 **사본에만** 반영하고 원본은 건드리지 않는다.
사본만 쓴 경우 자식은 미반영 상태로 남으므로 `finalize` 도 건너뛴다.

### 3. 큐 상태 변경마다 시리즈 파일을 두 번 읽었다

`update_series_queue_runtime` 이 이미 갱신된 매니페스트를 돌려주는데 호출자가
`load_series_project` 로 전체를 다시 읽었다. 큐 런타임은 manifest 필드라 items 는
바뀌지 않는다. 재로드를 없앴다.

**이 과정에서 sentinel 결함이 드러났다.** 컨트롤러가 상태 모듈과 **다른** `object()` 를
`_UNSET` 으로 쓰고 있어, 인자를 생략하면 저쪽에서 "빈 값이 주어졌다"로 해석한다.
지금까지 호출자가 전부 모든 인자를 나열해서 드러나지 않았을 뿐이고, 부분 호출을
시도하면 `TypeError` 가 난다. `SERIES_FIELD_UNSET` 공개 별칭으로 통일했다.

---

## #282 — 366장을 넣었는데 347장이 나왔다

`fed8799e` `fix(pipeline): stop the batch from running itself out of memory`
`917e5724` `fix(pipeline): never drop a page from the output because a stage failed`
`3282e711` `fix(ocr): keep a truncated bubble from failing its whole page`
`c50169b2` `feat(pipeline): record how long a run actually took and what each page did`
`a86eeacb` `docs(runtime): document where run logs and reports land`

### 1. 배치가 스스로를 메모리 고갈로 몰았다

4K 페이지 인페인팅에서 두 종류로 죽었다.

```
Unable to allocate 190. MiB … shape (2160, 3840, 3) float64
Unable to allocate 23.7 MiB … shape (2160, 3840, 3) uint8
```

**23.7 MiB 할당 실패는 하나의 연산이 큰 게 아니라 프로세스가 이미 한계였다는 뜻이다.**
원인은 둘이었다.

**페이지 버퍼가 해제되지 않았다.** `StagePageContext` 는 원본·인페인팅 결과·마스크
둘·패치를 들고 있고 스윕은 모든 페이지의 컨텍스트를 실행 내내 살려둔다. 4K 페이지
하나가 적재부터 배치 종료까지 60 MiB 를 붙잡았다. 이제 렌더 정산이 끝나면 해제하고,
실패 페이지는 실패로 표시되는 즉시 해제한다. 배열보다 오래 살아야 하는 유일한 집계인
마스크 픽셀 수는 먼저 복사해 둔다.

**인페인팅 블렌드가 페이지 전체를 float64 로 승격시켰다.**

```python
result = result * (mask / 255) + image * (1 - (mask / 255))
```

uint8 마스크에 `mask / 255` 는 float64 가 된다. 전체 크기 임시본을 여섯 개 만들고
반환값도 float64 라 하류로 8배 크기가 전파됐다. 크래시가 난 그 페이지에서 측정:
**569.5 MiB → 53.1 MiB.** 대체 구현은 float32 로, 목표 바이트 수에 맞춘 행 블록 단위로
블렌드한다. 작업 메모리가 1024행부터 16384행까지 **29.3 MiB 로 평평하다.** 소프트
마스크 동작은 보존된다.

같은 경로의 작은 범인 둘: `count_changed_outside_edit_mask` 가 int16 으로 승격
(150.3 → 41.2 MiB), `normalize_edit_mask` 가 `np.where(arr > 0, 255, 0)` 의 파이썬 int
리터럴 때문에 uint8 결과를 위해 전체 크기 int64 배열을 만들어 4K 페이지당 66 MiB.

**크기 제한은 두지 않았다.** 경계는 임시본에만 건다.

### 2. 단계 실패가 페이지를 출력에서 조용히 지웠다

빠진 19장은 파일도 사유도 남기지 않았다. 잘린 OCR 을 거부하는 것은 의도된 올바른
동작이지만, 예외가 페이지 전체를 실패로 표시했고 `ctx.failed_stage` 는 스윕의 열 군데를
게이트한다. 마지막이 렌더 제출이라 아무것도 쓰지 않고 반환했다. 레거시 폴백
`skip_save` 는 오래전부터 빈 함수였고 단계 배치 파이프라인은 부르지도 않았다.

**렌더 스윕 뒤, 배치 완료 표시 전에 조정 패스를 넣었다.** 기록된 출력이 없는 페이지는
가용한 최선의 이미지 — 인페인팅 결과, 없으면 원본, 버퍼가 이미 해제됐으면 디스크에서
재적재 — 로 **같은 export 경로를 통해** 쓴다. 아카이브 페이지 번호가 어긋나면 안 되기
때문이다. 모든 폴백은 페이지 요약·이벤트·로그에 남는다. 그래도 못 쓰는 페이지가
있으면 입출력 개수와 누락 파일명을 배치 리포트에 남기고, **조용히 끝내지 않는다.**

### 3. 잘린 말풍선 하나가 페이지 전체를 실패시켰다

`finish_reason == "length"` 는 전용 `PaddleDirectOcrTruncatedError` 를 던진다(여전히
`LocalServiceResponseError` 하위라 기존 핸들러는 영향 없음). 절단은 원인이 분명하므로
토큰 예산 3배(상한 4096)로 한 번 재시도하고, 이미 상한이면 재시도하지 않는다. 재시도도
잘리면 **그 블록만** 비우고 페이지는 계속한다.

### 4. 로깅 — 위 셋이 진단 불가능했던 이유

`logging.basicConfig` 에 핸들러가 없어 전부 stderr 로 갔고 콘솔 없이 실행하면 사라졌다.
출력물 옆 실행별 로그 디렉터리는 만들어지기만 하고 비어 있었다. 회전 파일 핸들러와
`sys.excepthook` / `threading.excepthook` 을 붙였다.

실행 리포트를 JSON·텍스트로 남긴다. **측정값만** 싣는다 — 파이프라인 시작이 아니라
**클릭 시점부터의** 전체 벽시계, 단계별 초·분과 비중, 입출력 개수, 폴백별 단계와 사유,
페이지별 초.

---

## #283 — 무거운 단계가 시작될 때마다 남은 시간이 위로 튄다

`ee3b3d27` `fix(progress): stop dropping container start and model load from the ETA`

### 원인

컨테이너 기동과 모델 적재가 **모델에 아예 없었다.** 단계 스윕 추정기는 페이지당 비용만
측정했고, 한 단계의 마지막 페이지와 다음 단계의 첫 보고 사이 간격 — 정확히 컨테이너
기동과 모델 적재 — 은 버려졌다. 단계가 바뀌면 페이지 시계를 `now` 로 리셋했기 때문에
그 시간은 아무 데에도 귀속되지 않았다.

죽은 코드 둘이 이를 확인해준다.

- `_runtime_swap_sec` 는 누적되기만 하고 `remaining_seconds()` 가 **읽지 않았다**
- `observe_runtime_swap` 은 제품 코드가 **부르지 않았다** — 테스트만 불렀다

가벼운 단계가 도는 동안 다음 단계의 기동 비용이 조용히 빠져 있다가, 그 단계가 시작되는
순간 통째로 나타난 것이다.

### 설계 판단

단계 간 간격을 **그 단계의 기동 비용**으로 잡고, `remaining_seconds()` 가 아직 시작하지
않은 단계의 기동 추정치를 더한다. 현재 단계의 기동은 이미 썼으므로 다시 더하지 않는다.

측정된 기동 비용은 페이지당 속도와 함께 내보내고 이력에서 씨앗으로 받는다. 첫 실행이
학습하고 다음 실행은 맞는 값으로 시작한다.

`observe_runtime_swap` 은 단계 경계가 아니라 단계 **내부**에서 일어나는 교체를 위해
남기되, 아무도 읽지 않던 카운터 대신 실행 중인 단계에 귀속시킨다.

결정적 테스트: 60초 기동 비용을 빼면 남은 시간이 정확히 60초 바뀐다.

---

## #284 — 실제 실행이 드러낸 세 결함

`27ef27c1` `fix(pipeline): close the three gaps a real 366-page run exposed`

366장이 전부 나왔지만 리포트를 보니 19장이 폴백으로만 디스크에 닿았고 단계 시간은
쓸 수 없는 값이었다.

**잘린 OCR 이 두 번째 작업 경로로 새어 나갔다.** 엔진에는 작업 처리기가 **둘** 있다 —
`_process_job` 은 일반 스윕, `_process_prepared_job` 은 영구 캐시 미스. 절단 봉쇄가 첫
번째에만 있어서 캐시 경로를 탄 실행은 여전히 페이지 전체를 실패시켰다. 로그가
정확히 그것을 보여줬다: 재시도는 동작하는데(`max_tokens=1024` → `3072`, 4페이지 × 2회)
봉쇄 메시지는 0회.

**같은 결함이 두 곳에 있으면 한 곳만 고치는 것은 고치지 않은 것과 같다.** 두 경로 모두에
봉쇄를 넣고 테스트로 고정했다.

---

## #285 — 렌더가 병목처럼 보였는데 계측 탓이었다

`31809444` `perf(pipeline): measure the render sweep instead of guessing at it`

### 잘못된 결론과 정정

렌더가 2.31초/페이지로 14분짜리 병목이라고 판단했다. **틀렸다.** 그 값은
`stage_work / count` 였고, 워커 시간과 큐 대기와 tail drain 이 한 바구니에 섞여 있었다.
실제는 **0.370초/페이지**이고 인페인팅 뒤에 완전히 가려져 있다.

계측을 `worker` / `queue_wait` / `tail_drain` 으로 분리한 뒤에야 이것이 보였다.

### 워커 증설을 권고하지 않은 이유

`scripts/verify_render_worker_scaling.py` 는 **정확성부터** 확인한다. 워커 수를 바꿔가며
렌더된 픽셀 합을 비교하고, 하나라도 다르면 증설을 권고하지 않는다. Qt 씬 렌더는 Qt
스레드가 아닌 곳에서 조용히 빈 이미지를 만들기 때문이다.

```
main-thread reference pixel sum: 1093865247
OK   workers=1     0.49s  0.030s/page  blank=0  identical=True
OK   workers=2     0.45s  0.028s/page  blank=0  identical=True
OK   workers=4     0.39s  0.024s/page  blank=0  identical=True
```

픽셀 동일, 빈 이미지 없음 — 조용한 실패는 나타나지 않았다. **그러나 이것은 속도 향상을
증명하지 않는다.** 합성 페이지는 0.03초에 렌더되는데 실제 페이지는 2.31초다. 텍스트
아이템 레이아웃과 PNG 인코딩을 건너뛰기 때문이고, 실제 비용은 거기에 있다. 실제 이득은
새 계측이 붙은 실행에서 워커 1개와 2개의 `tail_drain` 을 비교해야 나온다.

### 번역을 일부러 건드리지 않았다

번역은 이미 `translate/inference_and_cache` 와 `translate/dictionary_and_sanitizer` 로
나뉘어 있고 둘 다 이제 operations 표에 뜬다. **sampler·chunk 크기·동시성은 하나도
건드리지 않았다.** 그 값들은 출력 품질을 움직이고 이 브랜치에는 품질에 대한 증거가
없다. 분리 계측이 2.77초/페이지가 모델 추론인지 후처리인지 보여준 다음에 손댄다.

---

## #286 — 로그를 보려면 경로부터 물어봐야 했다

`616e18ee` `chore(logs): keep app logs and run reports inside the repository`

`%LOCALAPPDATA%\ComicTranslate\logs` 대신 저장소의 `logs/` 로 옮겼다. 그것이 설명하는
소스와 출력물 옆이다. `/logs/` 는 `.gitignore` 에 넣어 아무것도 추적하지 않는다.

**위치를 정하는 곳을 `get_log_dir()` 하나로 만들었다.** 이전에는 `comic.py`, `memlog.py`,
실행 리포트 세 곳이 각자 골랐는데, 그것이 정확히 다시 어긋나는 방식이다. 세 호출부가
모두 헬퍼를 쓰고 아무도 사용자 데이터 디렉터리에 손대지 않음을 테스트로 고정했다.

`COMIC_TRANSLATE_LOG_DIR` 로 재정의할 수 있다. 자기 파일 옆에 쓸 수 없는 설치본은 사용자
데이터 디렉터리로 폴백한다 — **로그 기록 실패가 앱을 멈춰서는 안 되기 때문이다.**
실제로 쓸 수 없는 경로로 그 폴백을 테스트한다.

---

## #287 — 텍스트 없는 페이지가 실패로 집계됐다

`f29119ea` `fix(pipeline): export text-free pages normally and make the work totals honest`

### B. 텍스트 없는 페이지가 렌더를 건너뛰었다

OCR 후 쓸만한 텍스트가 없다고 판정된 페이지는 인페인팅을 건너뛴다 — 옳다. 그런데
**렌더 제출도** 건너뛰었다 — 옳지 않다. 366장 중 15장이 배치 종료 폴백으로 떨어졌고,
아무것도 실패하지 않았는데 실패로 보고됐다.

**그릴 텍스트가 없다는 것은 페이지를 내주지 않을 이유가 아니다.** 이제 렌더를 제출한 뒤
계속하고, `continue` 앞에 제출이 오는 것을 테스트로 고정했다.

### A. 인페인팅 cleanup 추출 (오프로딩은 준비만)

페이지당 실측, 완전 순차:

| 구간 | 시간 | 장치 |
| --- | --- | --- |
| `mask_generation` | 1.335초 | **GPU** (CTD refiner) |
| `model_forward` | 1.309초 | **GPU** (LaMa) |
| `cleanup_and_composite` | 0.991초 | CPU |

**앞서 마스크 생성을 CPU 작업이라고 한 것은 틀렸다.** CTD 네트워크가 CUDA 에서 돈다
(`ctd_refiner.py:671`). 따라서 앞 두 구간을 겹쳐도 같은 장치를 두고 경합할 뿐이다.
CPU 구간은 cleanup 하나이고, 옆의 렌더 워커는 거의 놀고 있다(0.370초/페이지, 큐 대기
0.0002초). 옮길 수 있는 것은 cleanup 이다. 상한 효과는 42분 중 약 5.8분.

**이 PR 은 추출만 하고 동작은 바꾸지 않는다.** 계산이 `pipeline/inpaint_cleanup_job.py`
로 값 in / 값 out 형태로 이동하고, 진행 보고·마스크 외부 침범 가드·체크포인트 기록은
파이프라인 스레드에 남는다. **순서가 결과를 정하므로** 테스트가 추출된 작업이 원래
시퀀스와 여러 장면에서 **바이트 단위로 일치**하고, 마스크 밖 픽셀을 건드리지 않으며,
입력을 변형하지 않음을 확인한다.

오프로딩은 별도 변경으로 남겼다. 정확성 가드와 체크포인트 기록을 옮겨야 하고, 그것은
곁다리로 묻어갈 게 아니라 자기 리뷰를 받아야 한다.

읽다 발견: `refine_bubble_residue_inpaint` 는 쓰지 **않는** `inpainter` 인자를 받는다.
즉 cleanup 은 모델에 손대지 않는다. 그래서 옮기는 것이 애초에 안전하다.

### C. 작업 총량이 말이 안 됐다

총량이 여러 출처를 한 바구니에 담는 `telemetry["stages"]` 에서 왔다 — 42분 실행이
인페인팅 **4030분**을 보고했다. 이제 각 단계 자신의 operations 합으로 낸다.

| 단계 | 창(window) | 작업(work) |
| --- | --- | --- |
| 인페인팅 | 21.6분 | **21.3분** (이전 4030분) |
| 번역 | 17.5분 | 38.1분 |
| 렌더 | 0.0분 | 2.2분 |

렌더가 창 0.0분에 작업 2.2분인 것이 융합이 동작한다는 증거다 — 인페인팅 뒤에 완전히
가려진 시간이다.

---

## #288 — 융합 단계가 ETA 모델을 망가뜨렸다 (본 PR)

증상은 인페인팅 47장부터 327장까지 **280장이 처리되는 동안 남은 시간이 11~12분에 못
박혀** 있던 것이다. 원인 셋과 UI 변경은 [01-eta-model-ko.md](01-eta-model-ko.md) 에
따로 적는다. 요약하면:

- 단계 전환 시 직전 단계를 완료로 표시하던 로직이, 렌더가 인페인팅과 교대로 보고하는
  융합 파이프라인에서 인페인팅을 첫 보고에 366/366 으로 만들었다
- 융합된 렌더가 감싸는 단계와 같은 페이지당 속도로 측정돼 남은 시간을 두 배로 만들었다
- `remaining_seconds()` 가 단계 순서에 의존해, 순서를 벗어난 보고에서 어긋났다

파이프라인 상태 패널에는 **남은 시간과 전체 예상 시간을 함께** 띄우고, 시간 라벨에
마우스를 올렸을 때만 단계별 예상이 목록으로 뜨는 툴팁을 붙였다.
