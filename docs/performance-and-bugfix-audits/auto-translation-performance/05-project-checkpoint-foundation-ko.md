# 프로젝트 stage checkpoint 기반 명세

이 문서는 `.ctpr` 옆 sidecar에 detection·OCR·translation·inpaint·render 재사용 정보를 저장하기 위한 공통 기반 계약을 고정한다. 이 단계에서는 stage별 제품 연결을 시작하지 않으며 설정 기본값은 꺼짐이다.

## 저장 구조

```text
chapter01.ctpr
chapter01.ctpr.cache/
├── README.txt
├── checkpoint.sqlite3
└── objects/
    └── sha256/
        └── ab/
            └── <64자리 SHA-256>
```

- `.ctpr`에는 checkpoint schema version, project UUID, cache ID, 상대 sidecar 이름만 저장한다.
- 큰 mask·inpaint·render 산출물은 `.ctpr`나 SQLite BLOB에 넣지 않고 immutable content-addressed object로 저장한다.
- object는 같은 디렉터리의 `.partial` 파일을 완성·동기화한 뒤 `os.replace`로 공개한다.
- stage manifest는 필요한 object가 모두 존재하고 SHA-256이 일치한 뒤 SQLite 트랜잭션으로 기록한다.
- sidecar의 `README.txt`는 이 폴더가 재계산 가능한 로컬 캐시임을 명시한다.

## stage DAG

- `detection`
- `detection -> ocr -> translation`
- `detection -> inpaint`
- `translation + inpaint -> render`

upstream fingerprint가 바뀌면 해당 stage의 downstream record만 무효화한다. translation과 inpaint는 서로 직렬 의존하지 않으며 render에서 합류한다. 페이지 키를 지정한 수동 수정은 해당 페이지만 무효화할 수 있다.

## 프로젝트 수명주기

- 기존 v1/v2 `.ctpr`는 checkpoint 참조가 없어도 그대로 열린다.
- 참조가 없거나 잘못된 프로젝트는 새 identity를 메모리에만 만들고, 사용자가 다음에 저장할 때 새 참조를 추가한다.
- Save As는 새 project UUID와 cache ID를 만들고, 기존 immutable object를 같은 volume에서 hardlink한다. hardlink가 불가능하면 파일을 복사한다.
- 앱 내 Rename/Move는 identity를 유지한 target sidecar를 먼저 완성한다. 새 프로젝트 파일 저장과 기존 프로젝트 파일 삭제가 성공한 뒤에만 source sidecar를 제거한다.
- 덮어쓸 target sidecar는 임시 backup으로 보존하며 프로젝트 저장 실패 시 복구한다.
- source 또는 target sidecar가 손상되거나 잠겨 있으면 프로젝트 저장 자체는 계속하고 stage를 재계산한다.

## 안전·실패 계약

- sidecar 이름은 프로젝트 폴더 바로 아래의 상대 `.ctpr.cache` 이름만 허용한다.
- 경로 탈출, symlink sidecar, symlink object는 거부한다.
- 기존 DB의 schema, project UUID, cache ID가 참조와 다르면 DB를 수정·삭제하지 않고 해당 실행의 checkpoint만 끈다.
- 누락된 object는 cache miss로 처리하고 manifest를 자동 삭제하거나 고치지 않는다.
- DB lock, 손상, schema mismatch, JSON 손상은 프로젝트 열기·저장·정상 stage 계산을 막지 않는다.
- Rename 정리에서는 DB identity가 정확히 일치하는 sidecar만 삭제한다.

## 관리 기능

`Settings > Project`에서 다음 기능을 제공한다.

- 프로젝트 stage checkpoint 사용: 현재 구현 단계에서는 기본 꺼짐
- cache folder 열기
- manifest에서 참조하지 않는 object 정리
- 전체 stage manifest 무효화 후 다음 실행에서 강제 재계산

강제 재계산은 원본 페이지와 사용자 편집을 삭제하지 않는다. 무효화 뒤 남은 object는 별도 정리 작업에서만 제거한다.

## Stage 연결 상태

1. detection과 OCR checkpoint 연결 완료
2. project checkpoint miss일 때만 global exact OCR cache를 조회
3. translation은 기존 SQLite result cache를 중복 저장하지 않고 `.ctpr` 번역 상태의 fingerprint 유효성만 기록
4. inpaint는 final mask와 lossless cleaned artifact를 content-addressed object로 저장
5. render는 output SHA가 일치하면 생략하고, output이 없으면 검증된 encoded object만 원래 안전한 경로에 materialize
6. 통합 cold/all-hit 검증 전까지 설정 기본값은 계속 꺼짐

## 검증 게이트

- old v1/v2 프로젝트는 저장 전 파일 변경 없이 열린다.
- object atomic write와 manifest transaction 사이 실패에서 불완전 hit가 없어야 한다.
- Save As hardlink/copy와 Rename/Move rollback이 원본 sidecar를 보존해야 한다.
- 누락·손상·잠금·schema mismatch·path traversal에서 fail-open해야 한다.
- 한 stage 변경은 같은 페이지의 해당 stage와 downstream만 무효화해야 한다.
- `.venv-win`, `.venv-win-cuda13`, headless smoke, 번역 자산 검사를 모두 통과해야 한다.
