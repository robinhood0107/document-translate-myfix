# CBZ/CBR Reader Integration

## Goal

`.cbz`와 `.cbr` 읽기 경로만 `cbz` 패키지로 보수화하고, 나머지 archive 형식과 `pdf` 경로는 그대로 유지한다.

## Scope

- 대상: `.cbz`, `.cbr`
- 비대상: `.pdf`, `.cb7`, `.cbt`, `.zip`, `.rar`, `.7z`, `.tar`, `.epub`
- 목적: 기존 lazy materialization 계약을 유지한 채 페이지 열거와 바이트 추출을 안정화

## Integration Rules

- 공용 진입점은 계속 `modules/utils/archives.py`를 사용한다.
- `.cbz`는 `ComicInfo.from_cbz(path)`로 로드한다.
- `.cbr`는 `ComicInfo.from_cbr(path)`로 로드한다.
- 페이지 바이트는 `page.content`를 사용한다.
- 엔트리 이름은 `page.name`, 확장자는 `page.suffix`를 사용한다.
- 앱이 기대하는 엔트리 구조는 계속 아래 형태를 유지한다.

```python
{
    "kind": "archive_entry",
    "entry_name": "...",
    "ext": ".png",
}
```

## Ordering and Compatibility

- 페이지 순서는 기존 앱 호환성을 우선한다.
- `FileHandler.prepare_files()`가 생성하는 lazy path 순서는 archive 리더 결과 순서를 그대로 따른다.
- project save/load, export directory resolution, archive source record 형식은 바꾸지 않는다.
- 메타데이터를 읽더라도 이번 단계에서는 UI나 project state에 저장하지 않는다.

## Failure Policy

- `.cbz/.cbr` 로드 실패 시 예외를 그대로 올려 상위 에러 흐름이 처리하게 한다.
- `pdf` 경로는 이번 변경에서 절대 건드리지 않는다.
- `rarfile`/외부 RAR 툴 요구사항은 기존과 동일하다.

## Validation

- `.cbz` fixture로 페이지 개수, 순서, `entry_name`, `ext`, byte materialization 검증
- `.cbr` adapter 단위 테스트
- 기존 `.pdf` 회귀 테스트
- 기존 기타 archive 포맷 회귀 테스트
