"""테스트는 사용자의 실제 설정을 잃게 만들지 않는다.

여러 테스트가 `QSettings("ComicLabs", "ComicTranslate")` 를 그대로 열고, 일부는 실제
윈도우를 만들어 `load_settings` / `save_main_page_settings` 경로를 태운다. Windows 에서
그 저장소는 `HKEY_CURRENT_USER\\Software\\ComicLabs\\ComicTranslate` 레지스트리다.
그래서 이 스위트를 한 번 돌리는 것만으로 사용자가 직접 맞춘 설정이 바뀌었다. 실제로
`text_rendering/font_family` 가 삭제되고 본문 색 두 개가 사라진 것을 확인했다.

형식을 바꿔 임시 파일로 돌리는 방법은 통하지 않는다. PySide6 의 두 인수 생성자는
`defaultFormat()` 이 `IniFormat` 이어도 레지스트리를 쓴다(실측 확인). 그래서 여기서는
세션 시작에 그 하위 키를 통째로 내보내고, 끝날 때 원래대로 되돌린다.

한계: pytest 가 강제 종료되면 복원이 돌지 않는다. 그때는 아래 백업 파일이 임시
디렉터리가 아니라 `banchmark_result_log/` 밖의 고정 경로에 남아 있어 손으로 되돌릴 수
있다. 근본 해결은 제품 코드가 테스트에서 다른 스코프를 쓰게 만드는 것이고, 그것은
별도 작업으로 둔다.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SETTINGS_KEY = r"HKCU\Software\ComicLabs\ComicTranslate"


def _reg(*args: str) -> subprocess.CompletedProcess:
    # 출력을 디코딩하지 않는다. `reg` 는 콘솔 코드페이지로 쓰기 때문에 UTF-8 로 읽으면
    # 한국어 Windows 에서 UnicodeDecodeError 가 난다. 우리는 종료 코드만 본다.
    return subprocess.run(
        ["reg", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


@pytest.fixture(scope="session", autouse=True)
def keep_the_user_settings_intact():
    """스위트가 사용자 설정을 바꿨다면 끝날 때 원래대로 되돌린다."""

    if sys.platform != "win32":
        yield
        return

    handle, backup = tempfile.mkstemp(
        prefix="comic-translate-settings-backup-", suffix=".reg"
    )
    os.close(handle)
    exported = _reg("export", SETTINGS_KEY, backup, "/y").returncode == 0
    try:
        yield
    finally:
        if exported:
            # 테스트가 새로 만든 키까지 정리하려면 지운 뒤 되돌려야 한다.
            _reg("delete", SETTINGS_KEY, "/f")
            _reg("import", backup)
        try:
            os.unlink(backup)
        except OSError:
            pass
