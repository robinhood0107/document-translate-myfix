import os
import platform
from pathlib import Path


def get_repo_root() -> str:
    """Return the repository root for the running application."""
    return str(Path(__file__).resolve().parents[2])


def get_project_models_dir() -> str:
    """Return the repository-local models directory."""
    return os.path.join(get_repo_root(), "models")


def get_log_dir(*parts: str) -> str:
    """앱 로그를 둘 디렉터리. 기본은 저장소 안의 ``logs/``.

    로그를 산출물·소스와 같은 곳에 두면 문제가 생겼을 때 바로 찾을 수 있다.
    ``%LOCALAPPDATA%`` 아래에 흩어져 있으면 경로부터 물어봐야 한다. 이 폴더는
    ``.gitignore`` 대상이라 저장소를 오염시키지 않는다.

    설치형처럼 저장소 폴더에 쓸 수 없는 환경에서는 사용자 데이터 폴더로
    물러난다. 로그를 못 써서 앱이 멈추는 일은 없어야 한다.
    """

    candidates = [os.path.join(get_repo_root(), "logs")]
    override = str(os.environ.get("COMIC_TRANSLATE_LOG_DIR", "") or "").strip()
    if override:
        candidates.insert(0, override)
    candidates.append(os.path.join(get_user_data_dir(), "logs"))

    for base in candidates:
        try:
            target = os.path.join(base, *parts)
            os.makedirs(target, exist_ok=True)
        except (OSError, ValueError):
            # ValueError 는 경로에 null 문자가 섞인 경우다. 어느 쪽이든 이
            # 후보를 쓸 수 없다는 뜻이므로 다음 후보로 넘어간다.
            continue
        if os.access(target, os.W_OK):
            return target
    # 어디에도 못 쓰면 호출부가 자기 방식으로 실패하도록 기본 경로를 돌려준다.
    return os.path.join(candidates[-1], *parts)


def get_user_data_dir(app_name: str = "ComicTranslate") -> str:
    """
    Returns the platform-specific user data directory for the application.

    Windows: %LOCALAPPDATA%/<app_name>
    macOS: ~/Library/Application Support/<app_name>
    Linux: $XDG_DATA_HOME/<app_name> or ~/.local/share/<app_name>
    """
    system = platform.system()

    if system == "Windows":
        base_dir = os.getenv('LOCALAPPDATA')
        if not base_dir:
            base_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local")
    elif system == "Darwin":
        base_dir = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        # Linux / Unix
        base_dir = os.getenv('XDG_DATA_HOME')
        if not base_dir:
            base_dir = os.path.join(os.path.expanduser("~"), ".local", "share")

    return os.path.join(base_dir, app_name)


def get_default_project_autosave_dir(folder_name: str = "Comic Translate") -> str:
    """
    Returns a user-facing default folder for project auto-save files.

    Windows/macOS/Linux: ~/Documents/<folder_name>
    """
    return os.path.join(os.path.expanduser("~"), "Documents", folder_name)
