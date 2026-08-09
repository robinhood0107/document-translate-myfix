from __future__ import annotations

import inspect
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.utils.paths import get_log_dir, get_repo_root  # noqa: E402


REPO = Path(get_repo_root())


class LogDirLocationTests(unittest.TestCase):
    def test_logs_live_inside_the_repository(self) -> None:
        # 로그를 소스·산출물과 같은 곳에 둔다. %LOCALAPPDATA% 아래에 흩어져 있으면
        # 문제가 생겼을 때 경로부터 물어봐야 한다.
        self.assertEqual(Path(get_log_dir()), REPO / "logs")

    def test_subdirectories_are_created_under_the_same_root(self) -> None:
        self.assertEqual(Path(get_log_dir("runs")), REPO / "logs" / "runs")
        self.assertTrue((REPO / "logs" / "runs").is_dir())

    def test_an_explicit_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ, {"COMIC_TRANSLATE_LOG_DIR": tmp}, clear=False
            ):
                self.assertEqual(Path(get_log_dir("runs")), Path(tmp) / "runs")

    def test_an_unwritable_override_falls_back_rather_than_failing(self) -> None:
        # 로그를 못 써서 앱이 멈추면 안 된다. 파일 아래에는 디렉터리를 만들 수
        # 없으므로 이 경로는 확실히 실패한다.
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not-a-directory"
            blocker.write_bytes(b"x")
            with mock.patch.dict(
                os.environ,
                {"COMIC_TRANSLATE_LOG_DIR": str(blocker / "logs")},
                clear=False,
            ):
                resolved = Path(get_log_dir())

        self.assertTrue(resolved.is_dir())
        self.assertEqual(resolved, REPO / "logs")


class LogDirWiringTests(unittest.TestCase):
    def test_every_log_writer_uses_the_shared_helper(self) -> None:
        # 위치를 한 곳에서만 정한다. 세 군데가 각자 정하면 다시 흩어진다.
        import comic
        from modules.utils.memlog import MemLogger
        from pipeline.stage_batched_processor import StageBatchedProcessor

        for func in (
            comic._configure_file_logging,
            MemLogger._resolve_log_path,
            StageBatchedProcessor._write_run_report,
        ):
            with self.subTest(func=func.__qualname__):
                source = inspect.getsource(func)
                self.assertIn("get_log_dir", source)
                self.assertNotIn("get_user_data_dir", source)


class GitIgnoreTests(unittest.TestCase):
    def test_the_log_directory_is_ignored(self) -> None:
        ignored = subprocess.run(
            ["git", "check-ignore", "logs/runs/run_example.json"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(ignored.returncode, 0, ignored.stderr)

    def test_the_app_log_is_ignored(self) -> None:
        ignored = subprocess.run(
            ["git", "check-ignore", "logs/comic-translate.log"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(ignored.returncode, 0, ignored.stderr)


if __name__ == "__main__":
    unittest.main()
