from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_common  # noqa: E402
import benchmark_suite  # noqa: E402


class BenchmarkRuntimePolicyTests(unittest.TestCase):
    def test_ensure_compose_groups_health_first_reuses_healthy_group(self) -> None:
        groups = [
            {
                "name": "gemma",
                "container_names": ["gemma-local-server"],
                "health_urls": ["http://127.0.0.1:18080/health"],
                "compose_path": ROOT / "docker-compose.yaml",
                "cwd": ROOT,
            }
        ]

        with mock.patch.object(benchmark_common, "url_available", return_value=True) as url_available, \
             mock.patch.object(benchmark_common, "compose_up_detached") as compose_up_detached:
            report = benchmark_common.ensure_compose_groups_health_first(groups, quick_timeout_sec=5)

        self.assertEqual(report[0]["action"], "reused")
        url_available.assert_called_once()
        compose_up_detached.assert_not_called()

    def test_ensure_compose_groups_health_first_restarts_only_failed_group(self) -> None:
        groups = [
            {
                "name": "gemma",
                "container_names": ["gemma-local-server"],
                "health_urls": ["http://127.0.0.1:18080/health"],
                "compose_path": ROOT / "docker-compose.yaml",
                "cwd": ROOT,
            },
            {
                "name": "ocr",
                "container_names": ["paddleocr-server", "paddleocr-vllm"],
                "health_urls": ["http://127.0.0.1:28118/docs"],
                "compose_path": ROOT / "paddleocr_vl_docker_files" / "docker-compose.yaml",
                "cwd": ROOT / "paddleocr_vl_docker_files",
            },
        ]

        def _url_available(url: str, timeout_sec: int = 8) -> bool:
            return "18080" in url

        with mock.patch.object(benchmark_common, "url_available", side_effect=_url_available), \
             mock.patch.object(benchmark_common, "wait_for_url") as wait_for_url, \
             mock.patch.object(benchmark_common, "compose_up_detached") as compose_up_detached:
            report = benchmark_common.ensure_compose_groups_health_first(
                groups,
                quick_timeout_sec=5,
                boot_timeout_sec=30,
            )

        self.assertEqual([item["action"] for item in report], ["reused", "restarted"])
        compose_up_detached.assert_called_once_with(
            ROOT / "paddleocr_vl_docker_files" / "docker-compose.yaml",
            cwd=ROOT / "paddleocr_vl_docker_files",
            project_directory=None,
        )
        wait_for_url.assert_called_once_with("http://127.0.0.1:28118/docs", timeout_sec=30)

    def test_restore_runtime_skips_when_attach_running_is_healthy(self) -> None:
        snapshot_dir = ROOT / "tmp" / "snapshot"
        with mock.patch.object(benchmark_suite, "url_available", return_value=True), \
             mock.patch.object(benchmark_suite, "ensure_compose_groups_health_first") as ensure_groups:
            benchmark_suite._restore_runtime(snapshot_dir)

        ensure_groups.assert_not_called()

    def test_restore_runtime_uses_health_first_group_recovery(self) -> None:
        snapshot_dir = ROOT / "tmp" / "snapshot"
        with mock.patch.object(benchmark_suite, "url_available", return_value=False), \
             mock.patch.object(benchmark_suite, "ensure_compose_groups_health_first") as ensure_groups:
            benchmark_suite._restore_runtime(snapshot_dir)

        self.assertEqual(ensure_groups.call_count, 1)
        groups = ensure_groups.call_args.args[0]
        self.assertEqual([group["name"] for group in groups], ["gemma", "ocr"])
        self.assertEqual(groups[0]["compose_path"], snapshot_dir / "docker-compose.yaml")
        self.assertEqual(
            groups[1]["compose_path"],
            snapshot_dir / "paddleocr_vl_docker_files" / "docker-compose.yaml",
        )


if __name__ == "__main__":
    unittest.main()
