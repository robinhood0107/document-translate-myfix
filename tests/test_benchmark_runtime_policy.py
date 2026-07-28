from __future__ import annotations

import sys
import tempfile
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
             mock.patch.object(
                 benchmark_common,
                 "existing_container_names",
                 side_effect=lambda names: list(names),
             ), \
             mock.patch.object(
                 benchmark_common,
                 "running_container_names",
                 side_effect=lambda names: list(names),
             ), \
             mock.patch.object(
                 benchmark_common,
                 "wait_for_health_urls",
                 side_effect=[
                     ["http://127.0.0.1:28118/docs"],
                     [],
                 ],
             ) as wait_for_health, \
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
            force_recreate=True,
        )
        self.assertEqual(wait_for_health.call_count, 2)

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

    def test_stopped_container_is_recreated_without_health_wait(self) -> None:
        groups = [
            {
                "name": "ocr",
                "container_names": ["paddleocr-server"],
                "health_urls": ["http://127.0.0.1:28118/docs"],
                "compose_path": ROOT
                / "paddleocr_vl_docker_files"
                / "docker-compose.yaml",
                "cwd": ROOT / "paddleocr_vl_docker_files",
            }
        ]

        with mock.patch.object(
            benchmark_common,
            "url_available",
            return_value=False,
        ), mock.patch.object(
            benchmark_common,
            "existing_container_names",
            return_value=["paddleocr-server"],
        ), mock.patch.object(
            benchmark_common,
            "running_container_names",
            return_value=[],
        ), mock.patch.object(
            benchmark_common,
            "wait_for_health_urls",
            return_value=[],
        ) as wait_for_health, mock.patch.object(
            benchmark_common,
            "compose_up_detached",
        ) as compose_up:
            report = benchmark_common.ensure_compose_groups_health_first(
                groups,
                quick_timeout_sec=1,
                boot_timeout_sec=30,
            )

        self.assertEqual(report[0]["action"], "restarted")
        self.assertEqual(wait_for_health.call_count, 1)
        compose_up.assert_called_once_with(
            groups[0]["compose_path"],
            cwd=groups[0]["cwd"],
            project_directory=None,
            force_recreate=True,
        )

    def test_staged_gemma_runtime_locks_no_spec_command(self) -> None:
        compose = {
            "services": {
                "gemma-local-server": {
                    "command": [
                        "-ctk",
                        "${LLAMA_CACHE_TYPE_K:-f16}",
                        "-ctv",
                        "${LLAMA_CACHE_TYPE_V:-f16}",
                        "--spec-type",
                        "${LLAMA_SPEC_TYPE:-ngram-mod}",
                        "--spec-draft-n-max",
                        "${LLAMA_SPEC_DRAFT_N_MAX:-16}",
                    ],
                    "volumes": [],
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            benchmark_common,
            "_python3_yaml_load",
            return_value=compose,
        ), mock.patch.object(
            benchmark_common,
            "_python3_yaml_dump",
        ) as yaml_dump, mock.patch.object(
            benchmark_common,
            "_resolve_testmodel_dir",
            return_value=Path(tmp),
        ):
            benchmark_common._stage_gemma_runtime(
                {
                    "gemma": {
                        "cache_type_k": "f16",
                        "cache_type_v": "f16",
                        "spec_type": "none",
                        "spec_draft_n_max": 8,
                    }
                },
                Path(tmp),
            )

        staged = yaml_dump.call_args.args[1]
        self.assertEqual(staged["name"], "comic-translate")
        command = staged["services"]["gemma-local-server"]["command"]
        self.assertEqual(command[command.index("-ctk") + 1], "f16")
        self.assertEqual(command[command.index("-ctv") + 1], "f16")
        self.assertNotIn("--cache-type-k", command)
        self.assertNotIn("--cache-type-v", command)
        self.assertEqual(
            command[command.index("--spec-type") + 1],
            "none",
        )
        self.assertEqual(
            command[command.index("--spec-draft-n-max") + 1],
            "8",
        )

    def test_staged_paddle_runtime_keeps_product_compose_project(
        self,
    ) -> None:
        compose = {
            "services": {
                "paddleocr-layout": {
                    "command": "--device cpu",
                    "depends_on": {},
                },
                "paddleocr-vllm": {},
            }
        }
        pipeline = {
            "SubModules": {
                "VLRecognition": {"genai_config": {}}
            }
        }
        vllm = {}
        loaded = [compose, pipeline, vllm]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            benchmark_common,
            "_python3_yaml_load",
            side_effect=loaded,
        ), mock.patch.object(
            benchmark_common,
            "_python3_yaml_dump",
        ) as yaml_dump:
            benchmark_common._stage_ocr_runtime({}, Path(tmp))

        staged_compose = yaml_dump.call_args_list[0].args[1]
        self.assertEqual(
            staged_compose["name"],
            "paddleocr_vl_docker_files",
        )


if __name__ == "__main__":
    unittest.main()
