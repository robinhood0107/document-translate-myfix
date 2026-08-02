from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_pipeline  # noqa: E402


class _Checkbox:
    def __init__(self) -> None:
        self.value = False

    def setChecked(self, value: bool) -> None:
        self.value = bool(value)

    def isChecked(self) -> bool:
        return self.value


class _TranslationMemoryPage:
    def __init__(self) -> None:
        self.values = {}

    def load_translation_memory_settings(self, values) -> None:
        self.values = dict(values)


class BenchmarkPipelineFinalizationHookTests(unittest.TestCase):
    def test_pagecache_command_is_read_only_low_priority_and_cpu_only(
        self,
    ) -> None:
        contract = SimpleNamespace(
            model_name="model.gguf",
            volume_name="comic-translate-gemma-models-v1",
            image_ref="example.invalid/llama@sha256:" + "a" * 64,
        )

        command = benchmark_pipeline._gemma_pagecache_command(
            contract,
            action="read",
            container_name="ct-prefetch-test",
        )

        self.assertEqual(command[:3], ["docker", "run", "--rm"])
        self.assertIn("--network", command)
        self.assertIn("none", command)
        self.assertIn("ct-prefetch-test", command)
        self.assertIn("ACTION=read", command)
        self.assertIn(
            "type=volume,source=comic-translate-gemma-models-v1,"
            "target=/models,readonly",
            command,
        )
        self.assertIn("/usr/bin/ionice", command)
        self.assertNotIn("--gpus", command)
        self.assertNotIn("down", command)

    def test_pagecache_command_rejects_unknown_action(self) -> None:
        contract = SimpleNamespace(
            model_name="model.gguf",
            volume_name="volume",
            image_ref="image",
        )
        with self.assertRaises(ValueError):
            benchmark_pipeline._gemma_pagecache_command(
                contract,
                action="drop-everything",
            )

    def test_completed_prefetch_does_not_accept_container_failure(self) -> None:
        prefetch = benchmark_pipeline._GemmaPagecachePrefetch(
            SimpleNamespace(),
            Path("unused"),
        )
        prefetch.started_at = 1.0
        prefetch.completed = SimpleNamespace(
            returncode=137,
            stdout="",
            stderr="unexpected exit",
        )
        thread = mock.Mock()
        thread.is_alive.return_value = False
        prefetch.thread = thread

        with self.assertRaises(RuntimeError):
            prefetch.finish()

    def test_cache_policy_controls_all_persistent_layers(self) -> None:
        dictionary_page = _TranslationMemoryPage()
        ui = SimpleNamespace(
            paddleocr_vl_persistent_cache_checkbox=_Checkbox(),
            project_checkpoint_enabled_checkbox=_Checkbox(),
            user_dictionaries_page=dictionary_page,
        )
        window = SimpleNamespace(
            settings_page=SimpleNamespace(ui=ui),
        )

        benchmark_pipeline._apply_benchmark_cache_policy(
            window,
            {
                "benchmark_cache_policy": {
                    "paddleocr_persistent": True,
                    "translation_persistent": False,
                    "exact_tm": False,
                    "project_checkpoint": True,
                }
            },
        )

        self.assertTrue(
            ui.paddleocr_vl_persistent_cache_checkbox.isChecked()
        )
        self.assertTrue(
            ui.project_checkpoint_enabled_checkbox.isChecked()
        )
        self.assertFalse(
            dictionary_page.values["persistent_cache_enabled"]
        )
        self.assertFalse(dictionary_page.values["exact_tm_enabled"])

    def test_http_experiment_replaces_modules_only_inside_context(self) -> None:
        from modules.ocr import ocr_paddle_VL as paddle_module
        from modules.translation.llm import custom_local_gemma as gemma_module

        original_gemma = gemma_module.requests
        original_paddle = paddle_module.requests

        with benchmark_pipeline._benchmark_http_clients(
            {
                "gemma_session": True,
                "paddle_thread_local_session": True,
                "pool_connections": 2,
                "pool_maxsize": 4,
            }
        ):
            self.assertIsNot(gemma_module.requests, original_gemma)
            self.assertIsNot(paddle_module.requests, original_paddle)
            self.assertTrue(callable(gemma_module.requests.post))
            self.assertTrue(callable(paddle_module.requests.post))
            self.assertIs(
                gemma_module.requests.exceptions,
                original_gemma.exceptions,
            )

        self.assertIs(gemma_module.requests, original_gemma)
        self.assertIs(paddle_module.requests, original_paddle)

    def test_http_experiment_injects_deterministic_gemma_seed_only_in_copy(
        self,
    ) -> None:
        from modules.translation.llm import custom_local_gemma as gemma_module

        original_gemma = gemma_module.requests
        payload = {"model": "example", "temperature": 0.7}
        with mock.patch.object(
            original_gemma,
            "post",
            return_value=SimpleNamespace(status_code=200),
        ) as post:
            with benchmark_pipeline._benchmark_http_clients(
                {"gemma_seed": 20260801}
            ):
                gemma_module.requests.post(
                    "http://127.0.0.1:18080/v1/chat/completions",
                    json=payload,
                )
            sent = post.call_args.kwargs["json"]

        self.assertEqual(sent["seed"], 20260801)
        self.assertNotIn("seed", payload)
        self.assertIs(gemma_module.requests, original_gemma)

    def test_http_experiment_records_gemma_requests_without_seed_or_session(
        self,
    ) -> None:
        from modules.translation.llm import custom_local_gemma as gemma_module

        original_gemma = gemma_module.requests
        response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            record_path = output_dir / "gemma-records.jsonl"
            with mock.patch.object(
                original_gemma,
                "post",
                return_value=response,
            ) as post, mock.patch.dict(
                os.environ,
                {"CT_BENCH_OUTPUT_DIR": str(output_dir)},
                clear=False,
            ):
                with benchmark_pipeline._benchmark_http_clients(
                    {"gemma_request_record_path": str(record_path)}
                ):
                    gemma_module.requests.post(
                        "http://127.0.0.1:18080/v1/chat/completions",
                        json={"model": "test"},
                    )

            self.assertEqual(post.call_count, 1)
            rows = [
                json.loads(line)
                for line in record_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(rows[0]["attempt_index"], 0)
        self.assertEqual(rows[0]["request"], {"model": "test"})
        self.assertEqual(rows[0]["status_code"], 200)
        self.assertEqual(
            rows[1],
            {"attempt_count": 1, "record_type": "summary", "write_error": ""},
        )
        self.assertIs(gemma_module.requests, original_gemma)

    def test_runtime_files_are_injected_only_inside_context(self) -> None:
        from modules.ocr import local_runtime as ocr_runtime_module
        from modules.translation import local_runtime as gemma_runtime_module

        original_gemma = gemma_runtime_module._RUNTIME_CONFIG[
            "compose_file"
        ]
        original_ocr = ocr_runtime_module._ENGINE_CONFIG[
            "PaddleOCR VL"
        ]["compose_file"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_dir = root / "run" / "runtime"
            runtime_dir.mkdir(parents=True)
            gemma_compose = runtime_dir / "gemma" / "docker-compose.yaml"
            ocr_compose = runtime_dir / "ocr" / "docker-compose.yaml"
            gemma_compose.parent.mkdir()
            ocr_compose.parent.mkdir()
            gemma_compose.write_text("services: {}\n", encoding="utf-8")
            ocr_compose.write_text("services: {}\n", encoding="utf-8")
            staged = {
                "gemma": {"compose_path": str(gemma_compose)},
                "ocr": {
                    "kind": "paddleocr_vl",
                    "compose_path": str(ocr_compose),
                },
            }
            with mock.patch.object(
                benchmark_pipeline,
                "stage_runtime_files",
                return_value=staged,
            ):
                with benchmark_pipeline._benchmark_runtime_files(
                    {"app": {"ocr": "PaddleOCR VL"}},
                    root / "run",
                ):
                    self.assertEqual(
                        gemma_runtime_module._RUNTIME_CONFIG[
                            "compose_file"
                        ],
                        gemma_compose,
                    )
                    self.assertEqual(
                        ocr_runtime_module._ENGINE_CONFIG[
                            "PaddleOCR VL"
                        ]["compose_file"],
                        ocr_compose,
                    )

        self.assertEqual(
            gemma_runtime_module._RUNTIME_CONFIG["compose_file"],
            original_gemma,
        )
        self.assertEqual(
            ocr_runtime_module._ENGINE_CONFIG[
                "PaddleOCR VL"
            ]["compose_file"],
            original_ocr,
        )

    def test_gemma_runtime_environment_is_explicit_and_restored(self) -> None:
        import os

        before = os.environ.get("LLAMA_SPEC_TYPE")
        before_parallel = os.environ.get("LLAMA_N_PARALLEL")
        try:
            os.environ["LLAMA_SPEC_TYPE"] = "ngram-mod"
            snapshot = benchmark_pipeline._apply_gemma_env(
                {
                    "n_parallel": 2,
                    "cache_type_k": "f16",
                    "cache_type_v": "f16",
                    "spec_type": "none",
                }
            )
            self.assertEqual(os.environ["LLAMA_N_PARALLEL"], "2")
            self.assertEqual(os.environ["LLAMA_SPEC_TYPE"], "none")
            benchmark_pipeline._restore_env(snapshot)
            self.assertEqual(
                os.environ.get("LLAMA_SPEC_TYPE"),
                "ngram-mod",
            )
            self.assertEqual(
                os.environ.get("LLAMA_N_PARALLEL"),
                before_parallel,
            )
        finally:
            if before is None:
                os.environ.pop("LLAMA_SPEC_TYPE", None)
            else:
                os.environ["LLAMA_SPEC_TYPE"] = before

    def test_shared_corpus_reuses_identical_file_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "page.png"
            source.parent.mkdir()
            source.write_bytes(b"same-image")
            shared = root / "shared"
            run_dir = root / "run"

            first = benchmark_pipeline._stage_selected_images(
                run_dir,
                [source],
                shared_corpus_dir=shared,
            )
            second = benchmark_pipeline._stage_selected_images(
                run_dir,
                [source],
                shared_corpus_dir=shared,
            )
            self.assertEqual(first, second)
            self.assertEqual(first[0].read_bytes(), b"same-image")

            first[0].write_bytes(b"different")
            with self.assertRaises(RuntimeError):
                benchmark_pipeline._stage_selected_images(
                    run_dir,
                    [source],
                    shared_corpus_dir=shared,
                )

    def test_controlled_project_invalidation_is_resume_only_and_scoped(
        self,
    ) -> None:
        window = object()
        with mock.patch(
            "app.projects.stage_checkpoints."
            "invalidate_project_page_checkpoints",
            return_value=4,
        ) as invalidate:
            removed = (
                benchmark_pipeline
                ._invalidate_project_checkpoint_for_benchmark(
                    window,
                    ["first.png", "second.png"],
                    project_action="resume",
                    page_index=1,
                    stage="ocr",
                )
            )

        self.assertEqual(removed, 4)
        invalidate.assert_called_once_with(
            window,
            "second.png",
            stage="ocr",
        )
        with self.assertRaises(ValueError):
            benchmark_pipeline._invalidate_project_checkpoint_for_benchmark(
                window,
                ["first.png"],
                project_action="create",
                page_index=0,
                stage="ocr",
            )

    def test_project_ui_restore_is_drained_before_pipeline(self) -> None:
        runner = SimpleNamespace(
            is_processing_queue=True,
            operation_queue=[object()],
        )
        window = SimpleNamespace(
            current_worker=object(),
            task_runner_ctrl=runner,
        )
        calls = 0

        def process_events() -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                window.current_worker = None
                runner.is_processing_queue = False
                runner.operation_queue.clear()

        app = SimpleNamespace(processEvents=process_events)
        benchmark_pipeline._wait_for_project_ui_idle(
            app,
            window,
            timeout_sec=0.5,
        )

        self.assertGreaterEqual(calls, 4)

    def test_project_ui_restore_timeout_is_explicit(self) -> None:
        runner = SimpleNamespace(
            is_processing_queue=True,
            operation_queue=[],
        )
        window = SimpleNamespace(
            current_worker=object(),
            task_runner_ctrl=runner,
        )
        app = SimpleNamespace(processEvents=lambda: None)

        with self.assertRaises(TimeoutError):
            benchmark_pipeline._wait_for_project_ui_idle(
                app,
                window,
                timeout_sec=0.02,
            )


if __name__ == "__main__":
    unittest.main()
