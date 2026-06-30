from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_no_gemma_replay_pipeline as replay  # noqa: E402


class _FakeCacheManager:
    def clear_ocr_cache(self) -> None:
        pass


class _FakePipeline:
    def __init__(self) -> None:
        self.stage_batched_processor = types.SimpleNamespace()
        self.cache_manager = _FakeCacheManager()
        self.batch_process_calls: list[list[str]] = []

    def batch_process(self, loaded_paths: list[str]) -> None:
        self.batch_process_calls.append(list(loaded_paths))

    def release_model_caches(self) -> None:
        pass


class _FakeWindow:
    last_instance: "_FakeWindow | None" = None

    def __init__(self) -> None:
        self.pipeline = _FakePipeline()
        self._current_batch_run_type = ""
        self._skip_close_prompt = False
        self.memlog_events: list[tuple[str, dict[str, object]]] = []
        _FakeWindow.last_instance = self

    def emit_memlog(self, event: str, **payload: object) -> None:
        self.memlog_events.append((event, dict(payload)))

    def close(self) -> None:
        pass


class NoGemmaRuntimeLifecycleTests(unittest.TestCase):
    def test_no_gemma_replay_lets_product_stage_batch_start_ocr_runtime(self) -> None:
        fake_controller = types.SimpleNamespace(ComicTranslate=_FakeWindow)
        fake_spec = replay.DatasetSpec(
            key="sample_japan",
            display_name="Sample/japan",
            snapshot_dir_name="sample_japan_product_stage_batch",
            source_kind="sample_japan",
            source_name="Sample/japan",
            source_lang="Japanese",
        )
        snapshot_pages = [{"image_name": "p_016.jpg", "blocks": []}]
        staged_path = Path("C:/tmp/no-gemma-runtime-test/p_016.jpg")

        with mock.patch.dict(sys.modules, {"controller": fake_controller}), \
            mock.patch.object(replay, "_load_snapshot_pages", return_value=snapshot_pages), \
            mock.patch.object(replay, "_stage_sources", return_value=[staged_path]), \
            mock.patch.object(replay, "resolve_product_benchmark_contract", return_value={
                "product_pipeline_entrypoint": True,
                "runner_render_mode": "product",
                "inpainter_family": "lama",
                "inpainter": "lama_large_512px",
                "mask_refiner": "ctd",
            }), \
            mock.patch.object(replay, "_settings_snapshot", return_value={}), \
            mock.patch.object(replay, "_restore_settings"), \
            mock.patch.object(replay, "_restore_env"), \
            mock.patch.object(replay, "_configure_window"), \
            mock.patch.object(replay, "_load_images", return_value=["loaded-p_016.jpg"]), \
            mock.patch.object(replay, "_write_page_snapshots", return_value=Path("page_snapshots.json")), \
            mock.patch.object(replay, "_audit_run", return_value={}), \
            mock.patch.object(replay, "render_summary_markdown", return_value="summary"), \
            mock.patch.object(replay, "_write_csv"), \
            mock.patch.object(replay, "write_json"):
            summary = replay._run_dataset(
                app=types.SimpleNamespace(processEvents=lambda: None),
                spec=fake_spec,
                source_root=Path("C:/unused"),
                previous_run_root=Path("C:/snapshots"),
                output_root=Path("C:/tmp/no-gemma-runtime-test"),
                base_preset={"app": {"inpainter": "lama_large_512px"}},
                use_gpu=True,
                match_threshold=0.25,
            )

        self.assertTrue(summary["product_pipeline_entrypoint"])
        self.assertEqual(_FakeWindow.last_instance.pipeline.batch_process_calls, [["loaded-p_016.jpg"]])
        self.assertFalse(hasattr(replay, "_ensure_managed_runtime"))


if __name__ == "__main__":
    unittest.main()
