from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import imkit as imk
import numpy as np
from PySide6.QtWidgets import QApplication

from app.ui.canvas.save_renderer import ImageSaveRenderer
from pipeline.inpainting import InpaintingHandler
from pipeline.stage_batched_processor import StageBatchedProcessor, StagePageContext


class InpainterReleaseTests(unittest.TestCase):
    def test_targeted_release_includes_positive_claim_onnx_cache(self) -> None:
        handler = InpaintingHandler(SimpleNamespace())
        before = {
            "process": {"available": False},
            "driver": {"available": False, "primary": None},
        }
        gate = {
            "required": True,
            "measurement_available": True,
            "observed": True,
            "status": "observed",
        }
        positive_release = {
            "cache_entry_count": 1,
            "cuda_session_count": 1,
            "cpu_session_count": 0,
            "unknown_session_count": 0,
            "expected_process_reclaim_mb": 0.0,
            "untracked_gpu_resource_count": 1,
            "gpu_release_expected": True,
        }

        with mock.patch(
            "pipeline.inpainting.query_cuda_handoff_metrics",
            return_value=before,
        ), mock.patch(
            "pipeline.inpainting.release_source_lama_cache",
            return_value={
                "cache_entry_count": 0,
                "loaded_model_count": 0,
                "gpu_loaded_model_count": 0,
                "expected_process_reclaim_mb": 0.0,
                "untracked_gpu_resource_count": 0,
                "gpu_release_expected": False,
            },
        ), mock.patch(
            "pipeline.inpainting.release_ctd_positive_claim_cache",
            return_value=positive_release,
        ) as release_positive, mock.patch(
            "pipeline.inpainting.cleanup_python_cuda_memory",
            return_value={"gc_collected": 0, "errors": []},
        ), mock.patch(
            "pipeline.inpainting.wait_for_vram_release",
            return_value=gate,
        ) as wait_for_release:
            report = handler.release_inpainter_resources()

        release_positive.assert_called_once_with()
        self.assertEqual(report["positive_claim_release"], positive_release)
        self.assertTrue(report["gpu_release_expected"])
        self.assertEqual(report["untracked_gpu_resource_count"], 1)
        wait_for_release.assert_called_once_with(
            before,
            gpu_release_expected=True,
            expected_process_drop_mb=0.0,
            untracked_gpu_resource_count=1,
            driver_baseline=None,
            timeout_sec=5.0,
            poll_interval_sec=0.1,
            min_drop_mb=16.0,
        )

    def test_cpu_inpainter_does_not_acquire_gpu_lease(self) -> None:
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(settings_page=object())
        processor.inpainting = SimpleNamespace(
            _ensure_inpainter=mock.Mock(),
        )

        with mock.patch(
            "pipeline.stage_batched_processor.get_inpainter_runtime",
            return_value={"device": "cpu", "backend": "torch"},
        ):
            processor._ensure_inpainter()

        snapshot = processor._runtime_resource_arbiter().snapshot()
        self.assertIsNone(snapshot.active_model)
        self.assertEqual(snapshot.states, {})

    def test_cuda_inpainter_lease_is_held_until_verified_release(self) -> None:
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(settings_page=object())
        processor.inpainting = SimpleNamespace(
            _ensure_inpainter=mock.Mock(),
            release_inpainter_resources=mock.Mock(
                return_value={
                    "gpu_release_expected": True,
                    "vram_release_gate": {
                        "required": True,
                        "observed": True,
                        "status": "observed",
                    },
                }
            ),
        )
        processor._emit_benchmark_event = mock.Mock()

        with mock.patch(
            "pipeline.stage_batched_processor.get_inpainter_runtime",
            return_value={"device": "cuda", "backend": "torch"},
        ):
            processor._ensure_inpainter()

        self.assertEqual(
            processor._runtime_resource_arbiter().snapshot().active_model,
            "inpainter",
        )
        self.assertEqual(
            processor._runtime_resource_arbiter().snapshot().states[
                "inpainter"
            ],
            "model_ready",
        )
        processor._release_inpainter_before_render([])
        snapshot = processor._runtime_resource_arbiter().snapshot()
        self.assertIsNone(snapshot.active_model)
        self.assertEqual(snapshot.states["inpainter"], "stopped")

    def test_failed_inpainter_vram_gate_preserves_failed_gpu_lease(self) -> None:
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(settings_page=object())
        processor.inpainting = SimpleNamespace(
            _ensure_inpainter=mock.Mock(),
            release_inpainter_resources=mock.Mock(
                return_value={
                    "gpu_release_expected": True,
                    "vram_release_gate": {
                        "required": True,
                        "observed": False,
                        "status": "timeout",
                    },
                }
            ),
        )
        processor._emit_benchmark_event = mock.Mock()

        with mock.patch(
            "pipeline.stage_batched_processor.get_inpainter_runtime",
            return_value={"device": "cuda", "backend": "torch"},
        ):
            processor._ensure_inpainter()

        # 강제가 켜져 있을 때만 실패 lease 를 남기고 중단한다.
        with mock.patch(
            "pipeline.stage_batched_processor.gpu_release_enforcement_enabled",
            return_value=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "VRAM release was not confirmed",
            ):
                processor._release_inpainter_before_render(
                    [],
                )

        snapshot = processor._runtime_resource_arbiter().snapshot()
        self.assertEqual(snapshot.active_model, "inpainter")
        self.assertEqual(snapshot.states["inpainter"], "release_failed")

    def test_unobserved_release_drops_logical_lease_when_enforcement_is_off(self) -> None:
        processor = object.__new__(StageBatchedProcessor)
        processor.main_page = SimpleNamespace(settings_page=object())
        processor.inpainting = SimpleNamespace(
            _ensure_inpainter=mock.Mock(),
            release_inpainter_resources=mock.Mock(
                return_value={
                    "gpu_release_expected": True,
                    "vram_release_gate": {
                        "required": True,
                        "observed": None,
                        "status": "unavailable",
                    },
                }
            ),
        )
        processor._emit_benchmark_event = mock.Mock()
        with mock.patch(
            "pipeline.stage_batched_processor.get_inpainter_runtime",
            return_value={"device": "cuda", "backend": "torch"},
        ):
            processor._ensure_inpainter()

        with mock.patch(
            "pipeline.stage_batched_processor.gpu_release_enforcement_enabled",
            return_value=False,
        ):
            processor._release_inpainter_before_render([])

        snapshot = processor._runtime_resource_arbiter().snapshot()
        self.assertIsNone(snapshot.active_model)
        self.assertEqual(snapshot.states["inpainter"], "stopped")

    def test_targeted_release_preserves_materialized_edit_mask(self) -> None:
        handler = InpaintingHandler(SimpleNamespace())
        cached_model = SimpleNamespace()
        handler.inpainter_cache = SimpleNamespace(
            runtime_device="cuda",
            model=cached_model,
            session=None,
        )
        handler.cached_inpainter_key = "lama_large_512px"
        edit_mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        handler.last_inpaint_edit_mask = edit_mask
        before = {
            "process": {
                "available": True,
                "allocated_mb": 1024.0,
                "reserved_mb": 1280.0,
            },
            "driver": {"available": False, "primary": None},
        }
        gate = {
            "required": True,
            "measurement_available": True,
            "observed": True,
            "status": "observed",
        }

        with mock.patch(
            "pipeline.inpainting.query_cuda_handoff_metrics",
            return_value=before,
        ), mock.patch(
            "pipeline.inpainting.release_source_lama_cache",
            return_value={
                "cache_entry_count": 1,
                "loaded_model_count": 1,
                "gpu_loaded_model_count": 1,
                "gpu_release_expected": True,
            },
        ), mock.patch(
            "pipeline.inpainting.cleanup_python_cuda_memory",
            return_value={"gc_collected": 1, "errors": []},
        ), mock.patch(
            "pipeline.inpainting.wait_for_vram_release",
            return_value=gate,
        ) as wait_for_release:
            report = handler.release_inpainter_resources()

        self.assertIsNone(handler.inpainter_cache)
        self.assertIsNone(handler.cached_inpainter_key)
        self.assertIs(handler.last_inpaint_edit_mask, edit_mask)
        np.testing.assert_array_equal(handler.last_inpaint_edit_mask, edit_mask)
        self.assertTrue(report["gpu_release_expected"])
        self.assertEqual(report["vram_release_gate"], gate)
        self.assertEqual(
            report["python_native_cleanup"],
            {"gc_collected": 1, "errors": []},
        )
        wait_for_release.assert_called_once_with(
            before,
            gpu_release_expected=True,
            expected_process_drop_mb=0.0,
            untracked_gpu_resource_count=1,
            driver_baseline=None,
            timeout_sec=5.0,
            poll_interval_sec=0.1,
            min_drop_mb=16.0,
        )

    def test_stage_handoff_preserves_page_outputs_before_rendering(self) -> None:
        """인페인터를 내려도 렌더가 쓸 페이지 산출물이 그대로 남아야 한다.

        순서가 번역 → 인페인팅으로 바뀌면서 이 해제 다음은 번역이 아니라 렌더다.
        예전에는 여기서 Gemma 예열을 시작했지만 그 훅은 도달할 수 없게 되어 걷어냈다.
        """

        processor = object.__new__(StageBatchedProcessor)
        events: list[str] = []
        release_report = {
            "gpu_release_expected": True,
            "vram_release_gate": {
                "required": True,
                "observed": True,
                "status": "observed",
                "elapsed_sec": 0.2,
            },
        }
        processor.inpainting = SimpleNamespace(
            release_inpainter_resources=lambda: (
                events.append("release") or release_report
            )
        )
        processor._emit_benchmark_event = lambda *_args, **_kwargs: events.append("telemetry")
        processor._raise_if_cancelled = lambda: events.append("cancel-check")

        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        patch_image = image[:1, :1].copy()
        viewer_state = {"text_items_state": []}
        page = StagePageContext(
            image_path="example.png",
            image_name="example.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=image.copy(),
            inpaint_input_img=image,
            mask=mask,
            patches=[{"bbox": [0, 0, 1, 1], "image": patch_image, "order": 1}],
            blk_list=[
                SimpleNamespace(
                    text="source",
                    translation="",
                    text_class="text_bubble",
                )
            ],
        )
        image_before = image.copy()
        mask_before = mask.copy()
        patch_before = patch_image.copy()
        viewer_state_before = dict(viewer_state)

        app = QApplication.instance() or QApplication([])

        def render_page() -> bytes:
            renderer = ImageSaveRenderer(page.image.copy())
            renderer.apply_patches(page.patches)
            renderer.add_state_to_image(viewer_state)
            return renderer.render_to_image().tobytes()

        rendered_before = render_page()
        processor._release_inpainter_before_render([page])
        rendered_after = render_page()

        # 해제와 그 기록만 일어난다. Gemma 기동은 이제 이 경로에 없다.
        self.assertEqual(events, ["release", "telemetry"])
        self.assertIsNotNone(app)
        np.testing.assert_array_equal(page.inpaint_input_img, image_before)
        np.testing.assert_array_equal(page.mask, mask_before)
        np.testing.assert_array_equal(page.patches[0]["image"], patch_before)
        self.assertEqual(viewer_state, viewer_state_before)
        self.assertEqual(rendered_after, rendered_before)

    def test_failed_vram_gate_blocks_gemma_start(self) -> None:
        processor = object.__new__(StageBatchedProcessor)
        started = False
        processor.inpainting = SimpleNamespace(
            release_inpainter_resources=lambda: {
                "gpu_release_expected": True,
                "vram_release_gate": {
                    "required": True,
                    "observed": False,
                    "status": "timeout",
                    "elapsed_sec": 5.0,
                },
            }
        )
        processor._emit_benchmark_event = lambda *_args, **_kwargs: None

        def start_gemma() -> None:
            nonlocal started
            started = True

        processor._start_gemma_prewarm = start_gemma
        # 비차단 경로는 Gemma 기동까지 계속 진행하므로, 그 뒤에 필요한 최소 표면을
        # 더블에 갖춰 둔다.
        processor._raise_if_cancelled = lambda: None
        page = StagePageContext(
            image_path="example.png",
            image_name="example.png",
            source_lang="Japanese",
            target_lang="Korean",
        )

        # 기본값은 강건성 우선이다. VRAM 확인에 실패해도 게이트에서 멈추지 않고
        # 번역 단계로 넘어간다. (이 페이지에는 번역 블록이 없어 예열까지 가지 않는
        # 것이 정상이며, 여기서 확인하려는 것은 게이트가 중단시키지 않는다는 점이다.)
        processor._release_inpainter_before_render([page])

        # 강제를 켜면 예전처럼 차단한다.
        with mock.patch(
            "pipeline.stage_batched_processor.gpu_release_enforcement_enabled",
            return_value=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "VRAM release was not confirmed"
            ):
                processor._release_inpainter_before_render([page])
        self.assertFalse(started)

    def test_aborted_inpaint_releases_resources_without_starting_gemma(self) -> None:
        processor = object.__new__(StageBatchedProcessor)
        events: list[str] = []
        processor._inpaint_pages = lambda _pages: (_ for _ in ()).throw(
            RuntimeError("inpaint aborted")
        )
        processor._release_inpainter_before_render = (
            lambda _pages, **kwargs: events.append(
                f"release:{kwargs['handoff_outcome']}"
            )
        )
        page = StagePageContext(
            image_path="example.png",
            image_name="example.png",
            source_lang="Japanese",
            target_lang="Korean",
        )

        with self.assertRaisesRegex(RuntimeError, "inpaint aborted"):
            processor._inpaint_all([page])

        self.assertEqual(events, ["release:aborted"])

    def test_handoff_preserves_debug_patch_mask_and_final_png_hashes(self) -> None:
        processor = object.__new__(StageBatchedProcessor)
        release_report = {
            "gpu_release_expected": True,
            "vram_release_gate": {
                "required": True,
                "observed": True,
                "status": "observed",
            },
        }
        processor._emit_benchmark_event = lambda *_args, **_kwargs: None
        processor._raise_if_cancelled = lambda: None
        processor._start_gemma_prewarm = lambda: None

        image = np.full((64, 64, 3), 240, dtype=np.uint8)
        inpainted = image.copy()
        inpainted[8:24, 8:24] = [210, 220, 230]
        raw_mask = np.zeros((64, 64), dtype=np.uint8)
        raw_mask[8:24, 8:24] = 255
        final_mask = raw_mask.copy()
        edit_mask = raw_mask.copy()
        patch_image = inpainted[8:24, 8:24].copy()
        viewer_state = {
            "text_items_state": [
                {
                    "block_id": "fixed-translation",
                    "text": "고정 번역",
                    "font_family": "Arial",
                    "font_size": 10,
                    "text_color": "#111111",
                    "position": (28, 20),
                    "width": 30,
                    "height": 18,
                    "source_rect": (28, 20, 30, 18),
                    "block_anchor": (28, 20, 30, 18),
                }
            ]
        }
        page = StagePageContext(
            image_path="example.png",
            image_name="example.png",
            source_lang="Japanese",
            target_lang="Korean",
            image=image,
            inpaint_input_img=inpainted,
            raw_mask=raw_mask,
            mask=final_mask,
            patches=[
                {
                    "bbox": [8, 8, 16, 16],
                    "image": patch_image,
                    "order": 1,
                }
            ],
        )
        handler = InpaintingHandler(SimpleNamespace())
        handler.last_inpaint_edit_mask = edit_mask
        before_paths: list[Path] = []

        def release_resources():
            self.assertTrue(before_paths)
            self.assertTrue(all(path.is_file() for path in before_paths))
            return release_report

        processor.inpainting = SimpleNamespace(
            release_inpainter_resources=release_resources
        )
        _app = QApplication.instance() or QApplication([])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def materialize(prefix: str) -> dict[str, str]:
                arrays = {
                    "inpainted": page.inpaint_input_img,
                    "raw_mask": page.raw_mask,
                    "final_mask": page.mask,
                    "edit_mask": handler.last_inpaint_edit_mask,
                    "patch_0": page.patches[0]["image"],
                    "debug_overlay": np.repeat(page.mask[:, :, None], 3, axis=2),
                }
                paths: dict[str, Path] = {}
                for name, array in arrays.items():
                    path = root / f"{prefix}-{name}.png"
                    imk.write_image(str(path), array)
                    paths[name] = path
                renderer = ImageSaveRenderer(page.image.copy())
                renderer.apply_patches(page.patches)
                renderer.add_state_to_image(viewer_state)
                final_path = root / f"{prefix}-final.png"
                imk.write_image(str(final_path), renderer.render_to_image())
                paths["final"] = final_path
                if prefix == "before":
                    before_paths.extend(paths.values())
                return {
                    name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in paths.items()
                }

            before_hashes = materialize("before")
            processor._release_inpainter_before_render([page])
            after_hashes = materialize("after")

        self.assertEqual(after_hashes, before_hashes)


if __name__ == "__main__":
    unittest.main()
