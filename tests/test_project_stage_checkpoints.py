from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from app.projects.checkpoint_store import (
    ProjectCheckpointStore,
    checkpoint_reference_for_save,
)
from app.projects.stage_checkpoints import (
    RenderCheckpointResult,
    apply_translation_checkpoint,
    build_detection_fingerprint,
    build_detection_identity,
    build_inpaint_fingerprint,
    build_inpaint_identity,
    build_project_ocr_fingerprint,
    build_project_ocr_identity,
    build_render_fingerprint,
    build_render_identity,
    build_skipped_stage_fingerprint,
    build_translation_fingerprint,
    build_translation_identity,
    decoded_image_sha256,
    detection_structure_signature,
    invalidate_project_page_checkpoints,
    lookup_inpaint_checkpoint,
    lookup_detection_checkpoint,
    lookup_ocr_checkpoint,
    lookup_render_checkpoint,
    lookup_translation_checkpoint,
    materialize_render_checkpoint_output,
    open_project_stage_checkpoint_store,
    project_checkpoint_page_key,
    record_inpaint_checkpoint,
    record_detection_checkpoint,
    record_ocr_checkpoint,
    record_render_checkpoint,
    record_translation_checkpoint,
    registered_inpainter_model_identity,
    restore_inpaint_block_state,
    snapshot_project_render_blocks,
    snapshot_project_translations,
)
from modules.ocr.persistent_cache import snapshot_raw_ocr_result
from modules.utils.download import ModelDownloader, ModelID, ModelSpec
from modules.utils.textblock import TextBlock


class ProjectStageCheckpointTests(unittest.TestCase):
    def _store(
        self,
        root: str,
    ) -> tuple[Path, object, ProjectCheckpointStore]:
        project_file = Path(root) / "chapter01.ctpr"
        project_file.write_bytes(b"project")
        reference = checkpoint_reference_for_save(None, project_file)
        store = ProjectCheckpointStore(
            project_file,
            reference,
            enabled=True,
        )
        self.assertTrue(store.ensure_initialized())
        return project_file, reference, store

    @staticmethod
    def _blocks() -> list[TextBlock]:
        first = TextBlock(
            text_bbox=np.array([10, 20, 110, 160], dtype=np.int32),
            bubble_bbox=np.array([5, 10, 120, 175], dtype=np.int32),
            text_class="text_bubble",
            block_id="block-a",
            direction="vertical",
            font_color=(0, 0, 0),
        )
        first._render_original_xyxy = [10, 20, 110, 160]
        first._render_area_source = "detected_bubble"
        first._render_area_xyxy = [5, 10, 120, 175]
        second = TextBlock(
            text_bbox=np.array([200, 30, 280, 90], dtype=np.int32),
            text_class="text_free",
            block_id="block-b",
            direction="horizontal",
            font_color=(255, 255, 255),
        )
        return [first, second]

    def test_decoded_content_hash_includes_dtype_shape_and_pixels(self) -> None:
        base = np.zeros((4, 5, 3), dtype=np.uint8)
        changed = base.copy()
        changed[0, 0, 0] = 1

        self.assertNotEqual(
            decoded_image_sha256(base),
            decoded_image_sha256(changed),
        )
        self.assertNotEqual(
            decoded_image_sha256(base),
            decoded_image_sha256(base.astype(np.uint16)),
        )
        self.assertNotEqual(
            decoded_image_sha256(base),
            decoded_image_sha256(np.zeros((5, 4, 3), dtype=np.uint8)),
        )

    def test_detection_round_trip_preserves_order_class_and_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _project, _reference, store = self._store(tmp)
            blocks = self._blocks()
            identity = {"detector": "RT-DETR-v2", "runtime": "test"}
            source_sha = "1" * 64
            fingerprint = build_detection_fingerprint(
                source_sha256=source_sha,
                identity=identity,
            )
            mask = {
                "raw": np.arange(16, dtype=np.uint8).reshape(4, 4),
                "kind": "detector",
            }

            self.assertTrue(
                record_detection_checkpoint(
                    store,
                    page_key="page:00000000",
                    fingerprint=fingerprint,
                    source_sha256=source_sha,
                    identity=identity,
                    blocks=blocks,
                    precomputed_mask_details=mask,
                )
            )
            restored = lookup_detection_checkpoint(
                store,
                page_key="page:00000000",
                fingerprint=fingerprint,
                source_sha256=source_sha,
                identity=identity,
            )

            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(
                [block.block_id for block in restored.blocks],
                ["block-a", "block-b"],
            )
            self.assertEqual(
                [block.text_class for block in restored.blocks],
                ["text_bubble", "text_free"],
            )
            self.assertEqual(
                detection_structure_signature(restored.blocks),
                detection_structure_signature(blocks),
            )
            np.testing.assert_array_equal(
                restored.precomputed_mask_details["raw"],
                mask["raw"],
            )

    def test_detection_metadata_mismatch_is_a_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _project, _reference, store = self._store(tmp)
            identity = {"detector": "RT-DETR-v2", "runtime": "test"}
            source_sha = "2" * 64
            fingerprint = build_detection_fingerprint(
                source_sha256=source_sha,
                identity=identity,
            )
            self.assertTrue(
                record_detection_checkpoint(
                    store,
                    page_key="page:00000000",
                    fingerprint=fingerprint,
                    source_sha256=source_sha,
                    identity=identity,
                    blocks=self._blocks(),
                    precomputed_mask_details=None,
                )
            )

            self.assertIsNone(
                lookup_detection_checkpoint(
                    store,
                    page_key="page:00000000",
                    fingerprint=fingerprint,
                    source_sha256="3" * 64,
                    identity=identity,
                )
            )

    def test_ocr_round_trip_restores_raw_results_and_retained_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _project, _reference, store = self._store(tmp)
            detection_blocks = self._blocks()
            detection_fingerprint = "4" * 64
            identity = build_project_ocr_identity(
                detection_fingerprint=detection_fingerprint,
                runtime_identity={"runtime_fingerprint": "official-image"},
                policy={
                    "primary_ocr_engine": "PaddleOCR VL",
                    "normalized_ocr_mode": "best_local",
                },
                paddle_settings={
                    "max_new_tokens": 512,
                    "prettify_markdown": False,
                    "visualize": False,
                },
                source_lang_english="Japanese",
            )
            fingerprint = build_project_ocr_fingerprint(identity)
            retained = [detection_blocks[1]]
            retained[0].text = "raw OCR"
            retained[0].texts = ["raw", "OCR"]
            retained[0].ocr_status = "ok"
            retained[0].ocr_raw_text = "raw OCR"
            raw_results = {
                retained[0].block_id: snapshot_raw_ocr_result(retained[0])
            }

            self.assertTrue(
                record_ocr_checkpoint(
                    store,
                    page_key="page:00000000",
                    fingerprint=fingerprint,
                    identity=identity,
                    blocks=retained,
                    raw_results=raw_results,
                    attempt_count=2,
                    engine_name="PaddleOCRVLEngine",
                    page_profile={"diagnostic": {"status": "ok"}},
                )
            )
            fresh_detection_blocks = self._blocks()
            restored = lookup_ocr_checkpoint(
                store,
                page_key="page:00000000",
                fingerprint=fingerprint,
                identity=identity,
                detection_blocks=fresh_detection_blocks,
            )

            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(
                [block.block_id for block in restored.blocks],
                ["block-b"],
            )
            self.assertEqual(restored.blocks[0].text, "raw OCR")
            self.assertEqual(restored.blocks[0].texts, ["raw", "OCR"])
            self.assertEqual(restored.attempt_count, 2)
            self.assertEqual(
                restored.page_profile,
                {"diagnostic": {"status": "ok"}},
            )

    def test_dictionary_is_not_part_of_project_ocr_identity(self) -> None:
        common = {
            "detection_fingerprint": "5" * 64,
            "runtime_identity": {"runtime_fingerprint": "runtime"},
            "policy": {
                "primary_ocr_engine": "PaddleOCR VL",
                "normalized_ocr_mode": "best_local",
            },
            "paddle_settings": {
                "max_new_tokens": 512,
                "prettify_markdown": False,
                "visualize": False,
            },
            "source_lang_english": "Japanese",
        }
        first = build_project_ocr_identity(**common)
        second = build_project_ocr_identity(**common)

        self.assertEqual(first, second)
        self.assertNotIn("dictionary", first)

    def test_translation_checkpoint_uses_ctpr_state_without_sidecar_copy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _project, _reference, store = self._store(tmp)
            blocks = self._blocks()
            blocks[0].text = "first"
            blocks[0].translation = "첫 번째"
            blocks[1].text = "second"
            blocks[1].translation = "두 번째"
            snapshot = snapshot_project_translations(blocks)
            identity = build_translation_identity(
                ocr_fingerprint="a" * 64,
                source_lang="Japanese",
                target_lang="Korean",
                extra_context="",
                translator_key="Custom Local Server(Gemma)",
                translator_engine="CustomLocalGemmaTranslation",
                translator_settings={"chunk_size": 6},
                runtime_identity={
                    "model_sha256": "b" * 64,
                    "runtime_fingerprint": "runtime",
                },
                dictionary_fingerprint="c" * 64,
            )
            fingerprint = build_translation_fingerprint(identity)

            self.assertTrue(
                record_translation_checkpoint(
                    store,
                    page_key="page:00000000",
                    fingerprint=fingerprint,
                    identity=identity,
                    blocks=blocks,
                )
            )
            manifest = store.lookup_stage(
                "page:00000000",
                "translation",
                fingerprint,
            )
            self.assertIsNotNone(manifest)
            assert manifest is not None
            self.assertEqual(manifest.objects, {})

            current = self._blocks()
            current[0].text = "first"
            current[1].text = "second"
            hit = lookup_translation_checkpoint(
                store,
                page_key="page:00000000",
                fingerprint=fingerprint,
                identity=identity,
                current_blocks=current,
                project_snapshot=snapshot,
            )
            self.assertIsNotNone(hit)
            assert hit is not None
            apply_translation_checkpoint(current, hit)
            self.assertEqual(
                [block.translation for block in current],
                ["첫 번째", "두 번째"],
            )

            changed = [dict(item) for item in snapshot]
            changed[0]["translation"] = "변경됨"
            self.assertIsNone(
                lookup_translation_checkpoint(
                    store,
                    page_key="page:00000000",
                    fingerprint=fingerprint,
                    identity=identity,
                    current_blocks=current,
                    project_snapshot=changed,
                )
            )

    def test_inpaint_checkpoint_restores_lossless_image_and_final_mask(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _project, _reference, store = self._store(tmp)
            blocks = self._blocks()
            blocks[0].text = "OCR"
            image = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(
                12,
                10,
                3,
            )
            raw_mask = np.zeros((12, 10), dtype=np.uint8)
            raw_mask[2:5, 3:8] = 255
            final_mask = raw_mask.copy()
            final_mask[5:7, 4:6] = 255
            blocks[0].block_final_mask_pixel_count = 15
            blocks[0].block_mask_bbox = [3, 2, 8, 7]
            blocks[0].block_mask_source = "ctd-refined"
            blocks[0].block_mask_decision = "accepted"
            identity = build_inpaint_identity(
                source_sha256="d" * 64,
                detection_fingerprint="e" * 64,
                ocr_fingerprint="f" * 64,
                blocks=blocks,
                brush_strokes=[],
                runtime={
                    "key": "AOT",
                    "backend": "torch",
                    "precision": "fp32",
                },
                model_identity={
                    "id": "aot",
                    "declared_digests": ["1" * 64],
                },
                hd_strategy={"strategy": "Original"},
                mask_settings={"mask_refiner": "ctd"},
            )
            fingerprint = build_inpaint_fingerprint(identity)

            self.assertTrue(
                record_inpaint_checkpoint(
                    store,
                    page_key="page:00000000",
                    fingerprint=fingerprint,
                    identity=identity,
                    blocks=blocks,
                    cleaned_image=image,
                    raw_mask=raw_mask,
                    final_mask=final_mask,
                    cleanup_stats={
                        "applied": True,
                        "component_count": 2,
                        "block_count": 1,
                    },
                )
            )
            hit = lookup_inpaint_checkpoint(
                store,
                page_key="page:00000000",
                fingerprint=fingerprint,
                identity=identity,
                source_shape=image.shape,
                current_blocks=blocks,
            )

            self.assertIsNotNone(hit)
            assert hit is not None
            np.testing.assert_array_equal(hit.cleaned_image, image)
            np.testing.assert_array_equal(hit.raw_mask, raw_mask)
            np.testing.assert_array_equal(hit.final_mask, final_mask)
            self.assertTrue(hit.cleanup_stats["applied"])
            self.assertEqual(
                hit.cleaned_decoded_sha256,
                decoded_image_sha256(image),
            )
            restored_blocks = [block.deep_copy() for block in blocks]
            restored_blocks[0].block_final_mask_pixel_count = 0
            restored_blocks[0].block_mask_bbox = None
            restored_blocks[0].block_mask_source = ""
            restored_blocks[0].block_mask_decision = ""
            restore_inpaint_block_state(
                restored_blocks,
                hit.block_states,
            )
            self.assertEqual(
                restored_blocks[0].block_final_mask_pixel_count,
                15,
            )
            self.assertEqual(
                restored_blocks[0].block_mask_bbox,
                [3, 2, 8, 7],
            )
            self.assertEqual(
                restored_blocks[0].block_mask_source,
                "ctd-refined",
            )
            self.assertEqual(
                restored_blocks[0].block_mask_decision,
                "accepted",
            )

    def test_inpaint_checkpoint_compresses_lossless_array_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _project, _reference, store = self._store(tmp)
            blocks = self._blocks()
            blocks[0].text = "OCR"
            image = np.zeros((128, 128, 3), dtype=np.uint8)
            raw_mask = np.zeros((128, 128), dtype=np.uint8)
            raw_mask[16:96, 20:104] = 255
            final_mask = raw_mask.copy()
            final_mask[8:12, 8:12] = 255
            identity = build_inpaint_identity(
                source_sha256="a" * 64,
                detection_fingerprint="b" * 64,
                ocr_fingerprint="c" * 64,
                blocks=blocks,
                brush_strokes=[],
                runtime={
                    "key": "AOT",
                    "backend": "torch",
                    "precision": "fp32",
                },
                model_identity={
                    "id": "aot",
                    "declared_digests": ["1" * 64],
                },
                hd_strategy={"strategy": "Original"},
                mask_settings={"mask_refiner": "ctd"},
            )
            fingerprint = build_inpaint_fingerprint(identity)

            self.assertTrue(
                record_inpaint_checkpoint(
                    store,
                    page_key="page:00000000",
                    fingerprint=fingerprint,
                    identity=identity,
                    blocks=blocks,
                    cleaned_image=image,
                    raw_mask=raw_mask,
                    final_mask=final_mask,
                    cleanup_stats=None,
                )
            )
            stage = store.lookup_stage(
                "page:00000000",
                "inpaint",
                fingerprint,
            )

            self.assertIsNotNone(stage)
            assert stage is not None
            stored_bytes = sum(
                (
                    store.object_root
                    / object_hash[:2]
                    / object_hash
                ).stat().st_size
                for object_hash in stage.objects.values()
            )
            raw_bytes = image.nbytes + raw_mask.nbytes + final_mask.nbytes
            self.assertLess(stored_bytes, raw_bytes // 4)

    def test_inpainter_identity_hashes_model_without_declared_sha256(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.bin"
            model_path.write_bytes(b"local-model")
            spec = ModelSpec(
                id=ModelID.AOT_TORCH,
                url="",
                files=["model.bin"],
                sha256=[None],
                save_dir=tmp,
            )

            with mock.patch.dict(
                ModelDownloader.registry,
                {ModelID.AOT_TORCH: spec},
            ):
                identity = registered_inpainter_model_identity(
                    "AOT",
                    "torch",
                )

            self.assertEqual(
                identity["file_identities"][0]["sha256"],
                hashlib.sha256(b"local-model").hexdigest(),
            )

    def test_render_checkpoint_skips_existing_output_and_materializes_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _project, _reference, store = self._store(tmp)
            output_base = Path(tmp) / "translated"
            output_root = Path(tmp) / "translated_001"
            output_root.mkdir()
            output_path = output_root / "page_translated.png"
            expected = b"lossless-render-output"
            output_path.write_bytes(expected)
            blocks = self._blocks()
            blocks[0].text = "first"
            blocks[0].translation = "첫 번째"
            viewer_state = {
                "text_items_state": [
                    {"block_id": "block-a", "text": "첫 번째"}
                ]
            }
            identity = build_render_identity(
                source_sha256="2" * 64,
                translation_fingerprint="3" * 64,
                inpaint_fingerprint="4" * 64,
                inpaint_artifact_sha256="5" * 64,
                blocks=blocks,
                render_settings={"font_family": "Arial"},
                export_settings={
                    "resolved_automatic_output_target": (
                        "individual_images"
                    )
                },
                font_identity={
                    "family": "Arial",
                    "file_sha256": "6" * 64,
                },
                target_language_code="ko",
                output_base_root=str(output_base),
            )
            fingerprint = build_render_fingerprint(identity)

            self.assertTrue(
                record_render_checkpoint(
                    store,
                    page_key="page:00000000",
                    fingerprint=fingerprint,
                    identity=identity,
                    blocks=blocks,
                    viewer_state=viewer_state,
                    output_path=str(output_path),
                    output_root=str(output_root),
                )
            )
            project_blocks = snapshot_project_render_blocks(blocks)
            with mock.patch.object(
                store,
                "read_object",
                side_effect=AssertionError(
                    "Existing verified output must not read the CAS object."
                ),
            ):
                existing = lookup_render_checkpoint(
                    store,
                    page_key="page:00000000",
                    fingerprint=fingerprint,
                    identity=identity,
                    project_blocks=project_blocks,
                    project_viewer_state=viewer_state,
                    current_output_base_root=str(output_base),
                )
            self.assertIsNotNone(existing)
            assert existing is not None
            self.assertTrue(existing.output_exists)
            self.assertIsNone(existing.output_bytes)

            output_path.unlink()
            missing = lookup_render_checkpoint(
                store,
                page_key="page:00000000",
                fingerprint=fingerprint,
                identity=identity,
                project_blocks=project_blocks,
                project_viewer_state=viewer_state,
                current_output_base_root=str(output_base),
            )
            self.assertIsNotNone(missing)
            assert missing is not None
            self.assertFalse(missing.output_exists)
            self.assertEqual(
                Path(materialize_render_checkpoint_output(missing)).read_bytes(),
                expected,
            )

    def test_render_materialization_rejects_tampered_bytes_before_writing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "translated"
            output_path = output_root / "page.png"
            result = RenderCheckpointResult(
                output_path=str(output_path),
                output_root=str(output_root),
                output_sha256=hashlib.sha256(b"expected").hexdigest(),
                output_bytes=b"tampered",
                output_exists=False,
            )

            with self.assertRaises(ValueError):
                materialize_render_checkpoint_output(result)

            self.assertFalse(output_root.exists())
            self.assertFalse(output_path.exists())

    def test_skipped_stage_fingerprint_is_stage_specific(self) -> None:
        common = {
            "source_sha256": "1" * 64,
            "detection_fingerprint": "2" * 64,
            "reason": "no_text_detected",
        }
        translation = build_skipped_stage_fingerprint(
            stage="translation",
            **common,
        )
        inpaint = build_skipped_stage_fingerprint(
            stage="inpaint",
            **common,
        )

        self.assertNotEqual(translation, inpaint)
        self.assertEqual(len(translation), 64)
        with self.assertRaises(ValueError):
            build_skipped_stage_fingerprint(stage="render", **common)

    def test_render_checkpoint_rejects_output_outside_owned_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _project, _reference, store = self._store(tmp)
            output_root = Path(tmp) / "owned"
            output_root.mkdir()
            outside = Path(tmp) / "outside.png"
            outside.write_bytes(b"outside")

            self.assertFalse(
                record_render_checkpoint(
                    store,
                    page_key="page:00000000",
                    fingerprint="7" * 64,
                    identity={"render": "identity"},
                    blocks=self._blocks(),
                    viewer_state={"text_items_state": []},
                    output_path=str(outside),
                    output_root=str(output_root),
                )
            )

    def test_manual_page_invalidation_removes_only_ocr_and_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_file, reference, store = self._store(tmp)
            fingerprints = {
                "detection": "6" * 64,
                "ocr": "7" * 64,
                "translation": "8" * 64,
                "inpaint": "a" * 64,
            }
            for page_key in ("page:00000000", "page:00000001"):
                for stage, fingerprint in fingerprints.items():
                    self.assertTrue(
                        store.record_stage(
                            page_key,
                            stage,
                            fingerprint,
                            payload={"stage": stage},
                        )
                    )
            main = SimpleNamespace(
                project_file=str(project_file),
                project_checkpoint_reference=reference.to_dict(),
                image_files=["page-1.png", "page-2.png"],
            )

            removed = invalidate_project_page_checkpoints(
                main,
                "page-1.png",
                stage="ocr",
            )

            self.assertEqual(removed, 3)
            self.assertIsNotNone(
                store.lookup_stage(
                    "page:00000000",
                    "detection",
                    fingerprints["detection"],
                )
            )
            self.assertIsNone(
                store.lookup_stage(
                    "page:00000000",
                    "ocr",
                    fingerprints["ocr"],
                )
            )
            self.assertIsNone(
                store.lookup_stage(
                    "page:00000000",
                    "inpaint",
                    fingerprints["inpaint"],
                )
            )
            self.assertIsNotNone(
                store.lookup_stage(
                    "page:00000001",
                    "ocr",
                    fingerprints["ocr"],
                )
            )
            self.assertIsNotNone(
                store.lookup_stage(
                    "page:00000001",
                    "inpaint",
                    fingerprints["inpaint"],
                )
            )

    def test_detection_identity_covers_runtime_models_and_reading_order(self) -> None:
        settings = SimpleNamespace(
            get_tool_selection=lambda _name: "RT-DETR-v2",
            is_gpu_enabled=lambda: False,
        )
        with mock.patch(
            "app.projects.stage_checkpoints.get_providers",
            return_value=["CPUExecutionProvider"],
        ):
            japanese = build_detection_identity(
                settings,
                source_lang_english="Japanese",
            )
            english = build_detection_identity(
                settings,
                source_lang_english="English",
            )

        self.assertIsNotNone(japanese)
        assert japanese is not None and english is not None
        self.assertTrue(japanese["right_to_left"])
        self.assertFalse(english["right_to_left"])
        self.assertEqual(len(japanese["model"]["sha256"]), 64)
        self.assertEqual(len(japanese["font_model"]["sha256"]), 64)
        self.assertEqual(
            japanese["runtime"]["selected_providers"],
            ["CPUExecutionProvider"],
        )
        self.assertNotEqual(
            build_detection_fingerprint(
                source_sha256="9" * 64,
                identity=japanese,
            ),
            build_detection_fingerprint(
                source_sha256="9" * 64,
                identity=english,
            ),
        )

    def test_page_key_is_project_order_stable_and_duplicate_safe(self) -> None:
        main = SimpleNamespace(
            image_files=["same.png", "same-copy.png"],
        )

        self.assertEqual(
            project_checkpoint_page_key(main, "same.png"),
            "page:00000000",
        )
        self.assertEqual(
            project_checkpoint_page_key(main, "same-copy.png"),
            "page:00000001",
        )

    def test_old_project_waits_for_saved_reference_before_creating_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_file = Path(tmp) / "legacy.ctpr"
            project_file.write_bytes(b"legacy")
            reference = checkpoint_reference_for_save(None, project_file)
            main = SimpleNamespace(
                project_file=str(project_file),
                project_kind="project",
                project_checkpoint_reference=reference.to_dict(),
                project_checkpoint_reference_persisted=False,
                settings_page=SimpleNamespace(
                    get_project_checkpoint_settings=lambda: {
                        "enabled": True,
                    }
                ),
            )

            self.assertIsNone(
                open_project_stage_checkpoint_store(
                    main,
                    initialize=True,
                )
            )
            self.assertFalse(Path(f"{project_file}.cache").exists())

            main.project_checkpoint_reference_persisted = True
            store = open_project_stage_checkpoint_store(
                main,
                initialize=True,
            )
            self.assertIsNotNone(store)
            self.assertTrue(Path(f"{project_file}.cache").is_dir())


if __name__ == "__main__":
    unittest.main()
