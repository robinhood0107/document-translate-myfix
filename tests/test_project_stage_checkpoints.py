from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from app.projects.checkpoint_store import (
    ProjectCheckpointStore,
    checkpoint_reference_for_save,
)
from app.projects.stage_checkpoints import (
    build_detection_fingerprint,
    build_detection_identity,
    build_project_ocr_fingerprint,
    build_project_ocr_identity,
    decoded_image_sha256,
    detection_structure_signature,
    invalidate_project_page_checkpoints,
    lookup_detection_checkpoint,
    lookup_ocr_checkpoint,
    open_project_stage_checkpoint_store,
    project_checkpoint_page_key,
    record_detection_checkpoint,
    record_ocr_checkpoint,
)
from modules.ocr.persistent_cache import snapshot_raw_ocr_result
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
