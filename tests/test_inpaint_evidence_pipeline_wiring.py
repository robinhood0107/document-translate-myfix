from __future__ import annotations

import inspect

import numpy as np

from modules.utils.inpaint_evidence import MaskPatch
from pipeline.batch_processor import BatchProcessor
from pipeline.stage_batched_processor import StageBatchedProcessor
from pipeline.webtoon_batch.chunk import ChunkMixin


def _assert_raw_mask_handoff_and_evidence_release(callable_object) -> None:
    source = inspect.getsource(callable_object)

    assert "raw_source_mask=" in source
    assert "last_inpaint_evidence" in source
    assert "last_inpaint_evidence = ()" in source


def test_regular_batch_hands_off_raw_mask_and_releases_evidence() -> None:
    _assert_raw_mask_handoff_and_evidence_release(BatchProcessor.batch_process)


def test_stage_batched_hands_off_raw_mask_and_releases_evidence() -> None:
    _assert_raw_mask_handoff_and_evidence_release(
        StageBatchedProcessor._inpaint_pages
    )


def test_webtoon_hands_off_raw_mask_and_releases_evidence() -> None:
    _assert_raw_mask_handoff_and_evidence_release(
        ChunkMixin._inpaint_image_with_blocks
    )


def test_mask_patch_exposes_contract_aliases_without_copying() -> None:
    local_mask = np.zeros((4, 5), dtype=np.uint8)
    local_mask[1:3, 2:4] = 255
    patch = MaskPatch((7, 9, 12, 13), local_mask)

    assert patch.roi == patch.xyxy == (7, 9, 12, 13)
    assert patch.local_mask is patch.mask
