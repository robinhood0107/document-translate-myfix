from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from modules.utils.inpaint_cleanup import refine_bubble_residue_inpaint


def test_residue_cleanup_reports_blocks_truncated_by_legacy_page_cap(
    monkeypatch,
) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.full((64, 64), 255, dtype=np.uint8)
    blocks = [
        SimpleNamespace(
            xyxy=[0, 0, 64, 64],
            cleanup_roi_xyxy=[0, 0, 64, 64],
            text_class="text_bubble",
        )
        for _ in range(5)
    ]
    boxes = [
        (x, y, x + 2, y + 2)
        for y in range(0, 20, 3)
        for x in range(0, 15, 3)
    ][:35]
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        lambda *_args, **_kwargs: list(boxes),
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup._build_bubble_faint_boxes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.build_text_prior_mask",
        lambda *_args, **_kwargs: np.full((64, 64), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        lambda source, _edit_mask: (source.copy(), "test_fill"),
    )

    _cleaned, _final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        (block for block in blocks),
        None,
        None,
    )

    assert stats["component_count"] == 120
    assert stats["residue_pass_truncated_block_count"] == 2
    assert stats["pass2_backend"] == "test_fill"


def test_residue_cleanup_reports_the_per_block_component_cap(monkeypatch) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.full((64, 64), 255, dtype=np.uint8)
    block = SimpleNamespace(
        xyxy=[0, 0, 64, 64],
        cleanup_roi_xyxy=[0, 0, 64, 64],
        text_class="text_bubble",
    )
    boxes = [
        (x, y, x + 2, y + 2)
        for y in range(0, 24, 3)
        for x in range(0, 15, 3)
    ][:40]
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        lambda *_args, **_kwargs: iter(boxes),
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup._build_bubble_faint_boxes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.build_text_prior_mask",
        lambda *_args, **_kwargs: np.full((64, 64), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        lambda source, _edit_mask: (source.copy(), "test_fill"),
    )

    _cleaned, _final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        (item for item in [block]),
        None,
        None,
    )

    assert stats["component_count"] == 35
    assert stats["residue_pass_truncated_block_count"] == 1
