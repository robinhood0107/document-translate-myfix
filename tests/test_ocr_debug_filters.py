from __future__ import annotations

from types import SimpleNamespace

import modules.utils.ocr_debug as ocr_debug_module
from modules.utils.ocr_debug import (
    drop_embedded_ui_ocr_blocks,
    is_embedded_ui_panel_layout_review_candidate,
)


def _block(
    text: str,
    xyxy: tuple[int, int, int, int],
    text_class: str = "text_free",
    bubble_xyxy: tuple[int, int, int, int] | None = None,
):
    return SimpleNamespace(
        text=text,
        xyxy=list(xyxy),
        text_class=text_class,
        bubble_xyxy=list(bubble_xyxy) if bubble_xyxy is not None else None,
    )


def test_embedded_ui_cluster_drops_dense_small_ui_labels_before_inpaint() -> None:
    blocks = [
        _block("メラ", (1934, 512, 2034, 586)),
        _block("最新 ユーザー", (1455, 507, 1864, 590)),
        _block("ソツイート", (1123, 512, 1355, 590)),
        _block("Al 3,748", (839, 601, 1033, 668)),
        _block("011", (315, 598, 467, 673)),
        _block("コスアニ虹川", (81, 740, 191, 923)),
        _block("メインメニュー", (1564, 2133, 1812, 2206)),
        _block("記憶アクセス券", (1605, 2244, 1998, 2343)),
        _block("・お手軽なりすまし", (1599, 2441, 1902, 2515)),
        _block("数据", (1952, 2730, 2037, 2783)),
        _block("校購入", (1951, 2789, 2086, 2856)),
        _block("(数：0枚", (1954, 2884, 2086, 2935)),
        _block(
            "の愛アなりすまし用記憶アクセスオプションってのン自然憶レ自記よ",
            (1697, 2552, 1916, 2917),
            "text_bubble",
            bubble_xyxy=(1603, 2525, 1943, 2948),
        ),
        _block("中身は男なのになぁ", (207, 767, 373, 1059), "text_bubble"),
        _block("便利だったな", (812, 2530, 895, 2804)),
    ]

    kept, dropped = drop_embedded_ui_ocr_blocks(blocks, (3035, 2150, 3))

    dropped_texts = {block.text for block in dropped}
    kept_texts = {block.text for block in kept}
    assert "メラ" in dropped_texts
    assert "最新 ユーザー" in dropped_texts
    assert "ソツイート" in dropped_texts
    assert "Al 3,748" in dropped_texts
    assert "011" in dropped_texts
    assert "コスアニ虹川" in dropped_texts
    assert "メインメニュー" in dropped_texts
    assert "記憶アクセス券" in dropped_texts
    assert "の愛アなりすまし用記憶アクセスオプションってのン自然憶レ自記よ" in kept_texts
    assert "中身は男なのになぁ" in kept_texts
    assert "便利だったな" in kept_texts
    assert all(block.ocr_reject_reason == "embedded_device_ui_cluster" for block in dropped)


def test_embedded_ui_cluster_drops_long_ui_token_without_bubble_protection() -> None:
    blocks = [
        _block("メラ", (1934, 512, 2034, 586)),
        _block("最新 ユーザー", (1455, 507, 1864, 590)),
        _block("ソツイート", (1123, 512, 1355, 590)),
        _block("メインメニュー", (1564, 2133, 1812, 2206)),
        _block("記憶アクセス券", (1605, 2244, 1998, 2343)),
        _block("・お手軽なりすまし", (1599, 2441, 1902, 2515)),
        _block("(数：0枚", (1954, 2884, 2086, 2935)),
        _block(
            "の愛アなりすまし用記憶アクセスオプションってのン自然憶レ自記よ",
            (1697, 2552, 1916, 2917),
            "text_bubble",
        ),
    ]

    kept, dropped = drop_embedded_ui_ocr_blocks(blocks, (3035, 2150, 3))

    assert "の愛アなりすまし用記憶アクセスオプションってのン自然憶レ自記よ" in {
        block.text for block in dropped
    }
    assert "メインメニュー" in {block.text for block in dropped}


def test_embedded_ui_filter_does_not_drop_isolated_labels() -> None:
    blocks = [
        _block("メインメニュー", (100, 100, 280, 170)),
        _block("普通のセリフ", (300, 300, 420, 550), "text_bubble"),
    ]

    kept, dropped = drop_embedded_ui_ocr_blocks(blocks, (1200, 900, 3))

    assert kept == blocks
    assert dropped == []


def test_bubble_protected_embedded_ui_panel_is_layout_review_candidate() -> None:
    block = _block(
        "の愛アなりすまし用記憶アクセスオプションってのン自然憶レ自記よ",
        (1640, 2571, 1906, 2901),
        "text_bubble",
        bubble_xyxy=(1604, 2525, 1943, 2948),
    )

    assert is_embedded_ui_panel_layout_review_candidate(block)


def test_bubble_protected_embedded_ui_panel_is_excluded_from_lama_mask_only() -> None:
    panel = _block(
        "の愛アなりすまし用記憶アクセスオプションってのン自然憶レ自記よ",
        (1640, 2571, 1906, 2901),
        "text_bubble",
        bubble_xyxy=(1604, 2525, 1943, 2948),
    )
    dialogue = _block(
        "中身は男なのになぁ",
        (207, 767, 373, 1059),
        "text_bubble",
        bubble_xyxy=(185, 727, 396, 1100),
    )

    assert hasattr(ocr_debug_module, "split_inpaint_protected_ocr_blocks")
    inpaint_blocks, protected_blocks = ocr_debug_module.split_inpaint_protected_ocr_blocks(
        [panel, dialogue]
    )

    assert inpaint_blocks == [dialogue]
    assert protected_blocks == [panel]
    assert panel._inpaint_protected_reason == "embedded_ui_panel_layout_review"


def test_bubble_protected_embedded_ui_panel_gets_preserve_and_preview_review_metadata() -> None:
    panel = _block(
        "の愛アなりすまし用記憶アクセスオプションってのン自然憶レ自記よ",
        (1640, 2571, 1906, 2901),
        "text_bubble",
        bubble_xyxy=(1604, 2525, 1943, 2948),
    )

    _, protected_blocks = ocr_debug_module.split_inpaint_protected_ocr_blocks([panel])

    assert protected_blocks == [panel]
    assert panel.ui_panel_mode == "preserve_original"
    assert panel.ui_panel_preview_path == ""
    assert panel.mask_decision == "review"
    assert panel.mask_reject_reason == "embedded_ui_panel_layout_review"

    payload = ocr_debug_module.build_ocr_debug_payload("p_016", "PaddleOCR-VL", "ja", [panel])
    block_payload = payload["blocks"][0]
    assert block_payload["ui_panel_mode"] == "preserve_original"
    assert block_payload["ui_panel_preview_path"] == ""
    assert block_payload["mask_decision"] == "review"
    assert block_payload["mask_reject_reason"] == "embedded_ui_panel_layout_review"


def test_normal_dialogue_bubble_is_not_embedded_ui_panel_review_candidate() -> None:
    dialogue = _block("中身は男なのになぁ", (207, 767, 373, 1059), "text_bubble", bubble_xyxy=(185, 727, 396, 1100))
    site_dialogue = _block(
        "偶然見つけたカルデア裏18禁サイト...",
        (100, 100, 380, 260),
        "text_bubble",
        bubble_xyxy=(80, 80, 420, 300),
    )
    free_ui = _block("記憶アクセスオプション", (100, 100, 300, 160), "text_free")

    assert not is_embedded_ui_panel_layout_review_candidate(dialogue)
    assert not is_embedded_ui_panel_layout_review_candidate(site_dialogue)
    assert not is_embedded_ui_panel_layout_review_candidate(free_ui)
