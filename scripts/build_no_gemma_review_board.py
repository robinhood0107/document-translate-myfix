#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import sys
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

FOCUS_PAGES = {
    "p_016.jpg",
    "p_018.jpg",
    "p_020.jpg",
    "i_099.jpg",
    "i_100.jpg",
    "i_101.jpg",
    "i_102.jpg",
    "i_103.jpg",
    "i_104.jpg",
    "i_105.jpg",
}

FORBIDDEN_RENDER_SYMBOLS = set("♡♥❤❥❣💗💖💘💕💞💓💝※♪♫→←↑↓↔↕⇒⇐⇔・●○◆◇■□▲△▼▽★☆")


@dataclass(frozen=True)
class AssetSet:
    original: str
    old_result: str
    new_result: str
    inpainted: str
    mask_overlay: str
    cleanup_delta: str
    detector_overlay: str
    raw_mask: str
    diff_heatmap: str
    original_crop: str
    old_crop: str
    new_crop: str


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _rel_asset(path: Path, board_dir: Path) -> str:
    try:
        return path.relative_to(board_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _copy_asset(src: Path | None, dst: Path, board_dir: Path) -> str:
    if not src or not src.is_file():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return _rel_asset(dst, board_dir)


def _read_image(path: Path | None) -> np.ndarray | None:
    if not path or not path.is_file():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def _write_image(path: Path, image: np.ndarray, board_dir: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        return ""
    encoded.tofile(str(path))
    return _rel_asset(path, board_dir)


def _safe_stem(image_name: str) -> str:
    return Path(image_name).stem.replace(" ", "_")


def _page_id(image_name: str) -> str:
    return _safe_stem(image_name)


def _resolve_run_dir(path: Path) -> Path:
    if (path / "page_snapshots.json").is_file():
        return path
    sample = path / "sample_japan"
    if (sample / "page_snapshots.json").is_file():
        return sample
    raise FileNotFoundError(f"Could not find page_snapshots.json in {path} or {sample}")


def _load_pages(run_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(run_dir / "page_snapshots.json", {})
    pages = payload.get("pages", [])
    if not isinstance(pages, list):
        raise RuntimeError(f"Invalid page_snapshots.json: {run_dir}")
    return [page for page in pages if isinstance(page, dict)]


def _pages_by_name(pages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for page in pages:
        name = str(page.get("image_name") or Path(str(page.get("image_path") or "")).name)
        if name:
            result[name] = page
    return result


def _result_image(run_dir: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    suffix = Path(image_name).suffix.lower()
    candidates = sorted((run_dir / "corpus").glob(f"result_*/*_translated*"))
    exact = [p for p in candidates if p.stem == f"{stem}_translated" and p.suffix.lower() == suffix]
    if exact:
        return exact[0]
    stem_matches = [p for p in candidates if p.stem == f"{stem}_translated"]
    return stem_matches[0] if stem_matches else None


def _latest_log_dir(run_dir: Path, image_name: str) -> Path | None:
    stem = _safe_stem(image_name)
    logs = sorted((run_dir / "corpus").glob(f"log_{stem}_*"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return logs[-1] if logs else None


def _first_existing(path: Path, *alternatives: Path) -> Path | None:
    for candidate in (path, *alternatives):
        if candidate.is_file():
            return candidate
    return None


def _debug_paths(run_dir: Path, image_name: str) -> dict[str, Path | None]:
    stem = _safe_stem(image_name)
    log_dir = _latest_log_dir(run_dir, image_name)
    if not log_dir:
        return {}
    inpaint_dir = log_dir / "inpainted_images" / stem
    return {
        "inpainted": _first_existing(
            inpaint_dir / f"{stem}_cleaned.png",
            inpaint_dir / f"{stem}_cleaned.jpg",
            inpaint_dir / f"{stem}_cleaned.jpeg",
            inpaint_dir / f"{stem}_cleaned.webp",
        ),
        "mask_overlay": log_dir / "mask_overlays" / stem / f"{stem}_mask_overlay.png",
        "cleanup_delta": log_dir / "cleanup_mask_delta" / stem / f"{stem}_cleanup_delta.png",
        "detector_overlay": log_dir / "detector_overlays" / stem / f"{stem}_detector_overlay.png",
        "raw_mask": log_dir / "raw_masks" / stem / f"{stem}_raw_mask.png",
        "debug_json": log_dir / "debug_metadata" / stem / f"{stem}_debug.json",
        "ocr_json": log_dir / "ocr_debugs" / stem / f"{stem}_ocr_debug.json",
    }


def _load_debug_metadata(run_dir: Path, image_name: str) -> dict[str, Any]:
    paths = _debug_paths(run_dir, image_name)
    return _read_json(paths.get("debug_json") or Path(), {})


def _load_ocr_debug(run_dir: Path, image_name: str) -> dict[str, Any]:
    paths = _debug_paths(run_dir, image_name)
    return _read_json(paths.get("ocr_json") or Path(), {})


def _load_replay_rows(run_dir: Path) -> dict[tuple[str, int], dict[str, str]]:
    path = run_dir / "translation_replay_matches.csv"
    if not path.is_file():
        return {}
    rows: dict[tuple[str, int], dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                block_index = int(row.get("block_index") or 0)
            except ValueError:
                block_index = 0
            rows[(str(row.get("page") or ""), block_index)] = dict(row)
    return rows


def _xyxy(block: dict[str, Any] | None) -> list[int] | None:
    if not block:
        return None
    raw = block.get("xyxy") or block.get("bbox")
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in raw[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _expanded_box(box: list[int] | None, width: int, height: int, pad: int = 64) -> list[int] | None:
    if not box:
        return None
    x1, y1, x2, y2 = box
    return [max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)]


def _crop_asset(src: Path | None, dst: Path, box: list[int] | None, board_dir: Path) -> str:
    image = _read_image(src)
    if image is None or not box:
        return ""
    height, width = image.shape[:2]
    crop_box = _expanded_box(box, width, height)
    if not crop_box:
        return ""
    x1, y1, x2, y2 = crop_box
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    return _write_image(dst, crop, board_dir)


def _diff_heatmap(old_path: Path | None, new_path: Path | None, dst: Path, board_dir: Path) -> tuple[str, float]:
    old = _read_image(old_path)
    new = _read_image(new_path)
    if old is None or new is None:
        return "", 0.0
    if old.shape[:2] != new.shape[:2]:
        new = cv2.resize(new, (old.shape[1], old.shape[0]), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(old, new)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mean_delta = float(np.mean(gray))
    heat = cv2.applyColorMap(np.clip(gray * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(new, 0.65, heat, 0.35, 0)
    return _write_image(dst, overlay, board_dir), mean_delta


def _build_assets(
    *,
    run_dir: Path,
    baseline_run_dir: Path | None,
    board_dir: Path,
    image_name: str,
    block: dict[str, Any] | None,
) -> tuple[AssetSet, float]:
    stem = _safe_stem(image_name)
    item_prefix = stem if block is None else f"{stem}_b{int(block.get('index', -1)):03d}"
    asset_dir = board_dir / "assets" / stem
    source = run_dir / "corpus" / image_name
    new_result = _result_image(run_dir, image_name)
    old_result = _result_image(baseline_run_dir, image_name) if baseline_run_dir else None
    debug = _debug_paths(run_dir, image_name)
    box = _xyxy(block)
    diff_path, mean_delta = _diff_heatmap(
        old_result,
        new_result,
        asset_dir / f"{item_prefix}_diff_heatmap.jpg",
        board_dir,
    )
    assets = AssetSet(
        original=_copy_asset(source, asset_dir / f"{item_prefix}_original{Path(image_name).suffix}", board_dir),
        old_result=_copy_asset(
            old_result,
            asset_dir / f"{item_prefix}_old_result{Path(old_result).suffix if old_result else '.jpg'}",
            board_dir,
        ),
        new_result=_copy_asset(
            new_result,
            asset_dir / f"{item_prefix}_new_result{Path(new_result).suffix if new_result else '.jpg'}",
            board_dir,
        ),
        inpainted=_copy_asset(debug.get("inpainted"), asset_dir / f"{item_prefix}_inpainted.png", board_dir),
        mask_overlay=_copy_asset(debug.get("mask_overlay"), asset_dir / f"{item_prefix}_mask_overlay.png", board_dir),
        cleanup_delta=_copy_asset(debug.get("cleanup_delta"), asset_dir / f"{item_prefix}_cleanup_delta.png", board_dir),
        detector_overlay=_copy_asset(debug.get("detector_overlay"), asset_dir / f"{item_prefix}_detector_overlay.png", board_dir),
        raw_mask=_copy_asset(debug.get("raw_mask"), asset_dir / f"{item_prefix}_raw_mask.png", board_dir),
        diff_heatmap=diff_path,
        original_crop=_crop_asset(source, asset_dir / f"{item_prefix}_original_crop.jpg", box, board_dir),
        old_crop=_crop_asset(old_result, asset_dir / f"{item_prefix}_old_crop.jpg", box, board_dir),
        new_crop=_crop_asset(new_result, asset_dir / f"{item_prefix}_new_crop.jpg", box, board_dir),
    )
    return assets, mean_delta


def _has_forbidden_symbols(text: str) -> bool:
    return any(char in FORBIDDEN_RENDER_SYMBOLS or ord(char) > 0xFFFF for char in text)


def _block_reasons(
    *,
    image_name: str,
    block: dict[str, Any],
    replay_row: dict[str, str] | None,
    debug_block: dict[str, Any] | None,
    ocr_block: dict[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    fit = str(block.get("text_fit_status") or "")
    text_class = str(block.get("text_class") or "")
    render_reasons = [str(v) for v in (block.get("render_normalization_reasons") or [])]
    render_text = str(block.get("render_text") or "")
    translation = str(block.get("translation") or "")
    if fit and fit != "fit":
        reasons.append(fit)
    if "ui" in fit.lower() or "embedded" in fit.lower():
        reasons.append("ui_panel_mode")
    if text_class == "text_free" and fit == "needs_review_text_free_layout":
        reasons.append("text_free_layout")
    if text_class == "text_free" and image_name in FOCUS_PAGES:
        reasons.append("focus_text_free")
    if any(
        reason in render_reasons
        for reason in ("forbidden_symbol_removed", "strict_symbol_removed", "render_sanitized_symbols")
    ):
        reasons.append("sanitizer_changed")
    if _has_forbidden_symbols(render_text):
        reasons.append("forbidden_symbol_residue")
    if translation.strip() and not render_text.strip():
        reasons.append("translation_not_rendered")
    if replay_row and replay_row.get("matched") != "1":
        reasons.append(str(replay_row.get("reason") or "translation_replay_unmatched"))
    if ocr_block:
        status = str(ocr_block.get("status") or "")
        reject = str(ocr_block.get("reject_reason") or ocr_block.get("empty_reason") or "")
        if status and status != "ok":
            reasons.append(f"ocr_{status}")
        if reject:
            reasons.append(f"ocr_{reject}")
    final_ui_panel_mode = str(block.get("ui_panel_mode") or "")
    final_mask_decision = str(block.get("mask_decision") or "")
    final_mask_reject = str(block.get("mask_reject_reason") or "")
    if final_ui_panel_mode:
        reasons.append("ui_panel_mode")
    if final_mask_decision and final_mask_decision not in {"accepted", "ok"}:
        reasons.append(f"mask_{final_mask_decision}")
    if final_mask_reject:
        reasons.append(f"mask_{final_mask_reject}")
    if bool(block.get("bubble_panel_text_candidate")):
        reasons.append("bubble_panel_text_candidate")
    if bool(block.get("bubble_merge_reocr_needed")):
        reasons.append("bubble_merge_reocr_needed")
    if debug_block:
        if debug_block.get("ui_panel_mode") and "ui_panel_mode" not in reasons:
            reasons.append("ui_panel_mode")
        mask_decision = str(debug_block.get("mask_decision") or "")
        if mask_decision and mask_decision not in {"accepted", "ok"} and f"mask_{mask_decision}" not in reasons:
            reasons.append(f"mask_{mask_decision}")
        mask_reject = str(debug_block.get("mask_reject_reason") or "")
        if mask_reject and f"mask_{mask_reject}" not in reasons:
            reasons.append(f"mask_{mask_reject}")
        if debug_block.get("bubble_panel_text_candidate"):
            reasons.append("bubble_panel_text_candidate")
        if debug_block.get("bubble_merge_reocr_needed"):
            reasons.append("bubble_merge_reocr_needed")
        skipped = str(debug_block.get("erase_skipped_reason") or "")
        if skipped and skipped not in {"line_art_intrusion", "empty_seed"}:
            reasons.append(f"erase_{skipped}")
        if debug_block.get("hard_box_applied"):
            reasons.append("hard_box_applied")
    return sorted(dict.fromkeys(reason for reason in reasons if reason))


def _page_reasons(image_name: str, debug: dict[str, Any], summary_page: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if image_name in FOCUS_PAGES:
        reasons.append("focus_page_full_check")
    if debug.get("hard_box_applied_count"):
        reasons.append("hard_box_applied_count")
    if debug.get("hard_box_rescue_mask_pixel_count"):
        reasons.append("hard_box_rescue_mask_pixel_count")
    if debug.get("mask_policy_outside_bubble_removed_pixel_count"):
        reasons.append("bubble_clamp_removed_outside_pixels")
    if debug.get("mask_policy_removed_pixel_count"):
        reasons.append("mask_policy_removed_pixels")
    if debug.get("mask_score_outline_damage") not in (None, "", 0, 0.0):
        reasons.append("outline_damage_score")
    if debug.get("ui_panel_mode") or debug.get("ui_panel_preview_path"):
        reasons.append("ui_panel_mode")
    if summary_page.get("page_failed"):
        reasons.append("page_failed")
    return sorted(dict.fromkeys(reasons))


def _decision_for_reasons(reasons: list[str], *, render_text: str = "") -> tuple[str, str, bool]:
    reason_set = set(reasons)
    if not reasons:
        return "pass_auto", "high", False
    user_required_tokens = {
        "focus_page_full_check",
        "focus_text_free",
        "needs_review",
        "needs_review_text_free_layout",
        "text_free_layout",
        "ui_panel_mode",
        "bubble_panel_text_candidate",
        "bubble_merge_reocr_needed",
        "outline_damage_score",
        "low_iou",
        "no_snapshot_translation_match",
        "translation_not_rendered",
        "hard_box_applied",
        "hard_box_applied_count",
        "hard_box_rescue_mask_pixel_count",
        "page_failed",
    }
    if "forbidden_symbol_residue" in reason_set:
        return "needs_user_review_symbol_residue", "high", True
    if reason_set == {"sanitizer_changed"} and not _has_forbidden_symbols(render_text):
        return "pass_sanitizer_removed", "medium", False
    if reason_set.intersection(user_required_tokens):
        return "needs_user_review", "medium", True
    if any(
        (reason.startswith("mask_") and not reason.startswith("mask_policy_"))
        or reason.startswith(("erase_", "ocr_"))
        for reason in reasons
    ):
        return "needs_user_review", "medium", True
    return "codex_reviewed_pass", "medium", False


def _metadata_blocks_by_index(debug: dict[str, Any]) -> dict[int, dict[str, Any]]:
    blocks = debug.get("blocks") or []
    result: dict[int, dict[str, Any]] = {}
    for block in blocks:
        if not isinstance(block, dict):
            continue
        try:
            index = int(block.get("index"))
        except (TypeError, ValueError):
            continue
        result[index] = block
    return result


def _ocr_blocks_by_index(debug: dict[str, Any]) -> dict[int, dict[str, Any]]:
    blocks = debug.get("blocks") or []
    result: dict[int, dict[str, Any]] = {}
    for block in blocks:
        if not isinstance(block, dict):
            continue
        try:
            index = int(block.get("index"))
        except (TypeError, ValueError):
            continue
        result[index] = block
    return result


def _make_item(
    *,
    run_dir: Path,
    baseline_run_dir: Path | None,
    board_dir: Path,
    image_name: str,
    page: dict[str, Any],
    block: dict[str, Any] | None,
    kind: str,
    reasons: list[str],
    debug_page: dict[str, Any],
    debug_block: dict[str, Any] | None,
    replay_row: dict[str, str] | None,
) -> dict[str, Any]:
    assets, mean_delta = _build_assets(
        run_dir=run_dir,
        baseline_run_dir=baseline_run_dir,
        board_dir=board_dir,
        image_name=image_name,
        block=block,
    )
    block_index = int(block.get("index", -1)) if block else -1
    render_text = str((block or {}).get("render_text") or "")
    codex_decision, confidence, user_required = _decision_for_reasons(reasons, render_text=render_text)
    item_id = f"{_page_id(image_name)}__page" if block is None else f"{_page_id(image_name)}__b{block_index:03d}"
    block_mask_bbox = (block or {}).get("block_mask_bbox")
    block_data = block or {}
    debug_data = debug_block or {}

    def _merged_value(key: str, default: Any = "") -> Any:
        value = block_data.get(key)
        if value not in (None, "", [], {}):
            return value
        debug_value = debug_data.get(key)
        if debug_value not in (None, "", [], {}):
            return debug_value
        return default

    return {
        "id": item_id,
        "kind": kind,
        "page": image_name,
        "block_index": block_index,
        "text_class": str((block or {}).get("text_class") or ""),
        "bbox": (block or {}).get("xyxy") or [],
        "bubble_xyxy": (block or {}).get("bubble_xyxy") or [],
        "ocr_text": str((block or {}).get("text") or ""),
        "translation": str((block or {}).get("translation") or ""),
        "render_text": render_text,
        "text_fit_status": str((block or {}).get("text_fit_status") or ""),
        "render_reasons": (block or {}).get("render_normalization_reasons") or [],
        "block_final_mask_pixel_count": int((block or {}).get("block_final_mask_pixel_count") or 0),
        "block_mask_iou": float((block or {}).get("block_mask_iou") or 0.0),
        "block_mask_span_coverage": float((block or {}).get("block_mask_span_coverage") or 0.0),
        "block_mask_bbox": block_mask_bbox if block_mask_bbox is not None else None,
        "block_mask_source": str((block or {}).get("block_mask_source") or ""),
        "block_mask_decision": str((block or {}).get("block_mask_decision") or ""),
        "render_restore_applied": bool((block or {}).get("render_restore_applied", False)),
        "ui_panel_mode": str((block or {}).get("ui_panel_mode") or ""),
        "ui_panel_preview_path": str((block or {}).get("ui_panel_preview_path") or ""),
        "mask_decision": str((block or {}).get("mask_decision") or ""),
        "mask_reject_reason": str((block or {}).get("mask_reject_reason") or ""),
        "bubble_panel_text_candidate": bool(_merged_value("bubble_panel_text_candidate", False)),
        "bubble_panel_group_id": str(_merged_value("bubble_panel_group_id", "")),
        "bubble_panel_member_indices": _merged_value("bubble_panel_member_indices", []),
        "bubble_panel_mask_pixel_count": int(_merged_value("bubble_panel_mask_pixel_count", 0) or 0),
        "bubble_panel_mask_source": str(_merged_value("bubble_panel_mask_source", "")),
        "bubble_panel_merge_decision": str(_merged_value("bubble_panel_merge_decision", "")),
        "bubble_merge_reocr_needed": bool(_merged_value("bubble_merge_reocr_needed", False)),
        "review_reasons": reasons,
        "codex_decision": codex_decision,
        "codex_confidence": confidence,
        "user_review_required": user_required,
        "diff_mean_delta": round(mean_delta, 3),
        "debug": {
            "inpainter": debug_page.get("inpainter", ""),
            "mask_refiner": debug_page.get("mask_refiner", ""),
            "mask_policy_version": debug_page.get("mask_policy_version", ""),
            "hard_box_applied_count": debug_page.get("hard_box_applied_count", 0),
            "hard_box_rescue_mask_pixel_count": debug_page.get("hard_box_rescue_mask_pixel_count", 0),
            "mask_policy_removed_pixel_count": debug_page.get("mask_policy_removed_pixel_count", 0),
            "mask_policy_outside_bubble_removed_pixel_count": debug_page.get("mask_policy_outside_bubble_removed_pixel_count", 0),
            "block_debug": debug_block or {},
            "replay": replay_row or {},
        },
        "assets": assets.__dict__,
    }


def _build_items(run_dir: Path, baseline_run_dir: Path | None, board_dir: Path) -> list[dict[str, Any]]:
    pages = _load_pages(run_dir)
    replay_rows = _load_replay_rows(run_dir)
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in pages:
        image_name = str(page.get("image_name") or Path(str(page.get("image_path") or "")).name)
        if not image_name:
            continue
        debug_page = _load_debug_metadata(run_dir, image_name)
        ocr_debug = _load_ocr_debug(run_dir, image_name)
        debug_blocks = _metadata_blocks_by_index(debug_page)
        ocr_blocks = _ocr_blocks_by_index(ocr_debug)
        page_reasons = _page_reasons(image_name, debug_page, page)
        if page_reasons:
            item = _make_item(
                run_dir=run_dir,
                baseline_run_dir=baseline_run_dir,
                board_dir=board_dir,
                image_name=image_name,
                page=page,
                block=None,
                kind="page",
                reasons=page_reasons,
                debug_page=debug_page,
                debug_block=None,
                replay_row=None,
            )
            items.append(item)
            seen.add(item["id"])
        for index, block in enumerate(page.get("blocks") or []):
            if not isinstance(block, dict):
                continue
            block = dict(block)
            block.setdefault("index", index)
            replay = replay_rows.get((image_name, index))
            debug_block = debug_blocks.get(index)
            ocr_block = ocr_blocks.get(index)
            reasons = _block_reasons(
                image_name=image_name,
                block=block,
                replay_row=replay,
                debug_block=debug_block,
                ocr_block=ocr_block,
            )
            if not reasons:
                continue
            item = _make_item(
                run_dir=run_dir,
                baseline_run_dir=baseline_run_dir,
                board_dir=board_dir,
                image_name=image_name,
                page=page,
                block=block,
                kind="block",
                reasons=reasons,
                debug_page=debug_page,
                debug_block=debug_block,
                replay_row=replay,
            )
            if item["id"] not in seen:
                items.append(item)
                seen.add(item["id"])
    items.sort(key=lambda item: (0 if item["user_review_required"] else 1, item["page"], item["block_index"]))
    return items


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "kind",
        "page",
        "block_index",
        "text_class",
        "text_fit_status",
        "codex_decision",
        "codex_confidence",
        "user_review_required",
        "review_reasons",
        "ocr_text",
        "translation",
        "render_text",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key, "")
                    for key in fields
                }
            )


def _summary(items: list[dict[str, Any]], run_dir: Path, baseline_run_dir: Path | None) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    for item in items:
        for reason in item.get("review_reasons") or []:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "run_dir": str(run_dir),
        "baseline_run_dir": str(baseline_run_dir or ""),
        "item_count": len(items),
        "user_review_required_count": sum(1 for item in items if item.get("user_review_required")),
        "codex_pass_count": sum(1 for item in items if not item.get("user_review_required")),
        "reason_counts": dict(sorted(reason_counts.items())),
        "pages": sorted({str(item.get("page") or "") for item in items}),
    }


def _server_py() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ReviewHandler(SimpleHTTPRequestHandler):
    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/decisions":
            path = Path("decisions.json")
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                payload = {}
            self._send_json(payload)
            return
        super().do_GET()

    def do_POST(self):
        if self.path not in {"/api/decision", "/api/decisions"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length") or "0")
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        path = Path("decisions.json")
        current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if self.path == "/api/decision":
            item_id = str(payload.get("id") or "")
            if not item_id:
                self._send_json({"ok": False, "error": "missing id"}, HTTPStatus.BAD_REQUEST)
                return
            current[item_id] = payload
        else:
            current = payload if isinstance(payload, dict) else {}
        path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        self._send_json({"ok": True, "count": len(current)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler)
    print(f"[review-board] http://127.0.0.1:{args.port}/index.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
'''


def _index_html() -> str:
    return r'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sample/japan CTD/LaMa Review Board</title>
  <style>
    :root { color-scheme: light; --ink:#172026; --muted:#63717c; --line:#d8e0e5; --blue:#175ddc; --bad:#b00020; --ok:#147a3f; --warn:#995f00; --bg:#f6f8fa; }
    * { box-sizing: border-box; }
    body { margin:0; font:14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
    header { position:sticky; top:0; z-index:10; background:#fff; border-bottom:1px solid var(--line); padding:12px 18px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
    h1 { font-size:18px; margin:0 10px 0 0; }
    button, select, input { font:inherit; }
    button { border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:6px; padding:7px 10px; cursor:pointer; }
    button:hover { border-color:#8ea2b1; }
    button.active { background:var(--blue); color:#fff; border-color:var(--blue); }
    .meta { color:var(--muted); }
    .layout { display:grid; grid-template-columns: 360px 1fr; gap:14px; padding:14px; }
    .list { height:calc(100vh - 82px); overflow:auto; background:#fff; border:1px solid var(--line); border-radius:8px; }
    .row { padding:10px 12px; border-bottom:1px solid var(--line); cursor:pointer; }
    .row.active { background:#eaf1ff; }
    .row .title { font-weight:700; }
    .row .tags { margin-top:6px; display:flex; flex-wrap:wrap; gap:4px; }
    .tag { border-radius:999px; padding:2px 7px; font-size:12px; background:#edf1f4; color:#334; }
    .tag.need { background:#ffe9e8; color:#8a1020; }
    .tag.pass { background:#e8f6ee; color:#11613a; }
    .viewer { min-width:0; }
    .panel { background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px; margin-bottom:12px; }
    .actions { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0; }
    .actions button[data-decision="pass"] { border-color:#98d4ad; }
    .actions button[data-decision^="fail"] { border-color:#eea4ad; }
    .actions button[data-decision="preserve_original"] { border-color:#d5b56d; }
    .decision { font-weight:700; margin-left:auto; }
    .grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:10px; }
    figure { margin:0; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#fff; }
    figcaption { padding:7px 9px; font-weight:700; border-bottom:1px solid var(--line); background:#f9fbfc; }
    img { width:100%; height:auto; display:block; }
    .crop-grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:10px; margin-top:10px; }
    pre { white-space:pre-wrap; word-break:break-word; background:#f8fafb; border:1px solid var(--line); border-radius:8px; padding:10px; max-height:280px; overflow:auto; }
    .empty { color:var(--muted); padding:20px; }
    @media (max-width: 980px) { .layout { grid-template-columns:1fr; } .list { height:260px; } .grid, .crop-grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Sample/japan CTD/LaMa 검수판</h1>
    <button id="filterNeed" class="active">검수 필요</button>
    <button id="filterAll">전체</button>
    <button id="filterPage">페이지</button>
    <button id="filterBlock">블록</button>
    <input id="search" placeholder="page / reason / text" />
    <span id="counts" class="meta"></span>
  </header>
  <main class="layout">
    <section class="list" id="list"></section>
    <section class="viewer" id="viewer"><div class="panel empty">항목을 선택하세요.</div></section>
  </main>
  <script>
    let items = [];
    let decisions = {};
    let selected = null;
    let filter = "need";
    const listEl = document.getElementById("list");
    const viewerEl = document.getElementById("viewer");
    const countsEl = document.getElementById("counts");

    const decisionLabels = [
      ["pass", "1 통과"],
      ["fail_mask", "2 마스크 문제"],
      ["fail_render", "3 렌더 문제"],
      ["preserve_original", "4 원본 보존"],
      ["accept_ui_preview", "5 UI preview 채택"],
      ["needs_reocr", "6 재OCR 필요"],
      ["hold", "7 보류"],
    ];

    async function boot() {
      items = await fetch("items.json").then(r => r.json());
      try { decisions = await fetch("/api/decisions").then(r => r.json()); } catch { decisions = {}; }
      render();
      const first = filtered()[0];
      if (first) select(first.id);
    }

    function filtered() {
      const q = document.getElementById("search").value.trim().toLowerCase();
      return items.filter(item => {
        if (filter === "need" && !item.user_review_required) return false;
        if (filter === "page" && item.kind !== "page") return false;
        if (filter === "block" && item.kind !== "block") return false;
        if (!q) return true;
        return JSON.stringify([item.page, item.review_reasons, item.ocr_text, item.translation, item.render_text]).toLowerCase().includes(q);
      });
    }

    function tag(text, cls="") { return `<span class="tag ${cls}">${escapeHtml(text)}</span>`; }
    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    function img(label, path) {
      if (!path) return `<figure><figcaption>${label}</figcaption><div class="empty">missing</div></figure>`;
      return `<figure><figcaption>${label}</figcaption><img src="${escapeHtml(path)}" loading="lazy"></figure>`;
    }

    function render() {
      document.querySelectorAll("header button").forEach(btn => btn.classList.remove("active"));
      document.getElementById(filter === "need" ? "filterNeed" : filter === "page" ? "filterPage" : filter === "block" ? "filterBlock" : "filterAll").classList.add("active");
      const rows = filtered();
      countsEl.textContent = `표시 ${rows.length} / 전체 ${items.length}, 검수 필요 ${items.filter(i => i.user_review_required).length}`;
      listEl.innerHTML = rows.map(item => {
        const decision = decisions[item.id]?.decision || item.codex_decision;
        const needCls = item.user_review_required ? "need" : "pass";
        return `<div class="row ${selected === item.id ? "active" : ""}" onclick="select('${item.id}')">
          <div class="title">${escapeHtml(item.page)} ${item.block_index >= 0 ? `#${item.block_index}` : "page"}</div>
          <div class="meta">${escapeHtml(item.text_class || item.kind)} · ${escapeHtml(decision)}</div>
          <div class="tags">${tag(item.user_review_required ? "검수" : "자동", needCls)} ${item.review_reasons.slice(0,4).map(r => tag(r)).join("")}</div>
        </div>`;
      }).join("") || `<div class="empty">조건에 맞는 항목이 없습니다.</div>`;
    }

    function select(id) {
      selected = id;
      const item = items.find(i => i.id === id);
      render();
      if (!item) return;
      const d = decisions[item.id] || {};
      const actionButtons = decisionLabels.map(([value, label]) => `<button data-decision="${value}" onclick="saveDecision('${item.id}', '${value}')">${label}</button>`).join("");
      viewerEl.innerHTML = `<div class="panel">
        <div style="display:flex; gap:10px; align-items:flex-start; flex-wrap:wrap;">
          <h2 style="margin:0; font-size:18px;">${escapeHtml(item.page)} ${item.block_index >= 0 ? `#${item.block_index}` : "page"}</h2>
          <span class="decision">현재: ${escapeHtml(d.decision || item.codex_decision)}</span>
        </div>
        <div class="tags" style="margin-top:8px;">${item.review_reasons.map(r => tag(r, item.user_review_required ? "need" : "")).join("")}</div>
        <div class="actions">${actionButtons}</div>
        <pre>${escapeHtml(JSON.stringify({
          text_class: item.text_class,
          bbox: item.bbox,
          bubble_xyxy: item.bubble_xyxy,
          text_fit_status: item.text_fit_status,
          ocr_text: item.ocr_text,
          translation: item.translation,
          render_text: item.render_text,
          codex_decision: item.codex_decision,
          debug: item.debug
        }, null, 2))}</pre>
      </div>
      <div class="panel crop-grid">
        ${img("original crop", item.assets.original_crop)}
        ${img("old crop", item.assets.old_crop)}
        ${img("new crop", item.assets.new_crop)}
      </div>
      <div class="panel grid">
        ${img("original", item.assets.original)}
        ${img("old result", item.assets.old_result)}
        ${img("new result", item.assets.new_result)}
        ${img("inpainted", item.assets.inpainted)}
        ${img("mask overlay", item.assets.mask_overlay)}
        ${img("cleanup delta", item.assets.cleanup_delta)}
        ${img("detector overlay", item.assets.detector_overlay)}
        ${img("raw mask", item.assets.raw_mask)}
        ${img("diff heatmap", item.assets.diff_heatmap)}
      </div>`;
    }

    async function saveDecision(id, decision) {
      const item = items.find(i => i.id === id);
      const payload = { id, decision, page: item?.page, block_index: item?.block_index, ts: new Date().toISOString() };
      decisions[id] = payload;
      await fetch("/api/decision", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
      render();
      select(id);
    }

    document.getElementById("filterNeed").onclick = () => { filter = "need"; render(); };
    document.getElementById("filterAll").onclick = () => { filter = "all"; render(); };
    document.getElementById("filterPage").onclick = () => { filter = "page"; render(); };
    document.getElementById("filterBlock").onclick = () => { filter = "block"; render(); };
    document.getElementById("search").oninput = render;
    document.addEventListener("keydown", event => {
      if (!selected) return;
      const index = Number(event.key);
      if (index >= 1 && index <= decisionLabels.length) saveDecision(selected, decisionLabels[index - 1][0]);
    });
    boot();
  </script>
</body>
</html>
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a no-Gemma Sample/japan CTD/LaMa review board.")
    parser.add_argument("--run-dir", required=True, help="Dataset run dir or output root containing sample_japan/")
    parser.add_argument("--baseline-run-dir", default="", help="Previous dataset run dir for old_result/diff comparison")
    parser.add_argument("--output-dir", default="", help="Review board output dir. Defaults to <run-dir>/review_board")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(Path(args.run_dir))
    baseline_run_dir = _resolve_run_dir(Path(args.baseline_run_dir)) if args.baseline_run_dir else None
    board_dir = Path(args.output_dir) if args.output_dir else run_dir / "review_board"
    if board_dir.exists():
        shutil.rmtree(board_dir)
    board_dir.mkdir(parents=True, exist_ok=True)

    items = _build_items(run_dir, baseline_run_dir, board_dir)
    summary = _summary(items, run_dir, baseline_run_dir)
    _write_json(board_dir / "items.json", items)
    _write_json(board_dir / "summary.json", summary)
    _write_json(board_dir / "decisions.json", {})
    _write_json(board_dir / "codex_pre_review.json", {"summary": summary, "items": items})
    _write_csv(board_dir / "codex_pre_review.csv", items)
    (board_dir / "index.html").write_text(_index_html(), encoding="utf-8")
    (board_dir / "review_server.py").write_text(_server_py(), encoding="utf-8")
    (board_dir / "README.md").write_text(
        "\n".join(
            [
                "# Sample/japan CTD/LaMa Review Board",
                "",
                f"- run_dir: `{run_dir}`",
                f"- baseline_run_dir: `{baseline_run_dir or ''}`",
                f"- item_count: `{summary['item_count']}`",
                f"- user_review_required_count: `{summary['user_review_required_count']}`",
                "",
                "Run:",
                "",
                "```bash",
                "python review_server.py --port 8768",
                "```",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[review-board] output: {board_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
