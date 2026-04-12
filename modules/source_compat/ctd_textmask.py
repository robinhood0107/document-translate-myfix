from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from modules.utils.textblock import TextBlock

REFINEMASK_INPAINT = 0
REFINEMASK_ANNOTATION = 1


def enlarge_window(rect, im_w, im_h, ratio=1.7, aspect_ratio=1.0) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in rect]
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return [x1, y1, x2, y2]
    coeff = [aspect_ratio, w + h * aspect_ratio, (1 - ratio) * w * h]
    roots = np.roots(coeff)
    roots.sort()
    delta = int(round(float(roots[-1]) / 2))
    delta_w = int(delta * aspect_ratio)
    delta_w = min(x1, im_w - x2, delta_w)
    delta = min(y1, im_h - y2, delta)
    out = np.array([x1 - delta_w, y1 - delta, x2 + delta_w, y2 + delta], dtype=np.int64)
    out[::2] = np.clip(out[::2], 0, im_w)
    out[1::2] = np.clip(out[1::2], 0, im_h)
    return out.tolist()


def union_area(rect_a, rect_b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in rect_a]
    bx1, by1, bx2, by2 = [float(v) for v in rect_b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def get_topk_color(color_list, bins, k=3, color_var=10, bin_tol=0.001):
    idx = np.argsort(bins * -1)
    color_list, bins = color_list[idx], bins[idx]
    top_colors = [color_list[0]]
    bin_tol = np.sum(bins) * bin_tol
    if len(color_list) > 1:
        for color, bin_value in zip(color_list[1:], bins[1:]):
            if np.abs(np.array(top_colors) - color).min() > color_var:
                top_colors.append(color)
            if len(top_colors) >= k or bin_value < bin_tol:
                break
    return top_colors


def minxor_thresh(threshed, mask, dilate=False):
    neg_threshed = 255 - threshed
    if dilate:
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))
        neg_threshed = cv2.dilate(neg_threshed, element, iterations=1)
        threshed = cv2.dilate(threshed, element, iterations=1)
    neg_xor_sum = cv2.bitwise_xor(neg_threshed, mask).sum()
    xor_sum = cv2.bitwise_xor(threshed, mask).sum()
    if neg_xor_sum < xor_sum:
        return neg_threshed, neg_xor_sum
    return threshed, xor_sum


def get_otsuthresh_masklist(img, pred_mask, per_channel=False):
    channels = [img[..., 0], img[..., 1], img[..., 2]]
    mask_list = []
    for channel in channels:
        _, threshed = cv2.threshold(channel, 1, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        threshed, xor_sum = minxor_thresh(threshed, pred_mask, dilate=False)
        mask_list.append([threshed, xor_sum])
    mask_list.sort(key=lambda item: item[1])
    if per_channel:
        return mask_list
    return [mask_list[0]]


def get_topk_masklist(im_grey, pred_mask):
    if len(im_grey.shape) == 3 and im_grey.shape[-1] == 3:
        im_grey = cv2.cvtColor(im_grey, cv2.COLOR_RGB2GRAY)
    msk = np.ascontiguousarray(pred_mask)
    candidate_grey_px = im_grey[np.where(cv2.erode(msk, np.ones((3, 3), np.uint8), iterations=1) > 127)]
    if candidate_grey_px.size == 0:
        return []
    hist, bins = np.histogram(candidate_grey_px, bins=255)
    topk_color = get_topk_color(bins[:-1], hist, color_var=10, k=3)
    color_range = 30
    mask_list = []
    for color in topk_color:
        c_top = min(color + color_range, 255)
        c_bottom = c_top - 2 * color_range
        threshed = cv2.inRange(im_grey, c_bottom, c_top)
        threshed, xor_sum = minxor_thresh(threshed, msk)
        mask_list.append([threshed, xor_sum])
    return mask_list


def merge_mask_list(mask_list, pred_mask, pred_thresh=30, refine_mode=REFINEMASK_INPAINT):
    if not mask_list:
        return np.zeros_like(pred_mask)
    mask_list.sort(key=lambda item: item[1])
    if pred_thresh > 0:
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (1, 1))
        pred_mask = cv2.erode(pred_mask, element, iterations=1)
        _, pred_mask = cv2.threshold(pred_mask, 60, 255, cv2.THRESH_BINARY)
    connectivity = 8
    mask_merged = np.zeros_like(pred_mask)
    for candidate_mask, _xor_sum in mask_list:
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate_mask, connectivity, cv2.CV_16U)
        for label_index, stat in zip(range(num_labels), stats):
            if label_index == 0:
                continue
            x, y, w, h, area = stat
            if w * h < 3:
                continue
            x1, y1, x2, y2 = x, y, x + w, y + h
            label_local = labels[y1:y2, x1:x2]
            coords = np.where(label_local == label_index)
            tmp_merged = np.zeros_like(label_local, np.uint8)
            tmp_merged[coords] = 255
            tmp_merged = cv2.bitwise_or(mask_merged[y1:y2, x1:x2], tmp_merged)
            xor_merged = cv2.bitwise_xor(tmp_merged, pred_mask[y1:y2, x1:x2]).sum()
            xor_origin = cv2.bitwise_xor(mask_merged[y1:y2, x1:x2], pred_mask[y1:y2, x1:x2]).sum()
            if xor_merged < xor_origin:
                mask_merged[y1:y2, x1:x2] = tmp_merged
    if refine_mode == REFINEMASK_INPAINT:
        mask_merged = cv2.dilate(mask_merged, np.ones((5, 5), np.uint8), iterations=1)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(255 - mask_merged, connectivity, cv2.CV_16U)
    sorted_area = np.sort(stats[:, -1])
    area_thresh = sorted_area[-2] if len(sorted_area) > 1 else sorted_area[-1]
    for label_index, stat in zip(range(num_labels), stats):
        x, y, w, h, area = stat
        if area < area_thresh:
            x1, y1, x2, y2 = x, y, x + w, y + h
            label_local = labels[y1:y2, x1:x2]
            coords = np.where(label_local == label_index)
            tmp_merged = np.zeros_like(label_local, np.uint8)
            tmp_merged[coords] = 255
            tmp_merged = cv2.bitwise_or(mask_merged[y1:y2, x1:x2], tmp_merged)
            xor_merged = cv2.bitwise_xor(tmp_merged, pred_mask[y1:y2, x1:x2]).sum()
            xor_origin = cv2.bitwise_xor(mask_merged[y1:y2, x1:x2], pred_mask[y1:y2, x1:x2]).sum()
            if xor_merged < xor_origin:
                mask_merged[y1:y2, x1:x2] = tmp_merged
    return np.where(mask_merged > 0, 255, 0).astype(np.uint8)


def refine_mask_crop(image: np.ndarray, pred_mask: np.ndarray, refine_mode: int = REFINEMASK_INPAINT) -> np.ndarray:
    if image.size == 0 or pred_mask.size == 0 or not np.any(pred_mask):
        return np.zeros_like(pred_mask)
    mask_list = get_topk_masklist(image, pred_mask)
    mask_list += get_otsuthresh_masklist(image, pred_mask, per_channel=False)
    return merge_mask_list(mask_list, pred_mask, refine_mode=refine_mode)


def _local_clamp(window_xyxy: list[int], clamp_xyxy: list[int] | tuple[int, int, int, int] | None):
    if clamp_xyxy is None:
        return None
    wx1, wy1, wx2, wy2 = [int(v) for v in window_xyxy]
    cx1, cy1, cx2, cy2 = [int(v) for v in clamp_xyxy]
    ix1 = max(wx1, cx1)
    iy1 = max(wy1, cy1)
    ix2 = min(wx2, cx2)
    iy2 = min(wy2, cy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return [ix1 - wx1, iy1 - wy1, ix2 - wx1, iy2 - wy1]


def refine_mask(
    image: np.ndarray,
    pred_mask: np.ndarray,
    blocks: list[TextBlock],
    refine_mode: int = REFINEMASK_INPAINT,
    window_resolver: Callable[[TextBlock, tuple[int, int]], list[int] | None] | None = None,
    clamp_resolver: Callable[[TextBlock, list[int], tuple[int, int]], list[int] | None] | None = None,
) -> np.ndarray:
    mask_refined = np.zeros_like(pred_mask)
    im_h, im_w = image.shape[:2]
    for block in blocks or []:
        if getattr(block, "xyxy", None) is None:
            continue
        if window_resolver is None:
            wx1, wy1, wx2, wy2 = enlarge_window(block.xyxy, im_w, im_h, ratio=1.7)
        else:
            window = window_resolver(block, (im_h, im_w))
            if window is None:
                continue
            wx1, wy1, wx2, wy2 = [int(v) for v in window]
        region = np.ascontiguousarray(image[wy1:wy2, wx1:wx2])
        region_mask = np.ascontiguousarray(pred_mask[wy1:wy2, wx1:wx2])
        if region.size == 0 or region_mask.size == 0 or not np.any(region_mask):
            continue
        merged = refine_mask_crop(region, region_mask, refine_mode=refine_mode)
        if clamp_resolver is not None:
            clamp_xyxy = clamp_resolver(block, [wx1, wy1, wx2, wy2], (im_h, im_w))
            local = _local_clamp([wx1, wy1, wx2, wy2], clamp_xyxy)
            if local is None:
                continue
            lx1, ly1, lx2, ly2 = local
            clamp_mask = np.zeros_like(merged, dtype=np.uint8)
            clamp_mask[ly1:ly2, lx1:lx2] = 255
            merged = cv2.bitwise_and(merged, clamp_mask)
        mask_refined[wy1:wy2, wx1:wx2] = cv2.bitwise_or(mask_refined[wy1:wy2, wx1:wx2], merged)
    return np.where(mask_refined > 0, 255, 0).astype(np.uint8)


def refine_undetected_mask(
    image: np.ndarray,
    pred_mask: np.ndarray,
    mask_refined: np.ndarray,
    blocks: list[TextBlock],
    refine_mode: int = REFINEMASK_INPAINT,
) -> np.ndarray:
    pred_mask_work = pred_mask.copy()
    pred_mask_work[np.where(mask_refined > 30)] = 0
    _, pred_mask_t = cv2.threshold(pred_mask_work, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(pred_mask_t, 4, cv2.CV_16U)
    valid_labels = np.where(stats[:, -1] > 50)[0]
    seg_blk_list: list[TextBlock] = []
    if len(valid_labels) > 0:
        for lab_index in valid_labels[1:]:
            x, y, w, h, area = stats[lab_index]
            bbox = [x, y, x + w, y + h]
            bbox_score = -1.0
            for blk in blocks or []:
                bbox_s = union_area(blk.xyxy, bbox)
                if bbox_s > bbox_score:
                    bbox_score = bbox_s
            if w > 0 and h > 0 and bbox_score / max(1.0, float(w * h)) < 0.5:
                seg_blk_list.append(TextBlock(np.array(bbox, dtype=np.int32)))
    if seg_blk_list:
        mask_refined = cv2.bitwise_or(mask_refined, refine_mask(image, pred_mask_work, seg_blk_list, refine_mode=refine_mode))
    return np.where(mask_refined > 0, 255, 0).astype(np.uint8)
