from __future__ import annotations

import cv2
import imkit as imk
import numpy as np

from modules.utils.textblock import TextBlock


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


def extract_ballon_mask(img: np.ndarray, mask: np.ndarray):
    if mask is None:
        return None, None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(mask):
        return None, None
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    h, w = img.shape[:2]
    text_sum = int(np.sum(mask))
    nz = cv2.findNonZero(mask)
    if nz is None:
        return None, None
    cannyed = cv2.Canny(img, 70, 140, L2gradient=True, apertureSize=3)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))
    cannyed = cv2.dilate(cannyed, element, iterations=1)
    br = cv2.boundingRect(nz)
    br_xyxy = [br[0], br[1], br[0] + br[2], br[1] + br[3]]
    cv2.rectangle(cannyed, (0, 0), (w - 1, h - 1), (255, 255, 255), 1, cv2.LINE_8)
    cannyed = cv2.bitwise_and(cannyed, 255 - mask)
    contours, _ = cv2.findContours(cannyed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    min_ballon_area = w * h
    ballon_mask = None
    non_text_mask = None
    for ii, con in enumerate(contours):
        br_c = cv2.boundingRect(con)
        br_c = [br_c[0], br_c[1], br_c[0] + br_c[2], br_c[1] + br_c[3]]
        if br_c[0] > br_xyxy[0] or br_c[1] > br_xyxy[1] or br_c[2] < br_xyxy[2] or br_c[3] < br_xyxy[3]:
            continue
        tmp = np.zeros_like(cannyed)
        cv2.drawContours(tmp, contours, ii, (255, 255, 255), -1, cv2.LINE_8)
        if cv2.bitwise_and(tmp, mask).sum() >= text_sum:
            con_area = cv2.contourArea(con)
            if con_area < min_ballon_area:
                min_ballon_area = con_area
                ballon_mask = tmp
    if ballon_mask is not None:
        non_text_mask = cv2.bitwise_and(ballon_mask, 255 - mask)
    return ballon_mask, non_text_mask


def _try_uniform_balloon_fill(image_crop: np.ndarray, mask_crop: np.ndarray):
    ballon_msk, non_text_msk = extract_ballon_mask(image_crop, mask_crop)
    if ballon_msk is None or non_text_msk is None:
        return False, image_crop
    non_text_region = np.where(non_text_msk > 0)
    if len(non_text_region[0]) == 0:
        return False, image_crop
    non_text_px = image_crop[non_text_region]
    if non_text_px.size == 0:
        return False, image_crop
    average_bg_color = np.median(non_text_px, axis=0)
    std_rgb = np.std(non_text_px - average_bg_color, axis=0)
    std_max = float(np.max(std_rgb))
    inpaint_thresh = 7 if float(np.std(std_rgb)) > 1 else 10
    if std_max >= inpaint_thresh:
        return False, image_crop
    out = image_crop.copy()
    out[np.where(ballon_msk > 0)] = average_bg_color.astype(np.uint8)
    return True, out


def blockwise_inpaint(image: np.ndarray, mask: np.ndarray, blk_list: list[TextBlock], inpainter, config, *, check_need_inpaint: bool = True) -> np.ndarray:
    if image is None or mask is None or not np.any(mask) or not blk_list:
        result = inpainter(image, mask, config)
        return imk.convert_scale_abs(result)

    inpainted = np.copy(image)
    work_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    im_h, im_w = inpainted.shape[:2]

    for blk in blk_list:
        xyxy = getattr(blk, 'xyxy', None)
        if xyxy is None:
            continue
        bx1, by1, bx2, by2 = [int(float(v)) for v in xyxy[:4]]
        bx1 = max(0, min(bx1, im_w))
        by1 = max(0, min(by1, im_h))
        bx2 = max(0, min(bx2, im_w))
        by2 = max(0, min(by2, im_h))
        if bx2 <= bx1 or by2 <= by1:
            continue
        if not np.any(work_mask[by1:by2, bx1:bx2] > 0):
            continue

        ex1, ey1, ex2, ey2 = enlarge_window([bx1, by1, bx2, by2], im_w, im_h, ratio=1.7)
        crop_img = inpainted[ey1:ey2, ex1:ex2].copy()
        crop_mask = work_mask[ey1:ey2, ex1:ex2].copy()
        if crop_img.size == 0 or crop_mask.size == 0 or not np.any(crop_mask > 0):
            work_mask[by1:by2, bx1:bx2] = 0
            continue

        need_inpaint = True
        if check_need_inpaint:
            flat_fill, filled = _try_uniform_balloon_fill(crop_img, crop_mask)
            if flat_fill:
                crop_result = filled
                need_inpaint = False
        if need_inpaint:
            crop_result = inpainter(crop_img, crop_mask, config)
            crop_result = imk.convert_scale_abs(crop_result)

        inpainted[ey1:ey2, ex1:ex2] = crop_result
        work_mask[by1:by2, bx1:bx2] = 0

    return inpainted
