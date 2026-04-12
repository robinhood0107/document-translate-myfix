from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from modules.masking.ctd_vendor.ctd import TextDetBase, TextDetBaseDNN
from modules.utils.download import ModelDownloader, ModelID
from modules.utils.textblock import TextBlock


@dataclass(slots=True)
class CTDRefinerSettings:
    detect_size: int = 1280
    det_rearrange_max_batches: int = 4
    device: str = "cuda"
    font_size_multiplier: float = 1.0
    font_size_max: int = -1
    font_size_min: int = -1
    mask_dilate_size: int = 2


@dataclass(slots=True)
class MaskGenerationResult:
    raw_mask: np.ndarray
    refined_mask: np.ndarray
    protect_mask: np.ndarray
    final_mask: np.ndarray
    backend: str
    device: str
    fallback_used: bool = False


def _enlarge_window(rect, im_w, im_h, ratio=2.5, aspect_ratio=1.0) -> list[int]:
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


def _letterbox(im, new_shape=(640, 640), color=(0, 0, 0), auto=False, stride=128):
    shape = im.shape[:2]
    if not isinstance(new_shape, tuple):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dh, dw = int(dh), int(dw)
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def _square_pad_resize(img: np.ndarray, tgt_size: int):
    h, w = img.shape[:2]
    pad_h, pad_w = 0, 0
    if w < h:
        pad_w = h - w
        w += pad_w
    elif h < w:
        pad_h = w - h
        h += pad_h
    pad_size = tgt_size - h
    if pad_size > 0:
        pad_h += pad_size
        pad_w += pad_size
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)
    down_scale_ratio = tgt_size / img.shape[0]
    if down_scale_ratio < 1:
        img = cv2.resize(img, (tgt_size, tgt_size), interpolation=cv2.INTER_AREA)
    return img, down_scale_ratio, pad_h, pad_w


def _postprocess_mask(img, thresh=None):
    if "torch" in str(type(img)).lower():
        img = img.squeeze()
        if getattr(img, "device", None) is not None and str(img.device) != "cpu":
            img = img.detach().cpu()
        img = img.numpy()
    else:
        img = np.asarray(img).squeeze()
    if thresh is not None:
        img = img > thresh
    return (img * 255).astype(np.uint8)


def _preprocess_img_rgb(image: np.ndarray, detect_size=(1024, 1024), device="cpu", half=False, to_tensor=True):
    if not isinstance(detect_size, tuple):
        detect_size = (detect_size, detect_size)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_in, ratio, (dw, dh) = _letterbox(bgr, new_shape=detect_size, auto=False, stride=64)
    if to_tensor:
        import torch

        img_in = img_in.transpose((2, 0, 1))[::-1]
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255.0
        img_in = torch.from_numpy(img_in).to(device)
        if half:
            img_in = img_in.half()
    return img_in, ratio, int(dw), int(dh)


def _get_topk_color(color_list, bins, k=3, color_var=10, bin_tol=0.001):
    idx = np.argsort(bins * -1)
    color_list, bins = color_list[idx], bins[idx]
    top_colors = [color_list[0]]
    bin_tol = np.sum(bins) * bin_tol
    if len(color_list) > 1:
        for color, bin_val in zip(color_list[1:], bins[1:]):
            if np.abs(np.array(top_colors) - color).min() > color_var:
                top_colors.append(color)
            if len(top_colors) >= k or bin_val < bin_tol:
                break
    return top_colors


def _minxor_thresh(threshed, mask, dilate=False):
    neg_threshed = 255 - threshed
    if dilate:
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))
        neg_threshed = cv2.dilate(neg_threshed, element, iterations=1)
        threshed = cv2.dilate(threshed, element, iterations=1)
    neg_xor_sum = cv2.bitwise_xor(neg_threshed, mask).sum()
    xor_sum = cv2.bitwise_xor(threshed, mask).sum()
    return (neg_threshed, neg_xor_sum) if neg_xor_sum < xor_sum else (threshed, xor_sum)


def _get_otsuthresh_masklist(img, pred_mask):
    channels = [img[..., 0], img[..., 1], img[..., 2]]
    mask_list = []
    for channel in channels:
        _, threshed = cv2.threshold(channel, 1, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        threshed, xor_sum = _minxor_thresh(threshed, pred_mask, dilate=False)
        mask_list.append([threshed, xor_sum])
    mask_list.sort(key=lambda item: item[1])
    return [mask_list[0]]


def _get_topk_masklist(image_rgb, pred_mask):
    if image_rgb.ndim == 3 and image_rgb.shape[-1] == 3:
        grey = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        grey = image_rgb.astype(np.uint8)
    candidate_px = grey[np.where(cv2.erode(pred_mask, np.ones((3, 3), np.uint8), iterations=1) > 127)]
    if candidate_px.size == 0:
        return []
    hist, bins = np.histogram(candidate_px, bins=255)
    topk_color = _get_topk_color(bins[:-1], hist, color_var=10, k=3)
    mask_list = []
    color_range = 30
    for color in topk_color:
        c_top = min(color + color_range, 255)
        c_bottom = c_top - 2 * color_range
        threshed = cv2.inRange(grey, c_bottom, c_top)
        threshed, xor_sum = _minxor_thresh(threshed, pred_mask)
        mask_list.append([threshed, xor_sum])
    return mask_list


def _merge_mask_list(mask_list, pred_mask):
    if not mask_list:
        return np.zeros_like(pred_mask)
    mask_list.sort(key=lambda item: item[1])
    mask_merged = np.zeros_like(pred_mask)
    pred_mask_work = cv2.erode(pred_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    _, pred_mask_work = cv2.threshold(pred_mask_work, 60, 255, cv2.THRESH_BINARY)
    num_connectivity = 8
    for candidate_mask, _xor in mask_list:
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate_mask, num_connectivity, cv2.CV_16U)
        for label_index, stat in zip(range(num_labels), stats):
            if label_index == 0:
                continue
            x, y, w, h, _area = stat
            if w * h < 3:
                continue
            x1, y1, x2, y2 = x, y, x + w, y + h
            label_local = labels[y1:y2, x1:x2]
            coords = np.where(label_local == label_index)
            tmp_merged = np.zeros_like(label_local, np.uint8)
            tmp_merged[coords] = 255
            tmp_merged = cv2.bitwise_or(mask_merged[y1:y2, x1:x2], tmp_merged)
            xor_merged = cv2.bitwise_xor(tmp_merged, pred_mask_work[y1:y2, x1:x2]).sum()
            xor_origin = cv2.bitwise_xor(mask_merged[y1:y2, x1:x2], pred_mask_work[y1:y2, x1:x2]).sum()
            if xor_merged < xor_origin:
                mask_merged[y1:y2, x1:x2] = tmp_merged
    mask_merged = cv2.dilate(mask_merged, np.ones((5, 5), np.uint8), iterations=1)
    return mask_merged


def _refine_mask(image: np.ndarray, pred_mask: np.ndarray, blocks: list[TextBlock]) -> np.ndarray:
    mask_refined = np.zeros_like(pred_mask)
    im_h, im_w = image.shape[:2]
    for block in blocks:
        if getattr(block, "xyxy", None) is None:
            continue
        bx1, by1, bx2, by2 = _enlarge_window(block.xyxy, im_w, im_h)
        region = np.ascontiguousarray(image[by1:by2, bx1:bx2])
        region_mask = np.ascontiguousarray(pred_mask[by1:by2, bx1:bx2])
        if region.size == 0 or region_mask.size == 0 or not np.any(region_mask):
            continue
        mask_list = _get_topk_masklist(region, region_mask)
        mask_list += _get_otsuthresh_masklist(region, region_mask)
        merged = _merge_mask_list(mask_list, region_mask)
        mask_refined[by1:by2, bx1:bx2] = cv2.bitwise_or(mask_refined[by1:by2, bx1:bx2], merged)
    return mask_refined


def _det_rearrange_forward(img, dbnet_batch_forward, tgt_size=1280, max_batch_size=4, device="cuda"):
    import einops

    def _unrearrange(patch_lst, transpose, channel=1, pad_num=0):
        _psize = patch_lst[0].shape[-1]
        _step = int(ph_step * _psize / patch_size)
        _pw = int(_psize / pw_num)
        _h = int(_pw / w * h)
        tgtmap = np.zeros((channel, _h, _pw), dtype=np.float32)
        num_patches = len(patch_lst) * pw_num - pad_num
        for ii, patch in enumerate(patch_lst):
            if transpose:
                patch = einops.rearrange(patch, "c h w -> c w h")
            for jj in range(pw_num):
                pidx = ii * pw_num + jj
                rel_t = rel_step_list[pidx]
                t = int(round(rel_t * _h))
                b = min(t + _psize, _h)
                l = jj * _pw
                r = l + _pw
                tgtmap[..., t:b, :] += patch[..., : b - t, l:r]
                if pidx > 0:
                    interleave = _psize - _step
                    tgtmap[..., t:t + interleave, :] /= 2.0
                if pidx >= num_patches - 1:
                    break
        if transpose:
            tgtmap = einops.rearrange(tgtmap, "c h w -> c w h")
        return tgtmap[None, ...]

    def _patch2batches(patch_lst, p_num, transpose):
        if transpose:
            patch_lst = einops.rearrange(patch_lst, "(p_num pw_num) ph pw c -> p_num (pw_num pw) ph c", p_num=p_num)
        else:
            patch_lst = einops.rearrange(patch_lst, "(p_num pw_num) ph pw c -> p_num ph (pw_num pw) c", p_num=p_num)
        batches = [[]]
        for patch in patch_lst:
            if len(batches[-1]) >= max_batch_size:
                batches.append([])
            square, _down_scale_ratio, _pad_h, _pad_w = _square_pad_resize(patch, tgt_size=tgt_size)
            batches[-1].append(square)
        return batches

    h, w = img.shape[:2]
    transpose = False
    if h < w:
        transpose = True
        h, w = img.shape[1], img.shape[0]

    asp_ratio = h / w
    down_scale_ratio = h / tgt_size
    require_rearrange = down_scale_ratio > 2.5 and asp_ratio > 3
    if not require_rearrange:
        return None, None

    if transpose:
        img = np.transpose(img, (1, 0, 2))

    pw_num = max(int(np.floor(2 * tgt_size / w)), 2)
    patch_size = ph = pw_num * w
    ph_num = int(np.ceil(h / ph))
    ph_step = int((h - ph) / (ph_num - 1)) if ph_num > 1 else 0
    rel_step_list = []
    patch_list = []
    for ii in range(ph_num):
        t = ii * ph_step
        b = t + ph
        rel_step_list.append(t / h)
        patch_list.append(img[t:b])

    p_num = int(np.ceil(ph_num / pw_num))
    pad_num = p_num * pw_num - ph_num
    for _ii in range(pad_num):
        patch_list.append(np.zeros_like(patch_list[0]))

    batches = _patch2batches(patch_list, p_num, transpose)
    db_lst, mask_lst = [], []
    for batch in batches:
        batch = np.array(batch)
        db, mask = dbnet_batch_forward(batch, device=device)
        for d, m in zip(db, mask):
            db_lst.append(d)
            mask_lst.append(m)

    db = _unrearrange(db_lst, transpose, channel=2, pad_num=pad_num)
    mask = _unrearrange(mask_lst, transpose, channel=1, pad_num=pad_num)
    return db, mask


class CTDRefiner:
    def __init__(self, settings: CTDRefinerSettings | None = None) -> None:
        self.settings = settings or CTDRefinerSettings()
        self.net = None
        self.backend = "torch"
        self.model_path = None

    def _choose_model_path(self) -> str:
        if self.settings.device.lower() != "cpu":
            return ModelDownloader.primary_path(ModelID.CTD_TORCH)
        return ModelDownloader.primary_path(ModelID.CTD_ONNX)

    def _ensure_model(self) -> None:
        model_path = self._choose_model_path()
        if self.net is not None and self.model_path == model_path:
            return
        if model_path.endswith(".onnx"):
            self.net = TextDetBaseDNN(1024, model_path)
            self.backend = "onnx"
        else:
            self.net = TextDetBase(model_path, device=self.settings.device, act="leaky", half=False)
            self.backend = "torch"
        self.model_path = model_path

    def _batch_forward(self, batch: np.ndarray, device: str):
        if isinstance(self.net, TextDetBase):
            import einops
            import torch

            batch = einops.rearrange(batch.astype(np.float32) / 255.0, "n h w c -> n c h w")
            batch_t = torch.from_numpy(batch).to(device)
            _blks, mask, lines = self.net(batch_t)
            return lines.detach().cpu().numpy(), mask.detach().cpu().numpy()

        mask_list, line_list = [], []
        for sample in batch:
            _blks, mask, lines = self.net(sample)
            if mask.shape[1] == 2:
                mask, lines = lines, mask
            mask_list.append(mask)
            line_list.append(lines)
        return np.concatenate(line_list, 0), np.concatenate(mask_list, 0)

    def _infer_raw_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        self._ensure_model()
        detect_size = self.settings.detect_size if self.backend != "onnx" else 1024
        lines_map, mask = _det_rearrange_forward(
            image_rgb,
            self._batch_forward,
            tgt_size=int(detect_size),
            max_batch_size=max(1, int(self.settings.det_rearrange_max_batches)),
            device=self.settings.device,
        )
        im_h, im_w = image_rgb.shape[:2]
        if lines_map is None:
            img_in, _ratio, dw, dh = _preprocess_img_rgb(
                image_rgb,
                detect_size=(detect_size, detect_size),
                device=self.settings.device,
                half=False,
                to_tensor=self.backend == "torch",
            )
            _blks, mask, _lines_map = self.net(img_in)
            if self.backend == "onnx" and mask.shape[1] == 2:
                mask, _lines_map = _lines_map, mask
            if "torch" in str(type(mask)).lower():
                mask = mask.detach()
                if getattr(mask, "device", None) is not None and str(mask.device) != "cpu":
                    mask = mask.cpu()
                mask = mask.numpy()
            else:
                mask = np.asarray(mask)
            mask = mask.squeeze()
            mask = mask[..., : mask.shape[0] - dh, : mask.shape[1] - dw]
        raw_mask = _postprocess_mask(mask)
        raw_mask = cv2.resize(raw_mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        if int(self.settings.mask_dilate_size) > 0:
            k = int(self.settings.mask_dilate_size)
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1), (k, k))
            raw_mask = cv2.dilate(raw_mask, element)
        return np.where(raw_mask > 0, 255, 0).astype(np.uint8)

    def _compose_by_blocks(self, image_rgb: np.ndarray, refined_mask: np.ndarray, blocks: list[TextBlock]) -> np.ndarray:
        composed = np.zeros_like(refined_mask)
        im_h, im_w = image_rgb.shape[:2]
        for block in blocks:
            if getattr(block, "xyxy", None) is None:
                continue
            if getattr(block, "text_class", "") == "text_bubble" and getattr(block, "bubble_xyxy", None) is not None:
                x1, y1, x2, y2 = [int(float(v)) for v in block.bubble_xyxy[:4]]
            else:
                x1, y1, x2, y2 = _enlarge_window(block.xyxy, im_w, im_h)
            if x2 <= x1 or y2 <= y1:
                continue
            block_slice = refined_mask[y1:y2, x1:x2]
            if not np.any(block_slice):
                continue
            composed[y1:y2, x1:x2] = cv2.bitwise_or(composed[y1:y2, x1:x2], block_slice)
        return np.where(composed > 0, 255, 0).astype(np.uint8)

    def refine(self, image_rgb: np.ndarray, blocks: Iterable[TextBlock]) -> MaskGenerationResult:
        block_list = list(blocks or [])
        raw_mask = self._infer_raw_mask(image_rgb)
        refined_mask = _refine_mask(image_rgb, raw_mask, block_list) if block_list else raw_mask.copy()
        composed_mask = self._compose_by_blocks(image_rgb, refined_mask, block_list) if block_list else refined_mask.copy()
        return MaskGenerationResult(
            raw_mask=np.where(raw_mask > 0, 255, 0).astype(np.uint8),
            refined_mask=np.where(refined_mask > 0, 255, 0).astype(np.uint8),
            protect_mask=np.zeros_like(raw_mask),
            final_mask=np.where(composed_mask > 0, 255, 0).astype(np.uint8),
            backend=self.backend,
            device=self.settings.device,
            fallback_used=False,
        )
