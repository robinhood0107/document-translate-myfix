import abc
from typing import Optional

import numpy as np
import imkit as imk
from PIL import Image
import logging
from contextlib import nullcontext

logger = logging.getLogger(__name__)

from ..utils.inpainting import (
    boxes_from_mask,
    resize_max_size,
    pad_img_to_modulo,
    # switch_mps_device,
)
from .schema import Config, HDStrategy


# 블록 하나가 만드는 float32 임시 배열의 목표 크기(바이트). 이미지 크기에 상한을
# 두는 게 아니라 임시 배열 크기에만 상한을 둔다.
_BLEND_CHUNK_WORKING_BYTES = 8 * 1024 * 1024


def blend_with_mask(
    result: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """``result*(mask/255) + image*(1-mask/255)`` 를 uint8 로 돌려준다.

    원래 구현은 ``result * (mask / 255) + image * (1 - (mask / 255))`` 한 줄이었다.
    ``mask`` 가 uint8 이라 ``mask / 255`` 가 float64 로 승격되고, 전체 크기 임시
    배열이 여섯 개나 만들어졌다. 4K 페이지(2160x3840x3) 기준으로 임시 배열
    하나가 190 MiB, 순간 사용량이 760 MiB 였고, 반환값마저 uint8(24 MiB)이 아니라
    float64(190 MiB)로 남아 하류로 8배 부풀어 전파됐다. 실제로 아래 오류로 죽었다.

        MemoryError: Unable to allocate 190. MiB for an array with
        shape (2160, 3840, 3) and data type float64

    여기서는 float32 로 계산하고 행 블록 단위로 잘라 쓴다. 임시 배열 크기가
    이미지 크기와 무관하게 일정해지므로, 어떤 해상도에서도 같은 메모리로 돈다.
    부드러운(0/255 가 아닌) 마스크의 의미는 그대로 보존한다.
    """

    image = np.asarray(image)
    result = np.asarray(result)
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]

    height = int(image.shape[0])
    # 블록 하나에서 살아 있는 float32 임시 배열은 alpha 와 chunk 두 개다.
    row_bytes = max(1, int(np.prod(image.shape[1:], dtype=np.int64)) * 4)
    rows_per_chunk = max(1, _BLEND_CHUNK_WORKING_BYTES // row_bytes)
    blended = np.empty(image.shape, dtype=np.uint8)
    for start in range(0, height, rows_per_chunk):
        stop = min(start + rows_per_chunk, height)
        alpha = mask[start:stop].astype(np.float32, copy=False) / np.float32(255.0)
        chunk = result[start:stop].astype(np.float32, copy=False) * alpha
        chunk += image[start:stop].astype(np.float32, copy=False) * (
            np.float32(1.0) - alpha
        )
        np.clip(chunk, 0.0, 255.0, out=chunk)
        np.rint(chunk, out=chunk)
        blended[start:stop] = chunk.astype(np.uint8, copy=False)
    return blended


class InpaintModel:
    name = "base"
    min_size: Optional[int] = None
    pad_mod = 8
    pad_to_square = False

    def __init__(self, device, **kwargs):
        """

        Args:
            device:
        """
        # device = switch_mps_device(self.name, device)
        self.device = device
        self.init_model(device, **kwargs)

    @abc.abstractmethod
    def init_model(self, device, **kwargs):
        ...

    @staticmethod
    @abc.abstractmethod
    def is_downloaded() -> bool:
        ...

    @abc.abstractmethod
    def forward(self, image, mask, config: Config):
        """Input images and output images have same size
        images: [H, W, C] RGB
        masks: [H, W, 1] 255 为 masks 区域
        return: BGR IMAGE
        """
        ...

    def _pad_forward(self, image, mask, config: Config):
        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_modulo(
            image, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )
        pad_mask = pad_img_to_modulo(
            mask, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )

        logger.info(f"final forward pad size: {pad_image.shape}")

        result = self.forward(pad_image, pad_mask, config)
        result = result[0:origin_height, 0:origin_width, :]

        result, image, mask = self.forward_post_process(result, image, mask, config)

        return blend_with_mask(result, image, mask)

    def forward_post_process(self, result, image, mask, config):
        return result, image, mask

    def __call__(self, image, mask, config: Config):
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: BGR IMAGE
        """
        # Only import torch if we're using a torch backend; otherwise avoid dependency
        backend = getattr(self, 'backend', 'torch')
        if backend == 'onnx':
            no_grad_ctx = nullcontext()
        else:
            try:
                import torch  # noqa
                no_grad_ctx = torch.no_grad()
            except ImportError as e:
                raise RuntimeError("Torch backend selected but torch is not installed. Install torch or use backend='onnx'.") from e
        with no_grad_ctx:
            inpaint_result = None
            logger.info(f"hd_strategy: {config.hd_strategy}")
            if config.hd_strategy == HDStrategy.CROP:
                if max(image.shape) > config.hd_strategy_crop_trigger_size:
                    logger.info(f"Run crop strategy")
                    boxes = boxes_from_mask(mask)
                    crop_result = []
                    for box in boxes:
                        crop_image, crop_box = self._run_box(image, mask, box, config)
                        crop_result.append((crop_image, crop_box))

                    inpaint_result = image
                    for crop_image, crop_box in crop_result:
                        x1, y1, x2, y2 = crop_box
                        inpaint_result[y1:y2, x1:x2, :] = crop_image

            elif config.hd_strategy == HDStrategy.RESIZE:
                if max(image.shape) > config.hd_strategy_resize_limit:
                    origin_size = image.shape[:2]
                    downsize_image = resize_max_size(
                        image, size_limit=config.hd_strategy_resize_limit
                    )
                    downsize_mask = resize_max_size(
                        mask, size_limit=config.hd_strategy_resize_limit
                    )

                    logger.info(
                        f"Run resize strategy, origin size: {image.shape} forward size: {downsize_image.shape}"
                    )
                    inpaint_result = self._pad_forward(
                        downsize_image, downsize_mask, config
                    )

                    # only paste masked area result
                    inpaint_result = imk.resize(
                        inpaint_result,
                        (origin_size[1], origin_size[0]),
                        mode=Image.Resampling.BICUBIC,
                    )
                    original_pixel_indices = mask < 127
                    inpaint_result[original_pixel_indices] = image[
                        original_pixel_indices
                    ]

            if inpaint_result is None:
                inpaint_result = self._pad_forward(image, mask, config)

            return inpaint_result

    def _crop_box(self, image, mask, box, config: Config):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE, (l, r, r, b)
        """
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        img_h, img_w = image.shape[:2]

        w = box_w + config.hd_strategy_crop_margin * 2
        h = box_h + config.hd_strategy_crop_margin * 2

        _l = cx - w // 2
        _r = cx + w // 2
        _t = cy - h // 2
        _b = cy + h // 2

        l = max(_l, 0)
        r = min(_r, img_w)
        t = max(_t, 0)
        b = min(_b, img_h)

        # try to get more context when crop around image edge
        if _l < 0:
            r += abs(_l)
        if _r > img_w:
            l -= _r - img_w
        if _t < 0:
            b += abs(_t)
        if _b > img_h:
            t -= _b - img_h

        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)

        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]

        logger.info(f"box size: ({box_h},{box_w}) crop size: {crop_img.shape}")

        return crop_img, crop_mask, [l, t, r, b]

    def _calculate_cdf(self, histogram):
        cdf = histogram.cumsum()
        normalized_cdf = cdf / float(cdf.max())
        return normalized_cdf

    def _calculate_lookup(self, source_cdf, reference_cdf):
        lookup_table = np.zeros(256)
        lookup_val = 0
        for source_index, source_val in enumerate(source_cdf):
            for reference_index, reference_val in enumerate(reference_cdf):
                if reference_val >= source_val:
                    lookup_val = reference_index
                    break
            lookup_table[source_index] = lookup_val
        return lookup_table

    def _match_histograms(self, source, reference, mask):
        transformed_channels = []
        for channel in range(source.shape[-1]):
            source_channel = source[:, :, channel]
            reference_channel = reference[:, :, channel]

            # only calculate histograms for non-masked parts
            source_histogram, _ = np.histogram(source_channel[mask == 0], 256, [0, 256])
            reference_histogram, _ = np.histogram(
                reference_channel[mask == 0], 256, [0, 256]
            )

            source_cdf = self._calculate_cdf(source_histogram)
            reference_cdf = self._calculate_cdf(reference_histogram)

            lookup = self._calculate_lookup(source_cdf, reference_cdf)

            transformed_channels.append(imk.lut(source_channel, lookup))

        result = imk.merge_channels(transformed_channels)
        result = imk.convert_scale_abs(result)

        return result

    def _apply_cropper(self, image, mask, config: Config):
        img_h, img_w = image.shape[:2]
        l, t, w, h = (
            config.croper_x,
            config.croper_y,
            config.croper_width,
            config.croper_height,
        )
        r = l + w
        b = t + h

        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)

        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]
        return crop_img, crop_mask, (l, t, r, b)

    def _run_box(self, image, mask, box, config: Config):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE
        """
        crop_img, crop_mask, [l, t, r, b] = self._crop_box(image, mask, box, config)

        return self._pad_forward(crop_img, crop_mask, config), [l, t, r, b]


class DiffusionInpaintModel(InpaintModel):
    def __call__(self, image, mask, config: Config):
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: BGR IMAGE
        """
        backend = getattr(self, 'backend', 'torch')
        if backend == 'onnx':
            no_grad_ctx = nullcontext()
        else:
            try:
                import torch  # noqa
                no_grad_ctx = torch.no_grad()
            except ImportError as e:
                raise RuntimeError("Torch backend selected but torch is not installed. Install torch or use backend='onnx'.") from e
        with no_grad_ctx:
            # boxes = boxes_from_mask(mask)
            if config.use_croper:
                crop_img, crop_mask, (l, t, r, b) = self._apply_cropper(image, mask, config)
                crop_image = self._scaled_pad_forward(crop_img, crop_mask, config)
                inpaint_result = image
                inpaint_result[t:b, l:r, :] = crop_image
            else:
                inpaint_result = self._scaled_pad_forward(image, mask, config)

            return inpaint_result

    def _scaled_pad_forward(self, image, mask, config: Config):
        longer_side_length = int(config.sd_scale * max(image.shape[:2]))
        origin_size = image.shape[:2]
        downsize_image = resize_max_size(image, size_limit=longer_side_length)
        downsize_mask = resize_max_size(mask, size_limit=longer_side_length)
        if config.sd_scale != 1:
            logger.info(
                f"Resize image to do sd inpainting: {image.shape} -> {downsize_image.shape}"
            )
        inpaint_result = self._pad_forward(downsize_image, downsize_mask, config)
        # only paste masked area result
        inpaint_result = imk.resize(
            inpaint_result,
            (origin_size[1], origin_size[0]),
            mode=Image.Resampling.BICUBIC,
        )
        original_pixel_indices = mask < 127
        inpaint_result[original_pixel_indices] = image[
            original_pixel_indices
        ]
        return inpaint_result
