from __future__ import annotations

import numpy as np
import imkit as imk
from PIL import Image

from modules.utils.device import get_providers
from modules.utils.download import ModelDownloader, ModelID
from modules.utils.onnx import make_session

from .base import InpaintModel
from .schema import Config


class AOT(InpaintModel):
    name = "aot"
    pad_mod = 8
    min_size = 128
    max_size = 2048

    def init_model(self, device, **kwargs):
        self.backend = str(kwargs.get("backend", "torch") or "torch")
        self.runtime_device = str(kwargs.get("runtime_device", device) or device)
        self.inpaint_size = int(kwargs.get("inpaint_size", 2048) or 2048)
        self.device = self.runtime_device
        if self.backend == "onnx":
            ModelDownloader.get(ModelID.AOT_ONNX)
            onnx_path = ModelDownloader.primary_path(ModelID.AOT_ONNX)
            providers = get_providers(device)
            self.session = make_session(onnx_path, providers=providers)
            self.model = None
        else:
            ModelDownloader.get(ModelID.AOT_TORCH)
            local_path = ModelDownloader.primary_path(ModelID.AOT_TORCH)
            from .aot_torch_network import load_aot_model
            self.model = load_aot_model(local_path, self.device)
            self.session = None

    @staticmethod
    def is_downloaded() -> bool:
        return ModelDownloader.is_downloaded(ModelID.AOT_TORCH)

    def _resize_keep_aspect(self, img, target_size):
        max_dim = max(img.shape[:2])
        scale = float(target_size) / float(max_dim)
        new_size = (round(img.shape[1] * scale), round(img.shape[0] * scale))
        return imk.resize(img, new_size, mode=Image.Resampling.BILINEAR)

    def forward(self, image, mask, config: Config):
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        im_h, im_w = image.shape[:2]
        if max(image.shape[:2]) > self.inpaint_size:
            image = self._resize_keep_aspect(image, self.inpaint_size)
            mask = self._resize_keep_aspect(mask, self.inpaint_size)

        backend = getattr(self, "backend", "torch")
        if backend == "onnx":
            img_np = (image.astype(np.float32) / 127.5) - 1.0
            mask_np = (mask.astype(np.float32) / 255.0)
            mask_np = (mask_np >= 0.5).astype(np.float32)
            img_np = img_np * (1 - mask_np[..., None])
            img_nchw = np.transpose(img_np, (2, 0, 1))[np.newaxis, ...]
            mask_nchw = mask_np[np.newaxis, np.newaxis, ...]
            ort_inputs = {
                self.session.get_inputs()[0].name: img_nchw,
                self.session.get_inputs()[1].name: mask_nchw,
            }
            out = self.session.run(None, ort_inputs)[0]
            img_inpainted = ((out[0].transpose(1, 2, 0) + 1.0) * 127.5)
            img_inpainted = np.clip(np.round(img_inpainted), 0, 255).astype(np.uint8)
        else:
            import torch

            img_torch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            mask_torch = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() / 255.0
            mask_torch = torch.where(mask_torch >= 0.5, 1.0, 0.0)
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)
            img_torch = img_torch * (1 - mask_torch)
            with torch.no_grad():
                img_inpainted_torch = self.model(img_torch, mask_torch)
            img_inpainted = ((img_inpainted_torch.to(device="cpu", dtype=torch.float32).squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5)
            img_inpainted = np.clip(np.round(img_inpainted), 0, 255).astype(np.uint8)

        new_shape = img_inpainted.shape[:2]
        if new_shape[0] != im_h or new_shape[1] != im_w:
            img_inpainted = imk.resize(img_inpainted, (im_w, im_h), mode=Image.Resampling.BILINEAR)
        return img_inpainted
