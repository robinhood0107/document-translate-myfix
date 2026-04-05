"""
PaddleOCR-VL-1.5 (0.9B) via Hugging Face Transformers.
Element-level recognition (OCR / table / chart / formula / spotting / seal).

This module intentionally keeps the same flow as `ocr_paddleocr_vl_hf.py` and
only applies 1.5-specific input changes from the official transformers sample:
- default model id: PaddlePaddle/PaddleOCR-VL-1.5
- task prompt selector
- spotting upscale rule
- images_kwargs size control
"""
from typing import List
import os
import tempfile
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

_PADDLEOCR_VL_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    import torch

    _PADDLEOCR_VL_AVAILABLE = True
except ImportError as e:
    import logging

    logging.getLogger("BallonTranslator").debug(
        f"PaddleOCR-VL-1.5 (HF) not available: {e}. Install: pip install \"transformers>=5.0.0\" torch pillow"
    )


PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _PADDLEOCR_VL_AVAILABLE:

    @register_OCR("paddleocr_vl_15_hf_personal")
    class PaddleOCRVL15HFOCR(OCRBase):
        """
        PaddleOCR-VL-1.5 via Hugging Face (element-level recognition).
        Keeps base flow compatible with existing paddleocr_vl_hf module.
        """

        params = {
            "model_name": {
                "type": "line_editor",
                "value": "PaddlePaddle/PaddleOCR-VL-1.5",
                "description": "Hugging Face model id for PaddleOCR-VL-1.5 (transformers 5.x).",
            },
            "task": {
                "type": "selector",
                "options": ["ocr", "table", "chart", "formula", "spotting", "seal"],
                "value": "ocr",
                "description": "Task prompt type used by PaddleOCR-VL-1.5.",
            },
            "spotting_upscale_threshold": {
                "type": "line_editor",
                "value": 1500,
                "description": "For spotting task, upscale x2 when both width and height are below threshold.",
            },
            "device": DEVICE_SELECTOR(),
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each box when cropping (0-24).",
            },
            "max_new_tokens": {
                "type": "line_editor",
                "value": 256,
                "description": "Max tokens per block.",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available (recommended).",
            },
            "use_flash_attn": {
                "type": "checkbox",
                "value": False,
                "description": "Use flash_attention_2 when available.",
            },
            "description": "PaddleOCR-VL-1.5 (HF transformers) with original module flow.",
        }
        _load_model_keys = {"processor", "model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.processor = None
            self.model = None
            self._model_name = None
            self._load_signature = None

        def _load_model(self):
            model_name = (
                (self.params.get("model_name") or {}).get("value", "PaddlePaddle/PaddleOCR-VL-1.5")
                or "PaddlePaddle/PaddleOCR-VL-1.5"
            )
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            use_flash_attn = self.params.get("use_flash_attn", {}).get("value", False)
            dtype = torch.bfloat16 if (
                use_bf16 and torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            ) else torch.float16
            if str(self.device) == "cpu":
                dtype = torch.float32

            load_sig = (model_name, bool(use_flash_attn), str(dtype), self.device)
            if self.processor is not None and self.model is not None and self._load_signature == load_sig:
                return

            self._model_name = model_name
            self.processor = AutoProcessor.from_pretrained(model_name)
            load_kwargs = {"torch_dtype": dtype}
            if use_flash_attn:
                load_kwargs["attn_implementation"] = "flash_attention_2"
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
            except Exception as e:
                if use_flash_attn:
                    self.logger.warning(
                        f"PaddleOCR-VL-1.5 load with flash-attn failed: {e}; retrying without flash-attn."
                    )
                    load_kwargs.pop("attn_implementation", None)
                    try:
                        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
                    except Exception as e2:
                        self.logger.warning(f"PaddleOCR-VL-1.5 load with dtype failed: {e2}; trying default dtype.")
                        self.model = AutoModelForImageTextToText.from_pretrained(model_name)
                else:
                    self.logger.warning(f"PaddleOCR-VL-1.5 load with dtype failed: {e}; trying default dtype.")
                    self.model = AutoModelForImageTextToText.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self._load_signature = load_sig

        def _run_one(self, pil_img: "Image.Image") -> str:
            tmp_path = None
            try:
                task = str((self.params.get("task") or {}).get("value", "ocr") or "ocr").strip().lower()
                if task not in PROMPTS:
                    task = "ocr"
                ocr_prompt = PROMPTS[task]

                proc_img = pil_img
                max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28
                if task == "spotting":
                    try:
                        spotting_threshold = int(
                            (self.params.get("spotting_upscale_threshold") or {}).get("value", 1500)
                        )
                    except (TypeError, ValueError):
                        spotting_threshold = 1500
                    ow, oh = proc_img.size
                    if ow < spotting_threshold and oh < spotting_threshold:
                        try:
                            resample_filter = Image.Resampling.LANCZOS
                        except AttributeError:
                            resample_filter = Image.LANCZOS
                        proc_img = proc_img.resize((ow * 2, oh * 2), resample_filter)

                size_dict = {"longest_edge": max_pixels}
                min_pixels = getattr(getattr(self.processor, "image_processor", None), "min_pixels", None)
                if min_pixels is not None:
                    size_dict["shortest_edge"] = min_pixels
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": proc_img},
                            {"type": "text", "text": ocr_prompt},
                        ],
                    }
                ]
                try:
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        images_kwargs={"size": size_dict},
                    )
                except Exception as image_template_err:
                    # Keep backward compatibility with processors that only accept URL image payloads.
                    self.logger.debug(
                        f"PaddleOCR-VL-1.5 image payload failed: {image_template_err}; retrying with URL payload."
                    )
                    fd, tmp_path = tempfile.mkstemp(suffix=".png")
                    os.close(fd)
                    proc_img.save(tmp_path)
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "url": tmp_path},
                                {"type": "text", "text": ocr_prompt},
                            ],
                        }
                    ]
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        images_kwargs={"size": size_dict},
                    )
                inputs = inputs.to(self.model.device)

                max_tokens = 256
                mt = self.params.get("max_new_tokens", {})
                if isinstance(mt, dict):
                    try:
                        max_tokens = max(32, min(1024, int(mt.get("value", 256))))
                    except (TypeError, ValueError):
                        pass

                generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
                in_ids = inputs["input_ids"]
                out_ids = generated_ids[0]
                generated_ids_trimmed = out_ids[in_ids.shape[-1] :]
                text = self.processor.decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                return text
            except Exception as e:
                self.logger.warning(f"PaddleOCR-VL-1.5 HF failed: {e}")
                return ""
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
            im_h, im_w = img.shape[:2]
            pad = 0
            cp = self.params.get("crop_padding", {})
            if isinstance(cp, dict):
                val = cp.get("value", 0)
            else:
                val = 0
            try:
                pad = max(0, min(24, int(val)))
            except (TypeError, ValueError):
                pass
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                x1 = max(0, min(int(round(float(x1))), im_w - 1))
                y1 = max(0, min(int(round(float(y1))), im_h - 1))
                x2 = max(x1 + 1, min(int(round(float(x2))), im_w))
                y2 = max(y1 + 1, min(int(round(float(y2))), im_h))
                if pad > 0:
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(im_w, x2 + pad)
                    y2 = min(im_h, y2 + pad)
                if not (x1 < x2 and y1 < y2 and x2 <= im_w and y2 <= im_h and x1 >= 0 and y1 >= 0):
                    blk.text = [""]
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    blk.text = [""]
                    continue
                pil_img = _cv2_to_pil_rgb(crop)
                text = self._run_one(pil_img)
                blk.text = [text if text else ""]

        def ocr_img(self, img: np.ndarray) -> str:
            pil_img = _cv2_to_pil_rgb(img)
            return self._run_one(pil_img)

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key == "device":
                self.device = self.params["device"]["value"]
                if self.model is not None:
                    self.model.to(self.device)
            elif param_key in ("model_name", "use_bf16", "use_flash_attn"):
                self.processor = None
                self.model = None
                self._model_name = None
                self._load_signature = None
