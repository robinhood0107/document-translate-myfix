from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np

from modules.masking.ctd_refiner import _letterbox, _square_pad_resize
from modules.masking.ctd_vendor.ctd import TextDetBase, TextDetBaseDNN
from modules.masking.ctd_vendor.yolov5.yolov5_utils import non_max_suppression


class _ReferenceTextBlock:
    def __init__(self, **values: Any) -> None:
        for key, value in values.items():
            setattr(self, key, value)


class _ReferenceBase:
    def get_param_value(self, key: str) -> Any:
        return self.params[key]["value"]

    def all_model_loaded(self) -> bool:
        return getattr(self, "model", None) is not None


class _ReferenceLogger:
    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None


def _package(name: str, path: Path | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)] if path is not None else []
    return module


def load_ballons_ctbd_reference(ballons_root: Path):
    """Load the original CTBD module without importing Ballons' whole UI stack."""

    root = ballons_root.resolve()
    source = root / "ballontranslator" / "modules" / "textdetector" / "detector_ctbd.py"
    if not source.is_file():
        raise FileNotFoundError(f"Ballons CTBD reference source not found: {source}")

    package_names = {
        "ballontranslator": root / "ballontranslator",
        "ballontranslator.modules": root / "ballontranslator" / "modules",
        "ballontranslator.modules.textdetector": root
        / "ballontranslator"
        / "modules"
        / "textdetector",
        "ballontranslator.utils": root / "ballontranslator" / "utils",
    }
    for name, path in package_names.items():
        sys.modules[name] = _package(name, path)

    base = types.ModuleType("ballontranslator.modules.textdetector.base")
    base.register_textdetectors = lambda _name: (lambda value: value)
    base.TextDetectorBase = _ReferenceBase
    base.TextBlock = _ReferenceTextBlock
    base.DEVICE_SELECTOR = lambda: {"value": "cpu"}
    base.ProjImgTrans = object
    sys.modules[base.__name__] = base

    logger_module = types.ModuleType("ballontranslator.utils.logger")
    logger_module.logger = _ReferenceLogger()
    sys.modules[logger_module.__name__] = logger_module

    module_name = "ballontranslator.modules.textdetector.detector_ctbd"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError("unable to build the Ballons CTBD module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_ballons_ctbd_detector(module, session):
    detector = module.RTDetrV2TextDetector.__new__(module.RTDetrV2TextDetector)
    detector.params = copy.deepcopy(module.RTDetrV2TextDetector.params)
    detector.model = session
    detector.logger = _ReferenceLogger()
    detector.name = "ctbd"
    return detector


def reference_ctbd_single_image(module, detector, image_bgr: np.ndarray):
    bubbles, texts = detector._detect_single_slice(image_bgr)
    blocks, mask = detector._create_text_blocks_and_mask(image_bgr, texts, bubbles)
    block_records = [
        {
            "xyxy": tuple(map(int, getattr(block, "xyxy"))),
            "text_class": str(getattr(block, "text_class")),
            "bubble_xyxy": (
                tuple(map(int, getattr(block, "bubble_xyxy")))
                if getattr(block, "bubble_xyxy", None) is not None
                else None
            ),
        }
        for block in blocks
    ]
    return np.asarray(bubbles), np.asarray(texts), np.asarray(mask), block_records


def _stub_module(name: str, **values: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, value in values.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def load_ballons_ctd_inference_reference(ballons_root: Path):
    """Load the original CTD inference functions with only runtime-neutral stubs."""

    root = ballons_root.resolve()
    source = (
        root
        / "ballontranslator"
        / "modules"
        / "textdetector"
        / "ctd"
        / "inference.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Ballons CTD reference source not found: {source}")

    package_names = {
        "ballontranslator": root / "ballontranslator",
        "ballontranslator.modules": root / "ballontranslator" / "modules",
        "ballontranslator.modules.textdetector": root
        / "ballontranslator"
        / "modules"
        / "textdetector",
        "ballontranslator.modules.textdetector.ctd": root
        / "ballontranslator"
        / "modules"
        / "textdetector"
        / "ctd",
        "ballontranslator.modules.textdetector.yolov5": root
        / "ballontranslator"
        / "modules"
        / "textdetector"
        / "yolov5",
        "ballontranslator.utils": root / "ballontranslator" / "utils",
    }
    for name, path in package_names.items():
        sys.modules[name] = _package(name, path)

    _stub_module(
        "ballontranslator.modules.textdetector.ctd.basemodel",
        TextDetBase=TextDetBase,
        TextDetBaseDNN=TextDetBaseDNN,
    )

    class _UnusedRepresenter:
        def __init__(self, *args, **kwargs):
            pass

    _stub_module(
        "ballontranslator.modules.textdetector.db_utils",
        SegDetectorRepresenter=_UnusedRepresenter,
    )
    _stub_module(
        "ballontranslator.modules.textdetector.yolov5.yolov5_utils",
        non_max_suppression=non_max_suppression,
    )
    _stub_module(
        "ballontranslator.modules.textdetector.ctd.textmask",
        refine_mask=lambda image, mask, blocks, refine_mode=0: mask,
        refine_undetected_mask=lambda image, mask, refined, blocks, refine_mode=0: refined,
        REFINEMASK_INPAINT=0,
        REFINEMASK_ANNOTATION=1,
    )
    _stub_module(
        "ballontranslator.utils.io_utils",
        find_all_imgs=lambda *args, **kwargs: [],
        NumpyEncoder=object,
    )
    _stub_module(
        "ballontranslator.utils.imgproc_utils",
        letterbox=_letterbox,
        square_pad_resize=_square_pad_resize,
        xyxy2yolo=lambda value, *args, **kwargs: value,
        get_yololabel_strings=lambda *args, **kwargs: "",
    )
    _stub_module(
        "ballontranslator.utils.textblock",
        TextBlock=_ReferenceTextBlock,
        group_output=lambda *args, **kwargs: [],
        mit_merge_textlines=lambda *args, **kwargs: [],
    )
    _stub_module("tqdm", tqdm=lambda value, *args, **kwargs: value)

    module_name = "ballontranslator.modules.textdetector.ctd.inference"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError("unable to build the Ballons CTD inference module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_ballons_ctd_runtime_reference(ballons_root: Path):
    """Load Ballons' full CTD Python runtime while bypassing unrelated UI imports."""

    root = ballons_root.resolve()
    package_names = {
        "ballontranslator": root / "ballontranslator",
        "ballontranslator.modules": root / "ballontranslator" / "modules",
        "ballontranslator.modules.textdetector": root
        / "ballontranslator"
        / "modules"
        / "textdetector",
        "ballontranslator.utils": root / "ballontranslator" / "utils",
    }
    for name, path in package_names.items():
        sys.modules[name] = _package(name, path)
    _stub_module("pillow_jxl")
    _stub_module("natsort", natsorted=sorted)
    _stub_module("tqdm", tqdm=lambda value, *args, **kwargs: value)

    import importlib

    return importlib.import_module("ballontranslator.modules.textdetector.ctd.inference")


def reference_ctd_raw_mask(module, net, image_rgb: np.ndarray, detect_size: int) -> np.ndarray:
    import cv2

    height, width = image_rgb.shape[:2]
    backend = "opencv" if isinstance(net, TextDetBaseDNN) else "torch"
    effective_size = 1024 if backend == "opencv" else int(detect_size)
    image_input, _ratio, dw, dh = module.preprocess_img(
        image_rgb,
        bgr2rgb=False,
        detect_size=effective_size,
        device="cpu",
        half=False,
        to_tensor=backend == "torch",
    )
    _blocks, mask, lines = net(image_input)
    if backend == "opencv" and mask.shape[1] == 2:
        mask, lines = lines, mask
    mask = np.asarray(mask.detach().cpu().numpy() if hasattr(mask, "detach") else mask).squeeze()
    mask = mask[..., : mask.shape[0] - dh, : mask.shape[1] - dw]
    raw = module.postprocess_mask(mask)
    return cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)


def load_ballons_lama_runtime_reference(ballons_root: Path):
    """Load Ballons' original LaMa class without importing its UI stack."""

    root = ballons_root.resolve()
    source = root / "ballontranslator" / "modules" / "inpaint" / "inpaint_default.py"
    if not source.is_file():
        raise FileNotFoundError(f"Ballons LaMa reference source not found: {source}")

    package_names = {
        "ballontranslator": root / "ballontranslator",
        "ballontranslator.modules": root / "ballontranslator" / "modules",
        "ballontranslator.modules.inpaint": root
        / "ballontranslator"
        / "modules"
        / "inpaint",
        "ballontranslator.modules.textdetector": root
        / "ballontranslator"
        / "modules"
        / "textdetector",
        "ballontranslator.utils": root / "ballontranslator" / "utils",
    }
    for name, path in package_names.items():
        sys.modules[name] = _package(name, path)

    import torch
    from modules.source_parity_vendor.utils.imgproc_utils import resize_keepasp

    def smart_resize(image, size):
        height, width = map(int, size)
        return __import__("cv2").resize(image, (width, height))

    _stub_module(
        "ballontranslator.utils.imgproc_utils",
        resize_keepasp=resize_keepasp,
        smart_resize=smart_resize,
    )
    _stub_module(
        "ballontranslator.modules.base",
        DEFAULT_DEVICE="cpu",
        DEVICE_SELECTOR=lambda **_kwargs: {"value": "cpu"},
        TORCH_DTYPE_MAP={"fp32": torch.float32, "bf16": torch.bfloat16},
        BF16_SUPPORTED="cuda" if torch.cuda.is_available() else "cpu",
    )
    textdetector = sys.modules["ballontranslator.modules.textdetector"]
    textdetector.TextBlock = _ReferenceTextBlock

    class _InpainterBase(_ReferenceBase):
        def __init__(self, **_params):
            self.params = copy.deepcopy(getattr(self.__class__, "params", {}))
            self.logger = _ReferenceLogger()

        def updateParam(self, key, value):
            self.params[key]["value"] = value

    _stub_module(
        "ballontranslator.modules.inpaint.base",
        InpainterBase=_InpainterBase,
        register_inpainter=lambda _name: (lambda value: value),
    )

    module_name = "ballontranslator.modules.inpaint.inpaint_default"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError("unable to build the Ballons LaMa module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
