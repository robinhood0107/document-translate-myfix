from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
import subprocess
import sys
import types
from pathlib import Path as _BootPath

PROJECT_ROOT = _BootPath(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from modules.inpainting.lama_variants import LaMaLarge512px
from modules.inpainting.schema import Config
from modules.source_parity_vendor import ExactSourceParityRuntime, source_parity_blockwise_inpaint
from modules.utils.download import ModelDownloader, ModelID


DEFAULT_SOURCE_ROOT = r"C:\Users\pjjpj\Desktop\openai_manga_translater\BallonsTranslator_dev_src_with_gitpython\이식\source_reference"
DEFAULT_IMAGES = [
    "banchmark_result_log/inpaint_debug/20260413_005001_sample-debug-export/japan/source_images/094_source.png",
    "banchmark_result_log/inpaint_debug/20260413_005001_sample-debug-export/japan/source_images/095_source.png",
    "banchmark_result_log/inpaint_debug/20260413_005001_sample-debug-export/japan/source_images/096_source.png",
    "banchmark_result_log/gemma_iq4nl_japan/20260411_160014_gemma_iq4nl_japan_fullgpu_suite/stage1/ov068-ngl14/attempt01_t07/corpus/101.png",
]


def _load_image_rgb(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    return arr


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image).astype(np.uint8)).save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)).save(path)


def _to_xyxy(obj: Any) -> list[int]:
    return [int(v) for v in list(getattr(obj, "xyxy", [0, 0, 0, 0]))]


def _serialize_lines(lines: Any) -> list[list[list[int]]]:
    out: list[list[list[int]]] = []
    for line in list(lines or []):
        out.append(np.asarray(line, dtype=np.int32).tolist())
    return out


def _serialize_current_blocks(result) -> list[dict[str, Any]]:
    details = result.mask_details
    source_blocks = details.get("source_blocks", []) or []
    serialized = []
    for idx, block in enumerate(source_blocks):
        text_getter = getattr(block, "get_text", None)
        serialized.append(
            {
                "index": int(idx),
                "xyxy": _to_xyxy(block),
                "language": str(getattr(block, "language", "unknown") or "unknown"),
                "vertical": bool(getattr(block, "vertical", False)),
                "src_is_vertical": bool(getattr(block, "src_is_vertical", False)),
                "angle": int(getattr(block, "angle", 0) or 0),
                "font_size": float(getattr(block, "font_size", -1) or -1),
                "detected_font_size": float(getattr(block, "_detected_font_size", -1) or -1),
                "text": text_getter() if callable(text_getter) else str(getattr(block, "text", "") or ""),
                "lines": _serialize_lines(getattr(block, "lines", [])),
            }
        )
    return serialized


def _build_cfg(device: str) -> dict[str, Any]:
    return {
        "ctd_detect_size": 1280,
        "ctd_det_rearrange_max_batches": 4,
        "ctd_device": device,
        "ctd_font_size_multiplier": 1.0,
        "ctd_font_size_max": -1,
        "ctd_font_size_min": -1,
        "ctd_mask_dilate_size": 2,
        "mask_inpaint_mode": "source_parity_ctd_lama",
    }


@dataclass
class ArtifactSet:
    name: str
    raw_mask: np.ndarray
    refined_mask: np.ndarray
    final_mask: np.ndarray
    cleaned: np.ndarray
    blocks: list[dict[str, Any]]
    meta: dict[str, Any]


class _CurrentRunner:
    def __init__(self, device: str, precision: str, inpaint_size: int):
        self.cfg = _build_cfg(device)
        self.runtime = ExactSourceParityRuntime()
        self.inpainter = LaMaLarge512px(
            device,
            backend="torch",
            runtime_device=device,
            inpaint_size=inpaint_size,
            precision=precision,
        )

    def run(self, image_rgb: np.ndarray) -> ArtifactSet:
        result = self.runtime.detect(image_rgb, self.cfg)
        details = result.mask_details
        final_mask = np.where(np.asarray(details["final_mask"]) > 0, 255, 0).astype(np.uint8)
        cleaned = source_parity_blockwise_inpaint(
            image_rgb,
            final_mask.copy(),
            result.blocks,
            self.inpainter,
            Config(hd_strategy="Original"),
            check_need_inpaint=True,
        )
        return ArtifactSet(
            name="current",
            raw_mask=np.where(np.asarray(details["raw_mask"]) > 0, 255, 0).astype(np.uint8),
            refined_mask=np.where(np.asarray(details["refined_mask"]) > 0, 255, 0).astype(np.uint8),
            final_mask=final_mask,
            cleaned=np.asarray(cleaned).astype(np.uint8),
            blocks=_serialize_current_blocks(result),
            meta={
                "backend": str(details.get("refiner_backend", "source_ctd")),
                "device": str(details.get("refiner_device", self.cfg["ctd_device"])),
            },
        )


EXTERNAL_CHILD = r'''
from __future__ import annotations
import importlib.machinery
import json
import os
import sys
import types
from pathlib import Path
from types import MethodType
import numpy as np
from PIL import Image


def install_external_packages(source_root: str):
    mods = os.path.join(source_root, 'modules')
    utils = os.path.join(source_root, 'utils')
    os.makedirs(os.path.join(source_root, 'translate'), exist_ok=True)
    for name, path in [
        ('modules', mods),
        ('modules.textdetector', os.path.join(mods, 'textdetector')),
        ('modules.inpaint', os.path.join(mods, 'inpaint')),
        ('utils', utils),
    ]:
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = mod
    for name in ['termcolor', 'colorama', 'docx', 'tqdm', 'pillow_jxl', 'natsort']:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        if name == 'termcolor':
            mod.colored = lambda text, *a, **k: text
        if name == 'colorama':
            mod.init = lambda *a, **k: None
        if name == 'tqdm':
            mod.tqdm = lambda iterable, *a, **k: iterable
        if name == 'natsort':
            mod.natsorted = lambda seq, *a, **k: sorted(seq)
        sys.modules[name] = mod
    from utils.textblock import TextBlock
    sys.modules['modules.textdetector'].TextBlock = TextBlock


def serialize_lines(lines):
    out = []
    for line in list(lines or []):
        out.append(np.asarray(line, dtype=np.int32).tolist())
    return out


def serialize_blocks(blocks):
    serialized = []
    for idx, block in enumerate(list(blocks or [])):
        text_getter = getattr(block, 'get_text', None)
        serialized.append({
            'index': int(idx),
            'xyxy': [int(v) for v in list(getattr(block, 'xyxy', [0, 0, 0, 0]))],
            'language': str(getattr(block, 'language', 'unknown') or 'unknown'),
            'vertical': bool(getattr(block, 'vertical', False)),
            'src_is_vertical': bool(getattr(block, 'src_is_vertical', False)),
            'angle': int(getattr(block, 'angle', 0) or 0),
            'font_size': float(getattr(block, 'font_size', -1) or -1),
            'detected_font_size': float(getattr(block, '_detected_font_size', -1) or -1),
            'text': text_getter() if callable(text_getter) else str(getattr(block, 'text', '') or ''),
            'lines': serialize_lines(getattr(block, 'lines', [])),
        })
    return serialized


def save_mask(path: Path, mask: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)).save(path)


def save_rgb(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image).astype(np.uint8)).save(path)


def main():
    image_path = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    source_root = sys.argv[4]
    device = sys.argv[5]
    precision = sys.argv[6]
    inpaint_size = int(sys.argv[7])
    ctd_torch_path = sys.argv[8]
    ctd_onnx_path = sys.argv[9]
    lama_path = sys.argv[10]

    install_external_packages(source_root)

    from modules.textdetector.ctd import CTDModel
    from modules.textdetector.ctd.textmask import REFINEMASK_INPAINT
    from modules.inpaint.base import LamaLarge, load_lama_mpe

    image_rgb = np.array(Image.open(image_path))
    if image_rgb.ndim == 2:
        image_rgb = np.stack([image_rgb, image_rgb, image_rgb], axis=2)
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        import cv2
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

    detect_size = 1280
    det_rearrange_max_batches = 4
    font_size_multiplier = 1.0
    font_size_max = -1
    font_size_min = -1
    mask_dilate_size = 2

    model_path = ctd_torch_path if str(device).lower() != 'cpu' else ctd_onnx_path
    model = CTDModel(model_path, detect_size=detect_size, device=device, det_rearrange_max_batches=det_rearrange_max_batches)
    raw_mask, refined_mask, blk_list = model(image_rgb, refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=False)

    for blk in blk_list:
        sz = float(getattr(blk, '_detected_font_size', -1) or -1)
        if sz <= 0:
            sz = float(getattr(blk, 'font_size', -1) or -1)
        if sz > 0:
            sz *= font_size_multiplier
            if font_size_max > 0:
                sz = min(font_size_max, sz)
            if font_size_min > 0:
                sz = max(font_size_min, sz)
            blk.font_size = sz
            blk._detected_font_size = sz

    raw_mask = np.where(np.asarray(raw_mask) > 0, 255, 0).astype(np.uint8)
    refined_mask = np.where(np.asarray(refined_mask) > 0, 255, 0).astype(np.uint8)
    final_mask = refined_mask.copy()
    if mask_dilate_size > 0:
        import cv2
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * mask_dilate_size + 1, 2 * mask_dilate_size + 1), (mask_dilate_size, mask_dilate_size))
        final_mask = cv2.dilate(final_mask, element)

    inpainter = LamaLarge(device=device, inpaint_size=inpaint_size, precision=precision)
    def _load_model_exact(self):
        device_local = self.params['device']['value']
        precision_local = self.params['precision']['value']
        self.model = load_lama_mpe(lama_path, device='cpu', use_mpe=False, large_arch=True)
        self.moveToDevice(device_local, precision=precision_local)
    inpainter._load_model = MethodType(_load_model_exact, inpainter)
    cleaned = inpainter.inpaint(image_rgb, final_mask.copy(), blk_list)

    save_mask(out_dir / 'raw_mask.png', raw_mask)
    save_mask(out_dir / 'refined_mask.png', refined_mask)
    save_mask(out_dir / 'final_mask.png', final_mask)
    save_rgb(out_dir / 'cleaned.png', cleaned)
    meta = {
        'blocks': serialize_blocks(blk_list),
        'device': device,
        'precision': precision,
        'inpaint_size': inpaint_size,
    }
    (out_dir / 'blocks.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
'''


def _image_diff_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim == 2:
        diff = cv2.absdiff(a, b)
    else:
        diff = np.max(cv2.absdiff(a, b), axis=2)
    return np.where(diff > 0, 255, 0).astype(np.uint8)


def _compare_blocks(current_blocks: list[dict[str, Any]], external_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    exact = current_blocks == external_blocks
    summary: dict[str, Any] = {
        "exact": bool(exact),
        "current_count": len(current_blocks),
        "external_count": len(external_blocks),
    }
    if exact:
        summary["first_difference"] = None
        return summary
    limit = min(len(current_blocks), len(external_blocks))
    first_difference = None
    for idx in range(limit):
        if current_blocks[idx] != external_blocks[idx]:
            first_difference = {
                "index": int(idx),
                "current": current_blocks[idx],
                "external": external_blocks[idx],
            }
            break
    if first_difference is None and len(current_blocks) != len(external_blocks):
        first_difference = {
            "index": int(limit),
            "current": current_blocks[limit] if len(current_blocks) > limit else None,
            "external": external_blocks[limit] if len(external_blocks) > limit else None,
        }
    summary["first_difference"] = first_difference
    return summary


def _run_external_subprocess(
    *,
    image_path: Path,
    out_dir: Path,
    source_root: str,
    device: str,
    precision: str,
    inpaint_size: int,
) -> None:
    python_exe = sys.executable
    image_path = image_path.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(Path(__file__).resolve()),
        "--external-only",
        str(image_path),
        str(out_dir),
        source_root,
        device,
        precision,
        str(inpaint_size),
        str(ModelDownloader.primary_path(ModelID.CTD_TORCH)),
        str(ModelDownloader.primary_path(ModelID.CTD_ONNX)),
        str(ModelDownloader.primary_path(ModelID.LAMA_LARGE_512PX)),
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def _load_external_artifacts(out_dir: Path) -> ArtifactSet:
    meta = json.loads((out_dir / "blocks.json").read_text(encoding="utf-8"))
    return ArtifactSet(
        name="external",
        raw_mask=np.array(Image.open(out_dir / "raw_mask.png")),
        refined_mask=np.array(Image.open(out_dir / "refined_mask.png")),
        final_mask=np.array(Image.open(out_dir / "final_mask.png")),
        cleaned=np.array(Image.open(out_dir / "cleaned.png")),
        blocks=list(meta.get("blocks", [])),
        meta={k: v for k, v in meta.items() if k != "blocks"},
    )


def _write_artifacts(base_dir: Path, artifact: ArtifactSet) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    _save_mask(base_dir / "raw_mask.png", artifact.raw_mask)
    _save_mask(base_dir / "refined_mask.png", artifact.refined_mask)
    _save_mask(base_dir / "final_mask.png", artifact.final_mask)
    _save_rgb(base_dir / "cleaned.png", artifact.cleaned)
    (base_dir / "blocks.json").write_text(
        json.dumps({"blocks": artifact.blocks, **artifact.meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_compare(args) -> None:
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    runner = _CurrentRunner(args.device, args.precision, args.inpaint_size)
    summary: dict[str, Any] = {
        "source_root": args.source_root,
        "device": args.device,
        "precision": args.precision,
        "inpaint_size": int(args.inpaint_size),
        "images": [],
    }

    for image_entry in args.images:
        image_path = Path(image_entry)
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        image_path = image_path.resolve()
        stem = image_path.stem.replace("_source", "")
        current_dir = out_root / stem / "current"
        external_dir = out_root / stem / "external"
        compare_dir = out_root / stem / "compare"
        compare_dir.mkdir(parents=True, exist_ok=True)

        image_rgb = _load_image_rgb(image_path)
        current = runner.run(image_rgb)
        _write_artifacts(current_dir, current)

        _run_external_subprocess(
            image_path=image_path,
            out_dir=external_dir,
            source_root=args.source_root,
            device=args.device,
            precision=args.precision,
            inpaint_size=args.inpaint_size,
        )
        external = _load_external_artifacts(external_dir)

        raw_diff = _image_diff_mask(current.raw_mask, external.raw_mask)
        refined_diff = _image_diff_mask(current.refined_mask, external.refined_mask)
        final_diff = _image_diff_mask(current.final_mask, external.final_mask)
        cleaned_diff = _image_diff_mask(current.cleaned, external.cleaned)

        _save_mask(compare_dir / "raw_mask_diff.png", raw_diff)
        _save_mask(compare_dir / "refined_mask_diff.png", refined_diff)
        _save_mask(compare_dir / "final_mask_diff.png", final_diff)
        _save_mask(compare_dir / "cleaned_diff.png", cleaned_diff)

        image_summary = {
            "image": str(image_path),
            "stem": stem,
            "raw_mask_equal": bool(not np.any(raw_diff)),
            "refined_mask_equal": bool(not np.any(refined_diff)),
            "final_mask_equal": bool(not np.any(final_diff)),
            "cleaned_equal": bool(not np.any(cleaned_diff)),
            "raw_mask_diff_pixels": int(np.count_nonzero(raw_diff)),
            "refined_mask_diff_pixels": int(np.count_nonzero(refined_diff)),
            "final_mask_diff_pixels": int(np.count_nonzero(final_diff)),
            "cleaned_diff_pixels": int(np.count_nonzero(cleaned_diff)),
            "blocks": _compare_blocks(current.blocks, external.blocks),
        }
        (compare_dir / "summary.json").write_text(json.dumps(image_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["images"].append(image_summary)

    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare current Source Parity runtime against external source_reference flow.")
    parser.add_argument("--images", nargs="*", default=DEFAULT_IMAGES)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--inpaint-size", type=int, default=1536)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for comparison artifacts. Defaults to banchmark_result_log/source_parity_compare/<timestamp>",
    )
    parser.add_argument("--external-only", action="store_true")
    return parser.parse_args()


def _default_output_dir() -> str:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("banchmark_result_log") / "source_parity_compare" / f"{stamp}_compare")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--external-only":
        exec(EXTERNAL_CHILD, {"__name__": "__main__"})
        return

    args = _parse_args()
    if args.output_dir is None:
        args.output_dir = _default_output_dir()
    _run_compare(args)


if __name__ == "__main__":
    main()
