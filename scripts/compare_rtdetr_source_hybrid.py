from __future__ import annotations

import argparse
import importlib.machinery
import json
import subprocess
import sys
import types
from pathlib import Path
from types import MethodType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import imkit as imk
import numpy as np
from PIL import Image
from PySide6.QtGui import QColor

from modules.detection.rtdetr_v2_onnx import RTDetrV2ONNXDetection
from modules.inpainting.lama_variants import LaMaLarge512px
from modules.inpainting.schema import Config
from modules.source_parity_vendor import source_parity_blockwise_inpaint
from modules.source_compat import build_exact_legacy_bbox_mask
from modules.utils.download import ModelDownloader, ModelID

DEFAULT_SOURCE_ROOT = r"C:\Users\pjjpj\Desktop\openai_manga_translater\BallonsTranslator_dev_src_with_gitpython\이식\source_reference"
DEFAULT_IMAGES = [
    "banchmark_result_log/inpaint_debug/20260413_005001_sample-debug-export/japan/source_images/094_source.png",
    "banchmark_result_log/inpaint_debug/20260413_005001_sample-debug-export/japan/source_images/095_source.png",
    "banchmark_result_log/inpaint_debug/20260413_005001_sample-debug-export/japan/source_images/096_source.png",
    "banchmark_result_log/gemma_iq4nl_japan/20260411_160014_gemma_iq4nl_japan_fullgpu_suite/stage1/ov068-ngl14/attempt01_t07/corpus/101.png",
]
LEGACY_BBOX_COMMIT = "1aec275f578a739a18a802595a609d67607c8c08"


def _load_image_rgb(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image).astype(np.uint8)).save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)).save(path)


def _serialize_lines(lines: Any) -> list[list[list[int]]]:
    return [np.asarray(line, dtype=np.int32).tolist() for line in list(lines or [])]


def _serialize_current_blocks(blocks: list[Any]) -> list[dict[str, Any]]:
    serialized = []
    for idx, block in enumerate(list(blocks or [])):
        serialized.append(
            {
                "index": int(idx),
                "xyxy": [int(v) for v in list(np.asarray(getattr(block, "xyxy", [0, 0, 0, 0]), dtype=np.int32))],
                "lines": _serialize_lines(getattr(block, "lines", [])),
                "angle": int(getattr(block, "angle", 0) or 0),
                "text": str(getattr(block, "text", "") or ""),
                "direction": str(getattr(block, "direction", "") or ""),
                "text_class": str(getattr(block, "text_class", "") or ""),
            }
        )
    return serialized


def _load_exact_legacy_generate_mask():
    source = subprocess.check_output(
        ["git", "show", f"{LEGACY_BBOX_COMMIT}:modules/utils/image_utils.py"],
        cwd=str(PROJECT_ROOT),
        text=True,
    )
    namespace: dict[str, Any] = {
        "np": np,
        "imk": imk,
        "QColor": QColor,
    }
    exec(compile(source, f"{LEGACY_BBOX_COMMIT}:modules/utils/image_utils.py", "exec"), namespace)
    return namespace["generate_mask"]


EXACT_LEGACY_GENERATE_MASK = _load_exact_legacy_generate_mask()


def _build_reference_mask(image_rgb: np.ndarray, blocks: list[Any]) -> np.ndarray:
    return np.where(np.asarray(EXACT_LEGACY_GENERATE_MASK(image_rgb, blocks)) > 0, 255, 0).astype(np.uint8)


class _CurrentRunner:
    def __init__(self, device: str, precision: str, inpaint_size: int):
        self.detector = RTDetrV2ONNXDetection()
        self.detector.initialize(device=device, confidence_threshold=0.3)
        self.inpainter = LaMaLarge512px(
            device,
            backend="torch",
            runtime_device=device,
            inpaint_size=inpaint_size,
            precision=precision,
        )

    def run(self, image_rgb: np.ndarray) -> tuple[list[Any], np.ndarray, np.ndarray]:
        blocks = self.detector.detect(image_rgb)
        final_mask = build_exact_legacy_bbox_mask(image_rgb, blocks)
        cleaned = source_parity_blockwise_inpaint(
            image_rgb,
            final_mask.copy(),
            blocks,
            self.inpainter,
            Config(hd_strategy="Original"),
            check_need_inpaint=True,
        )
        return blocks, final_mask, np.asarray(cleaned).astype(np.uint8)


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
import cv2
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
    for name in ['termcolor', 'colorama', 'docx', 'docx2txt', 'piexif', 'tqdm', 'pillow_jxl', 'natsort']:
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


def save_rgb(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image).astype(np.uint8)).save(path)


def serialize_lines(lines):
    out = []
    for line in list(lines or []):
        out.append(np.asarray(line, dtype=np.int32).tolist())
    return out


def serialize_blocks(blocks):
    payload = []
    for idx, block in enumerate(list(blocks or [])):
        text_getter = getattr(block, 'get_text', None)
        payload.append({
            'index': int(idx),
            'xyxy': [int(v) for v in list(getattr(block, 'xyxy', [0, 0, 0, 0]))],
            'angle': int(getattr(block, 'angle', 0) or 0),
            'vertical': bool(getattr(block, 'vertical', False)),
            'src_is_vertical': bool(getattr(block, 'src_is_vertical', False)),
            'text': text_getter() if callable(text_getter) else str(getattr(block, 'text', '') or ''),
            'lines': serialize_lines(getattr(block, 'lines', [])),
        })
    return payload


def main():
    image_path = Path(sys.argv[2])
    mask_path = Path(sys.argv[3])
    blocks_path = Path(sys.argv[4])
    out_dir = Path(sys.argv[5])
    source_root = sys.argv[6]
    device = sys.argv[7]
    precision = sys.argv[8]
    inpaint_size = int(sys.argv[9])
    lama_path = sys.argv[10]

    install_external_packages(source_root)

    from utils.textblock import TextBlock
    from modules.inpaint.base import LamaLarge, load_lama_mpe

    image_rgb = np.array(Image.open(image_path))
    if image_rgb.ndim == 2:
        image_rgb = np.stack([image_rgb, image_rgb, image_rgb], axis=2)
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
    final_mask = np.array(Image.open(mask_path))
    payload = json.loads(blocks_path.read_text(encoding='utf-8'))

    blk_list = []
    for item in list(payload or []):
        blk = TextBlock(
            xyxy=[int(v) for v in list(item.get('xyxy', [0, 0, 0, 0]))],
            lines=[np.asarray(line, dtype=np.int32).tolist() for line in list(item.get('lines', []))],
            angle=int(item.get('angle', 0) or 0),
            text=[str(item.get('text', '') or '')],
        )
        if bool(item.get('vertical', False)):
            blk.vertical = True
        if bool(item.get('src_is_vertical', False)):
            blk.src_is_vertical = True
        blk_list.append(blk)

    inpainter = LamaLarge(device=device, inpaint_size=inpaint_size, precision=precision)
    def _load_model_exact(self):
        device_local = self.params['device']['value']
        precision_local = self.params['precision']['value']
        self.model = load_lama_mpe(lama_path, device='cpu', use_mpe=False, large_arch=True)
        self.moveToDevice(device_local, precision=precision_local)
    inpainter._load_model = MethodType(_load_model_exact, inpainter)
    cleaned = inpainter.inpaint(image_rgb, final_mask.copy(), blk_list)

    save_rgb(out_dir / 'cleaned.png', cleaned)
    (out_dir / 'blocks.json').write_text(json.dumps({'blocks': serialize_blocks(blk_list)}, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
'''


def _run_external_inpaint(*, image_path: Path, mask_path: Path, blocks_path: Path, out_dir: Path, source_root: str, device: str, precision: str, inpaint_size: int) -> None:
    child = out_dir / '_external_inpaint_child.py'
    child.write_text(EXTERNAL_CHILD, encoding='utf-8')
    cmd = [
        sys.executable,
        str(child),
        '--external-only',
        str(image_path.resolve()),
        str(mask_path.resolve()),
        str(blocks_path.resolve()),
        str(out_dir.resolve()),
        source_root,
        device,
        precision,
        str(inpaint_size),
        str(ModelDownloader.primary_path(ModelID.LAMA_LARGE_512PX)),
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def _mask_diff(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(a) != np.asarray(b)))


def _image_diff_pixels(a: np.ndarray, b: np.ndarray) -> int:
    diff = cv2.absdiff(np.asarray(a), np.asarray(b))
    if diff.ndim == 3:
        diff = np.max(diff, axis=2)
    return int(np.count_nonzero(diff))


def _write_diff_overlay(path: Path, source_image: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
    diff = cv2.absdiff(np.asarray(a), np.asarray(b))
    if diff.ndim == 3:
        diff = np.max(diff, axis=2)
    overlay = np.asarray(source_image).copy()
    overlay[np.where(diff > 0)] = np.array([255, 0, 255], dtype=np.uint8)
    _save_rgb(path, overlay)


def _run_compare(args) -> None:
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    runner = _CurrentRunner(args.device, args.precision, args.inpaint_size)
    summary = {
        'source_root': args.source_root,
        'device': args.device,
        'precision': args.precision,
        'inpaint_size': int(args.inpaint_size),
        'images': [],
    }

    for entry in args.images:
        image_path = (PROJECT_ROOT / entry).resolve()
        image_rgb = _load_image_rgb(image_path)
        blocks, current_final_mask, current_cleaned = runner.run(image_rgb)
        reference_mask = _build_reference_mask(image_rgb, blocks)

        image_out = out_root / image_path.stem
        current_dir = image_out / 'current'
        reference_dir = image_out / 'reference'
        current_dir.mkdir(parents=True, exist_ok=True)
        reference_dir.mkdir(parents=True, exist_ok=True)

        _save_rgb(current_dir / 'source.png', image_rgb)
        _save_mask(current_dir / 'raw_mask.png', current_final_mask)
        _save_mask(current_dir / 'final_mask.png', current_final_mask)
        _save_rgb(current_dir / 'cleaned.png', current_cleaned)

        blocks_payload = _serialize_current_blocks(blocks)
        blocks_path = image_out / 'rtdetr_blocks.json'
        blocks_path.write_text(json.dumps(blocks_payload, ensure_ascii=False, indent=2), encoding='utf-8')
        _save_mask(reference_dir / 'raw_mask.png', reference_mask)
        _save_mask(reference_dir / 'final_mask.png', reference_mask)
        _run_external_inpaint(
            image_path=image_path,
            mask_path=reference_dir / 'final_mask.png',
            blocks_path=blocks_path,
            out_dir=reference_dir,
            source_root=args.source_root,
            device=args.device,
            precision=args.precision,
            inpaint_size=args.inpaint_size,
        )
        reference_cleaned = _load_image_rgb(reference_dir / 'cleaned.png')

        _write_diff_overlay(image_out / 'mask_diff_overlay.png', image_rgb, current_final_mask, reference_mask)
        _write_diff_overlay(image_out / 'cleaned_diff_overlay.png', image_rgb, current_cleaned, reference_cleaned)

        record = {
            'image': image_path.name,
            'block_count': len(blocks_payload),
            'raw_mask_equal': bool(np.array_equal(current_final_mask, reference_mask)),
            'refined_mask_equal': bool(np.array_equal(current_final_mask, reference_mask)),
            'final_mask_equal': bool(np.array_equal(current_final_mask, reference_mask)),
            'cleaned_equal': bool(np.array_equal(current_cleaned, reference_cleaned)),
            'raw_mask_diff_pixels': _mask_diff(current_final_mask, reference_mask),
            'final_mask_diff_pixels': _mask_diff(current_final_mask, reference_mask),
            'cleaned_diff_pixels': _image_diff_pixels(current_cleaned, reference_cleaned),
            'artifacts': {
                'current_source': str((current_dir / 'source.png').relative_to(out_root)),
                'current_final_mask': str((current_dir / 'final_mask.png').relative_to(out_root)),
                'current_cleaned': str((current_dir / 'cleaned.png').relative_to(out_root)),
                'reference_final_mask': str((reference_dir / 'final_mask.png').relative_to(out_root)),
                'reference_cleaned': str((reference_dir / 'cleaned.png').relative_to(out_root)),
                'mask_diff_overlay': str((image_out / 'mask_diff_overlay.png').relative_to(out_root)),
                'cleaned_diff_overlay': str((image_out / 'cleaned_diff_overlay.png').relative_to(out_root)),
            },
        }
        summary['images'].append(record)

    (out_root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out_root)


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare RT-DETR + exact 1aec275 bbox mask + exact source LaMa against reference runners.')
    parser.add_argument('--external-only', action='store_true')
    parser.add_argument('--source-root', default=DEFAULT_SOURCE_ROOT)
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'banchmark_result_log' / 'source_parity_compare' / 'rtdetr_legacy_bbox_source_compare'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', default='bf16')
    parser.add_argument('--inpaint-size', type=int, default=1536)
    parser.add_argument('images', nargs='*', default=DEFAULT_IMAGES)
    args = parser.parse_args()
    if args.external_only:
        raise SystemExit('external mode is handled by generated child scripts only')
    _run_compare(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
