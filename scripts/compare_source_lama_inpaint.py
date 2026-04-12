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
import numpy as np
from PIL import Image

from modules.inpainting.lama_variants import LaMaLarge512px
from modules.inpainting.schema import Config
from modules.source_parity_vendor import source_parity_blockwise_inpaint
from modules.utils.textblock import TextBlock
from modules.utils.download import ModelDownloader, ModelID

DEFAULT_SOURCE_ROOT = r"C:\Users\pjjpj\Desktop\openai_manga_translater\BallonsTranslator_dev_src_with_gitpython\이식\source_reference"


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


def _deserialize_blocks(path: Path) -> list[TextBlock]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    blocks = []
    for item in list(payload or []):
        blocks.append(
            TextBlock(
                text_bbox=np.asarray(item.get('xyxy', [0, 0, 0, 0]), dtype=np.int32),
                lines=[np.asarray(line, dtype=np.int32).tolist() for line in list(item.get('lines', []))],
                text_class=str(item.get('text_class', '') or ''),
                angle=int(item.get('angle', 0) or 0),
                text=str(item.get('text', '') or ''),
                direction=str(item.get('direction', '') or 'horizontal'),
            )
        )
    return blocks


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


def main():
    image_path = Path(sys.argv[2])
    mask_path = Path(sys.argv[3])
    blocks_path = Path(sys.argv[4])
    out_path = Path(sys.argv[5])
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
        if str(item.get('direction', '') or '') == 'vertical':
            blk.vertical = True
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
    save_rgb(out_path, cleaned)

if __name__ == '__main__':
    main()
'''


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare current shared source LaMa runtime with external source inpaint on the same image/mask/block list.')
    parser.add_argument('image')
    parser.add_argument('mask')
    parser.add_argument('blocks')
    parser.add_argument('--source-root', default=DEFAULT_SOURCE_ROOT)
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'banchmark_result_log' / 'source_parity_compare' / 'inpaint_only_compare'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', default='bf16')
    parser.add_argument('--inpaint-size', type=int, default=1536)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image = _load_image_rgb(Path(args.image))
    mask = np.where(np.asarray(Image.open(args.mask)) > 0, 255, 0).astype(np.uint8)
    blocks = _deserialize_blocks(Path(args.blocks))

    inpainter = LaMaLarge512px(
        args.device,
        backend='torch',
        runtime_device=args.device,
        inpaint_size=args.inpaint_size,
        precision=args.precision,
    )
    current = source_parity_blockwise_inpaint(image, mask.copy(), blocks, inpainter, Config(hd_strategy='Original'), check_need_inpaint=True)
    current = np.asarray(current).astype(np.uint8)
    current_path = out_dir / 'current_cleaned.png'
    _save_rgb(current_path, current)

    child = out_dir / '_external_source_lama_child.py'
    child.write_text(EXTERNAL_CHILD, encoding='utf-8')
    reference_path = out_dir / 'reference_cleaned.png'
    cmd = [
        sys.executable,
        str(child),
        '--external-only',
        str(Path(args.image).resolve()),
        str(Path(args.mask).resolve()),
        str(Path(args.blocks).resolve()),
        str(reference_path.resolve()),
        args.source_root,
        args.device,
        args.precision,
        str(args.inpaint_size),
        str(ModelDownloader.primary_path(ModelID.LAMA_LARGE_512PX)),
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    reference = _load_image_rgb(reference_path)

    diff = cv2.absdiff(current, reference)
    diff_mask = np.max(diff, axis=2) if diff.ndim == 3 else diff
    overlay = image.copy()
    overlay[np.where(diff_mask > 0)] = np.array([255, 0, 255], dtype=np.uint8)
    _save_rgb(out_dir / 'diff_overlay.png', overlay)

    summary = {
        'image': str(Path(args.image).resolve()),
        'mask': str(Path(args.mask).resolve()),
        'blocks': str(Path(args.blocks).resolve()),
        'equal': bool(np.array_equal(current, reference)),
        'diff_pixels': int(np.count_nonzero(diff_mask)),
        'current_cleaned': str(current_path),
        'reference_cleaned': str(reference_path),
        'diff_overlay': str(out_dir / 'diff_overlay.png'),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
