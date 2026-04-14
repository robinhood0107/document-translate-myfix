#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imkit as imk
import numpy as np
from PIL import Image, ImageDraw

from modules.detection.processor import TextBlockDetector
from modules.ocr.mangalmm_ocr import (
    DEFAULT_MANGALMM_DEBUG_EXPORT_LIMIT,
    DEFAULT_MANGALMM_SERVER_URL,
    DEFAULT_MANGALMM_TEMPERATURE,
    DEFAULT_MANGALMM_TOP_K,
    MangaLMMOCREngine,
)
from modules.utils.ocr_debug import ensure_three_channel


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class _UIStub:
    value_mappings: dict[str, str]

    def tr(self, text: str) -> str:
        return text


class _SettingsStub:
    def __init__(
        self,
        *,
        use_gpu: bool,
        server_url: str,
        temperature: float,
        top_k: int,
        max_completion_tokens: int,
    ) -> None:
        self._use_gpu = use_gpu
        self._server_url = server_url
        self._temperature = temperature
        self._top_k = top_k
        self._max_completion_tokens = max_completion_tokens
        self.ui = _UIStub(value_mappings={})

    def get_tool_selection(self, tool_type: str) -> str:
        if tool_type == "detector":
            return "RT-DETR-v2"
        raise KeyError(tool_type)

    def is_gpu_enabled(self) -> bool:
        return self._use_gpu

    def get_credentials(self, _provider_name: str) -> dict:
        return {}

    def get_mangalmm_ocr_settings(self) -> dict:
        return {
            "server_url": self._server_url,
            "max_completion_tokens": self._max_completion_tokens,
            "parallel_workers": 1,
            "request_timeout_sec": 300,
            "raw_response_logging": False,
            "safe_resize": True,
            "max_pixels": 2_116_800,
            "max_long_side": 1728,
            "temperature": self._temperature,
            "top_k": self._top_k,
        }


def _iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _normalize_box(value) -> list[int] | None:
    if value is None:
        return None
    try:
        coords = [int(float(item)) for item in value]
    except Exception:
        return None
    return coords if len(coords) == 4 else None


def _draw_overlay(image: np.ndarray, blocks, page_regions: list[dict[str, object]]) -> np.ndarray:
    canvas = ensure_three_channel(image)
    overlay = Image.fromarray(canvas)
    draw = ImageDraw.Draw(overlay)

    for blk in blocks:
        text_box = _normalize_box(getattr(blk, "xyxy", None))
        bubble_box = _normalize_box(getattr(blk, "bubble_xyxy", None))
        if bubble_box is not None:
            draw.rectangle(bubble_box, outline=(0, 128, 255), width=3)
        if text_box is not None:
            draw.rectangle(text_box, outline=(40, 220, 40), width=3)

    for region in page_regions:
        box = _normalize_box(region.get("bbox_xyxy"))
        if box is None:
            continue
        draw.rectangle(box, outline=(255, 64, 64), width=3)

    return np.asarray(overlay)


def _serialize_block(blk) -> dict[str, object]:
    return {
        "xyxy": _normalize_box(getattr(blk, "xyxy", None)),
        "bubble_xyxy": _normalize_box(getattr(blk, "bubble_xyxy", None)),
        "text": str(getattr(blk, "text", "") or ""),
        "texts": [str(item or "") for item in getattr(blk, "texts", []) or []],
        "ocr_status": str(getattr(blk, "ocr_status", "") or ""),
        "ocr_empty_reason": str(getattr(blk, "ocr_empty_reason", "") or ""),
        "ocr_crop_bbox": _normalize_box(getattr(blk, "ocr_crop_bbox", None)),
        "ocr_resize_scale": float(getattr(blk, "ocr_resize_scale", 1.0) or 1.0),
        "ocr_regions": list(getattr(blk, "ocr_regions", []) or []),
    }


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imk.write_image(str(path), ensure_three_channel(image))


def _default_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "banchmark_result_log" / "mangalmm_fullpage_ocr_debug" / f"{timestamp}_mangalmm_fullpage_ocr_debug"


def _process_image(
    *,
    image_path: Path,
    output_root: Path,
    detector: TextBlockDetector,
    engine: MangaLMMOCREngine,
) -> dict[str, object]:
    image = imk.read_image(str(image_path))
    if image is None:
        raise RuntimeError(f"failed to read {image_path.name}")
    image = ensure_three_channel(image)
    page_output = output_root / image_path.stem
    page_output.mkdir(parents=True, exist_ok=True)

    engine.debug_root = page_output / "engine_debug"
    engine.debug_root.mkdir(parents=True, exist_ok=True)
    engine.debug_export_limit = DEFAULT_MANGALMM_DEBUG_EXPORT_LIMIT
    engine._debug_export_counter = count(1)

    blocks = detector.detect(image) or []
    failure = ""
    try:
        engine.process_image(image, blocks)
    except Exception as exc:
        failure = str(exc)

    overlay = _draw_overlay(image, blocks, engine.last_page_regions)
    metadata = {
        "image": image_path.name,
        "source_path": str(image_path),
        "block_count": len(blocks),
        "mapped_region_count": len(engine.last_page_regions),
        "non_empty_block_count": sum(1 for blk in blocks if str(getattr(blk, "text", "") or "").strip()),
        "status": "success" if not failure and engine.last_page_regions else "failure",
        "failure": failure,
        "request": dict(engine.last_request_metadata),
        "blocks": [_serialize_block(blk) for blk in blocks],
    }

    _write_image(page_output / "source.png", image)
    _write_image(page_output / "mapped_overlay.png", overlay)
    _write_json(page_output / "page_summary.json", metadata)
    _write_text(page_output / "raw_response.txt", str(engine.last_request_metadata.get("raw_response", "") or ""))

    return {
        "image": image_path.name,
        "status": metadata["status"],
        "block_count": metadata["block_count"],
        "mapped_region_count": metadata["mapped_region_count"],
        "non_empty_block_count": metadata["non_empty_block_count"],
        "resize_profile": metadata["request"].get("resize_profile", ""),
        "request_shape": metadata["request"].get("request_shape", []),
        "output_dir": str(page_output),
        "failure": failure,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug full-page single-shot MangaLMM OCR with Sample/simpletest.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "Sample" / "simpletest",
        help="Input directory with page images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults under banchmark_result_log/mangalmm_fullpage_ocr_debug/.",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_MANGALMM_SERVER_URL,
        help="MangaLMM server URL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_MANGALMM_TEMPERATURE,
        help=f"Sampling temperature. Default is {DEFAULT_MANGALMM_TEMPERATURE}.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_MANGALMM_TOP_K,
        help=f"Top-k sampler value. Default is {DEFAULT_MANGALMM_TOP_K}.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=256,
        help="Base max completion tokens before profile floors are applied.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU detection runtime.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise SystemExit(f"input directory not found: {input_dir}")

    output_root = args.output_dir.resolve() if args.output_dir is not None else _default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)

    settings = _SettingsStub(
        use_gpu=not args.cpu,
        server_url=args.server_url,
        temperature=args.temperature,
        top_k=args.top_k,
        max_completion_tokens=args.max_completion_tokens,
    )

    detector = TextBlockDetector(settings)
    engine = MangaLMMOCREngine()
    engine.initialize(settings)

    images = _iter_images(input_dir)
    if not images:
        raise SystemExit(f"no images found under {input_dir}")

    results: list[dict[str, object]] = []
    for image_path in images:
        result = _process_image(
            image_path=image_path,
            output_root=output_root,
            detector=detector,
            engine=engine,
        )
        results.append(result)
        print(
            f"{result['image']}: status={result['status']} "
            f"profile={result['resize_profile']} request_shape={result['request_shape']} "
            f"regions={result['mapped_region_count']} output={result['output_dir']}"
        )
        if result["failure"]:
            print(f"  failure={result['failure']}")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_root),
        "server_url": args.server_url,
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "results": results,
    }
    _write_json(output_root / "summary.json", summary)
    print(f"summary={output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
